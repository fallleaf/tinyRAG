#!/usr/bin/env python3
"""
mcp_server/server.py - Production MCP RAG Server (v1.1.2)

tinyRAG v1.1.2 稳定版本功能:
1. ✅ Tools 接口: search, scan_index, rebuild_index
2. ✅ Resources 接口: 知识库统计信息、文档内容
3. ✅ Prompts 接口: 检索增强提示词模板

修复记录:
- F1: read_resource 的 uri 参数 AnyUrl 类型转换
- F2: summarize_document 路径多级匹配策略
"""

import asyncio
import contextlib
import importlib.util
import json
import os
import re
import sys
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, ClassVar

from utils.logger import setup_logger

# MCP - 导入所有需要的类型
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import (
        TextContent,
        Tool,
        Resource,
        ResourceTemplate,
        Prompt,
        PromptArgument,
        PromptMessage,
        GetPromptResult,
    )

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

from chunker.markdown_splitter import MarkdownSplitter
from config import Settings, get_merged_exclude, load_config
from embedder.embed_engine import EmbeddingEngine
from retriever.hybrid_engine import HybridEngine
from scanner.scan_engine import Scanner
from storage.database import DatabaseManager

# =====================
# Logger & Config
# =====================
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
log_file = PROJECT_ROOT / "logs" / "mcp_server.log"
logger = setup_logger(level="INFO", log_file=str(log_file))

_BUILD_INDEX_MODULE_KEY = "_tinyrag_build_index"


def _load_build_index_main():
    if _BUILD_INDEX_MODULE_KEY in sys.modules:
        return sys.modules[_BUILD_INDEX_MODULE_KEY].main

    module_path = PROJECT_ROOT / "build_index.py"
    if not module_path.exists():
        raise ImportError(f"build_index.py 未找到: {module_path}")

    spec = importlib.util.spec_from_file_location(_BUILD_INDEX_MODULE_KEY, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[_BUILD_INDEX_MODULE_KEY] = module
    spec.loader.exec_module(module)
    return module.main


# =====================
# App Context
# =====================
class AppContext:
    def __init__(self):
        self.config: Settings | None = None
        self.db: DatabaseManager | None = None
        self.scanner: Scanner | None = None
        self.retriever: HybridEngine | None = None
        self.splitter: MarkdownSplitter | None = None
        self.vault_excludes: dict[str, tuple[frozenset[str], list[str]]] = {}
        self._initialized = False
        self._lock = asyncio.Lock()
        self._background_tasks: list[asyncio.Task] = []

    async def initialize(self):
        if self._initialized:
            return
        async with self._lock:
            if self._initialized:
                return

        config = None
        db = None
        scanner = None
        retriever = None
        splitter = None
        vault_excludes = {}
        try:
            config_path = PROJECT_ROOT / "config.yaml"
            config = load_config(str(config_path))

            db = DatabaseManager(config.db_path, vec_dimension=config.embedding_model.dimensions)

            # 合并默认跳过目录和全局排除目录
            from scanner.scan_engine import DEFAULT_SKIP_DIRS

            global_skip_dirs = DEFAULT_SKIP_DIRS | frozenset(config.exclude.dirs)
            scanner = Scanner(db, global_skip_dirs, config.exclude.patterns)

            for v in config.vaults:
                if v.enabled:
                    if v.exclude:
                        vault_excludes[v.name] = (frozenset(v.exclude.dirs), v.exclude.patterns)
                    else:
                        vault_excludes[v.name] = (frozenset(), [])

            embed_engine = EmbeddingEngine(
                model_name=config.embedding_model.name,
                cache_dir=config.embedding_model.cache_dir,
                batch_size=config.embedding_model.batch_size,
                unload_after_seconds=config.embedding_model.unload_after_seconds,
            )

            retriever = HybridEngine(config=config, db=db, embed_engine=embed_engine)
            splitter = MarkdownSplitter(config)

            self.config = config
            self.db = db
            self.scanner = scanner
            self.retriever = retriever
            self.splitter = splitter
            self.vault_excludes = vault_excludes
            self._initialized = True
            logger.info("MCP components initialized successfully")
        except Exception:
            if db is not None:
                with contextlib.suppress(Exception):
                    db.close()
            logger.critical("MCP initialization failed", exc_info=True)
            raise

    def add_background_task(self, task: asyncio.Task):
        self._background_tasks.append(task)

        def _on_done(t: asyncio.Task):
            with contextlib.suppress(ValueError):
                self._background_tasks.remove(t)

        task.add_done_callback(_on_done)

    async def shutdown(self):
        logger.info("Shutting down server...")
        if self._background_tasks:
            for t in self._background_tasks:
                t.cancel()
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._background_tasks.clear()

        if self.db:
            try:
                self.db.close()
            except Exception as e:
                logger.warning(f"Database close error (ignored): {e}")
        logger.info("Resources released")


# =====================
# Error Wrapper & Tool Base
# =====================
def mcp_safe(func: Callable[..., Awaitable[list[TextContent]]]):
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Tool error: {e}", exc_info=True)
            return [TextContent(type="text", text=json.dumps({"error": str(e)}, ensure_ascii=False))]

    return wrapper


class BaseTool:
    name: str
    description: str
    schema: dict[str, Any]

    def __init__(self, ctx: AppContext):
        self.ctx = ctx

    async def run(self, args: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def to_mcp_tool(self) -> Tool:
        return Tool(name=self.name, description=self.description, inputSchema=self.schema)


# =====================
# Tools
# =====================
class SearchTool(BaseTool):
    name, description = "search", "Hybrid knowledge retrieval with RRF fusion (tinyRAG)"
    schema: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "mode": {
                "type": "string",
                "enum": ["semantic", "keyword", "hybrid"],
                "default": "hybrid",
            },
            "top_k": {"type": "integer", "default": 10, "minimum": 1, "maximum": 100},
        },
        "required": ["query"],
    }

    async def run(self, args: dict[str, Any]) -> dict[str, Any]:
        await self.ctx.initialize()
        query = args.get("query", "")
        mode = args.get("mode", "hybrid")
        top_k = min(max(args.get("top_k", 10), 1), 100)

        if mode == "keyword":
            alpha, beta = 0.0, 1.0
        elif mode == "semantic":
            alpha, beta = 1.0, 0.0
        else:
            alpha, beta = None, None

        # 构建 vault_filter（与 rag_cli.py 保持一致）
        vaults = [v.name for v in self.ctx.config.vaults if v.enabled]
        vault_filter = vaults if vaults else None

        results = await asyncio.to_thread(
            self.ctx.retriever.search, query, limit=top_k, vault_filter=vault_filter, alpha=alpha, beta=beta
        )
        return {
            "query": query,
            "total": len(results),
            "results": [
                {
                    "rank": i + 1,
                    "file": r.file_path,
                    "abs_path": r.absolute_path,
                    "content": r.content[:300],
                    "score": round(r.final_score, 4),
                    "confidence": round(r.confidence_score, 4),
                    "confidence_reason": r.confidence_reason,
                }
                for i, r in enumerate(results)
            ],
        }


class ScanTool(BaseTool):
    name, description = "scan_index", "Incrementally scan and update file index (tinyRAG)"
    schema: ClassVar[dict] = {"type": "object", "properties": {}}

    def _process_file_worker(self, file_item: dict, splitter: MarkdownSplitter) -> tuple[int, list, str]:
        abs_path = Path(file_item["absolute_path"])
        file_id = file_item["id"]
        file_path = file_item["file_path"]

        # 防御性检查：文件是否存在
        if not abs_path.exists():
            logger.warning(f"⚠️ 文件不存在，跳过：{abs_path}")
            return file_id, [], file_path

        try:
            content = abs_path.read_text(encoding="utf-8")
            mtime = file_item.get("mtime")
            # 修复：split() 第二个参数是 file_mtime (时间戳)，不是 file_path
            chunks = splitter.split(content, mtime)

            # 详细日志：分块结果
            if not chunks:
                logger.warning(f"⚠️ 文件分块为空：{file_path} (内容长度: {len(content)} 字符)")

            return file_id, chunks, file_path
        except Exception as e:
            logger.error(f"❌ 读取/分块失败：{abs_path} - {type(e).__name__}: {e}")
            return file_id, [], file_path

    async def _index_changed_files(self, changed_paths: list[str]) -> None:
        if not changed_paths:
            return

        placeholders = ",".join(["?"] * len(changed_paths))
        cursor = self.ctx.db.conn.execute(
            f"SELECT id, absolute_path, file_path, mtime FROM files WHERE absolute_path IN ({placeholders})",
            changed_paths,
        )
        files_to_index = [dict(row) for row in cursor.fetchall()]

        if not files_to_index:
            return

        all_pending_chunks = []
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            results = executor.map(
                lambda f: self._process_file_worker(f, self.ctx.splitter),
                files_to_index,
            )
            for f_id, chunks, f_path in results:
                for c in chunks:
                    all_pending_chunks.append((f_id, c, f_path))

        if not all_pending_chunks:
            return

        batch_size = self.ctx.config.embedding_model.batch_size
        processed = 0

        for i in range(0, len(all_pending_chunks), batch_size):
            batch = all_pending_chunks[i : i + batch_size]
            texts = [item[1].content for item in batch]

            try:
                embeddings = await asyncio.to_thread(self.ctx.retriever.embed_engine.embed, texts)
            except Exception as e:
                logger.error(f"❌ 批次向量化失败: {e}")
                continue

            try:
                for idx, ((file_id, chunk, f_path), emb) in enumerate(zip(batch, embeddings)):
                    global_idx = processed + idx
                    metadata_json = json.dumps(chunk.metadata or {}, ensure_ascii=False, default=_json_serialize)
                    confidence_json = json.dumps(
                        chunk.confidence_metadata or {}, ensure_ascii=False, default=_json_serialize
                    )

                    cursor = self.ctx.db.conn.execute(
                        """INSERT INTO chunks (file_id, chunk_index, content, content_type, section_title, section_path,
                        start_pos, end_pos, confidence_final_weight, metadata, confidence_json)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            file_id,
                            global_idx,
                            chunk.content,
                            chunk.content_type.value,
                            chunk.section_title,
                            chunk.section_path,
                            chunk.start_pos,
                            chunk.end_pos,
                            1.0,
                            metadata_json,
                            confidence_json,
                        ),
                    )
                    new_chunk_id = cursor.lastrowid

                    if self.ctx.db.vec_support:
                        import array

                        self.ctx.db.conn.execute(
                            "INSERT INTO vectors (chunk_id, embedding) VALUES (?, ?)",
                            (new_chunk_id, array.array("f", emb).tobytes()),
                        )

                    fts_text = _prepare_fts_content(chunk, f_path)
                    self.ctx.db.conn.execute(
                        "INSERT INTO fts5_index (rowid, content) VALUES (?, ?)",
                        (new_chunk_id, fts_text),
                    )

                self.ctx.db.conn.commit()
                processed += len(batch)

            except Exception as e:
                self.ctx.db.conn.rollback()
                logger.error(f"❌ 批次提交失败：{e}")
                raise

        logger.success(f"🎉 增量索引完成！共处理 {processed} 个 chunks")

    async def run(self, args: dict[str, Any]) -> dict[str, Any]:
        await self.ctx.initialize()
        vault_configs = [(v.name, v.path) for v in self.ctx.config.vaults if v.enabled]
        # 传入 per-vault 排除规则
        report = await asyncio.to_thread(self.ctx.scanner.scan_vaults, vault_configs, self.ctx.vault_excludes)
        await asyncio.to_thread(self.ctx.scanner.process_report, report)

        changed_paths = [f.absolute_path for f in report.new_files + report.modified_files]
        changed_paths.extend([f.new_absolute_path for f in report.moved_files])
        if changed_paths:
            await self._index_changed_files(changed_paths)

        return {
            "status": "success",
            "summary": report.summary(),
            "new": len(report.new_files),
            "modified": len(report.modified_files),
            "moved": len(report.moved_files),
            "deleted": len(report.deleted_files),
            "touched": len(report.touched_files),
        }


class RebuildTool(BaseTool):
    name, description = "rebuild_index", "Force rebuild full knowledge index (tinyRAG)"
    schema: ClassVar[dict] = {"type": "object", "properties": {}}

    async def run(self, args: dict[str, Any]) -> dict[str, Any]:
        task = asyncio.create_task(self._background_job(args))
        self.ctx.add_background_task(task)
        return {"status": "started", "message": "Index rebuild running in background"}

    async def _background_job(self, args: dict[str, Any]):
        try:
            await self.ctx.initialize()
            logger.info("Starting background index rebuild...")
            build_main = _load_build_index_main()
            import argparse

            # 从配置读取 batch_size，而不是硬编码
            batch_size = self.ctx.config.embedding_model.batch_size
            build_args = argparse.Namespace(force=True, batch_size=batch_size)
            await asyncio.to_thread(build_main, build_args)
            logger.info("Index rebuild completed successfully")
        except Exception as e:
            logger.error(f"Index rebuild failed: {e}", exc_info=True)
            raise


# =====================
# Resources 实现
# =====================
class ResourceManager:
    """MCP Resources 管理器"""

    def __init__(self, ctx: AppContext):
        self.ctx = ctx

    def list_resources(self) -> list[Resource]:
        return [
            Resource(
                uri="tinyrag://stats",
                name="Knowledge Base Statistics",
                description="知识库统计信息：文件数、分块数、vault 配置等",
                mimeType="application/json",
            ),
            Resource(
                uri="tinyrag://config",
                name="tinyRAG Configuration",
                description="当前配置信息：模型、权重、排除规则等",
                mimeType="application/json",
            ),
        ]

    def list_resource_templates(self) -> list[ResourceTemplate]:
        return [
            ResourceTemplate(
                uriTemplate="tinyrag://vault/{vault_name}",
                name="Vault Statistics",
                description="指定 vault 的统计信息",
                mimeType="application/json",
            ),
            ResourceTemplate(
                uriTemplate="tinyrag://file/{file_id}",
                name="File Content",
                description="获取指定文件的完整内容和元数据",
                mimeType="application/json",
            ),
            ResourceTemplate(
                uriTemplate="tinyrag://chunks/{file_id}",
                name="File Chunks",
                description="获取指定文件的所有分块",
                mimeType="application/json",
            ),
        ]

    async def read_resource(self, uri: Any) -> str:
        """
        读取资源内容
        注意：uri 参数是 AnyUrl 类型，需要转换为字符串
        """
        await self.ctx.initialize()

        # 🔧 F1: 将 AnyUrl 转换为字符串
        uri_str = str(uri)

        # 静态资源
        if uri_str == "tinyrag://stats":
            return self._get_stats()

        if uri_str == "tinyrag://config":
            return self._get_config()

        # 模板资源：tinyrag://vault/{vault_name}
        if uri_str.startswith("tinyrag://vault/"):
            vault_name = uri_str.split("/")[-1]
            return self._get_vault_stats(vault_name)

        # 模板资源：tinyrag://file/{file_id}
        if uri_str.startswith("tinyrag://file/"):
            file_id = uri_str.split("/")[-1]
            return self._get_file_content(file_id)

        # 模板资源：tinyrag://chunks/{file_id}
        if uri_str.startswith("tinyrag://chunks/"):
            file_id = uri_str.split("/")[-1]
            return self._get_file_chunks(file_id)

        raise ValueError(f"Unknown resource URI: {uri_str}")

    def _get_stats(self) -> str:
        db = self.ctx.db

        files_total = db.conn.execute("SELECT COUNT(*) FROM files WHERE is_deleted = 0").fetchone()[0]
        files_by_vault = db.conn.execute(
            "SELECT vault_name, COUNT(*) as cnt FROM files WHERE is_deleted = 0 GROUP BY vault_name"
        ).fetchall()

        chunks_total = db.conn.execute("SELECT COUNT(*) FROM chunks WHERE is_deleted = 0").fetchone()[0]
        vectors_total = db.conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]

        db_size = os.path.getsize(self.ctx.config.db_path) if self.ctx.config.db_path else 0

        stats = {
            "files": {
                "total": files_total,
                "by_vault": {row["vault_name"]: row["cnt"] for row in files_by_vault},
            },
            "chunks": {
                "total": chunks_total,
                "avg_per_file": round(chunks_total / max(files_total, 1), 1),
            },
            "vectors": {
                "total": vectors_total,
                "dimensions": self.ctx.config.embedding_model.dimensions,
            },
            "storage": {
                "db_size_mb": round(db_size / 1024 / 1024, 2),
                "db_path": self.ctx.config.db_path,
            },
            "model": {
                "name": self.ctx.config.embedding_model.name,
                "size": self.ctx.config.embedding_model.size,
            },
        }

        return json.dumps(stats, ensure_ascii=False, indent=2)

    def _get_config(self) -> str:
        cfg = self.ctx.config

        config_info = {
            "vaults": [
                {
                    "name": v.name,
                    "path": v.path,
                    "enabled": v.enabled,
                    "exclude": v.exclude.model_dump() if v.exclude else None,
                }
                for v in cfg.vaults
            ],
            "embedding_model": {
                "name": cfg.embedding_model.name,
                "dimensions": cfg.embedding_model.dimensions,
                "batch_size": cfg.embedding_model.batch_size,
            },
            "chunking": cfg.chunking,
            "confidence": {
                "doc_type_rules": cfg.confidence.doc_type_rules,
                "status_rules": cfg.confidence.status_rules,
                "date_decay": {
                    "enabled": cfg.confidence.date_decay.enabled,
                    "half_life_days": cfg.confidence.date_decay.half_life_days,
                    "type_specific": cfg.confidence.date_decay.type_specific_decay,
                },
            },
            "retrieval": cfg.retrieval,
            "exclude": cfg.exclude.model_dump(),
        }

        return json.dumps(config_info, ensure_ascii=False, indent=2)

    def _get_vault_stats(self, vault_name: str) -> str:
        db = self.ctx.db

        files_count = db.conn.execute(
            "SELECT COUNT(*) FROM files WHERE vault_name = ? AND is_deleted = 0", (vault_name,)
        ).fetchone()[0]

        chunks_count = db.conn.execute(
            "SELECT COUNT(*) FROM chunks c JOIN files f ON c.file_id = f.id WHERE f.vault_name = ? AND c.is_deleted = 0",
            (vault_name,),
        ).fetchone()[0]

        recent_files = db.conn.execute(
            """SELECT file_path, mtime, file_size FROM files 
               WHERE vault_name = ? AND is_deleted = 0 
               ORDER BY updated_at DESC LIMIT 10""",
            (vault_name,),
        ).fetchall()

        doc_types = db.conn.execute(
            """SELECT json_extract(confidence_json, '$.doc_type') as doc_type, COUNT(*) as cnt 
               FROM chunks c JOIN files f ON c.file_id = f.id 
               WHERE f.vault_name = ? AND c.is_deleted = 0 
               GROUP BY doc_type""",
            (vault_name,),
        ).fetchall()

        stats = {
            "vault_name": vault_name,
            "files_count": files_count,
            "chunks_count": chunks_count,
            "recent_files": [
                {"path": row["file_path"], "mtime": row["mtime"], "size": row["file_size"]} for row in recent_files
            ],
            "doc_type_distribution": {row["doc_type"] or "unknown": row["cnt"] for row in doc_types},
        }

        return json.dumps(stats, ensure_ascii=False, indent=2)

    def _get_file_content(self, file_id: str) -> str:
        try:
            fid = int(file_id)
        except ValueError:
            return json.dumps({"error": "Invalid file_id, must be integer"})

        row = self.ctx.db.conn.execute(
            """SELECT f.id, f.vault_name, f.file_path, f.absolute_path, f.file_hash, f.mtime, f.file_size
               FROM files f WHERE f.id = ? AND f.is_deleted = 0""",
            (fid,),
        ).fetchone()

        if not row:
            return json.dumps({"error": f"File not found: {file_id}"})

        try:
            with open(row["absolute_path"], "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            content = f"[Error reading file: {e}]"

        file_info = {
            "id": row["id"],
            "vault_name": row["vault_name"],
            "file_path": row["file_path"],
            "absolute_path": row["absolute_path"],
            "file_hash": row["file_hash"],
            "mtime": row["mtime"],
            "file_size": row["file_size"],
            "content": content[:10000],
            "content_truncated": len(content) > 10000,
        }

        return json.dumps(file_info, ensure_ascii=False, indent=2)

    def _get_file_chunks(self, file_id: str) -> str:
        try:
            fid = int(file_id)
        except ValueError:
            return json.dumps({"error": "Invalid file_id, must be integer"})

        chunks = self.ctx.db.conn.execute(
            """SELECT c.id, c.chunk_index, c.content, c.content_type, c.section_title, 
                      c.section_path, c.confidence_json
               FROM chunks c WHERE c.file_id = ? AND c.is_deleted = 0 
               ORDER BY c.chunk_index""",
            (fid,),
        ).fetchall()

        if not chunks:
            return json.dumps({"error": f"No chunks found for file: {file_id}"})

        result = {
            "file_id": fid,
            "total_chunks": len(chunks),
            "chunks": [
                {
                    "id": row["id"],
                    "index": row["chunk_index"],
                    "type": row["content_type"],
                    "section": row["section_title"],
                    "section_path": row["section_path"],
                    "content": row["content"][:500],
                    "confidence": json.loads(row["confidence_json"] or "{}"),
                }
                for row in chunks
            ],
        }

        return json.dumps(result, ensure_ascii=False, indent=2)


# =====================
# Prompts 实现
# =====================
class PromptManager:
    """MCP Prompts 管理器"""

    def __init__(self, ctx: AppContext):
        self.ctx = ctx

    def list_prompts(self) -> list[Prompt]:
        return [
            Prompt(
                name="search_with_context",
                description="基于知识库检索结果回答问题的提示词模板",
                arguments=[
                    PromptArgument(
                        name="query",
                        description="用户的问题或查询",
                        required=True,
                    ),
                    PromptArgument(
                        name="top_k",
                        description="检索结果数量，默认 5",
                        required=False,
                    ),
                ],
            ),
            Prompt(
                name="summarize_document",
                description="总结指定文档核心内容的提示词模板",
                arguments=[
                    PromptArgument(
                        name="file_path",
                        description="文档路径（相对路径）",
                        required=True,
                    ),
                ],
            ),
        ]

    async def get_prompt(self, name: str, arguments: dict[str, str]) -> GetPromptResult:
        await self.ctx.initialize()

        if name == "search_with_context":
            return self._prompt_search_with_context(arguments)
        elif name == "summarize_document":
            return self._prompt_summarize_document(arguments)
        else:
            raise ValueError(f"Unknown prompt: {name}")

    def _prompt_search_with_context(self, args: dict[str, str]) -> GetPromptResult:
        query = args.get("query", "")
        top_k = int(args.get("top_k", "5"))

        results = self.ctx.retriever.search(query, limit=top_k)

        context_parts = []
        for i, r in enumerate(results, 1):
            context_parts.append(
                f"【文档 {i}】{r.file_path}\n"
                f"置信度: {r.confidence_score:.3f} ({r.confidence_reason})\n"
                f"内容: {r.content}\n"
            )

        context = "\n".join(context_parts) if context_parts else "未找到相关文档"

        prompt_text = f"""你是一个知识库助手。请基于以下检索结果回答用户问题。

## 用户问题
{query}

## 知识库检索结果
{context}

## 回答要求
1. 仅基于检索结果回答，不要编造信息
2. 如果检索结果不足，明确说明
3. 引用文档时标注来源（如：【文档 1】）
4. 如果有多个相关观点，综合呈现

## 回答"""

        return GetPromptResult(
            description=f"检索增强回答: {query}",
            messages=[PromptMessage(role="user", content=TextContent(type="text", text=prompt_text))],
        )

    def _prompt_summarize_document(self, args: dict[str, str]) -> GetPromptResult:
        file_path = args.get("file_path", "")

        # 多级匹配策略：精确匹配 > 绝对路径匹配 > 模糊匹配
        row = None

        # 1. 精确匹配相对路径 (file_path)
        row = self.ctx.db.conn.execute(
            "SELECT id, absolute_path, file_path FROM files WHERE file_path = ? AND is_deleted = 0",
            (file_path,),
        ).fetchone()

        # 2. 如果是绝对路径，尝试精确匹配 absolute_path
        if not row and os.path.isabs(file_path):
            row = self.ctx.db.conn.execute(
                "SELECT id, absolute_path, file_path FROM files WHERE absolute_path = ? AND is_deleted = 0",
                (file_path,),
            ).fetchone()

        # 3. 最后尝试后缀模糊匹配（文件名匹配）
        if not row:
            # 只匹配文件名，避免匹配到多个文件
            filename = os.path.basename(file_path)
            rows = self.ctx.db.conn.execute(
                "SELECT id, absolute_path, file_path FROM files WHERE file_path LIKE ? AND is_deleted = 0",
                (f"%{filename}",),
            ).fetchall()

            if len(rows) == 1:
                row = rows[0]
            elif len(rows) > 1:
                # 多个匹配，选择最接近的
                for r in rows:
                    if r["file_path"].endswith(file_path):
                        row = r
                        break
                if not row:
                    row = rows[0]  # 默认取第一个

        if not row:
            prompt_text = f"未找到文档: {file_path}，请确认路径是否正确。"
        else:
            actual_file_path = row["file_path"]  # 使用数据库中的相对路径

            # 从 chunks 表获取内容和 confidence_json
            chunks = self.ctx.db.conn.execute(
                "SELECT content, section_title, confidence_json FROM chunks WHERE file_id = ? AND is_deleted = 0 ORDER BY chunk_index",
                (row["id"],),
            ).fetchall()

            content = "\n\n".join(f"### {c['section_title'] or '正文'}\n{c['content']}" for c in chunks)

            # 从第一个 chunk 获取 confidence 信息
            confidence = {}
            if chunks and chunks[0]["confidence_json"]:
                try:
                    confidence = json.loads(chunks[0]["confidence_json"])
                except json.JSONDecodeError:
                    pass

            prompt_text = f"""请总结以下文档的核心内容。

## 文档信息
- 路径: {actual_file_path}
- 类型: {confidence.get('doc_type', 'unknown')}
- 状态: {confidence.get('status', 'unknown')}
- 日期: {confidence.get('final_date', 'unknown')}

## 文档内容
{content[:8000]}

## 摘要要求
1. 提炼核心观点（3-5 条）
2. 说明文档的主要价值
3. 指出适用场景

## 摘要"""

        return GetPromptResult(
            description=f"文档摘要: {file_path}",
            messages=[PromptMessage(role="user", content=TextContent(type="text", text=prompt_text))],
        )


# =====================
# Helper Functions
# =====================
# 日期保护正则（与 build_index.py、hybrid_engine.py 保持一致）
_DATE_PATTERN = re.compile(r"\d{4}(?:-\d{2}(?:-\d{2})?|年(?:\d{1,2}(?:月(?:\d{1,2}日)?)?)?)")


def _jieba_segment(text: str) -> str:
    """对中文文本进行 jieba 分词，保护日期格式免被拆分"""
    if not text or not text.strip():
        return ""
    import jieba

    # 保护日期格式
    date_placeholders = {}
    protected_text = text
    for i, match in enumerate(_DATE_PATTERN.finditer(text)):
        placeholder = f"__DATE_{i}__"
        date_placeholders[placeholder] = match.group()
        protected_text = protected_text.replace(match.group(), placeholder, 1)

    # jieba 分词
    segmented = " ".join(jieba.cut_for_search(protected_text))

    # 修复被 jieba 拆分的占位符（如 "__ DATE _ 0 __" -> "__DATE_0__"）
    broken_pattern = re.compile(r"__\s*DATE\s*_\s*(\d+)\s*__")
    segmented = broken_pattern.sub(r"__DATE_\1__", segmented)

    # 恢复日期格式
    for placeholder, date_str in date_placeholders.items():
        segmented = segmented.replace(placeholder, date_str)

    # 清理点号前后的空格（如 "2026-04-13. md" -> "2026-04-13.md"）
    segmented = re.sub(r"\s*\.\s*", ".", segmented)

    return segmented


def _prepare_fts_content(chunk, file_path: str) -> str:
    metadata = chunk.metadata or {}
    tags = metadata.get("tags", [])
    if tags is None:
        tags = []
    if isinstance(tags, str):
        tags = [tags]
    tag_str = " ".join([f"#{t.strip()}" for t in tags if t])

    doc_type = metadata.get("doc_type") or ""
    filename = os.path.basename(file_path)
    section_title = chunk.section_title or ""

    parts = [
        _jieba_segment(filename),
        _jieba_segment(filename),
        _jieba_segment(chunk.section_path or ""),
        _jieba_segment(section_title),
        _jieba_segment(section_title),
        _jieba_segment(tag_str),
        _jieba_segment(doc_type),
        _jieba_segment(chunk.content),
    ]
    return " ".join(filter(None, parts)).strip()


def _json_serialize(obj):
    from datetime import date, datetime

    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# =====================
# Registry & Server
# =====================
class ToolRegistry:
    def __init__(self, ctx: AppContext):
        self.tools: dict[str, BaseTool] = {}
        self.ctx = ctx

    def register(self, cls):
        t = cls(self.ctx)
        self.tools[t.name] = t

    def list_tools(self):
        return [t.to_mcp_tool() for t in self.tools.values()]

    async def execute(self, name: str, args: dict[str, Any]) -> list[TextContent]:
        if name not in self.tools:
            raise ValueError(f"Unknown tool: {name}")
        result = await self.tools[name].run(args)
        return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, default=str))]


class RagServer:
    """tinyRAG MCP Server v1.1.2: Tools + Resources + Prompts"""

    def __init__(self):
        self.ctx = AppContext()
        self.registry = ToolRegistry(self.ctx)
        self.registry.register(SearchTool)
        self.registry.register(ScanTool)
        self.registry.register(RebuildTool)

        self.resource_manager = ResourceManager(self.ctx)
        self.prompt_manager = PromptManager(self.ctx)

        if MCP_AVAILABLE:
            self.server = Server("tinyRAG")
            self._register()
        else:
            logger.warning("MCP package not installed, running in mock mode")

    def _register(self):
        # ── Tools ──
        @self.server.list_tools()
        async def list_tools():
            return self.registry.list_tools()

        @self.server.call_tool()
        @mcp_safe
        async def call_tool(name: str, arguments: dict[str, Any]):
            return await self.registry.execute(name, arguments)

        # ── Resources ──
        @self.server.list_resources()
        async def list_resources():
            return self.resource_manager.list_resources()

        @self.server.list_resource_templates()
        async def list_resource_templates():
            return self.resource_manager.list_resource_templates()

        @self.server.read_resource()
        async def read_resource(uri):
            # uri 是 AnyUrl 类型，在方法内部转换为字符串
            return await self.resource_manager.read_resource(uri)

        # ── Prompts ──
        @self.server.list_prompts()
        async def list_prompts():
            return self.prompt_manager.list_prompts()

        @self.server.get_prompt()
        async def get_prompt(name: str, arguments: dict[str, str]):
            return await self.prompt_manager.get_prompt(name, arguments)

    async def run(self):
        if not MCP_AVAILABLE:
            logger.error("Please install mcp: pip install mcp")
            return
        logger.info("MCP Stdio Server v1.1.2 starting (Tools + Resources + Prompts)...")
        try:
            async with stdio_server() as (r, w):
                await self.server.run(r, w, self.server.create_initialization_options())
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        finally:
            await self.ctx.shutdown()


if __name__ == "__main__":
    asyncio.run(RagServer().run())
