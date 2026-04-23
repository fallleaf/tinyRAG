#!/usr/bin/env python3
"""
mcp_server/server.py - Production MCP RAG Server (v2.1 - Full Optimized)
优化记录:
- P0: ConfigTool 返回完整配置 (model_dump mode='json')
- P1: 新增 reload_config Tool (热重载配置，无需重启)
- P1: 新增 maintenance Tool (LLM 可触发的 DB 清理 & VACUUM)
- P1: 修复 PromptManager 括号嵌套语法错误
- P2: DRY 重构: 移除重复的 _jieba_segment / _DATE_PATTERN，统一导入 utils.jieba_helper
- P2: 增强异步锁边界，修复资源释放竞态条件
"""

import asyncio
import contextlib
import importlib.util
import json
import os
import sys
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, ClassVar

from utils.jieba_helper import jieba_segment  # ✅ DRY 重构：统一分词入口
from utils.logger import setup_logger

# MCP - 导入所有需要的类型
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import (
        GetPromptResult,
        Prompt,
        PromptArgument,
        PromptMessage,
        ReadResourceResult,
        Resource,
        ResourceTemplate,
        TextContent,
        TextResourceContents,
        Tool,
    )

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

from chunker.markdown_splitter import MarkdownSplitter
from config import Settings, load_config
from embedder.embed_engine import EmbeddingEngine
from retriever.hybrid_engine import HybridEngine
from scanner.scan_engine import DEFAULT_SKIP_DIRS, Scanner
from storage.database import DatabaseManager

# =====================
# Logger & Config
# =====================
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
logger = setup_logger(level="INFO")
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
            try:
                config_path = PROJECT_ROOT / "config.yaml"
                config = load_config(str(config_path))
                db = DatabaseManager(config.db_path, vec_dimension=config.embedding_model.dimensions)

                global_skip_dirs = DEFAULT_SKIP_DIRS | frozenset(config.exclude.dirs)
                scanner = Scanner(db, global_skip_dirs, config.exclude.patterns)
                vault_excludes = {}
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
            "query": {"type": "string", "description": "搜索关键词，支持自然语言查询"},
            "mode": {
                "type": "string",
                "enum": ["semantic", "keyword", "hybrid"],
                "default": "hybrid",
                "description": "检索模式：semantic(纯语义), keyword(纯关键词), hybrid(混合)",
            },
            "top_k": {
                "type": "integer",
                "default": 10,
                "minimum": 1,
                "maximum": 100,
                "description": "返回结果数量，默认 10",
            },
            "alpha": {
                "type": "number",
                "default": None,
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "语义检索权重 (0.0-1.0)，默认使用 config.yaml 中的值",
            },
            "beta": {
                "type": "number",
                "default": None,
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "关键词检索权重 (0.0-1.0)，默认使用 config.yaml 中的值",
            },
            "vaults": {
                "type": "array",
                "items": {"type": "string"},
                "default": None,
                "description": "指定检索的仓库名称列表，不指定则检索所有启用的 vault",
            },
        },
        "required": ["query"],
    }

    async def run(self, args: dict[str, Any]) -> dict[str, Any]:
        await self.ctx.initialize()
        query = args.get("query", "")
        mode = args.get("mode", "hybrid")
        top_k = min(max(args.get("top_k", 10), 1), 100)

        # 获取 alpha/beta 参数
        alpha = args.get("alpha")
        beta = args.get("beta")

        # 如果指定了 mode，覆盖 alpha/beta
        if mode == "keyword":
            alpha, beta = 0.0, 1.0
        elif mode == "semantic":
            alpha, beta = 1.0, 0.0
        elif alpha is not None or beta is not None:
            # 用户指定了 alpha 或 beta，需要验证
            if alpha is None:
                alpha = 1.0 - (beta if beta is not None else 0.4)
            if beta is None:
                beta = 1.0 - alpha
            # 验证 alpha + beta = 1
            if abs((alpha + beta) - 1.0) > 0.01:
                return {
                    "error": f"alpha + beta 必须等于 1，当前：alpha={alpha}, beta={beta}",
                    "query": query,
                    "total": 0,
                    "results": [],
                }
        else:
            # 使用 config.yaml 中的默认值
            alpha, beta = None, None

        # 处理 vaults 参数
        vaults_arg = args.get("vaults")
        if vaults_arg:
            # 用户指定了 vaults，验证是否存在
            available_vaults = [v.name for v in self.ctx.config.vaults if v.enabled]
            vault_filter = [v for v in vaults_arg if v in available_vaults]
            if not vault_filter:
                return {
                    "error": f"指定的 vaults 不存在或未启用：{vaults_arg}",
                    "query": query,
                    "total": 0,
                    "results": [],
                }
        else:
            # 使用所有启用的 vaults
            vault_filter = [v.name for v in self.ctx.config.vaults if v.enabled]

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
        if not abs_path.exists():
            logger.warning(f"⚠️ 文件不存在，跳过：{abs_path}")
            return file_item["id"], [], file_item["file_path"]
        try:
            content = abs_path.read_text(encoding="utf-8")
            chunks = splitter.split(content, file_item.get("mtime"))
            return file_item["id"], chunks, file_item["file_path"]
        except Exception as e:
            logger.error(f"❌ 读取/分块失败：{abs_path} - {type(e).__name__}: {e}")
            return file_item["id"], [], file_item["file_path"]

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
            results = executor.map(lambda f: self._process_file_worker(f, self.ctx.splitter), files_to_index)
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
                self.ctx.db.conn.execute("PRAGMA synchronous = OFF;")
                for idx, ((file_id, chunk, f_path), emb) in enumerate(zip(batch, embeddings, strict=False)):
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
                    self.ctx.db.conn.execute(
                        "INSERT INTO fts5_index (rowid, content) VALUES (?, ?)",
                        (new_chunk_id, _prepare_fts_content(chunk, f_path)),
                    )
                self.ctx.db.conn.commit()
                processed += len(batch)
            except Exception as e:
                self.ctx.db.conn.rollback()
                logger.error(f"❌ 批次提交失败：{e}")
                raise
            finally:
                self.ctx.db.conn.execute("PRAGMA synchronous = NORMAL;")
        logger.success(f"🎉 增量索引完成！共处理 {processed} 个 chunks")

    async def run(self, args: dict[str, Any]) -> dict[str, Any]:
        await self.ctx.initialize()
        vault_configs = [(v.name, v.path) for v in self.ctx.config.vaults if v.enabled]
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

            batch_size = self.ctx.config.embedding_model.batch_size
            build_args = argparse.Namespace(force=True, batch_size=batch_size)
            await asyncio.to_thread(build_main, build_args)
            logger.info("Index rebuild completed successfully")
        except Exception as e:
            logger.error(f"Index rebuild failed: {e}", exc_info=True)
            raise


class StatsTool(BaseTool):
    name, description = "stats", "Get knowledge base statistics (tinyRAG)"
    schema: ClassVar[dict] = {"type": "object", "properties": {}}

    async def run(self, args: dict[str, Any]) -> dict[str, Any]:
        await self.ctx.initialize()
        db = self.ctx.db
        files_total = db.conn.execute("SELECT COUNT(*) FROM files WHERE is_deleted = 0").fetchone()[0]
        files_by_vault = db.conn.execute(
            "SELECT vault_name, COUNT(*) as cnt FROM files WHERE is_deleted = 0 GROUP BY vault_name"
        ).fetchall()
        chunks_total = db.conn.execute("SELECT COUNT(*) FROM chunks WHERE is_deleted = 0").fetchone()[0]
        try:
            vectors_total = db.conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
        except Exception:
            vectors_total = 0
        db_size = os.path.getsize(self.ctx.config.db_path) if self.ctx.config.db_path else 0
        return {
            "files": {"total": files_total, "by_vault": {row["vault_name"]: row["cnt"] for row in files_by_vault}},
            "chunks": {"total": chunks_total, "avg_per_file": round(chunks_total / max(files_total, 1), 1)},
            "vectors": {"total": vectors_total, "dimensions": self.ctx.config.embedding_model.dimensions},
            "storage": {"db_size_mb": round(db_size / 1024 / 1024, 2), "db_path": self.ctx.config.db_path},
            "model": {"name": self.ctx.config.embedding_model.name, "size": self.ctx.config.embedding_model.size},
        }


# ✅ P0 修复：返回完整配置，不再硬编码遗漏字段
class ConfigTool(BaseTool):
    name, description = "config", "Get complete tinyRAG configuration (includes exclude, cache, maintenance, etc.)"
    schema: ClassVar[dict] = {"type": "object", "properties": {}}

    async def run(self, args: dict[str, Any]) -> dict[str, Any]:
        await self.ctx.initialize()
        cfg_dict = self.ctx.config.model_dump(mode="json")
        # ✅ 防御性保障：确保每个 vault 的 exclude 字段必定可见且结构完整
        if "vaults" in cfg_dict and isinstance(cfg_dict["vaults"], list):
            for v in cfg_dict["vaults"]:
                if not v.get("exclude") or v["exclude"] is None:
                    v["exclude"] = {"dirs": [], "patterns": []}
        return cfg_dict


# ✅ P1: 新增热重载 Tool
class ReloadConfigTool(BaseTool):
    name, description = "reload_config", "Hot-reload config.yaml and reinit retriever/splitter without restart"
    schema: ClassVar[dict] = {"type": "object", "properties": {}}

    async def run(self, args: dict[str, Any]) -> dict[str, Any]:
        await self.ctx.initialize()
        new_cfg = load_config(str(PROJECT_ROOT / "config.yaml"))
        async with self.ctx._lock:
            self.ctx.config = new_cfg
            self.ctx.splitter = MarkdownSplitter(new_cfg)
            self.ctx.retriever = HybridEngine(
                config=new_cfg, db=self.ctx.db, embed_engine=self.ctx.retriever.embed_engine
            )
            new_excludes = {}
            for v in new_cfg.vaults:
                if v.enabled:
                    new_excludes[v.name] = (
                        (frozenset(v.exclude.dirs), v.exclude.patterns) if v.exclude else (frozenset(), [])
                    )
            self.ctx.vault_excludes = new_excludes
        logger.info("✅ Configuration reloaded successfully")
        return {"status": "reloaded", "retrieval": new_cfg.retrieval}


# ✅ P1: 新增运维 Tool
class MaintenanceTool(BaseTool):
    name, description = "maintenance", "Run DB cleanup & VACUUM (tinyRAG)"
    schema: ClassVar[dict] = {"type": "object", "properties": {"dry_run": {"type": "boolean", "default": False}}}

    async def run(self, args: dict[str, Any]) -> dict[str, Any]:
        await self.ctx.initialize()
        dry_run = args.get("dry_run", False)
        import vacuum as vac_mod

        stats = vac_mod.check_vacuum_needed(self.ctx.db, self.ctx.config)
        if dry_run:
            return {"status": "dry_run", "stats": stats, "recommendation": "Execute without dry_run to reclaim space."}
        vac_mod.clean_deleted_records(self.ctx.db, dry_run=False)
        vac_mod.execute_vacuum(self.ctx.db, dry_run=False)
        return {"status": "completed", "reclaimed_space_mb": stats.get("file_size_mb", 0)}


# =====================
# Resources & Prompts
# =====================
class ResourceManager:
    def __init__(self, ctx: AppContext):
        self.ctx = ctx

    def list_resources(self) -> list[Resource]:
        return []

    def list_resource_templates(self) -> list[ResourceTemplate]:
        return [
            ResourceTemplate(
                uriTemplate="tinyrag://vault/{vault_name}",
                name="Vault Statistics",
                description="Vault stats",
                mimeType="application/json",
            ),
            ResourceTemplate(
                uriTemplate="tinyrag://file/{file_id}",
                name="File Content",
                description="File content",
                mimeType="application/json",
            ),
            ResourceTemplate(
                uriTemplate="tinyrag://chunks/{file_id}",
                name="File Chunks",
                description="File chunks",
                mimeType="application/json",
            ),
        ]

    async def read_resource(self, uri: Any) -> ReadResourceResult:
        await self.ctx.initialize()
        uri_str = str(uri)
        content = ""
        if uri_str.startswith("tinyrag://vault/"):
            content = self._get_vault_stats(uri_str.split("/")[-1])
        elif uri_str.startswith("tinyrag://file/"):
            content = self._get_file_content(uri_str.split("/")[-1])
        elif uri_str.startswith("tinyrag://chunks/"):
            content = self._get_file_chunks(uri_str.split("/")[-1])
        else:
            raise ValueError(f"Unknown resource URI: {uri_str}")
        return ReadResourceResult(
            contents=[TextResourceContents(uri=uri_str, mimeType="application/json", text=content)]
        )

    def _get_vault_stats(self, vault_name: str) -> str:
        db = self.ctx.db
        files_count = db.conn.execute(
            "SELECT COUNT(*) FROM files WHERE vault_name = ? AND is_deleted = 0", (vault_name,)
        ).fetchone()[0]
        chunks_count = db.conn.execute(
            "SELECT COUNT(*) FROM chunks c JOIN files f ON c.file_id = f.id WHERE f.vault_name = ? AND c.is_deleted = 0",
            (vault_name,),
        ).fetchone()[0]
        recent = db.conn.execute(
            "SELECT file_path, mtime, file_size FROM files WHERE vault_name = ? AND is_deleted = 0 ORDER BY updated_at DESC LIMIT 5",
            (vault_name,),
        ).fetchall()
        return json.dumps(
            {
                "vault_name": vault_name,
                "files": files_count,
                "chunks": chunks_count,
                "recent": [dict(r) for r in recent],
            },
            ensure_ascii=False,
            indent=2,
        )

    def _get_file_content(self, file_id: str) -> str:
        try:
            fid = int(file_id)
        except:
            return json.dumps({"error": "Invalid file_id"})
        row = self.ctx.db.conn.execute("SELECT * FROM files WHERE id = ? AND is_deleted = 0", (fid,)).fetchone()
        if not row:
            return json.dumps({"error": "File not found"})
        try:
            content = Path(row["absolute_path"]).read_text(encoding="utf-8")[:10000]
        except:
            content = "[Error reading file]"
        return json.dumps({**dict(row), "content_preview": content}, ensure_ascii=False, indent=2)

    def _get_file_chunks(self, file_id: str) -> str:
        try:
            fid = int(file_id)
        except:
            return json.dumps({"error": "Invalid file_id"})
        chunks = self.ctx.db.conn.execute(
            "SELECT id, chunk_index, content, content_type, section_title FROM chunks WHERE file_id = ? AND is_deleted = 0 ORDER BY chunk_index",
            (fid,),
        ).fetchall()
        return json.dumps(
            {"file_id": fid, "total": len(chunks), "chunks": [dict(c) for c in chunks]}, ensure_ascii=False, indent=2
        )


PROMPTS_DIR = PROJECT_ROOT / "prompts"
DEFAULT_PROMPT_SEARCH = (
    "你是一个知识库助手。请基于以下检索结果回答用户问题。\n## 用户问题\n{{query}}\n## 知识库检索结果\n{{context}}"
)
DEFAULT_PROMPT_SUMMARIZE = "请总结以下文档的核心内容。\n## 文档信息\n路径: {{file_path}}\n## 文档内容\n{{content}}"


def _load_prompt_template(filename: str, default: str) -> str:
    p = PROMPTS_DIR / filename
    if p.exists():
        try:
            return p.read_text(encoding="utf-8")
        except:
            return default
    return default


class PromptManager:
    def __init__(self, ctx: AppContext):
        self.ctx = ctx
        self._tpl_search = _load_prompt_template("prompt_search_with_context.md", DEFAULT_PROMPT_SEARCH)
        self._tpl_sum = _load_prompt_template("prompt_summarize_document.md", DEFAULT_PROMPT_SUMMARIZE)

    def list_prompts(self) -> list[Prompt]:
        return [
            Prompt(
                name="search_with_context",
                description="RAG answer prompt",
                arguments=[
                    PromptArgument(name="query", required=True),
                    PromptArgument(name="top_k", required=False),
                    PromptArgument(name="alpha", required=False),
                    PromptArgument(name="beta", required=False),
                    PromptArgument(name="vaults", required=False),
                ],
            ),
            Prompt(
                name="summarize_document",
                description="Summarize doc prompt",
                arguments=[PromptArgument(name="file_path", required=True)],
            ),
        ]

    async def get_prompt(self, name: str, arguments: dict[str, str]) -> GetPromptResult:
        await self.ctx.initialize()
        if name == "search_with_context":
            return self._prompt_search(arguments)
        elif name == "summarize_document":
            return self._prompt_summarize(arguments)
        raise ValueError(f"Unknown prompt: {name}")

    def _render(self, tpl: str, vars: dict) -> str:
        for k, v in vars.items():
            tpl = tpl.replace(f"{{{{{k}}}}}", str(v))
        return tpl

    # ✅ P1 修复：解决原代码括号嵌套不匹配导致的 SyntaxError
    def _prompt_search(self, args: dict) -> GetPromptResult:
        query = args.get("query", "")
        top_k = int(args.get("top_k", "5"))

        # 获取 alpha/beta 参数（转换为浮点数）
        alpha_str = args.get("alpha")
        beta_str = args.get("beta")
        alpha = float(alpha_str) if alpha_str else None
        beta = float(beta_str) if beta_str else None

        # 处理 vaults 参数
        vaults_arg = args.get("vaults")
        if vaults_arg:
            # 用户指定了 vaults，验证是否存在
            available_vaults = [v.name for v in self.ctx.config.vaults if v.enabled]
            vault_filter = [v for v in vaults_arg if v in available_vaults]
            if not vault_filter:
                ctx = f"错误：指定的 vaults 不存在或未启用：{vaults_arg}"
                return GetPromptResult(
                    description=f"检索回答: {query}",
                    messages=[
                        PromptMessage(
                            role="user",
                            content=TextContent(
                                type="text", text=self._render(self._tpl_search, {"query": query, "context": ctx})
                            ),
                        )
                    ],
                )
        else:
            # 使用所有启用的 vaults
            vault_filter = [v.name for v in self.ctx.config.vaults if v.enabled]

        # 执行检索
        results = self.ctx.retriever.search(query, limit=top_k, vault_filter=vault_filter, alpha=alpha, beta=beta)
        ctx = (
            "\n".join([f"【文档 {i+1}】{r.file_path}\n内容: {r.content[:300]}" for i, r in enumerate(results)])
            or "无结果"
        )
        return GetPromptResult(
            description=f"检索回答: {query}",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text", text=self._render(self._tpl_search, {"query": query, "context": ctx})
                    ),
                )
            ],
        )

    # ✅ P1 修复：同上
    def _prompt_summarize(self, args: dict) -> GetPromptResult:
        fp = args.get("file_path", "")
        row = self.ctx.db.conn.execute(
            "SELECT id, file_path FROM files WHERE file_path = ? AND is_deleted = 0", (fp,)
        ).fetchone()
        if not row:
            row = self.ctx.db.conn.execute(
                "SELECT id, file_path FROM files WHERE file_path LIKE ? AND is_deleted = 0 LIMIT 1",
                (f"%{os.path.basename(fp)}",),
            ).fetchone()
        if not row:
            return GetPromptResult(
                description=f"摘要: {fp}",
                messages=[PromptMessage(role="user", content=TextContent(type="text", text=f"未找到: {fp}"))],
            )
        chunks = self.ctx.db.conn.execute(
            "SELECT content, section_title, confidence_json FROM chunks WHERE file_id = ? AND is_deleted = 0 ORDER BY chunk_index",
            (row["id"],),
        ).fetchall()
        content = "\n".join(f"### {c['section_title'] or '正文'}\n{c['content']}" for c in chunks)[:8000]
        conf = json.loads(chunks[0]["confidence_json"]) if chunks and chunks[0]["confidence_json"] else {}
        return GetPromptResult(
            description=f"摘要: {fp}",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=self._render(
                            self._tpl_sum,
                            {
                                "file_path": row["file_path"],
                                "doc_type": conf.get("doc_type", ""),
                                "status": conf.get("status", ""),
                                "final_date": conf.get("final_date", ""),
                                "content": content,
                            },
                        ),
                    ),
                )
            ],
        )


# =====================
# Helpers (已替换为 utils.jieba_helper)
# =====================
def _prepare_fts_content(chunk, file_path: str) -> str:
    """FTS5 复合检索文本构建 (使用统一 jieba_segment)"""
    metadata = chunk.metadata or {}
    tags = metadata.get("tags", []) or []
    if isinstance(tags, str):
        tags = [tags]
    tag_str = " ".join([f"#{t.strip()}" for t in tags if t])
    doc_type = metadata.get("doc_type") or ""
    filename = os.path.basename(file_path)
    section = chunk.section_title or ""
    parts = [
        jieba_segment(filename),
        jieba_segment(chunk.section_path or ""),
        jieba_segment(section),
        jieba_segment(tag_str),
        jieba_segment(doc_type),
        jieba_segment(chunk.content),
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
    def __init__(self):
        self.ctx = AppContext()
        self.registry = ToolRegistry(self.ctx)
        self.registry.register(SearchTool)
        self.registry.register(ScanTool)
        self.registry.register(RebuildTool)
        self.registry.register(StatsTool)
        self.registry.register(ConfigTool)
        self.registry.register(ReloadConfigTool)  # ✅ P1
        self.registry.register(MaintenanceTool)  # ✅ P1
        self.resource_manager = ResourceManager(self.ctx)
        self.prompt_manager = PromptManager(self.ctx)
        if MCP_AVAILABLE:
            self.server = Server("tinyRAG")
            self._register()
        else:
            logger.warning("MCP package not installed, running in mock mode")

    def _register(self):
        @self.server.list_tools()
        async def list_tools():
            return self.registry.list_tools()

        @self.server.call_tool()
        @mcp_safe
        async def call_tool(name: str, arguments: dict[str, Any]):
            return await self.registry.execute(name, arguments)

        @self.server.list_resources()
        async def list_resources():
            return self.resource_manager.list_resources()

        @self.server.list_resource_templates()
        async def list_resource_templates():
            return self.resource_manager.list_resource_templates()

        @self.server.read_resource()
        async def read_resource(uri):
            return await self.resource_manager.read_resource(uri)

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
        logger.info("MCP Stdio Server v2.0 starting...")
        try:
            async with stdio_server() as (r, w):
                await self.server.run(r, w, self.server.create_initialization_options())
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        finally:
            await self.ctx.shutdown()


if __name__ == "__main__":
    asyncio.run(RagServer().run())
