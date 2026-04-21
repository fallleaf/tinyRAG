#!/usr/bin/env python3
"""
mcp_server/server.py - Production MCP RAG Server (v2.2 - 插件支持)
优化记录:
- P0: ConfigTool 返回完整配置 (model_dump mode='json')
- P1: 新增 reload_config Tool (热重载配置，无需重启)
- P1: 新增 maintenance Tool (LLM 可触发的 DB 清理 & VACUUM)
- P1: 修复 PromptManager 括号嵌套语法错误
- P2: DRY 重构: 移除重复的 _jieba_segment / _DATE_PATTERN，统一导入 utils.jieba_helper
- P2: 增强异步锁边界，修复资源释放竞态条件
- P3: 集成插件系统，支持动态加载扩展功能
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
from plugins.bootstrap import PluginLoader, init_plugins, shutdown_plugins  # P3: 插件支持
from retriever.hybrid_engine import HybridEngine
from scanner.scan_engine import DEFAULT_SKIP_DIRS, Scanner
from storage.database import DatabaseManager

# =====================
# Logger & Config
# =====================
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
logger = setup_logger(level="INFO", enable_stderr=False)  # MCP 模式禁用 stderr，避免干扰 JSON 协议
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
        self.plugin_loader: PluginLoader | None = None  # P3: 插件加载器
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

                # 先设置 self.db，再初始化插件系统（插件需要访问 db）
                self.config = config
                self.db = db
                self.scanner = scanner
                self.retriever = retriever
                self.splitter = splitter
                self.vault_excludes = vault_excludes

                # P3: 初始化插件系统（在设置 self.db 之后）
                plugin_loader = None
                if config.plugins.enabled:
                    try:
                        plugin_loader = init_plugins(config, self)
                        logger.info(f"✅ 插件系统已初始化，加载 {len(plugin_loader.get_all_plugins())} 个插件")
                    except Exception as e:
                        logger.warning(f"⚠️ 插件系统初始化失败: {e}")

                self.plugin_loader = plugin_loader  # P3
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
        # P3: 关闭插件系统
        if self.plugin_loader:
            try:
                shutdown_plugins()
                logger.info("Plugins shutdown completed")
            except Exception as e:
                logger.warning(f"Plugin shutdown error: {e}")
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
            "mode": {"type": "string", "enum": ["semantic", "keyword", "hybrid"], "default": "hybrid"},
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
        vaults = [v.name for v in self.ctx.config.vaults if v.enabled]
        vault_filter = vaults if vaults else None
        results = await asyncio.to_thread(
            self.ctx.retriever.search, query, limit=top_k, vault_filter=vault_filter, alpha=alpha, beta=beta
        )

        # P3: 插件增强检索（统一评分版）
        plugin_enabled = self.ctx.plugin_loader is not None
        query_vec = None
        if plugin_enabled and results:
            try:
                query_vec = await asyncio.to_thread(self.ctx.retriever.embed_engine.embed, [query])
                query_vec = query_vec[0] if query_vec else None

                enhanced_results = self.ctx.plugin_loader.invoke_hook(
                    "on_search",
                    query=query,
                    results=[r.__dict__ for r in results],
                    query_vec=query_vec,
                    base_alpha=alpha,  # 传递基础检索的 alpha 权重
                    base_beta=beta,  # 传递基础检索的 beta 权重
                )

                if enhanced_results and isinstance(enhanced_results, list) and len(enhanced_results) > 0:
                    first_result = enhanced_results[0]
                    if isinstance(first_result, list) and len(first_result) > 0:
                        from retriever.hybrid_engine import RetrievalResult

                        plugin_results = []
                        for r in first_result:
                            if isinstance(r, RetrievalResult):
                                plugin_results.append(r)
                            elif isinstance(r, dict):
                                plugin_results.append(
                                    RetrievalResult(
                                        chunk_id=r.get("chunk_id", 0),
                                        content=r.get("content", ""),
                                        file_path=r.get("file_path", ""),
                                        absolute_path=r.get("absolute_path", r.get("file_path", "")),
                                        section=r.get("section", ""),
                                        start_pos=r.get("start_pos", 0),
                                        end_pos=r.get("end_pos", 0),
                                        vault_name=r.get("vault_name", ""),
                                        chunk_type=r.get("chunk_type", ""),
                                        semantic_score=r.get("semantic_score", r.get("vector_score", 0.0)),
                                        keyword_score=r.get("keyword_score", 0.0),
                                        confidence_score=r.get("confidence_score", 1.0),
                                        final_score=r.get("final_score", r.get("score", 0.0)),
                                        confidence_reason=r.get("confidence_reason", ""),
                                        file_hash=r.get("file_hash", ""),
                                        graph_score=r.get("graph_score", 0.0),
                                        preference_score=r.get("preference_score", 0.0),
                                        hop_distance=r.get("hop_distance", 0),
                                        # 基础检索分数
                                        base_final_score=r.get(
                                            "base_final_score", r.get("final_score", r.get("score", 0.0))
                                        ),
                                    )
                                )
                        if plugin_results:
                            results = plugin_results
                            logger.info(f"✅ MCP 插件增强了 {len(results)} 条结果（统一评分 + 图谱分值）")
            except Exception as e:
                logger.warning(f"⚠️ MCP 插件增强失败: {e}")

        output_results = []
        for i, r in enumerate(results):
            result_item = {
                "rank": i + 1,
                "file": r.file_path,
                "abs_path": r.absolute_path,
                "content": r.content[:300],
                "score": round(r.final_score, 4),
                "semantic_score": round(r.semantic_score, 4),
                "keyword_score": round(r.keyword_score, 4),
                "confidence": round(r.confidence_score, 4),
                "confidence_reason": r.confidence_reason,
            }
            if plugin_enabled:
                result_item["graph_score"] = round(getattr(r, "graph_score", 0.0), 4)
                result_item["preference_score"] = round(getattr(r, "preference_score", 0.0), 4)
                result_item["hop_distance"] = getattr(r, "hop_distance", 0)
                # 添加基础检索分数便于调试和验证
                result_item["base_final_score"] = round(getattr(r, "base_final_score", r.final_score), 4)
            output_results.append(result_item)

        return {
            "query": query,
            "total": len(results),
            "plugin_enabled": plugin_enabled,
            "results": output_results,
        }


class ScanTool(BaseTool):
    name, description = "scan_index", "Incrementally scan and update file index (tinyRAG)"
    schema: ClassVar[dict] = {"type": "object", "properties": {}}

    def _process_file_worker(self, file_item: dict, splitter: MarkdownSplitter) -> tuple[int, list, str, str]:
        """处理单个文件，返回 (file_id, chunks, file_path, absolute_path)"""
        abs_path = Path(file_item["absolute_path"])
        if not abs_path.exists():
            logger.warning(f"⚠️ 文件不存在，跳过：{abs_path}")
            return file_item["id"], [], file_item["file_path"], file_item["absolute_path"]
        try:
            content = abs_path.read_text(encoding="utf-8")
            chunks = splitter.split(content, file_item.get("mtime"))
            return file_item["id"], chunks, file_item["file_path"], file_item["absolute_path"]
        except Exception as e:
            logger.error(f"❌ 读取/分块失败：{abs_path} - {type(e).__name__}: {e}")
            return file_item["id"], [], file_item["file_path"], file_item["absolute_path"]

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
            for f_id, chunks, f_path, abs_path in results:
                for c in chunks:
                    all_pending_chunks.append((f_id, c, f_path, abs_path))
        if not all_pending_chunks:
            return

        batch_size = self.ctx.config.embedding_model.batch_size
        processed = 0
        inserted_chunk_ids = []
        file_chunks_collector: dict = {}

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
                for idx, ((file_id, chunk, f_path, abs_path), emb) in enumerate(zip(batch, embeddings, strict=False)):
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
                    inserted_chunk_ids.append((new_chunk_id, file_id, chunk, f_path))

                    if file_id not in file_chunks_collector:
                        file_chunks_collector[file_id] = {"chunks": [], "file_path": f_path, "absolute_path": abs_path}
                    file_chunks_collector[file_id]["chunks"].append(
                        {
                            "id": new_chunk_id,
                            "content": chunk.content,
                            "metadata": chunk.metadata,
                        }
                    )

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

                if self.ctx.plugin_loader and inserted_chunk_ids:
                    try:
                        for chunk_id, file_id, chunk_obj, f_path in inserted_chunk_ids:
                            self.ctx.plugin_loader.invoke_hook(
                                "on_chunks_indexed",
                                chunk_id=chunk_id,
                                file_id=file_id,
                                content=chunk_obj.content,
                                metadata=chunk_obj.metadata,
                            )
                    except Exception as e:
                        logger.warning(f"⚠️ 插件钩子执行失败: {e}")
                inserted_chunk_ids.clear()

            except Exception as e:
                self.ctx.db.conn.rollback()
                logger.error(f"❌ 批次提交失败：{e}")
                raise
            finally:
                self.ctx.db.conn.execute("PRAGMA synchronous = NORMAL;")

        if self.ctx.plugin_loader and file_chunks_collector:
            try:
                logger.info(f"🔧 触发插件 on_file_indexed 钩子处理 {len(file_chunks_collector)} 个文件...")
                for file_id, data in file_chunks_collector.items():
                    self.ctx.plugin_loader.invoke_hook(
                        "on_file_indexed",
                        file_id=file_id,
                        chunks=data["chunks"],
                        filepath=data["file_path"],
                        absolute_path=data.get("absolute_path", ""),  # 传递绝对路径
                    )
                logger.info("✅ 插件钩子处理完成")
            except Exception as e:
                logger.warning(f"⚠️ on_file_indexed 钩子执行失败: {e}")

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
            logger.info("Starting background index rebuild...")

            # 临时关闭主数据库连接，避免冲突
            if self.ctx.db:
                self.ctx.db.close()
                self.ctx.db = None

            build_main = _load_build_index_main()
            import argparse

            batch_size = self.ctx.config.embedding_model.batch_size
            build_args = argparse.Namespace(force=True, batch_size=batch_size)
            await asyncio.to_thread(build_main, build_args)

            # 重建后重新初始化数据库连接
            await self.ctx.initialize()

            logger.info("Index rebuild completed successfully")
        except Exception as e:
            logger.error(f"Index rebuild failed: {e}", exc_info=True)
            # 尝试恢复数据库连接
            try:
                await self.ctx.initialize()
            except:
                pass
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


class ConfigTool(BaseTool):
    name, description = "config", "Get complete tinyRAG configuration (includes exclude, cache, maintenance, etc.)"
    schema: ClassVar[dict] = {"type": "object", "properties": {}}

    async def run(self, args: dict[str, Any]) -> dict[str, Any]:
        await self.ctx.initialize()
        cfg_dict = self.ctx.config.model_dump(mode="json")
        if "vaults" in cfg_dict and isinstance(cfg_dict["vaults"], list):
            for v in cfg_dict["vaults"]:
                if not v.get("exclude") or v["exclude"] is None:
                    v["exclude"] = {"dirs": [], "patterns": []}
        return cfg_dict


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
        except ValueError:
            return json.dumps({"error": "Invalid file_id"})
        row = self.ctx.db.conn.execute("SELECT * FROM files WHERE id = ? AND is_deleted = 0", (fid,)).fetchone()
        if not row:
            return json.dumps({"error": "File not found"})
        try:
            content = Path(row["absolute_path"]).read_text(encoding="utf-8")[:10000]
        except OSError:
            content = "[Error reading file]"
        return json.dumps({**dict(row), "content_preview": content}, ensure_ascii=False, indent=2)

    def _get_file_chunks(self, file_id: str) -> str:
        try:
            fid = int(file_id)
        except ValueError:
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
        except OSError:
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
                arguments=[PromptArgument(name="query", required=True), PromptArgument(name="top_k", required=False)],
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

    def _prompt_search(self, args: dict) -> GetPromptResult:
        query = args.get("query", "")
        top_k = int(args.get("top_k", "5"))
        results = self.ctx.retriever.search(query, limit=top_k)
        ctx = (
            "\n".join([f"【文档 {i + 1}】{r.file_path}\n内容: {r.content[:300]}" for i, r in enumerate(results)])
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
# Helpers
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
        self._plugin_tools_registered = False

    def register(self, cls):
        t = cls(self.ctx)
        self.tools[t.name] = t

    def list_tools(self):
        self._register_plugin_tools()
        return [t.to_mcp_tool() for t in self.tools.values()]

    async def execute(self, name: str, args: dict[str, Any]) -> list[TextContent]:
        self._register_plugin_tools()
        if name not in self.tools:
            raise ValueError(f"Unknown tool: {name}")
        result = await self.tools[name].run(args)
        return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, default=str))]

    def _register_plugin_tools(self) -> None:
        if self._plugin_tools_registered:
            return
        self._plugin_tools_registered = True

        if self.ctx.plugin_loader:
            try:
                count = self.ctx.plugin_loader.register_tools_to_registry(self)
                if count > 0:
                    logger.info(f"✅ 已注册 {count} 个插件工具")
            except Exception as e:
                logger.warning(f"⚠️ 注册插件工具失败: {e}")


class RagServer:
    def __init__(self):
        self.ctx = AppContext()
        self.registry = ToolRegistry(self.ctx)
        self.registry.register(SearchTool)
        self.registry.register(ScanTool)
        self.registry.register(RebuildTool)
        self.registry.register(StatsTool)
        self.registry.register(ConfigTool)
        self.registry.register(ReloadConfigTool)
        self.registry.register(MaintenanceTool)
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
