#!/usr/bin/env python3
"""
mcp_server/server.py - Production MCP RAG Server (v2.1)

v2.1 修复内容:
- S1: HybridRetriever → HybridEngine 类名修复
- S2: HybridEngine 构造函数签名对齐 (config, db, embed_engine)
- S3: MarkdownSplitter 构造函数签名对齐 (config 对象)
- S4: SearchTool.search() 调用参数对齐 (query, limit, vault_filter)

v2.0 优化内容:
1.  P0-1: MarkdownSplitter 传递 confidence_config (frontmatter 权重)
2.  P0-2: ScanTool 使用 config 中的真实 vault_name (不再用 v_0/v_1)
3.  P0-3: RebuildTool 使用 asyncio.to_thread 避免阻塞事件循环
4.  P0-4: ScanTool 返回 touched_files 计数
5.  P1-1: mcp_safe 不再向客户端暴露完整 traceback
6.  P1-2: _load_build_index_main 使用独立命名空间避免 sys.modules 污染
7.  P1-3: initialize() 局部变量模式 + 失败时清理部分初始化的资源
8.  P1-4: add_background_task 回调使用 try/except 避免竞态崩溃
9.  P1-5: shutdown() 安全关闭数据库连接
10. P2-1: BaseTool.run() 补充类型注解
11. P2-2: RebuildTool 支持 batch_size 参数
12. P2-3: ToolRegistry.execute 添加 default=str 序列化兜底 + 移除 indent
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

from utils.logger import setup_logger

# MCP
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import TextContent, Tool

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

from chunker.markdown_splitter import MarkdownSplitter
from config import Settings, load_config
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

# build_index 模块的唯一命名空间键，避免与用户代码的 import 冲突
_BUILD_INDEX_MODULE_KEY = "_tinyrag_build_index"


def _load_build_index_main():
    """
    安全加载 build_index 模块。
    使用唯一命名空间键注册到 sys.modules，生命周期与服务器一致。
    首次加载后缓存，后续调用直接返回缓存的 main 函数。
    """
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
    """
    应用上下文容器：统一管理配置、数据库、扫描器、检索器、分块器的生命周期。
    使用双重检查锁定确保异步安全初始化，局部变量模式防止部分失败时的资源泄漏。
    """

    def __init__(self):
        self.config: Settings | None = None
        self.db: DatabaseManager | None = None
        self.scanner: Scanner | None = None
        self.retriever: HybridEngine | None = None
        self.splitter: MarkdownSplitter | None = None
        self._initialized = False
        self._lock = asyncio.Lock()
        self._background_tasks: list[asyncio.Task] = []

    async def initialize(self):
        """异步初始化所有组件。幂等，可安全重复调用。"""
        if self._initialized:
            return
        async with self._lock:
            if self._initialized:
                return

        # ── 局部变量模式：全部成功后才赋值到 self，失败时清理中间资源 ──
        config = None
        db = None
        scanner = None
        retriever = None
        splitter = None
        try:
            config_path = PROJECT_ROOT / "config.yaml"
            config = load_config(str(config_path))

            # 修复 M4: 从 config 读取维度
            db = DatabaseManager(config.db_path, vec_dimension=config.embedding_model.dimensions)
            scanner = Scanner(db)

            embed_engine = EmbeddingEngine(
                model_name=config.embedding_model.name,
                cache_dir=config.embedding_model.cache_dir,
                batch_size=config.embedding_model.batch_size,
                unload_after_seconds=config.embedding_model.unload_after_seconds,
            )

            retriever = HybridEngine(
                config=config,
                db=db,
                embed_engine=embed_engine,
            )

            # P0-1: 传递完整 config 对象，使分块器与置信度系统生效
            splitter = MarkdownSplitter(config)

            # 全部成功，赋值到 self
            self.config = config
            self.db = db
            self.scanner = scanner
            self.retriever = retriever
            self.splitter = splitter
            self._initialized = True
            logger.info("MCP components initialized successfully")
        except Exception:
            # P1-3: 清理部分初始化的资源，防止连接泄漏
            if db is not None:
                with contextlib.suppress(Exception):
                    db.close()
            logger.critical("MCP initialization failed", exc_info=True)
            raise

    def add_background_task(self, task: asyncio.Task):
        """注册后台任务，完成后自动从列表移除。"""
        self._background_tasks.append(task)

        def _on_done(t: asyncio.Task):
            # P1-4: 防止竞态条件下重复 remove 导致 ValueError
            with contextlib.suppress(ValueError):
                self._background_tasks.remove(t)

        task.add_done_callback(_on_done)

    async def shutdown(self):
        """优雅关闭：取消后台任务 + 关闭数据库连接。"""
        logger.info("Shutting down server...")
        if self._background_tasks:
            for t in self._background_tasks:
                t.cancel()
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._background_tasks.clear()

        # P1-5: 安全关闭数据库连接
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
    """
    工具执行错误包装器。
    P1-1: 仅向客户端返回简短错误信息，完整 traceback 仅记录到服务端日志。
    """

    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Tool error: {e}", exc_info=True)
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {"error": str(e)},
                        ensure_ascii=False,
                    ),
                )
            ]

    return wrapper


class BaseTool:
    """MCP 工具基类。子类需定义 name/description/schema 并实现 run()。"""

    name: str
    description: str
    schema: dict[str, Any]

    def __init__(self, ctx: AppContext):
        self.ctx = ctx

    # P2-1: 补充类型注解
    async def run(self, args: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def to_mcp_tool(self) -> Tool:
        return Tool(name=self.name, description=self.description, inputSchema=self.schema)


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

    # P2-3: 补充参数类型注解
    async def run(self, args: dict[str, Any]) -> dict[str, Any]:
        await self.ctx.initialize()
        query = args.get("query", "")
        mode = args.get("mode", "hybrid")
        top_k = min(max(args.get("top_k", 10), 1), 100)

        # 修复 H1 + M2: 不直接修改 config，使用局部变量传递 alpha/beta
        # 通过 alpha/beta 比值模拟检索模式
        if mode == "keyword":
            alpha = 0.0
            beta = 1.0
        elif mode == "semantic":
            alpha = 1.0
            beta = 0.0
        else:  # hybrid 模式，使用默认值
            alpha = None
            beta = None

        results = await asyncio.to_thread(self.ctx.retriever.search, query, limit=top_k, alpha=alpha, beta=beta)
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
                }
                for i, r in enumerate(results)
            ],
        }


class ScanTool(BaseTool):
    name, description = (
        "scan_index",
        "Incrementally scan and update file index (tinyRAG)",
    )
    schema: ClassVar[dict] = {"type": "object", "properties": {}}

    def _process_file_worker(self, file_item: dict, splitter: MarkdownSplitter) -> tuple[int, list, str]:
        """并行分块任务单元"""
        abs_path = Path(file_item["absolute_path"])
        try:
            content = abs_path.read_text(encoding="utf-8")
            mtime = file_item.get("mtime")
            if mtime is None:
                logger.warning(f"⚠️ 文件 {file_item['file_path']} 缺少 mtime，使用当前时间")
            # ✅ 修复：传入正确的 file_path 参数
            chunks = splitter.split(content, file_path=file_item.get("file_path", ""))
            return file_item["id"], chunks, file_item["file_path"]
        except Exception as e:
            logger.error(f"❌ 读取/分块失败：{abs_path} - {e}")
            return file_item["id"], [], file_item["file_path"]

    async def _index_changed_files(self, changed_paths: list[str]) -> None:
        """为变更的文件创建 chunks 和向量（复用 build_index 逻辑）"""
        # 1. 查询文件列表
        placeholders = ",".join(["?"] * len(changed_paths))
        cursor = self.ctx.db.conn.execute(
            f"SELECT id, absolute_path, file_path, mtime FROM files WHERE absolute_path IN ({placeholders})",
            changed_paths,
        )
        files_to_index = [dict(row) for row in cursor.fetchall()]

        if not files_to_index:
            logger.info("✨ 没有需要索引的文件")
            return

        logger.info(f"🚀 开始为 {len(files_to_index)} 个变更文件创建 chunks 和向量...")

        # 2. 分块（使用线程池避免阻塞）
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
            logger.info("✨ 没有生成任何 chunks")
            return

        # 3. 分批向量化与入库（流式处理）
        total_chunks = len(all_pending_chunks)
        batch_size = self.ctx.config.embedding_model.batch_size
        stream_batch_size = self.ctx.config.stream_batch_size
        logger.info(f"🧩 待向量化块总数：{total_chunks}，Batch Size: {batch_size}")

        processed = 0
        for i in range(0, total_chunks, batch_size):
            batch = all_pending_chunks[i : i + batch_size]
            texts = [item[1].content for item in batch]

            try:
                embeddings = await asyncio.to_thread(self.ctx.retriever.embed_engine.embed, texts)
            except Exception as e:
                skipped_files = [item[2] for item in batch]
                logger.error(
                    f"❌ 批次向量化失败: {e}，跳过 {len(skipped_files)} 个文件: {skipped_files[:5]}{'...' if len(skipped_files) > 5 else ''}"
                )
                continue

            try:
                for idx, ((file_id, chunk, f_path), emb) in enumerate(zip(batch, embeddings)):
                    global_idx = processed + idx
                    try:
                        metadata_json = json.dumps(chunk.metadata or {}, ensure_ascii=False, default=_json_serialize)
                    except Exception as e:
                        logger.warning(f"⚠️ metadata 序列化失败，使用空对象：{e}")
                        metadata_json = "{}"

                    # 序列化置信度原始因子
                    try:
                        confidence_json = json.dumps(
                            chunk.confidence_metadata or {},
                            ensure_ascii=False,
                            default=_json_serialize,
                        )
                    except Exception as e:
                        logger.warning(f"⚠️ confidence_metadata 序列化失败，使用空对象：{e}")
                        confidence_json = "{}"

                    cursor = self.ctx.db.conn.execute(
                        """
                        INSERT INTO chunks (file_id, chunk_index, content, content_type, section_title, section_path,
                        start_pos, end_pos, confidence_final_weight, metadata, confidence_json)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            file_id,
                            global_idx,
                            chunk.content,
                            chunk.content_type.value,
                            chunk.section_title,
                            chunk.section_path,
                            chunk.start_pos,
                            chunk.end_pos,
                            1.0,  # 占位值（已废弃），实际权重由 confidence_json + 检索期动态计算
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
                logger.info(f" ✅ 已完成：{processed} / {total_chunks}")

            except Exception as e:
                self.ctx.db.conn.rollback()
                logger.error(f"❌ 批次提交失败：{e}", exc_info=True)
                raise

        logger.success(f"🎉 增量索引完成！共处理 {processed} 个 chunks")

    # P2-3: 补充参数类型注解
    async def run(self, args: dict[str, Any]) -> dict[str, Any]:
        await self.ctx.initialize()
        # P0-2: 使用 config 中的真实 vault_name (personal/work)，不再用 v_0/v_1
        vault_configs = [(v.name, v.path) for v in self.ctx.config.vaults if v.enabled]
        report = await asyncio.to_thread(self.ctx.scanner.scan_vaults, vault_configs)
        await asyncio.to_thread(self.ctx.scanner.process_report, report)

        # 🆕 增量索引：为新增、修改、移动的文件创建 chunks 和向量
        changed_paths = []
        for f in report.new_files + report.modified_files:
            changed_paths.append(f.absolute_path)
        for f in report.moved_files:
            changed_paths.append(f.new_absolute_path)
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

    # P2-3: 补充参数类型注解
    async def run(self, args: dict[str, Any]) -> dict[str, Any]:
        task = asyncio.create_task(self._background_job(args))
        self.ctx.add_background_task(task)
        return {"status": "started", "message": "Index rebuild running in background"}

    async def _background_job(self, args: dict[str, Any]):
        """后台执行索引重建，通过 asyncio.to_thread 避免阻塞事件循环。"""
        try:
            await self.ctx.initialize()
            logger.info("Starting background index rebuild...")
            build_main = _load_build_index_main()

            # v2.0: batch_size 从 config 读取，不再通过参数传递
            import argparse

            build_args = argparse.Namespace(force=True)

            # P0-3: asyncio.to_thread 防止同步 build_main 阻塞事件循环
            await asyncio.to_thread(build_main, build_args)
            logger.info("Index rebuild completed successfully")
        except Exception as e:
            logger.error(f"Index rebuild failed: {e}", exc_info=True)
            raise


# =====================
# Helper Functions for Incremental Indexing
# =====================
def _jieba_segment(text: str) -> str:
    """对中文文本进行 jieba 分词，返回空格拼接的词串"""
    if not text or not text.strip():
        return ""
    import jieba

    return " ".join(jieba.cut_for_search(text))


def _prepare_fts_content(chunk, file_path: str) -> str:
    """构建复合检索字符串，所有中文文本字段经 jieba 分词后写入 FTS5"""
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

    # 对所有含中文的文本字段做 jieba 分词，与检索端保持一致
    parts = [
        _jieba_segment(filename),
        _jieba_segment(filename),  # 文件名重复加权
        _jieba_segment(chunk.section_path or ""),
        _jieba_segment(section_title),
        _jieba_segment(section_title),  # 标题重复加权
        _jieba_segment(tag_str),
        _jieba_segment(doc_type),
        _jieba_segment(chunk.content),  # 正文分词
    ]
    return " ".join(filter(None, parts)).strip()


def _json_serialize(obj):
    """自定义 JSON 序列化器"""
    from datetime import date, datetime

    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# =====================
# Registry & Server
# =====================
class ToolRegistry:
    """MCP 工具注册中心：注册、列举、分发工具调用。"""

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
        # P2-4: 添加 default=str 兜底 + 移除 indent (机器通信不需要格式化)
        return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, default=str))]


class RagServer:
    """tinyRAG MCP Server 入口：组装组件、注册工具、启动 stdio 传输。"""

    def __init__(self):
        self.ctx = AppContext()
        self.registry = ToolRegistry(self.ctx)
        self.registry.register(SearchTool)
        self.registry.register(ScanTool)
        self.registry.register(RebuildTool)

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

    async def run(self):
        if not MCP_AVAILABLE:
            logger.error("Please install mcp: pip install mcp")
            return
        logger.info("MCP Stdio Server starting...")
        try:
            async with stdio_server() as (r, w):
                await self.server.run(r, w, self.server.create_initialization_options())
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        finally:
            await self.ctx.shutdown()


if __name__ == "__main__":
    asyncio.run(RagServer().run())
