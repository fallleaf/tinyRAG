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
import sys
from collections.abc import Awaitable, Callable
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

            db = DatabaseManager(config.db_path)
            scanner = Scanner(db)

            embed_engine = EmbeddingEngine(
                model_name=config.embedding_model.name,
                cache_dir=config.embedding_model.cache_dir,
                batch_size=32,
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
        return Tool(
            name=self.name, description=self.description, inputSchema=self.schema
        )


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

        # 通过 alpha/beta 比值模拟检索模式
        if mode == "keyword":
            self.ctx.config.retrieval["alpha"] = 0.0
            self.ctx.config.retrieval["beta"] = 1.0
        elif mode == "semantic":
            self.ctx.config.retrieval["alpha"] = 1.0
            self.ctx.config.retrieval["beta"] = 0.0

        results = await asyncio.to_thread(self.ctx.retriever.search, query, limit=top_k)
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

    # P2-3: 补充参数类型注解
    async def run(self, args: dict[str, Any]) -> dict[str, Any]:
        await self.ctx.initialize()
        # P0-2: 使用 config 中的真实 vault_name (personal/work)，不再用 v_0/v_1
        vault_configs = [(v.name, v.path) for v in self.ctx.config.vaults if v.enabled]
        report = await asyncio.to_thread(self.ctx.scanner.scan_vaults, vault_configs)
        await asyncio.to_thread(self.ctx.scanner.process_report, report)
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
    schema: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "batch_size": {
                "type": "integer",
                "default": 128,
                "minimum": 16,
                "maximum": 512,
                "description": "Embedding batch size for vectorization",
            },
        },
    }

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

            # P2-2: 支持通过参数自定义 batch_size
            batch_size = min(max(args.get("batch_size", 128), 16), 512)

            class BuildArgs:
                force = True
                batch_size = batch_size

            # P0-3: asyncio.to_thread 防止同步 build_main 阻塞事件循环
            await asyncio.to_thread(build_main, BuildArgs)
            logger.info("Index rebuild completed successfully")
        except Exception as e:
            logger.error(f"Index rebuild failed: {e}", exc_info=True)
            raise


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
        return [
            TextContent(
                type="text", text=json.dumps(result, ensure_ascii=False, default=str)
            )
        ]


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
