#!/usr/bin/env python3
"""
mcp_server/server.py - Production MCP RAG Server
修复: sys.path hack, 后台任务清理, 路径解析, 异步安全
"""

import asyncio
import importlib.util
import json
import sys
import traceback
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
from retriever.hybrid_engine import HybridRetriever
from scanner.scan_engine import Scanner
from storage.database import DatabaseManager

# =====================
# Logger & Config
# =====================
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
log_file = PROJECT_ROOT / "logs" / "mcp_server.log"
logger = setup_logger(level="INFO", log_file=str(log_file))


def _load_build_index_main():
    """安全加载 build_index 模块,避免 sys.path 污染"""
    module_path = PROJECT_ROOT / "build_index.py"
    if not module_path.exists():
        raise ImportError("build_index.py 未找到")
    spec = importlib.util.spec_from_file_location("build_index", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_index"] = module
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
        self.retriever: HybridRetriever | None = None
        self.splitter: MarkdownSplitter | None = None
        self._initialized = False
        self._lock = asyncio.Lock()
        self._background_tasks: list[asyncio.Task] = []

    async def initialize(self):
        if self._initialized:
            return
        async with self._lock:
            if self._initialized:
                return
            try:
                config_path = PROJECT_ROOT / "config.yaml"
                self.config = load_config(str(config_path))
                self.db = DatabaseManager(self.config.db_path)
                self.scanner = Scanner(self.db)
                self.retriever = HybridRetriever(
                    db=self.db,
                    alpha=self.config.confidence.fusion["alpha"],
                    beta=self.config.confidence.fusion["beta"],
                    model_name=self.config.embedding_model.name,
                    cache_dir=self.config.embedding_model.cache_dir,
                )
                self.splitter = MarkdownSplitter(
                    max_tokens=self.config.chunking["max_tokens"],
                    overlap=self.config.chunking["overlap"],
                )
                self._initialized = True
                logger.success("✅ MCP 组件初始化完成")
            except Exception as e:
                logger.critical(f"❌ 初始化失败: {e}")
                raise

    def add_background_task(self, task: asyncio.Task):
        self._background_tasks.append(task)
        task.add_done_callback(lambda t: self._background_tasks.remove(t))

    async def shutdown(self):
        logger.info("🛑 正在关闭服务...")
        # 安全取消并等待后台任务
        if self._background_tasks:
            for t in self._background_tasks:
                t.cancel()
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        if self.db:
            self.db.close()
        logger.info("✅ 资源释放完成")


# =====================
# Error Wrapper & Tool Base
# =====================
def mcp_safe(func: Callable[..., Awaitable[list[TextContent]]]):
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Tool error: {e}", exc_info=True)
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {"error": str(e), "traceback": traceback.format_exc()},
                        ensure_ascii=False,
                    ),
                )
            ]

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


class SearchTool(BaseTool):
    name, description = "search", "Hybrid knowledge retrieval with RRF fusion"
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

    async def run(self, args):
        await self.ctx.initialize()
        query, mode, top_k = (
            args.get("query", ""),
            args.get("mode", "hybrid"),
            min(max(args.get("top_k", 10), 1), 100),
        )
        results = await asyncio.to_thread(self.ctx.retriever.search, query, mode=mode, top_k=top_k)
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
                }
                for i, r in enumerate(results)
            ],
        }


class ScanTool(BaseTool):
    name, description = "scan_index", "Incrementally scan and update file index"
    schema: ClassVar[dict] = {"type": "object", "properties": {}}

    async def run(self, args):
        await self.ctx.initialize()
        vault_configs = [(f"v_{i}", p) for i, p in enumerate(self.ctx.config.vaults)] if self.ctx.config else []
        report = await asyncio.to_thread(self.ctx.scanner.scan_vaults, vault_configs)
        await asyncio.to_thread(self.ctx.scanner.process_report, report)
        return {
            "status": "success",
            "new": len(report.new_files),
            "modified": len(report.modified_files),
            "moved": len(report.moved_files),
            "deleted": len(report.deleted_files),
        }


class RebuildTool(BaseTool):
    name, description = "rebuild_index", "Force rebuild full knowledge index"
    schema: ClassVar[dict] = {"type": "object", "properties": {}}

    async def run(self, args):
        task = asyncio.create_task(self._background_job())
        self.ctx.add_background_task(task)
        return {"status": "started", "message": "Index rebuild running in background"}

    async def _background_job(self):
        try:
            await self.ctx.initialize()
            logger.info("🔄 开始后台强制重建索引...")
            build_main = _load_build_index_main()

            # 模拟 argparse.Namespace 传递参数
            class BuildArgs:
                force, batch_size = True, 128

            build_main(BuildArgs)
            logger.success("✅ 索引重建完成")
        except Exception as e:
            logger.error(f"❌ 索引重建失败：{e}", exc_info=True)
            raise


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
        return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]


class RagServer:
    def __init__(self):
        self.ctx = AppContext()
        self.registry = ToolRegistry(self.ctx)
        self.registry.register(SearchTool)
        self.registry.register(ScanTool)
        self.registry.register(RebuildTool)

        if MCP_AVAILABLE:
            self.server = Server("rag-system")
            self._register()
        else:
            logger.warning("⚠️ MCP 未安装，运行在 Mock 模式")

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
            logger.error("❌ 请安装 mcp: pip install mcp")
            return
        logger.info("🚀 MCP Stdio Server 启动中...")
        try:
            async with stdio_server() as (r, w):
                await self.server.run(r, w, self.server.create_initialization_options())
        except KeyboardInterrupt:
            logger.info("👋 收到中断信号")
        finally:
            await self.ctx.shutdown()


if __name__ == "__main__":
    asyncio.run(RagServer().run())
