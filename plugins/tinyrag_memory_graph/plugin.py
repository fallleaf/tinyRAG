#!/usr/bin/env python3
"""
plugin.py - Memory Graph 插件主类

实现插件的核心逻辑，整合所有功能模块。
继承 PluginBase 以支持插件加载器。
"""

import hashlib
import sqlite3
import time

from loguru import logger

from plugins.bootstrap import PluginBase  # 继承插件基类
from plugins.tinyrag_memory_graph.config import MemoryGraphConfig
from plugins.tinyrag_memory_graph.graph_builder import GraphBuilder
from plugins.tinyrag_memory_graph.hooks import HookContext, HookResult
from plugins.tinyrag_memory_graph.hybrid_retriever import HybridRetriever
from plugins.tinyrag_memory_graph.memify import MemifyEngine, PrincipleManager
from plugins.tinyrag_memory_graph.metrics import MetricsCollector, get_metrics_collector
from plugins.tinyrag_memory_graph.storage import GraphStorage


class MemoryGraphPlugin(PluginBase):
    """
    Memory Graph 插件主类

    提供图 - 向量混合记忆功能，通过钩子协议与 tinyRAG 核心集成。

    使用示例:
        config = MemoryGraphConfig.from_yaml("config.yaml")
        plugin = MemoryGraphPlugin(config)

        # 设置数据库连接
        plugin.set_db_connection(db_conn)

        # 启动插件
        await plugin.start()

        # 处理钩子
        result = await plugin.on_add_document(ctx)
    """

    # 插件元信息 (覆盖基类)
    NAME = "tinyrag_memory_graph"
    VERSION = "1.0.0"
    DESCRIPTION = "Graph-Vector Hybrid Memory Plugin for tinyRAG"

    def __init__(self, config: dict | None = None, context: object | None = None):
        """
        初始化插件

        Args:
            config: 插件配置字典（来自 config.yaml 中的 plugins.config）
            context: AppContext 上下文对象
        """
        # 调用基类初始化
        super().__init__(config, context)

        # 解析配置 - 支持多种格式
        if isinstance(config, dict):
            # 兼容格式：{graph: {...}, retrieval: {...}, memify: {...}}
            if "graph" in config and "enabled" not in config:
                # 从 graph 中提取 enabled
                config = dict(config)  # 复制一份
                if "graph" in config:
                    config["enabled"] = config["graph"].get("enabled", True)
            self.plugin_config = MemoryGraphConfig.from_dict(config)
        elif isinstance(config, MemoryGraphConfig):
            self.plugin_config = config
        else:
            self.plugin_config = MemoryGraphConfig()

        self._initialized = False
        self._started = False

        # 组件（延迟初始化）
        self._db: sqlite3.Connection | None = None
        self._storage: GraphStorage | None = None
        self._graph_builder: GraphBuilder | None = None
        self._hybrid_retriever: HybridRetriever | None = None
        self._memify_engine: MemifyEngine | None = None
        self._principle_manager: PrincipleManager | None = None
        self._metrics: MetricsCollector | None = None

        # 统计
        self._stats = {
            "documents_processed": 0,
            "entities_extracted": 0,
            "relations_created": 0,
            "searches_enhanced": 0,
        }

        # 注册钩子
        self.register_hook("on_file_indexed", self._on_file_indexed_hook)
        self.register_hook("on_search", self._on_search_hook)
        self.register_hook("on_response_generated", self._on_response_hook)
        self.register_hook("on_index_rebuild", self._on_index_rebuild_hook)

    def on_load(self) -> bool:
        """
        插件加载时调用
        """
        # 如果有上下文，尝试获取数据库连接
        if self.ctx and hasattr(self.ctx, "db") and self.ctx.db:
            self._db = self.ctx.db.conn if hasattr(self.ctx.db, "conn") else self.ctx.db
        return True

    def on_enable(self) -> bool:
        """
        插件启用时调用
        """
        if not self._db and self.ctx and hasattr(self.ctx, "db"):
            self._db = self.ctx.db.conn if hasattr(self.ctx.db, "conn") else self.ctx.db

        if self._db:
            try:
                # 同步初始化
                self._initialize_sync()
                return True
            except Exception as e:
                # 打印错误信息，便于调试
                logger.error(f"[MemoryGraphPlugin] ❌ 初始化失败：{e}")
                import traceback

                traceback.print_exc()
                return False
        return True  # 允许延迟初始化（等待 set_db_connection）

    def on_disable(self) -> None:
        """插件禁用时调用"""
        if self._started:
            # 直接停止后台任务，不依赖异步上下文
            if self._graph_builder:
                try:
                    self._graph_builder.stop()
                except Exception as e:
                    logger.warning(f"[MemoryGraphPlugin] ⚠️ 停止图谱构建器失败：{e}")

            # 停止代谢引擎（同步方式）
            if self._memify_engine:
                try:
                    import asyncio

                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(self._memify_engine.stop())
                    finally:
                        loop.close()
                except Exception as e:
                    logger.warning(f"[MemoryGraphPlugin] ⚠️ 停止代谢引擎失败：{e}")

            self._started = False

    def on_unload(self) -> None:
        """插件卸载时调用"""
        self.on_disable()
        self._initialized = False
        self._started = False

    def _initialize_sync(self):
        """同步初始化组件"""
        if self._initialized:
            return

        if not self._db:
            raise RuntimeError("Database connection not set")

        # 初始化存储层
        self._storage = GraphStorage(self._db, self.plugin_config)
        self._storage.initialize()

        # 初始化图谱构建器
        self._graph_builder = GraphBuilder(self._db, self.plugin_config)
        self._graph_builder.start()  # 启动后台处理线程

        # 初始化混合检索器
        self._hybrid_retriever = HybridRetriever(self._db, self.plugin_config)

        # 初始化代谢引擎
        self._memify_engine = MemifyEngine(self._db, self.plugin_config)

        # 初始化原则管理器
        self._principle_manager = PrincipleManager(self._db)

        # 初始化指标收集器
        if self.plugin_config.metrics.enabled:
            self._metrics = get_metrics_collector()

        self._initialized = True

    # 钩子处理函数
    def _on_index_rebuild_hook(self, force: bool = False, **kwargs):
        """索引重建时清理图谱数据"""
        if not force:
            return None

        if not self._db:
            return None

        # 确保已初始化（表已创建）
        if not self._initialized:
            try:
                self._initialize_sync()
            except Exception as e:
                logger.warning(f"[MemoryGraphPlugin] ⚠️ 初始化失败，跳过清理：{e}")
                return None

        try:
            # 清理所有图谱相关的表
            # 注意：chunks 表可能有 note_id 列引用 notes(note_id)，需要先处理
            # 临时禁用外键约束
            self._db.execute("PRAGMA foreign_keys = OFF")

            # 清理引用 notes 的 chunks.note_id 列（如果存在）
            from contextlib import suppress

            with suppress(Exception):
                self._db.execute("UPDATE chunks SET note_id = NULL WHERE note_id IS NOT NULL")

            # 清理插件表数据
            self._db.execute("DELETE FROM graph_build_jobs")
            self._db.execute("DELETE FROM relations")
            self._db.execute("DELETE FROM entities")
            self._db.execute("DELETE FROM notes")
            self._db.execute("DELETE FROM tag_co_occurrence")
            self._db.execute("DELETE FROM principles")
            self._db.commit()

            # 重新启用外键约束
            self._db.execute("PRAGMA foreign_keys = ON")
            logger.info("[MemoryGraphPlugin] 🧹 索引重建，已清理图谱数据")
            return {"cleaned": True}
        except Exception as e:
            logger.error(f"[MemoryGraphPlugin] ❌ 清理图谱数据失败：{e}")
            # 确保外键约束恢复
            from contextlib import suppress

            with suppress(Exception):
                self._db.execute("PRAGMA foreign_keys = ON")
            return None

    async def _on_file_indexed_hook(
        self, file_id: int, chunks: list, filepath: str = "", absolute_path: str = "", **kwargs
    ):
        """文件索引完成钩子 - 读取原始文件内容获取完整 frontmatter"""
        if not self.plugin_config.enabled:
            return None

        # 读取原始文件内容以获取完整 frontmatter（优先使用绝对路径）
        full_content = ""
        file_to_read = absolute_path or filepath
        if file_to_read:
            try:
                with open(file_to_read, encoding="utf-8") as f:
                    full_content = f.read()
            except Exception as e:
                logger.warning(f"[MemoryGraphPlugin] ⚠️ 无法读取文件 {file_to_read}: {e}")
                full_content = "\n".join(
                    [c.get("content", "") if isinstance(c, dict) else getattr(c, "content", "") for c in chunks]
                )

        # 提取 chunk_ids 和内容
        chunk_ids = [c.get("id") if isinstance(c, dict) else getattr(c, "id", 0) for c in chunks]
        chunks_content = [c.get("content", "") if isinstance(c, dict) else getattr(c, "content", "") for c in chunks]

        ctx = HookContext(
            document=full_content or "\n".join(chunks_content),
            chunk_ids=chunk_ids,
            chunks_content=chunks_content,
            metadata={"file_id": file_id, "filepath": filepath, "absolute_path": absolute_path, **kwargs},
        )
        result = await self.on_add_document(ctx)
        return result

    async def _on_search_hook(self, query: str, results: list, **kwargs):
        """搜索钩子"""
        if not self.plugin_config.enabled:
            return results

        ctx = HookContext(
            query=query,
            query_vec=kwargs.get("query_vec"),
            results=results,
            # 修复问题 1：接收基础检索的权重参数
            base_alpha=kwargs.get("base_alpha"),
            base_beta=kwargs.get("base_beta"),
        )
        result = await self.on_search_after(ctx)
        if result.modified:
            return ctx.results
        return results

    async def _on_response_hook(self, chunk_ids: list, **kwargs):
        """响应生成钩子"""
        if not self.plugin_config.enabled:
            return None

        ctx = HookContext()
        ctx.get_result_chunk_ids = lambda: chunk_ids
        return await self.on_response(ctx)

    def set_db_connection(self, db_conn: sqlite3.Connection):
        """
        设置数据库连接

        必须在 start() 之前调用。
        """
        self._db = db_conn

        # 如果插件已启用但未初始化，尝试初始化
        if self._enabled and not self._initialized:
            try:
                self._initialize_sync()
                logger.info("[MemoryGraphPlugin] ✅ 数据库连接已设置，Schema 初始化完成")
            except Exception as e:
                logger.error(f"[MemoryGraphPlugin] ❌ 初始化失败：{e}")
                import traceback

                traceback.print_exc()

    async def initialize(self):
        """初始化插件组件"""
        if self._initialized:
            return

        if not self._db:
            raise RuntimeError("Database connection not set. Call set_db_connection() first.")

        # 初始化存储层
        self._storage = GraphStorage(self._db, self.plugin_config)
        self._storage.initialize()

        # 初始化图谱构建器
        self._graph_builder = GraphBuilder(self._db, self.plugin_config)

        # 初始化混合检索器
        self._hybrid_retriever = HybridRetriever(self._db, self.plugin_config)

        # 初始化代谢引擎
        self._memify_engine = MemifyEngine(self._db, self.plugin_config)

        # 初始化原则管理器
        self._principle_manager = PrincipleManager(self._db)

        # 初始化指标收集器
        if self.plugin_config.metrics.enabled:
            self._metrics = get_metrics_collector()

        self._initialized = True

    async def start(self):
        """启动插件"""
        if not self._initialized:
            await self.initialize()

        if self._started:
            return

        # 启动后台处理
        if self._graph_builder:
            self._graph_builder.start()

        # 启动代谢引擎
        if self._memify_engine:
            await self._memify_engine.start()

        self._started = True

    async def stop(self):
        """停止插件"""
        if not self._started:
            return

        # 停止图谱构建器
        if self._graph_builder:
            self._graph_builder.stop()

        # 停止代谢引擎
        if self._memify_engine:
            await self._memify_engine.stop()

        self._started = False

    # ==================== 钩子实现 ====================

    async def on_add_document(self, ctx: HookContext) -> HookResult:
        """
        文档入库后钩子（FR-1.1）

        触发建图任务。在 CLI 模式下同步执行，确保任务完成后再退出。
        """
        if not self.plugin_config.enabled or not self._graph_builder:
            return HookResult.ok("Plugin disabled or not initialized")

        start_time = time.time()

        try:
            # 生成 note_id
            note_id = hashlib.md5(ctx.metadata.get("filepath", "").encode()).hexdigest()[:16]

            # 判断执行模式：如果没有启动后台线程（CLI 模式），使用同步执行
            sync_mode = not self._started

            # 提交建图任务
            job_id = self._graph_builder.build_for_document(
                filepath=ctx.metadata.get("filepath", ""),
                content=ctx.document or "",
                chunk_ids=ctx.chunk_ids,
                chunks_content=ctx.chunks_content,
                sync=sync_mode,  # CLI 模式下同步执行
            )

            self._stats["documents_processed"] += 1

            latency = (time.time() - start_time) * 1000
            if self._metrics:
                self._metrics.histogram("on_add_document_latency_ms", latency)

            return HookResult.ok(
                message=f"Graph build job submitted: {job_id}", metrics={"latency_ms": latency, "job_id": job_id}
            )

        except Exception as e:
            return HookResult.fail(f"on_add_document error: {e}")

    async def on_search_after(self, ctx: HookContext) -> HookResult:
        """
        检索融合后钩子（FR-2）

        增强检索结果，应用图谱扩展。
        """
        logger.info(
            f"[MemoryGraphPlugin] on_search_after called, enabled={self.plugin_config.enabled}, retriever={self._hybrid_retriever is not None}"
        )
        if not self.plugin_config.enabled or not self._hybrid_retriever:
            return HookResult.ok("Plugin disabled or not initialized")

        if not ctx.query_vec or not ctx.results:
            logger.info("[MemoryGraphPlugin] No query vector or results to enhance")
            return HookResult.ok("No query vector or results to enhance")

        start_time = time.time()
        logger.info(f"[MemoryGraphPlugin] Starting graph enhancement, results count={len(ctx.results)}")

        try:
            # 提取向量检索结果
            vector_results = []
            for r in ctx.results:
                chunk_id = r.get("chunk_id") or r.get("id")
                if chunk_id:
                    vector_results.append(
                        {
                            "chunk_id": chunk_id,
                            "semantic_score": r.get("semantic_score", 0.0),
                            "keyword_score": r.get("keyword_score", 0.0),
                            "confidence_score": r.get("confidence_score", 1.0),
                            "base_final_score": r.get("base_final_score", r.get("final_score", 0.0)),
                            "final_score": r.get("final_score", 0.0),
                        }
                    )

            logger.info(f"[MemoryGraphPlugin] Extracted {len(vector_results)} vector results")

            # 修复问题 1：使用从 rag_cli 传递的基础检索权重，如果没有则使用插件配置
            # 注意：这里的 alpha 是基础分数保留系数，不是 HybridEngine 的语义权重
            # 当基础检索使用了特定的 alpha/beta 时，图谱增强应该感知但不重复计算
            graph_alpha = self.plugin_config.retrieval.alpha if ctx.base_alpha is None else ctx.base_alpha
            graph_beta = self.plugin_config.retrieval.beta

            logger.info(f"[MemoryGraphPlugin] graph_alpha={graph_alpha}, graph_beta={graph_beta}")

            # 执行图谱增强检索
            logger.info(
                f"[MemoryGraphPlugin] Calling HybridRetriever.search, query={ctx.query}, top_k={len(ctx.results)}"
            )
            enhanced_results = self._hybrid_retriever.search(
                query=ctx.query,
                query_vec=ctx.query_vec,
                top_k=len(ctx.results),
                base_results=ctx.results,  # 传递基础检索结果，保留 (v×alpha + beta) × conf
                alpha=graph_alpha,  # 基础分数保留系数
                beta=graph_beta,  # 图谱增强权重
            )
            logger.info(f"[MemoryGraphPlugin] HybridRetriever.search returned {len(enhanced_results)} results")

            # 合并结果
            if enhanced_results:
                # 转换为标准格式
                new_results = []
                # 构建原始结果的映射（用于获取关键词得分）
                orig_results_map = {r.get("chunk_id"): r for r in ctx.results}
                for er in enhanced_results:
                    # 从原始结果中获取关键词得分
                    orig = orig_results_map.get(er.chunk_id, {})
                    # 修复：正确获取语义分数，确保 keyword 模式下显示为 0
                    # 优先使用原始结果中的 semantic_score，如果不存在则使用插件的 semantic_score
                    # 不再使用 vector_score 作为后备，因为它可能包含不正确的值
                    orig_semantic = orig.get("semantic_score")
                    plugin_semantic = er.semantic_score
                    final_semantic = orig_semantic if orig_semantic is not None else plugin_semantic

                    new_results.append(
                        {
                            "chunk_id": er.chunk_id,
                            "content": er.content,
                            "file_path": er.file_path,
                            "section": er.section,
                            "final_score": er.final_score,  # 包含图谱增强的最终分数
                            "base_final_score": er.base_final_score,  # 保留基础检索分数
                            "semantic_score": final_semantic,  # 保留原始语义分（keyword 模式下应为 0）
                            "keyword_score": orig.get("keyword_score", er.keyword_score),  # 保留原始关键词得分
                            "confidence_score": orig.get("confidence_score", 0.69),  # 保留原始置信度分数
                            "confidence_reason": orig.get("confidence_reason", ""),  # 保留置信度原因
                        }
                    )

                # 更新上下文中的结果
                ctx.results = new_results
                logger.info(f"[MemoryGraphPlugin] Graph enhancement completed, updated {len(new_results)} results")
                return HookResult.ok("Graph enhancement completed", modified=True)
            else:
                logger.warning("[MemoryGraphPlugin] No enhanced results returned")
                return HookResult.ok("No enhanced results returned")

        except Exception as e:
            logger.exception(f"[MemoryGraphPlugin] Error during graph enhancement: {e}")
            return HookResult.fail(f"Graph enhancement failed: {e}")
