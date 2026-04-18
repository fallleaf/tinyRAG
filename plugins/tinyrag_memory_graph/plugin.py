#!/usr/bin/env python3
"""
plugin.py - Memory Graph 插件主类

实现插件的核心逻辑，整合所有功能模块。
继承 PluginBase 以支持插件加载器。
"""
import contextlib
import hashlib
import sqlite3
import time
from typing import Any

from plugins.bootstrap import PluginBase  # 继承插件基类
from plugins.tinyrag_memory_graph.config import MemoryGraphConfig
from plugins.tinyrag_memory_graph.graph_builder import GraphBuilder
from plugins.tinyrag_memory_graph.hooks import HookContext, HookResult
from plugins.tinyrag_memory_graph.hybrid_retriever import HybridRetriever, HybridSearchResult
from plugins.tinyrag_memory_graph.memify import MemifyEngine, PrincipleManager
from plugins.tinyrag_memory_graph.metrics import MetricsCollector, get_metrics_collector
from plugins.tinyrag_memory_graph.storage import GraphStorage


class MemoryGraphPlugin(PluginBase):
    """
    Memory Graph 插件主类

    提供图-向量混合记忆功能，通过钩子协议与 tinyRAG 核心集成。

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

    def __init__(self, config: dict | None = None, context: Any | None = None):
        """
        初始化插件

        Args:
            config: 插件配置字典（来自 config.yaml 中的 plugins.config）
            context: AppContext 上下文对象
        """
        # 调用基类初始化
        super().__init__(config, context)

        # 解析配置
        if isinstance(config, dict):
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
        self.register_hook("on_chunks_indexed", self._on_chunks_indexed_hook)  # 兼容 build_index.py
        self.register_hook("on_search", self._on_search_hook)
        self.register_hook("on_response_generated", self._on_response_hook)

    def on_load(self) -> bool:
        """
        插件加载时调用
        """
        # 如果有上下文，尝试获取数据库连接
        if self.ctx and hasattr(self.ctx, 'db') and self.ctx.db:
            self._db = self.ctx.db.conn if hasattr(self.ctx.db, 'conn') else self.ctx.db
        return True

    def on_enable(self) -> bool:
        """
        插件启用时调用
        """
        if not self._db and self.ctx and hasattr(self.ctx, 'db'):
            # 尝试从上下文获取
            self._db = self.ctx.db.conn if hasattr(self.ctx.db, 'conn') else self.ctx.db

        if self._db:
            try:
                # 同步初始化
                self._initialize_sync()
                return True
            except Exception as e:
                return False
        return True  # 允许延迟初始化

    def on_disable(self) -> None:
        """插件禁用时调用"""
        if self._started:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    _stop_task = asyncio.create_task(self.stop())  # noqa: RUF006
                else:
                    loop.run_until_complete(self.stop())
            except Exception:
                pass

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
    async def _on_file_indexed_hook(self, file_id: int, chunks: list, filepath: str = "", **kwargs):
        """文件索引完成钩子"""
        if not self.plugin_config.enabled:
            return None

        # 提取 chunk_ids 和内容
        chunk_ids = [c.get('id') if isinstance(c, dict) else getattr(c, 'id', 0) for c in chunks]
        chunks_content = [c.get('content', '') if isinstance(c, dict) else getattr(c, 'content', '') for c in chunks]

        ctx = HookContext(
            document='\n'.join(chunks_content),
            chunk_ids=chunk_ids,
            chunks_content=chunks_content,
            metadata={"file_id": file_id, "filepath": filepath, **kwargs}
        )
        result = await self.on_add_document(ctx)
        return result

    async def _on_chunks_indexed_hook(self, chunk_id: int, file_id: int, content: str, metadata: dict, **kwargs):
        """
        单个 chunk 索引完成钩子（由 build_index.py 调用）

        收集同一文件的 chunks，在文件处理完成后批量构建图谱。
        """
        if not self.plugin_config.enabled:
            return None

        # 收集 chunk 信息，稍后批量处理
        if not hasattr(self, '_pending_chunks'):
            self._pending_chunks = {}

        if file_id not in self._pending_chunks:
            self._pending_chunks[file_id] = {
                'chunk_ids': [],
                'contents': [],
                'metadata': metadata or {}
            }

        self._pending_chunks[file_id]['chunk_ids'].append(chunk_id)
        self._pending_chunks[file_id]['contents'].append(content)

        return None

    async def _on_search_hook(self, query: str, results: list, **kwargs):
        """搜索钩子"""
        if not self.plugin_config.enabled:
            return results

        ctx = HookContext(
            query=query,
            query_vec=kwargs.get('query_vec'),
            results=results,
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
            with contextlib.suppress(Exception):
                self._initialize_sync()

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

        触发异步建图任务。
        """
        if not self.plugin_config.enabled or not self._graph_builder:
            return HookResult.ok("Plugin disabled or not initialized")

        start_time = time.time()

        try:
            # 生成 note_id
            note_id = hashlib.md5(ctx.metadata.get("filepath", "").encode()).hexdigest()[:16]

            # 提交建图任务
            job_id = self._graph_builder.build_for_document(
                filepath=ctx.metadata.get("filepath", ""),
                content=ctx.document or "",
                chunk_ids=ctx.chunk_ids,
                chunks_content=ctx.chunks_content,
            )

            self._stats["documents_processed"] += 1

            latency = (time.time() - start_time) * 1000
            if self._metrics:
                self._metrics.histogram("on_add_document_latency_ms", latency)

            return HookResult.ok(
                message=f"Graph build job submitted: {job_id}",
                metrics={"latency_ms": latency, "job_id": job_id}
            )

        except Exception as e:
            return HookResult.fail(f"on_add_document error: {e}")

    async def on_search_after(self, ctx: HookContext) -> HookResult:
        """
        检索融合后钩子（FR-2）- 统一评分版

        增强检索结果，应用图谱扩展。
        关键改进：保留基础检索分数，图谱分数作为增强加成。
        """
        if not self.plugin_config.enabled or not self._hybrid_retriever:
            return HookResult.ok("Plugin disabled or not initialized")

        if not ctx.query_vec or not ctx.results:
            return HookResult.ok("No query vector or results to enhance")

        start_time = time.time()

        try:
            # 执行图谱增强检索，传递基础检索结果
            enhanced_results = self._hybrid_retriever.search(
                query=ctx.query,
                query_vec=ctx.query_vec,
                top_k=len(ctx.results),
                base_results=ctx.results,  # 🔑 传递基础检索结果，保留完整评分
            )

            # 合并结果
            if enhanced_results:
                # 转换为标准格式，保留完整评分信息
                new_results = []
                for er in enhanced_results:
                    new_results.append({
                        "chunk_id": er.chunk_id,
                        "content": er.content,
                        "file_path": er.file_path,
                        "section": er.section,
                        # 统一评分字段
                        "score": er.final_score,
                        "final_score": er.final_score,
                        # 保留基础检索分数
                        "semantic_score": getattr(er, 'semantic_score', er.vector_score),
                        "keyword_score": getattr(er, 'keyword_score', 0.0),
                        "confidence_score": getattr(er, 'confidence_score', 1.0),
                        "base_final_score": getattr(er, 'base_final_score', er.final_score),
                        # 图谱增强分数
                        "graph_score": er.graph_score,
                        "preference_score": getattr(er, 'preference_score', 0.0),
                        "hop_distance": er.hop_distance,
                        # 元数据
                        "tags": er.tags,
                        "note_title": er.note_title,
                    })

                # 更新上下文
                ctx.results = new_results

                # 记录检索命中
                if self._memify_engine:
                    self._memify_engine.record_search_hit([er.chunk_id for er in enhanced_results[:5]])

            self._stats["searches_enhanced"] += 1

            latency = (time.time() - start_time) * 1000
            if self._metrics:
                self._metrics.record_retrieval_latency(latency, "graph_enhanced")

            return HookResult.ok(
                message=f"Enhanced {len(enhanced_results)} results",
                modified=True,
                metrics={"latency_ms": latency, "results_count": len(enhanced_results)}
            )

        except Exception as e:
            return HookResult.fail(f"on_search_after error: {e}")

    async def on_response(self, ctx: HookContext) -> HookResult:
        """
        LLM 输出后钩子（FR-3.2）

        更新访问计数和边权重。
        """
        if not self.plugin_config.enabled or not self._memify_engine:
            return HookResult.ok("Plugin disabled or not initialized")

        try:
            # 更新检索命中的关系访问记录
            chunk_ids = ctx.get_result_chunk_ids()
            if chunk_ids and self._storage:
                # 更新 Chunk 访问计数
                for cid in chunk_ids[:5]:  # 最多更新前 5 个
                    self._db.execute(
                        """UPDATE chunks SET
                               access_count = access_count + 1,
                               last_accessed = strftime('%s', 'now')
                           WHERE id = ?""",
                        (cid,)
                    )

                # 记录标签共现
                self._memify_engine.record_search_hit(chunk_ids)

            self._db.commit()

            return HookResult.ok("Access records updated")

        except Exception as e:
            return HookResult.fail(f"on_response error: {e}")

    async def on_delete_document(self, ctx: HookContext) -> HookResult:
        """文档删除后钩子"""
        if not self.plugin_config.enabled or not self._graph_builder:
            return HookResult.ok("Plugin disabled or not initialized")

        try:
            filepath = ctx.metadata.get("filepath", "")
            if filepath:
                self._graph_builder.delete_document(filepath)

            return HookResult.ok(f"Document graph data deleted: {filepath}")

        except Exception as e:
            return HookResult.fail(f"on_delete_document error: {e}")

    # ==================== 公开 API ====================

    def search(self, query: str, query_vec: list[float],
               top_k: int = 10, preferences: dict | None = None) -> list[HybridSearchResult]:
        """
        执行图谱增强检索

        Args:
            query: 查询文本
            query_vec: 查询向量
            top_k: 返回结果数
            preferences: 用户偏好

        Returns:
            检索结果列表
        """
        if not self._hybrid_retriever:
            return []

        return self._hybrid_retriever.search(
            query=query,
            query_vec=query_vec,
            top_k=top_k,
            user_preferences=preferences,
        )

    def get_stats(self) -> dict:
        """获取插件统计"""
        stats = {
            "enabled": self.plugin_config.enabled,
            "initialized": self._initialized,
            "started": self._started,
            **self._stats,
        }

        if self._graph_builder:
            stats["graph_builder"] = self._graph_builder.get_stats()

        if self._memify_engine:
            stats["memify"] = self._memify_engine.get_stats()

        if self._hybrid_retriever:
            stats["retrieval"] = self._hybrid_retriever.get_metrics()

        return stats

    def get_pending_principles(self, limit: int = 20) -> list[dict]:
        """获取待审核原则"""
        if not self._principle_manager:
            return []
        return self._principle_manager.get_pending_principles(limit)

    def approve_principle(self, principle_id: int) -> bool:
        """批准原则"""
        if not self._principle_manager:
            return False
        return self._principle_manager.approve_principle(principle_id)

    def reject_principle(self, principle_id: int) -> bool:
        """拒绝原则"""
        if not self._principle_manager:
            return False
        return self._principle_manager.reject_principle(principle_id)

    def export_metrics(self, format: str = "csv") -> str:
        """
        导出指标

        Args:
            format: csv / prometheus / json

        Returns:
            导出文件路径或内容
        """
        if not self._metrics:
            return ""

        if format == "csv":
            return self._metrics.export_csv()
        elif format == "prometheus":
            return self._metrics.export_prometheus()
        else:
            return self._metrics.export_json()

    async def run_memify(self) -> dict:
        """手动触发记忆代谢"""
        if not self._memify_engine:
            return {"error": "Memify engine not initialized"}

        stats = await self._memify_engine.run_memify()
        return {
            "relations_decayed": stats.relations_decayed,
            "relations_marked_stale": stats.relations_marked_stale,
            "relations_boosted": stats.relations_boosted,
            "principles_generated": stats.principles_generated,
            "processing_time_ms": stats.processing_time_ms,
        }


# 插件入口点
def create_plugin(config: dict | None = None) -> MemoryGraphPlugin:
    """
    创建插件实例（入口点函数）

    Args:
        config: 配置字典

    Returns:
        插件实例
    """
    cfg = MemoryGraphConfig.from_dict(config) if config else MemoryGraphConfig()
    return MemoryGraphPlugin(cfg)


__all__ = ["MemoryGraphPlugin", "create_plugin"]
