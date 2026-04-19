#!/usr/bin/env python3
"""
hybrid_retriever.py - 混合检索增强引擎

实现 FR-2 混合检索需求，包括：
- 向量基座召回
- 图扩展遍历
- RRF 融合重排
- 上下文组装
"""
import sqlite3
import time
from dataclasses import dataclass

from plugins.tinyrag_memory_graph.config import MemoryGraphConfig, RetrievalConfig
from plugins.tinyrag_memory_graph.storage import GraphStorage


@dataclass
class HybridSearchResult:
    """混合检索结果 - 统一评分版"""
    chunk_id: int
    content: str
    file_path: str
    note_title: str
    section: str

    # 基础检索分数（来自 HybridEngine）
    semantic_score: float = 0.0      # 向量语义分数
    keyword_score: float = 0.0       # FTS5 关键词分数
    confidence_score: float = 1.0    # 置信度分数
    base_final_score: float = 0.0    # 基础检索最终分数

    # 图谱增强分数
    graph_score: float = 0.0         # 图谱关联分数
    preference_score: float = 0.0    # 偏好匹配分数
    final_score: float = 0.0         # 综合最终分数

    # 兼容旧字段
    vector_score: float = 0.0        # 等同于 semantic_score

    # 图谱信息
    hop_distance: int = 0
    path_weight: float = 0.0
    path: str = ""

    # 元数据
    tags: list[str] = None
    inherited_meta: dict = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.inherited_meta is None:
            self.inherited_meta = {}
        # 兼容：vector_score 等同于 semantic_score
        if self.vector_score == 0.0 and self.semantic_score > 0.0:
            self.vector_score = self.semantic_score


class HybridRetriever:
    """
    混合检索引擎（FR-2）

    协调向量召回和图扩展，实现知识图谱增强的检索。
    """

    def __init__(self, db_conn: sqlite3.Connection,
                 config: MemoryGraphConfig,
                 base_retriever=None):
        """
        初始化混合检索器

        Args:
            db_conn: 数据库连接
            config: 插件配置
            base_retriever: tinyRAG 基础检索器（HybridEngine）
        """
        self.db = db_conn
        self.config = config
        self.retrieval_config = config.retrieval
        self.base_retriever = base_retriever
        self.storage = GraphStorage(db_conn, config)

        # 性能统计
        self._metrics = {
            "vector_latency_ms": [],
            "graph_latency_ms": [],
            "total_latency_ms": [],
            "graph_nodes_traversed": [],
        }

    def search(self, query: str, query_vec: list[float],
               top_k: int = 10,
               alpha: float | None = None,
               beta: float | None = None,
               gamma: float | None = None,
               user_preferences: dict | None = None,
               base_results: list[dict] | None = None) -> list[HybridSearchResult]:
        """
        执行混合检索（FR-2）

        Args:
            query: 查询文本
            query_vec: 查询向量
            top_k: 返回结果数量
            alpha: 向量权重（None 则使用配置）
            beta: 图权重
            gamma: 偏好加成
            user_preferences: 用户偏好 {author, project, status, tags}
            base_results: 基础检索结果（来自 HybridEngine），包含完整的评分信息

        Returns:
            混合检索结果列表
        """
        start_time = time.time()

        # 使用配置默认值
        alpha = alpha if alpha is not None else self.retrieval_config.alpha
        beta = beta if beta is not None else self.retrieval_config.beta
        gamma = gamma if gamma is not None else self.retrieval_config.gamma

        # 1. 向量基座召回（FR-2.1）
        # 优先使用基础检索结果，避免重复查询
        if base_results and len(base_results) > 0:
            vector_results = self._convert_base_results(base_results)
            vector_latency = 0  # 复用基础结果，无额外延迟
        else:
            vector_results = self._vector_recall(query_vec)
            vector_latency = (time.time() - start_time) * 1000
        self._metrics["vector_latency_ms"].append(vector_latency)

        if not vector_results:
            return []

        # 2. 图扩展遍历（FR-2.2）
        graph_start = time.time()
        seed_chunk_ids = [r["chunk_id"] for r in vector_results[:3]]  # Top 3 作为种子
        graph_results = self._graph_traverse(seed_chunk_ids)
        graph_latency = (time.time() - graph_start) * 1000
        self._metrics["graph_latency_ms"].append(graph_latency)
        self._metrics["graph_nodes_traversed"].append(len(graph_results))

        # 3. 合并候选集
        all_candidates = self._merge_candidates(vector_results, graph_results)

        # 4. RRF 融合重排（FR-2.3）
        ranked_results = self._rrf_fusion(
            all_candidates,
            vector_results,
            graph_results,
            alpha, beta, gamma,
            user_preferences
        )

        # 5. 填充内容（FR-2.4）
        final_results = self._hydrate_results(ranked_results[:top_k])

        total_latency = (time.time() - start_time) * 1000
        self._metrics["total_latency_ms"].append(total_latency)

        return final_results

    def _convert_base_results(self, base_results: list[dict]) -> list[dict]:
        """
        将基础检索结果转换为内部格式，保留完整评分信息

        Args:
            base_results: HybridEngine 的检索结果

        Returns:
            内部格式的向量召回结果
        """
        converted = []
        for r in base_results:
            # 支持字典和对象两种格式
            if isinstance(r, dict):
                chunk_id = r.get("chunk_id") or r.get("id", 0)
                # 修复：不使用默认值，正确获取语义分数
                # 如果 semantic_score 不存在，使用 0.0 而非 0.5
                semantic_score = r.get("semantic_score", 0.0)
                keyword_score = r.get("keyword_score", 0.0)
                confidence_score = r.get("confidence_score", 1.0)
                final_score = r.get("final_score", r.get("score", 0.0))
                content = r.get("content", "")
                file_path = r.get("file_path", "")
            else:
                chunk_id = getattr(r, "chunk_id", getattr(r, "id", 0))
                # 修复：不使用默认值，正确获取语义分数
                semantic_score = getattr(r, "semantic_score", 0.0)
                keyword_score = getattr(r, "keyword_score", 0.0)
                confidence_score = getattr(r, "confidence_score", 1.0)
                final_score = getattr(r, "final_score", 0.0)
                content = getattr(r, "content", "")
                file_path = getattr(r, "file_path", "")

            converted.append({
                "chunk_id": chunk_id,
                "score": semantic_score,  # 向量分数
                "semantic_score": semantic_score,
                "keyword_score": keyword_score,
                "confidence_score": confidence_score,
                "base_final_score": final_score,  # 保留基础检索的最终分数
                "content": content,
                "file_path": file_path,
            })
        return converted

    def _vector_recall(self, query_vec: list[float]) -> list[dict]:
        """
        向量基座召回（FR-2.1）

        使用 sqlite-vec 进行 L2 距离计算。
        """
        if not query_vec:
            return []

        try:
            import array
            query_blob = array.array("f", query_vec).tobytes()

            # 使用 sqlite-vec 的 vec_distance_L2 函数
            cursor = self.db.execute(
                f"""SELECT v.chunk_id, vec_distance_L2(v.embedding, ?) as distance,
                          c.content, f.file_path
                   FROM vectors v
                   JOIN chunks c ON v.chunk_id = c.id
                   JOIN files f ON c.file_id = f.id
                   WHERE c.is_deleted = 0
                   ORDER BY distance
                   LIMIT {self.retrieval_config.vector_top_k}""",
                (query_blob,)
            )

            results = []
            for row in cursor.fetchall():
                # L2 距离转相似度
                distance = row[1]
                similarity = 1.0 / (1.0 + distance)
                results.append({
                    "chunk_id": row[0],
                    "score": similarity,
                    "content": row[2],
                    "file_path": row[3],
                })
            return results

        except Exception as e:
            print(f"[HybridRetriever] Vector recall error: {e}")
            return []

    def _graph_traverse(self, seed_chunk_ids: list[int]) -> list[dict]:
        """
        图扩展遍历（FR-2.2）

        以 Top 3 Chunk 为种子，递归 CTE 有限遍历。
        """
        if not seed_chunk_ids:
            return []

        graph_results = self.storage.traverse_graph(
            seed_chunk_ids,
            max_hops=self.retrieval_config.max_hops,
            min_weight=self.retrieval_config.min_edge_weight,
            max_nodes=self.retrieval_config.max_traverse_nodes,
        )

        # 转换为标准格式
        results = []
        for r in graph_results:
            results.append({
                "chunk_id": r["chunk_id"],
                "hop": r["hop"],
                "path_weight": r["path_weight"],
                "path": r["path"],
            })

        return results

    def _merge_candidates(self, vector_results: list[dict],
                         graph_results: list[dict]) -> dict[int, dict]:
        """合并向量和图搜索结果"""
        candidates = {}

        # 向量结果
        for r in vector_results:
            cid = r["chunk_id"]
            candidates[cid] = {
                "chunk_id": cid,
                "vector_score": r["score"],
                "semantic_score": r.get("semantic_score", r["score"]),
                "keyword_score": r.get("keyword_score", 0.0),
                "confidence_score": r.get("confidence_score", 1.0),
                # 修复：优先获取 base_final_score（来自 _convert_base_results）
                "base_final_score": r.get("base_final_score", r.get("final_score", r["score"])),
                "graph_score": 0.0,
                "hop": 0,
                "path_weight": 0.0,
            }

        # 图结果
        for r in graph_results:
            cid = r["chunk_id"]
            if cid in candidates:
                candidates[cid]["hop"] = r["hop"]
                candidates[cid]["path_weight"] = r["path_weight"]
            else:
                candidates[cid] = {
                    "chunk_id": cid,
                    "vector_score": 0.0,
                    "semantic_score": r.get("semantic_score", 0.0),
                    "keyword_score": r.get("keyword_score", 0.0),
                    "confidence_score": r.get("confidence_score", 1.0),
                    # 修复：保持与向量结果处理一致的获取逻辑
                    "base_final_score": r.get("base_final_score", r.get("final_score", 0.0)),
                    "graph_score": 0.0,
                    "hop": r["hop"],
                    "path_weight": r["path_weight"],
                }

        return candidates

    def _rrf_fusion(self, candidates: dict[int, dict],
                    vector_results: list[dict],
                    graph_results: list[dict],
                    alpha: float, beta: float, gamma: float,
                    user_preferences: dict | None = None) -> list[dict]:
        """
        RRF 融合重排（FR-2.3）- 统一评分版

        新公式: FinalScore = base_final_score + β×graph_score + γ×preference_score

        说明:
        - base_final_score: 来自 HybridEngine 的基础检索分数
          （已包含 α×semantic + β×keyword）× confidence
        - graph_score: 图谱关联分数（归一化到 0-1）
        - preference_score: 用户偏好匹配分数
        - alpha 此处作为基础分数的保留系数（默认 1.0，完全保留）
        - beta: 图谱分数的增强权重
        - gamma: 偏好分数的增强权重
        """
        # 构建排序索引
        vector_ranks = {r["chunk_id"]: i + 1 for i, r in enumerate(vector_results)}
        graph_weights = {r["chunk_id"]: r["path_weight"] for r in graph_results}

        # 用户偏好匹配
        pref_matches = {}
        if user_preferences:
            for cid, cand in candidates.items():
                pref_score = self._calculate_preference_match(cid, user_preferences)
                pref_matches[cid] = pref_score

        # 计算最终分数
        ranked = []
        for cid, cand in candidates.items():
            # 获取基础检索分数（优先使用保留的基础分数）
            # 后备链：base_final_score -> final_score -> 0.0
            # 修复：移除 vector_score 作为后备，确保语义分数正确
            base_final_score = cand.get("base_final_score", 
                            cand.get("final_score", 0.0))
            # 修复：直接获取 semantic_score，不使用 vector_score 作为后备
            # 这样确保 keyword 模式下语义分数正确显示为 0
            semantic_score = cand.get("semantic_score", 0.0)
            keyword_score = cand.get("keyword_score", 0.0)
            confidence_score = cand.get("confidence_score", 1.0)
            v_score = cand.get("vector_score", 0.0)

            # 图分数（路径权重，越近越好）
            g_score = graph_weights.get(cid, 0.0)
            if cand["hop"] > 0:
                # 根据跳数衰减
                g_score = g_score / (1.0 + 0.5 * (cand["hop"] - 1))

            # 偏好匹配分数
            p_score = pref_matches.get(cid, 0.0)

            # 🔢 统一评分融合
            # 方案一：完全保留基础分数，图谱作为加成
            # final_score = base_final_score + beta * g_score + gamma * p_score
            #
            # 方案二：基础分数作为主体，图谱分数归一化后增强
            # 这样保证评分标准与基础检索一致，图谱只做正向增强
            final_score = alpha * base_final_score + beta * g_score + gamma * p_score

            ranked.append({
                **cand,
                "final_score": final_score,
                "preference_score": p_score,
                "graph_score": g_score,
                # 保留原始分数供调试
                "base_final_score": base_final_score,
                "semantic_score": semantic_score,
                "keyword_score": keyword_score,
                "confidence_score": confidence_score,
            })

        # 按最终分数排序
        ranked.sort(key=lambda x: x["final_score"], reverse=True)
        return ranked

    def _calculate_preference_match(self, chunk_id: int,
                                     preferences: dict) -> float:
        """计算用户偏好匹配分数"""
        if not preferences:
            return 0.0

        try:
            # 获取 Chunk 的继承元数据
            row = self.db.execute(
                "SELECT inherited_meta, note_id FROM chunks WHERE id = ?",
                (chunk_id,)
            ).fetchone()

            if not row:
                return 0.0

            import json
            inherited = json.loads(row[0] or "{}")

            score = 0.0
            count = 0

            # 匹配作者
            if preferences.get("author"):
                if inherited.get("author") == preferences["author"]:
                    score += 1.0
                count += 1

            # 匹配项目
            if preferences.get("project"):
                if inherited.get("project") == preferences["project"]:
                    score += 1.0
                count += 1

            # 匹配状态
            if preferences.get("status"):
                if inherited.get("status") == preferences["status"]:
                    score += 0.5
                count += 1

            # 匹配标签
            pref_tags = set(preferences.get("tags", []))
            chunk_tags = set(inherited.get("tags", []))
            if pref_tags and chunk_tags:
                overlap = len(pref_tags & chunk_tags)
                if overlap > 0:
                    score += overlap / max(len(pref_tags), 1)
                count += 1

            return score / max(count, 1)

        except Exception:
            return 0.0

    def _hydrate_results(self, ranked_results: list[dict]) -> list[HybridSearchResult]:
        """
        填充结果内容（FR-2.4）

        严格 Token 预算，仅注入 note_title + Top 3 标签。
        """
        if not ranked_results:
            return []

        chunk_ids = [r["chunk_id"] for r in ranked_results]
        placeholders = ",".join(["?"] * len(chunk_ids))

        query = f"""
            SELECT c.id, c.content, c.section_title, c.inherited_meta,
                   f.file_path, n.title as note_title
            FROM chunks c
            JOIN files f ON c.file_id = f.id
            LEFT JOIN notes n ON c.note_id = n.note_id
            WHERE c.id IN ({placeholders})
        """

        rows = self.db.execute(query, chunk_ids).fetchall()
        row_map = {r[0]: r for r in rows}

        results = []
        for ranked in ranked_results:
            cid = ranked["chunk_id"]
            row = row_map.get(cid)
            if not row:
                continue

            import json
            inherited = json.loads(row[3] or "{}") if row[3] else {}

            result = HybridSearchResult(
                chunk_id=cid,
                content=row[1],
                file_path=row[4],
                note_title=row[5] or "",
                section=row[2] or "",
                # 基础检索分数
                # 修复：直接获取 semantic_score，不使用 vector_score 作为后备
                # 这样确保 keyword 模式下语义分数正确显示为 0
                semantic_score=ranked.get("semantic_score", 0.0),
                keyword_score=ranked.get("keyword_score", 0.0),
                confidence_score=ranked.get("confidence_score", 1.0),
                base_final_score=ranked.get("base_final_score", ranked.get("final_score", 0.0)),
                # 图谱增强分数
                graph_score=ranked.get("graph_score", 0.0),
                preference_score=ranked.get("preference_score", 0.0),
                final_score=ranked.get("final_score", 0.0),
                # 兼容旧字段
                vector_score=ranked.get("semantic_score", ranked.get("vector_score", 0.0)),
                # 图谱信息
                hop_distance=ranked.get("hop", 0),
                path_weight=ranked.get("path_weight", 0.0),
                tags=inherited.get("tags", [])[:3],  # Top 3 标签
                inherited_meta=inherited,
            )
            results.append(result)

        return results

    def assemble_context(self, results: list[HybridSearchResult],
                         max_tokens: int | None = None) -> str:
        """
        组装上下文（FR-2.4）

        严格 Token 预算，仅注入必要信息。
        """
        max_tokens = max_tokens or self.retrieval_config.max_context_tokens

        # 估算 Token：中文约 1.5 字/token，英文约 4 字/token
        def estimate_tokens(text: str) -> int:
            chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
            other_chars = len(text) - chinese_chars
            return int(chinese_chars / 1.5 + other_chars / 4)

        context_parts = []
        current_tokens = 0

        for i, result in enumerate(results):
            # 构建条目
            entry = f"\n【文档 {i + 1}】{result.note_title or result.file_path}\n"

            # 添加标签
            if result.tags:
                entry += f"标签: {', '.join(result.tags)}\n"

            # 添加内容（截断）
            content = result.content[:500]
            entry += f"内容: {content}\n"

            entry_tokens = estimate_tokens(entry)

            if current_tokens + entry_tokens > max_tokens:
                break

            context_parts.append(entry)
            current_tokens += entry_tokens

        return "".join(context_parts)

    def get_metrics(self) -> dict:
        """获取性能指标"""
        def avg(lst):
            return sum(lst) / len(lst) if lst else 0

        return {
            "avg_vector_latency_ms": avg(self._metrics["vector_latency_ms"]),
            "avg_graph_latency_ms": avg(self._metrics["graph_latency_ms"]),
            "avg_total_latency_ms": avg(self._metrics["total_latency_ms"]),
            "avg_graph_nodes_traversed": avg(self._metrics["graph_nodes_traversed"]),
            "total_searches": len(self._metrics["total_latency_ms"]),
        }


class ContextAssembler:
    """
    上下文组装器

    负责将检索结果组装为适合 LLM 输入的上下文。
    """

    def __init__(self, config: RetrievalConfig):
        self.config = config

    def assemble(self, results: list[HybridSearchResult],
                 query: str,
                 template: str | None = None) -> str:
        """
        组装 LLM 提示上下文

        Args:
            results: 检索结果
            query: 用户查询
            template: 提示模板（可选）

        Returns:
            组装后的上下文
        """
        # 默认模板
        if not template:
            template = """你是一个知识库助手。请基于以下检索结果回答用户问题。

## 用户问题
{query}

## 知识库检索结果
{context}

## 回答要求
1. 优先引用知识库中的信息
2. 标注信息来源（文档标题）
3. 如果知识库中没有相关信息，请明确说明"""

        # 组装上下文
        context = self._build_context(results)

        return template.format(query=query, context=context)

    def _build_context(self, results: list[HybridSearchResult]) -> str:
        """构建上下文内容"""
        parts = []

        for i, result in enumerate(results[:10]):  # 最多 10 条
            part = f"### [{i + 1}] {result.note_title or result.file_path}\n"

            # 元数据
            meta_parts = []
            if result.tags:
                meta_parts.append(f"标签: {', '.join(result.tags)}")
            if result.section:
                meta_parts.append(f"章节: {result.section}")
            if meta_parts:
                part += " | ".join(meta_parts) + "\n"

            # 内容
            part += f"{result.content[:400]}\n"

            # 图谱信息
            if result.hop_distance > 0:
                part += f"_（通过 {result.hop_distance} 跳关联，权重: {result.path_weight:.2f}）_\n"

            parts.append(part)

        return "\n---\n".join(parts)


__all__ = ["ContextAssembler", "HybridRetriever", "HybridSearchResult"]
