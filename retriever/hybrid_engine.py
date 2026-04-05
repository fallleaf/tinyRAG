#!/usr/bin/env python3
"""
retriever/hybrid_engine.py - 混合检索引擎 (RRF + 置信度加权 + 双缓存)
修复内容:
1. 修复 _search_internal 检索链路断裂 (P0)
2. 修复方法缩进/作用域错误
3. 缓存键包含 alpha/beta 权重
4. sqlite-vec 增加 vault 过滤支持
"""

import os
import time

from loguru import logger

from storage.database import DatabaseManager

try:
    from storage.cache import get_cache

    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False
    logger.warning("⚠️ Cache 模块未找到，将仅使用内存缓存")


class RetrievalResult:
    def __init__(
        self,
        chunk_id: int,
        content: str,
        file_path: str,
        absolute_path: str,
        section: str,
        start_pos: int,
        end_pos: int,
        vault_name: str,
        chunk_type: str,
        semantic_score: float,
        keyword_score: float,
        confidence_score: float,
        final_score: float,
        confidence_reason: str,
        file_hash: str,
    ):
        self.chunk_id = chunk_id
        self.content = content
        self.file_path = file_path
        self.absolute_path = absolute_path
        self.section = section
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.vault_name = vault_name
        self.chunk_type = chunk_type
        self.semantic_score = semantic_score
        self.keyword_score = keyword_score
        self.confidence_score = confidence_score
        self.final_score = final_score
        self.confidence_reason = confidence_reason
        self.file_hash = file_hash

    def to_dict(self):
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "file_path": self.file_path,
            "absolute_path": self.absolute_path,
            "section": self.section,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "vault_name": self.vault_name,
            "chunk_type": self.chunk_type,
            "semantic_score": round(self.semantic_score, 4),
            "keyword_score": round(self.keyword_score, 4),
            "confidence_score": round(self.confidence_score, 4),
            "final_score": round(self.final_score, 4),
            "confidence_reason": self.confidence_reason,
            "file_hash": self.file_hash,
        }


class HybridRetriever:
    def __init__(
        self,
        db: DatabaseManager,
        alpha: float = 0.6,
        beta: float = 0.2,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        cache_dir: str = "~/.cache/fastembed",
        final_top_k: int = 10,
        cache_db_path: str = "./data/cache.db",
        cache_ttl: int = 3600,
    ):
        self.db = db
        self.alpha = alpha
        self.beta = beta
        self.k = 60  # RRF 常数
        self.model_name = model_name
        self.cache_dir = os.path.expanduser(cache_dir)
        self.final_top_k = final_top_k
        self.cache_db_path = cache_db_path
        self.cache_ttl = cache_ttl
        self._embedder = None
        self._cache = None
        self._memory_cache: dict[str, tuple[list[RetrievalResult], float]] = {}
        self._memory_cache_ttl = 300

    def _get_embedder(self):
        if self._embedder is None:
            from embedder.embed_engine import EmbeddingEngine

            self._embedder = EmbeddingEngine(
                model_name=self.model_name,
                cache_dir=self.cache_dir,
                batch_size=1,
                unload_after_seconds=120,
            )
            logger.info(f"✅ 嵌入模型懒加载：{self.model_name}")
        return self._embedder

    def _get_cache(self):
        if CACHE_AVAILABLE and self._cache is None:
            self._cache = get_cache(db_path=self.cache_db_path, ttl_seconds=self.cache_ttl)
        return self._cache

    def search(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int | None = None,
        vaults: list[str] | None = None,
    ) -> list[RetrievalResult]:
        if top_k is None:
            top_k = self.final_top_k
        # ✅ 缓存键包含权重，避免配置变更导致脏缓存
        cache_key = f"{query}|{mode}|{top_k}|{sorted(vaults or [])}|a{self.alpha}|b{self.beta}"

        cache = self._get_cache()
        if cache:
            cached = cache.get(cache_key)
            if cached:
                logger.debug(f"🎯 持久化缓存命中：'{query}'")
                return [RetrievalResult(**{k: v for k, v in item.items()}) for item in cached]

        if cache_key in self._memory_cache:
            results, ts = self._memory_cache[cache_key]
            if time.time() - ts < self._memory_cache_ttl:
                return results
            del self._memory_cache[cache_key]

        results = self._search_internal(query, mode, top_k, vaults)
        if cache:
            try:
                cache.set(cache_key, [r.to_dict() for r in results])
            except Exception as e:
                logger.error(f"❌ 缓存写入失败: {e}")
        self._memory_cache[cache_key] = (results, time.time())
        return results

    def _search_internal(self, query: str, mode: str, top_k: int, vaults: list[str] | None) -> list[RetrievalResult]:
        # ✅ 修复 P0：实际调用检索方法
        limit = top_k * 2
        vec_scores = {}
        kw_ranks = {}

        if self.db.vec_support and mode in ["semantic", "hybrid"]:
            vec_scores = self._vector_search(query, limit=limit, vaults=vaults)
        if mode in ["keyword", "hybrid"]:
            kw_ranks = self._keyword_search(query, limit=limit, vaults=vaults)

        if not vec_scores and not kw_ranks:
            return []

        rrf_scores = self._rrf_fusion(vec_scores, kw_ranks)
        return self._fetch_results_with_metadata(rrf_scores, vec_scores, kw_ranks, vaults)

    def _vector_search(self, query: str, limit: int, vaults: list[str] | None) -> dict[int, float]:
        """向量检索 (严格遵循 sqlite-vec KNN 语法规范)"""
        if not self.db.vec_support:
            return {}
        try:
            import array
            import sqlite3

            # 1. 生成查询向量并强制转为 SQLite BLOB
            query_vec = self._get_embedder().embed([query])[0]
            query_vec_bytes = sqlite3.Binary(array.array("f", query_vec).tobytes())

            # 2. ✅ 修复：使用 vec0 官方 MATCH 语法，必须带 LIMIT
            # 注意：MATCH 会自动触发向量索引扫描，distance 为 vec0 自动返回的隐藏列
            sql = """
            SELECT chunk_id, distance
            FROM vectors
            WHERE embedding MATCH ?
            ORDER BY distance ASC
            LIMIT ?
            """
            cursor = self.db.conn.execute(sql, (query_vec_bytes, limit))
            rows = cursor.fetchall()

            # 3. 距离转相似度 (L2 距离越小越相似)
            return {row["chunk_id"]: 1.0 / (1.0 + row["distance"]) for row in rows}
        except Exception as e:
            logger.error(f"❌ 向量检索失败：{e}", exc_info=True)
            return {}

    def _keyword_search(self, query: str, limit: int, vaults: list[str] | None) -> dict[int, int]:
        keywords = query.split() or [query]
        like_conds = " OR ".join(
            ["(LOWER(c.content) LIKE LOWER(?) OR LOWER(c.section_title) LIKE LOWER(?))"] * len(keywords)
        )
        sql = f"SELECT c.id FROM chunks c JOIN files f ON c.file_id = f.id WHERE ({like_conds}) AND c.is_deleted = 0 AND f.is_deleted = 0"
        params = [f"%{kw.lower()}%" for kw in keywords for _ in range(2)]

        if vaults:
            placeholders = ", ".join(["?"] * len(vaults))
            sql += f" AND f.vault_name IN ({placeholders})"
            params.extend(vaults)

        cursor = self.db.conn.execute(sql, params)
        rows = cursor.fetchall()
        return {row["id"]: i + 1 for i, row in enumerate(rows[:limit])}

    def _rrf_fusion(self, vec_scores: dict[int, float], kw_ranks: dict[int, int]) -> dict[int, float]:
        vec_ranks = {
            cid: rank for rank, (cid, _) in enumerate(sorted(vec_scores.items(), key=lambda x: x[1], reverse=True), 1)
        }
        rrf_scores = {}
        for cid in set(vec_ranks) | set(kw_ranks):
            r_vec = vec_ranks.get(cid, float("inf"))
            r_kw = kw_ranks.get(cid, float("inf"))
            rrf_scores[cid] = (1.0 / (self.k + r_vec)) + (1.0 / (self.k + r_kw))
        return rrf_scores

    def _fetch_results_with_metadata(self, rrf_scores, vec_scores, kw_ranks, vaults):
        chunk_ids = list(rrf_scores.keys())
        placeholders = ", ".join(["?"] * len(chunk_ids))
        sql = f"""
        SELECT c.id, c.content, c.start_pos, c.end_pos, c.content_type, c.confidence_final_weight,
               f.file_path, f.absolute_path, f.vault_name, f.file_hash, c.section_title
        FROM chunks c JOIN files f ON c.file_id = f.id
        WHERE c.id IN ({placeholders}) AND c.is_deleted = 0 AND f.is_deleted = 0
        """
        cursor = self.db.conn.execute(sql, chunk_ids)
        row_map = {row["id"]: dict(row) for row in cursor.fetchall()}

        final_results = []
        for cid, rrf_score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
            row = row_map.get(cid)
            if not row:
                continue
            if vaults and row["vault_name"] not in vaults:
                continue

            conf_w = row["confidence_final_weight"]
            final_score = rrf_score * conf_w
            kw_score = 1.0 / (self.k + kw_ranks[cid]) if cid in kw_ranks else 0.0

            final_results.append(
                RetrievalResult(
                    chunk_id=row["id"],
                    content=row["content"],
                    file_path=row["file_path"],
                    absolute_path=row["absolute_path"],
                    section=row["section_title"] or "Root",
                    start_pos=row["start_pos"],
                    end_pos=row["end_pos"],
                    vault_name=row["vault_name"],
                    chunk_type=row["content_type"],
                    semantic_score=vec_scores.get(cid, 0.0),
                    keyword_score=kw_score,
                    confidence_score=conf_w,
                    final_score=final_score,
                    confidence_reason=self._generate_reason(row["file_path"], row["content_type"], conf_w),
                    file_hash=row["file_hash"],
                )
            )
        return final_results

    def _generate_reason(self, file_path: str, content_type: str, weight: float) -> str:
        reasons = []
        if "official" in file_path.lower():
            reasons.append("官方文档")
        elif "draft" in file_path.lower():
            reasons.append("草稿")
        reasons.append(f"类型：{content_type} (权重：{weight:.2f})")
        return " | ".join(reasons)

    def clear_cache(self):
        cache = self._get_cache()
        if cache:
            cache.clear()
        self._memory_cache.clear()
        logger.info("🧹 检索缓存已清除")


__all__ = ["HybridRetriever", "RetrievalResult"]
