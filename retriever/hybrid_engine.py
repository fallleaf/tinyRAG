#!/usr/bin/env python3
"""
retriever/hybrid_engine.py - 混合检索引擎 (v2.0 - 动态置信度/双层缓存/DRY重构)
优化记录:
- P0: 移除重复的 jieba/日期保护逻辑，统一导入 utils.jieba_helper
- P1: 优化缓存键生成，修复时间范围查询缓存隔离
- P2: 严格隔离向量/FTS5 得分量级 (log1p 平滑)
"""

import contextlib
import hashlib
import importlib.util
import json
import math
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from storage.database import DatabaseManager
from utils.jieba_helper import jieba_segment  # ✅ DRY 重构：统一分词入口
from utils.logger import logger

# 时间范围查询检测正则
_TIME_RANGE_PATTERNS = [
    r"(\d{4})(?:-(\d{2})(?:-(\d{2}))?|年(\d{1,2})(?:月(\d{1,2})日)?)",
    r"(\d{4})(?:-(\d{2})|年(\d{1,2})月)(?![\-日\d])",
    r"(\d{4})年(?!\d)",
]

# 检查 jieba 是否可用
try:
    import jieba

    JIEBA_AVAILABLE = True
    jieba.initialize()
except ImportError:
    JIEBA_AVAILABLE = False
    logger.warning("⚠️ jieba 未安装，关键词检索功能将降级")

# 检查 cache 模块是否可用
_cache_spec = importlib.util.find_spec("storage.cache")
if _cache_spec:
    try:
        from storage.cache import get_cache

        CACHE_AVAILABLE = True
    except ImportError:
        CACHE_AVAILABLE = False
        logger.warning("⚠️ Cache 模块导入失败，将仅使用内存缓存")
else:
    CACHE_AVAILABLE = False


@dataclass
class RetrievalResult:
    chunk_id: int
    content: str
    file_path: str
    absolute_path: str
    section: str
    start_pos: int
    end_pos: int
    vault_name: str
    chunk_type: str
    semantic_score: float
    keyword_score: float
    confidence_score: float
    final_score: float
    confidence_reason: str
    file_hash: str
    # 图谱增强分值（由 tinyrag_memory_graph 插件填充）
    graph_score: float = 0.0
    preference_score: float = 0.0
    hop_distance: int = 0
    tags: list[str] = None
    note_title: str = ""
    # 基础检索分数（修复问题3：添加此字段便于调试和验证）
    base_final_score: float = 0.0


class HybridEngine:
    def __init__(self, config: Any, db: DatabaseManager, embed_engine: Any):
        self.config = config
        self.db = db
        self.embed_engine = embed_engine
        self.alpha = config.retrieval.get("alpha", 0.7)
        self.beta = config.retrieval.get("beta", 0.3)

        # 双层缓存
        self._memory_cache: OrderedDict = OrderedDict()
        self._memory_cache_max_size = 500
        self._cache_lock = threading.Lock()
        self._cache = None
        if CACHE_AVAILABLE:
            try:
                cache_cfg = config.cache if hasattr(config, "cache") else None
                if cache_cfg:
                    self._cache = get_cache(
                        db_path=getattr(cache_cfg, "db_path", "./data/cache.db"),
                        ttl_seconds=getattr(cache_cfg, "ttl_seconds", 3600),
                        max_entries=getattr(cache_cfg, "max_entries", 1000),
                    )
                    logger.info("✅ 持久化缓存已启用")
            except Exception as e:
                logger.warning(f"⚠️ 持久化缓存初始化失败，降级为内存缓存: {e}")

        # 加载 jieba 自定义词典 (已由 helper 接管，此处保留兼容性日志)
        if JIEBA_AVAILABLE and hasattr(config, "jieba_user_dict") and config.jieba_user_dict:
            from pathlib import Path

            dict_path = Path(config.jieba_user_dict).expanduser()
            if dict_path.exists():
                logger.info(f"✅ jieba 自定义词典已就绪: {dict_path}")

    def _extract_time_range_from_query(self, query: str) -> dict | None:
        match = re.search(r"(\d{4})(?:-(\d{1,2})(?:-(\d{1,2}))?|年(\d{1,2})(?:月(\d{1,2})日)?)", query)
        if match:
            year = int(match.group(1))
            month = int(match.group(2)) if match.group(2) else (int(match.group(4)) if match.group(4) else None)
            day = int(match.group(3)) if match.group(3) else (int(match.group(5)) if match.group(5) else None)
            return {"year": year, "month": month, "day": day}
        match = re.search(r"(\d{4})年(?!\d)", query)
        if match:
            return {"year": int(match.group(1)), "month": None, "day": None}
        return None

    def _calculate_time_match_score(self, doc_date_str: str, query_time: dict) -> float:
        try:
            doc_date = datetime.strptime(doc_date_str, "%Y-%m-%d")
        except ValueError:
            return 1.0
        qy, qm, qd = query_time.get("year"), query_time.get("month"), query_time.get("day")
        if doc_date.year == qy:
            if qd and qm and doc_date.month == qm and doc_date.day == qd:
                return 2.0
            if qm and doc_date.month == qm:
                return 1.8
            return 1.5
        return max(0.2, 1.0 - abs(doc_date.year - qy) * 0.3)

    def _make_cache_key(
        self,
        query: str,
        limit: int,
        vault_filter: list[str] | None,
        alpha: float | None = None,
        beta: float | None = None,
        query_time: dict | None = None,
    ) -> str:
        raw = f"{query}|{limit}"
        if vault_filter:
            raw += f"|{','.join(sorted(vault_filter))}"
        if alpha is not None:
            raw += f"|a={alpha:.2f}"
        if beta is not None:
            raw += f"|b={beta:.2f}"
        if query_time:
            raw += f"|t={query_time}"
        return hashlib.sha256(raw.encode()).hexdigest()

    @staticmethod
    def _serialize_results(results: list[RetrievalResult]) -> list[dict]:
        return [r.__dict__ for r in results]

    @staticmethod
    def _deserialize_results(data: list[dict]) -> list[RetrievalResult]:
        return [RetrievalResult(**item) for item in data]

    def _cache_get(self, cache_key: str) -> list[RetrievalResult] | None:
        if self._cache is not None:
            try:
                cached = self._cache.get(cache_key)
                if cached is not None:
                    return self._deserialize_results(cached)
            except Exception as e:
                logger.warning(f"持久化缓存读取失败: {e}")
        with self._cache_lock:
            cached = self._memory_cache.get(cache_key)
            if cached is not None:
                self._memory_cache.move_to_end(cache_key)
                return self._deserialize_results(cached)
        return None

    def _cache_set(self, cache_key: str, results: list[RetrievalResult]) -> None:
        serialized = self._serialize_results(results)
        if self._cache is not None:
            try:
                self._cache.set(cache_key, serialized)
            except Exception as e:
                logger.warning(f"持久化缓存写入失败: {e}")
        with self._cache_lock:
            self._memory_cache[cache_key] = serialized
            self._memory_cache.move_to_end(cache_key)
            if len(self._memory_cache) > self._memory_cache_max_size:
                self._memory_cache.popitem(last=False)

    def search(
        self,
        query: str,
        limit: int = 10,
        vault_filter: list[str] | None = None,
        alpha: float | None = None,
        beta: float | None = None,
    ) -> list[RetrievalResult]:
        if not query.strip():
            return []
        effective_alpha = alpha if alpha is not None else self.alpha
        effective_beta = beta if beta is not None else self.beta
        query_time = self._extract_time_range_from_query(query)

        cache_key = self._make_cache_key(query, limit, vault_filter, effective_alpha, effective_beta, query_time)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        query_vector = self.embed_engine.embed([query])[0]
        # ✅ 使用统一 helper 进行分词与日期保护
        keywords = jieba_segment(query) if JIEBA_AVAILABLE else query
        clean_keywords = re.sub(r"[^\w\s\u4e00-\u9fa5\-]", " ", keywords).strip()
        clean_keywords = re.sub(r"\s+", " ", clean_keywords)

        results = self._search_internal(
            query_vector, clean_keywords, limit, vault_filter, effective_alpha, effective_beta, query_time
        )
        self._cache_set(cache_key, results)
        return results

    def _calculate_dynamic_confidence(self, conf_json_str: str, query_time: dict | None = None) -> tuple[float, str]:
        try:
            data = json.loads(conf_json_str or "{}")
        except json.JSONDecodeError:
            data = {}
        conf_cfg = self.config.confidence
        doc_type = data.get("doc_type", "technical")
        status = data.get("status", "active")
        final_date_str = data.get("final_date")
        dt_w = conf_cfg.doc_type_rules.get(doc_type, 1.0)
        st_w = conf_cfg.status_rules.get(status, 1.0)

        date_w, days_passed = 1.0, 365
        time_match_mode = False
        if final_date_str:
            if query_time:
                time_match_mode = True
                date_w = self._calculate_time_match_score(final_date_str, query_time)
                with contextlib.suppress(BaseException):
                    days_passed = (datetime.now() - datetime.strptime(final_date_str, "%Y-%m-%d")).days
            else:
                try:
                    final_dt = datetime.strptime(final_date_str, "%Y-%m-%d")
                    days_passed = (datetime.now() - final_dt).days
                    half_life = conf_cfg.date_decay.type_specific_decay.get(
                        doc_type, conf_cfg.date_decay.half_life_days
                    )
                    date_w = max(conf_cfg.date_decay.min_weight, math.pow(0.5, days_passed / half_life))
                except Exception:
                    date_w = conf_cfg.date_decay.min_weight

        raw_factor = dt_w * st_w * date_w
        # 保护：确保 raw_factor 至少为 0.1，避免 conf_score = 0 导致结果无效
        raw_factor = max(0.1, raw_factor)
        conf_score = math.log1p(raw_factor)
        reason = f"Type:{doc_type}({dt_w}) | Status:{status}({st_w}) | {'TimeMatch' if time_match_mode else f'Age:{days_passed}d'}"
        return conf_score, reason

    def _search_internal(
        self,
        query_vector: Any,
        keywords: str,
        limit: int,
        vault_filter: list[str] | None,
        alpha: float,
        beta: float,
        query_time: dict | None = None,
    ) -> list[RetrievalResult]:
        vec_results = self.db.search_vectors(query_vector, limit=limit * 2)
        vec_scores = {r[0]: r[1] for r in vec_results}
        kw_results = self.db.search_fts(keywords, limit=limit * 2)
        kw_scores = {r[0]: r[1] for r in kw_results}

        # ✅ P2 修复：归一化 FTS5 分数，避免 BM25 原始分数差异过大干扰排序
        # FTS5 BM25 分数范围可能从 <1 到 >600000，使用 log1p 压缩到合理范围
        if kw_scores:
            max_kw = max(kw_scores.values())
            if max_kw > 1.0:
                # 如果只有一个结果，归一化后总是 1.0，可能导致分数偏高
                # 因此对单一结果使用 log1p 压缩但不归一化
                if len(kw_scores) == 1:
                    for cid in kw_scores:
                        kw_scores[cid] = math.log1p(kw_scores[cid]) / math.log1p(10)  # 假设 10 为参考值
                else:
                    for cid in kw_scores:
                        kw_scores[cid] = math.log1p(kw_scores[cid]) / math.log1p(max_kw)

        candidate_ids = list(set(vec_scores.keys()) | set(kw_scores.keys()))
        if not candidate_ids:
            return []

        placeholders = ",".join(["?"] * len(candidate_ids))
        query_sql = f"""
                    SELECT c.*, f.file_path, f.absolute_path, f.vault_name, f.file_hash,
                    c.inherited_meta
                    FROM chunks c
                    JOIN files f ON c.file_id = f.id
                    WHERE c.id IN ({placeholders}) AND c.is_deleted = 0
                    """
        query_params = list(candidate_ids)
        if vault_filter:
            query_sql += " AND f.vault_name IN ({})".format(",".join(["?"] * len(vault_filter)))
            query_params.extend(vault_filter)

        rows = self.db.conn.execute(query_sql, query_params).fetchall()
        final_results = []
        for row in rows:
            cid = row["id"]
            conf_score, conf_reason = self._calculate_dynamic_confidence(row["confidence_json"], query_time)
            # 原始分数
            raw_v_score = vec_scores.get(cid, 0.0)
            raw_k_score = max(0, kw_scores.get(cid, 0.0))
            # 加权分数
            v_score = raw_v_score * alpha
            k_score = raw_k_score * beta
            final_score = (v_score + k_score) * conf_score
            # 显示分数：当权重为 0 时，对应得分显示为 0（表示该检索模式未参与）
            # 这让用户能清楚看到哪些检索模式实际贡献了分数
            display_v_score = raw_v_score if alpha > 0 else 0.0
            display_k_score = raw_k_score if beta > 0 else 0.0
            # 提取标签
            tags = []
            try:
                inherited = json.loads(row["inherited_meta"] or "{}") if row["inherited_meta"] else {}
                tags = inherited.get("tags", [])[:3]
            except (json.JSONDecodeError, TypeError):
                pass
            final_results.append(
                RetrievalResult(
                    chunk_id=cid,
                    content=row["content"],
                    file_path=row["file_path"],
                    absolute_path=row["absolute_path"],
                    section=row["section_title"] or "Root",
                    start_pos=row["start_pos"],
                    end_pos=row["end_pos"],
                    vault_name=row["vault_name"],
                    chunk_type=row["content_type"],
                    semantic_score=display_v_score,
                    keyword_score=display_k_score,
                    confidence_score=conf_score,
                    final_score=final_score,
                    confidence_reason=conf_reason,
                    file_hash=row["file_hash"],
                    tags=tags,
                    note_title="",
                    # 基础检索分数（修复：正确设置 base_final_score）
                    base_final_score=final_score,
                )
            )
        final_results.sort(key=lambda x: x.final_score, reverse=True)
        return final_results[:limit]
