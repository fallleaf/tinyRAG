#!/usr/bin/env python3
"""
retriever/hybrid_engine.py - 混合检索引擎 (v4.1 - 置信度计算修复)

v4.1 修复内容:
1. ✅ 修复缺省值不一致：统一使用 technical/completed
2. ✅ 添加 status 中文→英文映射：已完成→completed, 进行中→active 等
3. ✅ 优化置信度计算：保留权重差异，避免对数过度压缩
4. ✅ 置信度归一化：输出范围 [0.5, 1.5]，便于理解
"""

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
from utils.logger import logger

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
else:
    CACHE_AVAILABLE = False


# 🔧 新增：status 中文→英文映射
STATUS_ZH_TO_EN = {
    "已完成": "completed",
    "完成": "completed",
    "进行中": "active",
    "进行": "active",
    "草稿": "draft",
    "已归档": "archived",
    "归档": "archived",
    "已发布": "published",
    "发布": "published",
}

# 🔧 新增：doc_type 中文→英文映射
DOC_TYPE_ZH_TO_EN = {
    "正式": "official",
    "项目": "project",
    "技术": "technical",
    "个人": "personal",
    "归档": "archive",
    "博客": "blog",
    "日记": "personal",
    "笔记": "technical",
}


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


class HybridEngine:
    def __init__(self, config: Any, db: DatabaseManager, embed_engine: Any):
        self.config = config
        self.db = db
        self.embed_engine = embed_engine
        self.alpha = config.retrieval.get("alpha", 0.7)
        self.beta = config.retrieval.get("beta", 0.3)

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
                logger.warning(f"⚠️ 持久化缓存初始化失败: {e}")

        if JIEBA_AVAILABLE and hasattr(config, "jieba_user_dict") and config.jieba_user_dict:
            from pathlib import Path
            dict_path = Path(config.jieba_user_dict).expanduser()
            if not dict_path.is_absolute():
                dict_path = Path("data") / dict_path.name
            if dict_path.exists():
                try:
                    jieba.load_userdict(str(dict_path))
                    logger.info(f"✅ jieba 自定义词典加载成功: {dict_path}")
                except Exception as e:
                    logger.warning(f"⚠️ jieba 自定义词典加载失败: {e}")

    def _make_cache_key(
        self,
        query: str,
        limit: int,
        vault_filter: list[str] | None,
        alpha: float | None = None,
        beta: float | None = None,
    ) -> str:
        raw = f"{query}|{limit}"
        if vault_filter:
            raw += f"|{','.join(sorted(vault_filter))}"
        if alpha is not None:
            raw += f"|a={alpha:.2f}"
        if beta is not None:
            raw += f"|b={beta:.2f}"
        return hashlib.sha256(raw.encode()).hexdigest()

    @staticmethod
    def _serialize_results(results: list[RetrievalResult]) -> list[dict]:
        return [
            {
                "chunk_id": r.chunk_id,
                "content": r.content,
                "file_path": r.file_path,
                "absolute_path": r.absolute_path,
                "section": r.section,
                "start_pos": r.start_pos,
                "end_pos": r.end_pos,
                "vault_name": r.vault_name,
                "chunk_type": r.chunk_type,
                "semantic_score": r.semantic_score,
                "keyword_score": r.keyword_score,
                "confidence_score": r.confidence_score,
                "final_score": r.final_score,
                "confidence_reason": r.confidence_reason,
                "file_hash": r.file_hash,
            }
            for r in results
        ]

    @staticmethod
    def _deserialize_results(data: list[dict]) -> list[RetrievalResult]:
        return [RetrievalResult(**item) for item in data]

    def _cache_get(self, cache_key: str) -> list[RetrievalResult] | None:
        if self._cache is not None:
            try:
                cached = self._cache.get(cache_key)
                if cached is not None:
                    return self._deserialize_results(cached)
            except Exception:
                pass

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
            except Exception:
                pass

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

        cache_key = self._make_cache_key(query, limit, vault_filter, effective_alpha, effective_beta)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        query_vector = self.embed_engine.embed([query])[0]
        keywords = " ".join(jieba.cut_for_search(query)) if JIEBA_AVAILABLE else query
        clean_keywords = re.sub(r"[^\w\s\u4e00-\u9fa5]", " ", keywords).strip()

        results = self._search_internal(
            query_vector, clean_keywords, limit, vault_filter, effective_alpha, effective_beta
        )
        self._cache_set(cache_key, results)
        return results

    def _normalize_status(self, status: str) -> str:
        """
        🔧 新增：标准化 status 值
        1. 尝试中文→英文映射
        2. 转小写后匹配
        3. 返回原始值作为 fallback
        """
        # 尝试中文映射
        if status in STATUS_ZH_TO_EN:
            return STATUS_ZH_TO_EN[status]
        
        # 转小写后直接匹配
        status_lower = status.lower()
        
        # 已知英文状态值
        known_status = {"published", "completed", "active", "draft", "archived"}
        if status_lower in known_status:
            return status_lower
        
        return status_lower  # 返回小写版本

    def _normalize_doc_type(self, doc_type: str) -> str:
        """
        🔧 新增：标准化 doc_type 值
        """
        # 尝试中文映射
        if doc_type in DOC_TYPE_ZH_TO_EN:
            return DOC_TYPE_ZH_TO_EN[doc_type]
        
        doc_type_lower = doc_type.lower()
        known_types = {"official", "project", "technical", "personal", "archive", "blog"}
        if doc_type_lower in known_types:
            return doc_type_lower
        
        return doc_type_lower

    def _calculate_dynamic_confidence(self, conf_json_str: str) -> tuple[float, str]:
        """
        🔧 重构：置信度计算（保留权重差异）
        
        v4.1 改进：
        1. 统一缺省值为 technical/completed
        2. 自动转换中文 status/doc_type
        3. 使用归一化公式保留权重差异
        """
        try:
            data = json.loads(conf_json_str or "{}")
        except json.JSONDecodeError:
            data = {}

        conf_cfg = self.config.confidence

        # A. 提取并标准化 doc_type
        raw_doc_type = data.get("doc_type", "technical")  # 🔧 统一缺省值为 technical
        doc_type = self._normalize_doc_type(raw_doc_type)
        
        # B. 提取并标准化 status
        raw_status = data.get("status", "completed")  # 🔧 统一缺省值为 completed
        status = self._normalize_status(raw_status)

        # C. 获取权重（使用标准化后的值）
        dt_w = conf_cfg.doc_type_rules.get(doc_type, conf_cfg.default_weight)
        st_w = conf_cfg.status_rules.get(status, conf_cfg.default_weight)

        # D. 日期衰减计算
        final_date_str = data.get("final_date")
        date_w = 1.0
        days_passed = 365
        
        if final_date_str:
            try:
                final_dt = datetime.strptime(final_date_str, "%Y-%m-%d")
                days_passed = max(0, (datetime.now() - final_dt).days)

                half_life = conf_cfg.date_decay.half_life_days
                if doc_type in conf_cfg.date_decay.type_specific_decay:
                    half_life = conf_cfg.date_decay.type_specific_decay[doc_type]

                decay = math.pow(0.5, days_passed / half_life)
                date_w = max(conf_cfg.date_decay.min_weight, decay)
            except (ValueError, AttributeError):
                date_w = conf_cfg.date_decay.min_weight

        # E. 🔧 核心修复：置信度计算公式
        # 原始公式：log1p(raw_factor) 会过度压缩权重差异
        # 新公式：归一化到 [0.5, 1.5] 范围，保留权重差异
        raw_factor = dt_w * st_w * date_w
        
        # 归一化公式：将 raw_factor (约 0.3~2.0) 映射到 0.5~1.5
        # 假设 raw_factor 范围为 [0.3, 2.0]
        MIN_FACTOR = 0.3
        MAX_FACTOR = 2.0
        TARGET_MIN = 0.5
        TARGET_MAX = 1.5
        
        # 线性归一化
        normalized = (raw_factor - MIN_FACTOR) / (MAX_FACTOR - MIN_FACTOR)
        conf_score = TARGET_MIN + normalized * (TARGET_MAX - TARGET_MIN)
        
        # 确保在目标范围内
        conf_score = max(TARGET_MIN, min(TARGET_MAX, conf_score))

        # F. 生成详细的理由描述（便于调试）
        reason = (
            f"Type:{raw_doc_type}→{doc_type}({dt_w:.2f}) | "
            f"Status:{raw_status}→{status}({st_w:.2f}) | "
            f"Age:{days_passed}d→{date_w:.2f} | "
            f"Raw:{raw_factor:.3f}→Score:{conf_score:.3f}"
        )

        return conf_score, reason

    def _search_internal(
        self,
        query_vector: Any,
        keywords: str,
        limit: int,
        vault_filter: list[str] | None,
        alpha: float,
        beta: float,
    ) -> list[RetrievalResult]:
        vec_results = self.db.search_vectors(query_vector, limit=limit * 2)
        vec_scores = {r[0]: r[1] for r in vec_results}

        kw_results = self.db.search_fts(keywords, limit=limit * 2)
        kw_scores = {r[0]: r[1] for r in kw_results}

        candidate_ids = list(set(vec_scores.keys()) | set(kw_scores.keys()))
        if not candidate_ids:
            return []

        placeholders = ",".join(["?"] * len(candidate_ids))
        query_sql = f"""
            SELECT c.*, f.file_path, f.absolute_path, f.vault_name, f.file_hash
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

            conf_score, conf_reason = self._calculate_dynamic_confidence(row["confidence_json"])

            raw_v = vec_scores.get(cid, 0.0)
            v_score = raw_v * alpha

            raw_kw = kw_scores.get(cid, 0.0)
            normalized_kw = 1.0 / (1.0 + math.exp(-math.log1p(max(0, raw_kw)) + 1.0))
            k_score = normalized_kw * beta

            final_score = (v_score + k_score) * conf_score

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
                    semantic_score=raw_v,
                    keyword_score=normalized_kw,
                    confidence_score=conf_score,
                    final_score=final_score,
                    confidence_reason=conf_reason,
                    file_hash=row["file_hash"],
                )
            )

        final_results.sort(key=lambda x: x.final_score, reverse=True)
        return final_results[:limit]
