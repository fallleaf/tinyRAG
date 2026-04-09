#!/usr/bin/env python3
"""
retriever/hybrid_engine.py - 混合检索引擎 (v4.0 - 动态置信度与对数平滑重构)

重构说明:
1. ✅ 动态计算: 从 chunks.confidence_json 读取原始因子，检索时实时计算得分。
2. ✅ 对数平滑: 采用 math.log1p 处理权重因子，防止极端权重破坏检索排序。
3. ✅ 实时衰减: 日期衰减基于检索时刻的 datetime.now() 计算。
4. ✅ 鲁棒降级: 自动处理缺失字段，注入 blog/已完成/365天 缺省逻辑。
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
    # 修复 L5: 模块级别调用 jieba.initialize()，避免首次使用时的延迟
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


# 修复 L7: 使用 @dataclass 简化 RetrievalResult
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
        self.alpha = config.retrieval.get("alpha", 0.7)  # 向量权重
        self.beta = config.retrieval.get("beta", 0.3)  # 关键词权重

        # 缓存初始化：优先使用持久化 QueryCache，降级为内存字典
        # 修复 M2: 使用 OrderedDict 实现 LRU 缓存
        self._memory_cache: OrderedDict = OrderedDict()
        self._memory_cache_max_size = 500
        self._cache_lock = threading.Lock()  # 添加锁保护
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
                else:
                    logger.info("ℹ️ 未配置 cache，使用内存缓存")
            except Exception as e:
                logger.warning(f"⚠️ 持久化缓存初始化失败，降级为内存缓存: {e}")

    # ─── 缓存键生成 ───
    def _make_cache_key(
        self,
        query: str,
        limit: int,
        vault_filter: list[str] | None,
        alpha: float | None = None,
        beta: float | None = None,
    ) -> str:
        """基于查询内容 + 参数生成确定性缓存键"""
        raw = f"{query}|{limit}"
        if vault_filter:
            raw += f"|{','.join(sorted(vault_filter))}"
        # 修复 M1: 使用实际生效的 alpha/beta
        if alpha is not None:
            raw += f"|a={alpha:.2f}"
        if beta is not None:
            raw += f"|b={beta:.2f}"
        # 修复 L4: 使用 hashlib.sha256 替代 hashlib.md5，降低碰撞风险
        return hashlib.sha256(raw.encode()).hexdigest()

    # ─── 序列化 / 反序列化 ───
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

    # ─── 缓存读写 ───
    def _cache_get(self, cache_key: str) -> list[RetrievalResult] | None:
        """读取缓存，优先持久化，降级内存"""
        # 1) 持久化缓存
        if self._cache is not None:
            try:
                cached = self._cache.get(cache_key)
                if cached is not None:
                    logger.debug("Cache HIT (persistent)")
                    return self._deserialize_results(cached)
            except Exception as e:
                logger.warning(f"持久化缓存读取失败: {e}")

        # 2) 内存缓存（带锁保护）
        with self._cache_lock:
            cached = self._memory_cache.get(cache_key)
            if cached is not None:
                # LRU: 移动到末尾
                self._memory_cache.move_to_end(cache_key)
                logger.debug("Cache HIT (memory)")
                return self._deserialize_results(cached)

        return None

    def _cache_set(self, cache_key: str, results: list[RetrievalResult]) -> None:
        """写入缓存，双层同步"""
        serialized = self._serialize_results(results)

        # 持久化缓存
        if self._cache is not None:
            try:
                self._cache.set(cache_key, serialized)
            except Exception as e:
                logger.warning(f"持久化缓存写入失败: {e}")

        # 内存缓存（带锁保护）
        with self._cache_lock:
            self._memory_cache[cache_key] = serialized
            # LRU: 移动到末尾
            self._memory_cache.move_to_end(cache_key)
            # LRU: 超过最大容量时删除最旧的
            if len(self._memory_cache) > self._memory_cache_max_size:
                self._memory_cache.popitem(last=False)
                logger.debug(f"Memory cache pruned to {len(self._memory_cache)} entries")

    def search(
        self,
        query: str,
        limit: int = 10,
        vault_filter: list[str] | None = None,
        alpha: float | None = None,
        beta: float | None = None,
    ) -> list[RetrievalResult]:
        """执行混合检索（带缓存）"""
        if not query.strip():
            return []

        # 使用传入的 alpha/beta 或默认值
        effective_alpha = alpha if alpha is not None else self.alpha
        effective_beta = beta if beta is not None else self.beta

        # 0. 缓存查询
        # 修复 M1: 使用 effective 值生成缓存键
        cache_key = self._make_cache_key(query, limit, vault_filter, effective_alpha, effective_beta)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        # 1. 生成查询向量
        query_vector = self.embed_engine.embed([query])[0]

        # 2. 关键词预处理 (jieba 分词 + 过滤)
        keywords = " ".join(jieba.cut_for_search(query)) if JIEBA_AVAILABLE else query
        clean_keywords = re.sub(r"[^\w\s\u4e00-\u9fa5]", " ", keywords).strip()

        # 3. 执行检索逻辑
        results = self._search_internal(
            query_vector, clean_keywords, limit, vault_filter, effective_alpha, effective_beta
        )

        # 4. 写入缓存
        self._cache_set(cache_key, results)

        return results

    def _calculate_dynamic_confidence(self, conf_json_str: str) -> tuple[float, str]:
        """
        核心重构：实现设想中的第 5 点（对数计算可信度）
        """
        try:
            data = json.loads(conf_json_str or "{}")
        except json.JSONDecodeError:
            data = {}

        # A. 获取配置
        conf_cfg = self.config.confidence

        # B. 基础因子提取 (含缺省逻辑)
        doc_type = data.get("doc_type", "blog")
        status = data.get("status", "已完成")
        final_date_str = data.get("final_date")

        dt_w = conf_cfg.doc_type_rules.get(doc_type, 1.0)
        st_w = conf_cfg.status_rules.get(status, 1.0)

        # C. 实时日期衰减计算
        date_w = 1.0
        days_passed = 365
        if final_date_str:
            try:
                final_dt = datetime.strptime(final_date_str, "%Y-%m-%d")
                days_passed = (datetime.now() - final_dt).days
                # 指数衰减公式: 2^(-days/half_life)
                decay = math.pow(0.5, days_passed / conf_cfg.date_decay.half_life_days)
                date_w = max(conf_cfg.date_decay.min_weight, decay)
            except (ValueError, AttributeError):
                date_w = conf_cfg.date_decay.min_weight

        # D. 对数平滑融合 (关键点)
        # 原始乘积
        raw_factor = dt_w * st_w * date_w
        # 对数化：使用 ln(1 + x) 保证非负且增长平滑
        conf_score = math.log1p(raw_factor)

        # E. 生成理由描述
        reason = f"Type:{doc_type}({dt_w}) | Status:{status}({st_w}) | Age:{days_passed}d"

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
        """内部检索逻辑：整合向量、FTS5 与 动态权重"""

        # 1. 向量检索 (获取 ID 和向量余弦分)
        vec_results = self.db.search_vectors(query_vector, limit=limit * 2)
        vec_scores = {r[0]: r[1] for r in vec_results}

        # 2. 关键词检索 (获取 ID 和 FTS5 BM25 分)
        kw_results = self.db.search_fts(keywords, limit=limit * 2)
        kw_scores = {r[0]: r[1] for r in kw_results}

        # 合并所有候选 ID
        candidate_ids = list(set(vec_scores.keys()) | set(kw_scores.keys()))
        if not candidate_ids:
            return []

        # 3. 从数据库拉取详细信息 (包括新增的 confidence_json)
        placeholders = ",".join(["?"] * len(candidate_ids))
        query_sql = f"""
            SELECT c.*, f.file_path, f.absolute_path, f.vault_name, f.file_hash
            FROM chunks c
            JOIN files f ON c.file_id = f.id
            WHERE c.id IN ({placeholders}) AND c.is_deleted = 0
        """

        # 修复 C2: 添加 vault_filter 过滤条件
        query_params = list(candidate_ids)
        if vault_filter:
            query_sql += " AND f.vault_name IN ({})".format(",".join(["?"] * len(vault_filter)))
            query_params.extend(vault_filter)

        rows = self.db.conn.execute(query_sql, query_params).fetchall()

        final_results = []
        for row in rows:
            cid = row["id"]

            # --- 动态计算置信度 ---
            conf_score, conf_reason = self._calculate_dynamic_confidence(row["confidence_json"])

            # --- 分值融合公式 ---
            # 向量分 (0-1 之间)
            v_score = vec_scores.get(cid, 0.0) * alpha

            # 关键词分 (FTS5 分数可能很大，同样采用对数平滑对齐量级)
            raw_kw = kw_scores.get(cid, 0.0)
            k_score = math.log1p(max(0, raw_kw)) * beta

            # 最终加权
            # (基础得分) * 动态置信度系数
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
                    semantic_score=v_score,
                    keyword_score=k_score,
                    confidence_score=conf_score,
                    final_score=final_score,
                    confidence_reason=conf_reason,
                    file_hash=row["file_hash"],
                )
            )

        # 4. 排序并截断
        final_results.sort(key=lambda x: x.final_score, reverse=True)
        return final_results[:limit]
