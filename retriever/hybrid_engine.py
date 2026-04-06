#!/usr/bin/env python3
"""
retriever/hybrid_engine.py - 混合检索引擎 (v4.0 - 动态置信度与对数平滑重构)

重构说明:
1. ✅ 动态计算: 从 chunks.confidence_json 读取原始因子，检索时实时计算得分。
2. ✅ 对数平滑: 采用 math.log1p 处理权重因子，防止极端权重破坏检索排序。
3. ✅ 实时衰减: 日期衰减基于检索时刻的 datetime.now() 计算。
4. ✅ 鲁棒降级: 自动处理缺失字段，注入 blog/已完成/365天 缺省逻辑。
"""

import importlib.util
import json
import math
import re
from datetime import datetime
from typing import Any

import jieba
from loguru import logger

from storage.database import DatabaseManager

# 检查 cache 模块是否可用
CACHE_AVAILABLE = importlib.util.find_spec("storage.cache") is not None
if not CACHE_AVAILABLE:
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


class HybridEngine:
    def __init__(self, config: Any, db: DatabaseManager, embed_engine: Any):
        self.config = config
        self.db = db
        self.embed_engine = embed_engine
        self.alpha = config.retrieval.get("alpha", 0.7)  # 向量权重
        self.beta = config.retrieval.get("beta", 0.3)  # 关键词权重
        self._memory_cache = {}

    def search(
        self, query: str, limit: int = 10, vault_filter: list[str] | None = None
    ) -> list[RetrievalResult]:
        """执行混合检索"""
        if not query.strip():
            return []

        # 1. 生成查询向量
        query_vector = self.embed_engine.embed([query])[0]

        # 2. 关键词预处理 (jieba 分词 + 过滤)
        keywords = " ".join(jieba.cut_for_search(query))
        clean_keywords = re.sub(r"[^\w\s\u4e00-\u9fa5]", " ", keywords).strip()

        # 3. 执行检索逻辑
        return self._search_internal(query_vector, clean_keywords, limit, vault_filter)

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
        reason = (
            f"Type:{doc_type}({dt_w}) | Status:{status}({st_w}) | Age:{days_passed}d"
        )

        return conf_score, reason

    def _search_internal(
        self,
        query_vector: Any,
        keywords: str,
        limit: int,
        vault_filter: list[str] | None,
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

        rows = self.db.conn.execute(query_sql, candidate_ids).fetchall()

        final_results = []
        for row in rows:
            cid = row["id"]

            # --- 动态计算置信度 ---
            conf_score, conf_reason = self._calculate_dynamic_confidence(
                row["confidence_json"]
            )

            # --- 分值融合公式 ---
            # 向量分 (0-1 之间)
            v_score = vec_scores.get(cid, 0.0) * self.alpha

            # 关键词分 (FTS5 分数可能很大，同样采用对数平滑对齐量级)
            raw_kw = kw_scores.get(cid, 0.0)
            k_score = math.log1p(max(0, raw_kw)) * self.beta

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
