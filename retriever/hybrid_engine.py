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

# 时间范围查询检测正则
_TIME_RANGE_PATTERNS = [
    # 完整日期: 2023-08-10, 2023年8月10日
    r"(\d{4})(?:-(\d{2})(?:-(\d{2}))?|年(\d{1,2})(?:月(\d{1,2})日)?)",
    # 年月: 2023-08, 2023年8月
    r"(\d{4})(?:-(\d{2})|年(\d{1,2})月)(?![\-日\d])",
    # 年份: 2023年
    r"(\d{4})年(?!\d)",
]

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

        # 加载 jieba 用户自定义词典
        if JIEBA_AVAILABLE and hasattr(config, "jieba_user_dict") and config.jieba_user_dict:
            from pathlib import Path

            dict_path = Path(config.jieba_user_dict).expanduser()
            if dict_path.exists():
                try:
                    jieba.load_userdict(str(dict_path))
                    logger.info(f"✅ jieba 自定义词典加载成功: {dict_path}")
                except Exception as e:
                    logger.warning(f"⚠️ jieba 自定义词典加载失败: {e}")
            else:
                logger.warning(f"⚠️ jieba 自定义词典文件不存在: {dict_path}")

    # ─── 时间范围查询检测 ───
    def _extract_time_range_from_query(self, query: str) -> dict | None:
        """
        从查询中提取时间范围意图
        
        返回: {"year": int, "month": int|None, "day": int|None} 或 None
        """
        # 匹配 YYYY-MM-DD 或 YYYY年M月D日 格式
        match = re.search(r"(\d{4})(?:-(\d{1,2})(?:-(\d{1,2}))?|年(\d{1,2})(?:月(\d{1,2})日)?)", query)
        if match:
            year = int(match.group(1))
            # 数字格式: 2023-08-10
            month = int(match.group(2)) if match.group(2) else None
            day = int(match.group(3)) if match.group(3) else None
            # 中文格式: 2023年8月10日
            if match.group(4):
                month = int(match.group(4))
            if match.group(5):
                day = int(match.group(5))
            return {"year": year, "month": month, "day": day}
        
        # 匹配 YYYY年 格式
        match = re.search(r"(\d{4})年(?!\d)", query)
        if match:
            return {"year": int(match.group(1)), "month": None, "day": None}
        
        return None

    def _calculate_time_match_score(self, doc_date_str: str, query_time: dict) -> float:
        """
        计算文档日期与查询时间范围的匹配度
        
        Args:
            doc_date_str: 文档的 final_date (YYYY-MM-DD 格式)
            query_time: {"year": int, "month": int|None, "day": int|None}
        
        Returns:
            匹配分数 (0.0 - 2.0)，越高越匹配
        """
        try:
            doc_date = datetime.strptime(doc_date_str, "%Y-%m-%d")
        except (ValueError, TypeError):
            return 1.0  # 解析失败，返回中性值
        
        query_year = query_time.get("year")
        query_month = query_time.get("month")
        query_day = query_time.get("day")
        
        # 完全匹配: 年月日都匹配
        if query_day and query_month:
            if (doc_date.year == query_year and 
                doc_date.month == query_month and 
                doc_date.day == query_day):
                return 2.0  # 最高匹配
            # 同一月
            if doc_date.year == query_year and doc_date.month == query_month:
                return 1.5
            # 同一年
            if doc_date.year == query_year:
                return 1.0
            # 不同年
            return 0.3
        
        # 年月匹配
        if query_month:
            if doc_date.year == query_year and doc_date.month == query_month:
                return 2.0
            if doc_date.year == query_year:
                return 1.0
            return 0.3
        
        # 年份匹配
        if doc_date.year == query_year:
            return 2.0
        
        # 距离目标年份越远，分数越低
        year_diff = abs(doc_date.year - query_year)
        if year_diff == 1:
            return 0.5
        return 0.2

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
        """执行混合检索（带缓存）
        
        自动检测时间范围查询，并调整日期权重计算策略：
        - 时间范围查询（如 "2023年8月的日记"）：使用时间匹配度
        - 普通查询：使用日期衰减
        """
        if not query.strip():
            return []

        # 使用传入的 alpha/beta 或默认值
        effective_alpha = alpha if alpha is not None else self.alpha
        effective_beta = beta if beta is not None else self.beta

        # 0. 检测时间范围查询
        query_time = self._extract_time_range_from_query(query)
        if query_time:
            logger.debug(f"🕐 检测到时间范围查询: {query_time}")

        # 1. 缓存查询（注意：需要包含 query_time 在缓存键中）
        cache_key = self._make_cache_key(
            query, limit, vault_filter, effective_alpha, effective_beta
        )
        # 如果是时间范围查询，添加时间标记到缓存键
        if query_time:
            time_key = f"{query_time.get('year')}-{query_time.get('month') or 0}-{query_time.get('day') or 0}"
            cache_key = hashlib.sha256((cache_key + time_key).encode()).hexdigest()
        
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        # 2. 生成查询向量
        query_vector = self.embed_engine.embed([query])[0]

        # 3. 关键词预处理 (jieba 分词 + 过滤)
        # 保护日期格式免被 jieba 拆分
        # 支持格式：
        # - 数字格式：2023-08-10, 2023-09
        # - 中文格式：2023年, 2023年9月, 2023年9月23日
        date_pattern = r"\d{4}(?:-\d{2}(?:-\d{2})?|年(?:\d{1,2}(?:月(?:\d{1,2}日)?)?)?)"
        date_placeholders = {}
        protected_query = query
        for i, match in enumerate(re.finditer(date_pattern, query)):
            placeholder = f"__DATE_{i}__"
            date_placeholders[placeholder] = match.group()
            protected_query = protected_query.replace(match.group(), placeholder, 1)
        
        # jieba 分词
        keywords = " ".join(jieba.cut_for_search(protected_query)) if JIEBA_AVAILABLE else protected_query
        
        # 修复被 jieba 拆分的占位符（如 "__ DATE _ 0 __" -> "__DATE_0__"）
        broken_pattern = re.compile(r'__\s*DATE\s*_\s*(\d+)\s*__')
        keywords = broken_pattern.sub(r'__DATE_\1__', keywords)
        
        # 恢复日期格式
        for placeholder, date_str in date_placeholders.items():
            keywords = keywords.replace(placeholder, date_str)
        
        # 清理特殊字符，但保留日期中的连字符
        clean_keywords = re.sub(r"[^\w\s\u4e00-\u9fa5\-]", " ", keywords).strip()
        clean_keywords = re.sub(r"\s+", " ", clean_keywords)

        # 4. 执行检索逻辑（传入时间范围参数）
        results = self._search_internal(
            query_vector, clean_keywords, limit, vault_filter, 
            effective_alpha, effective_beta, query_time
        )

        # 5. 写入缓存
        self._cache_set(cache_key, results)

        return results

    def _calculate_dynamic_confidence(
        self, 
        conf_json_str: str, 
        query_time: dict | None = None
    ) -> tuple[float, str]:
        """
        核心重构：实现设想中的第 5 点（对数计算可信度）
        
        Args:
            conf_json_str: 从数据库读取的 confidence_json
            query_time: 如果是时间范围查询，包含 {"year", "month", "day"}
                        此时使用时间匹配度替代日期衰减
        """
        try:
            data = json.loads(conf_json_str or "{}")
        except json.JSONDecodeError:
            data = {}

        # A. 获取配置
        conf_cfg = self.config.confidence

        # B. 基础因子提取 (含缺省逻辑)
        doc_type = data.get("doc_type", "technical")
        status = data.get("status", "active")
        final_date_str = data.get("final_date")

        dt_w = conf_cfg.doc_type_rules.get(doc_type, 1.0)
        st_w = conf_cfg.status_rules.get(status, 1.0)

        # C. 日期权重计算
        date_w = 1.0
        days_passed = 365
        time_match_mode = False
        
        if final_date_str:
            # 如果是时间范围查询，使用时间匹配度替代日期衰减
            if query_time:
                time_match_mode = True
                date_w = self._calculate_time_match_score(final_date_str, query_time)
                # 解析文档日期用于显示
                try:
                    final_dt = datetime.strptime(final_date_str, "%Y-%m-%d")
                    days_passed = (datetime.now() - final_dt).days
                except ValueError:
                    days_passed = 0
            else:
                # 普通查询：使用日期衰减
                try:
                    final_dt = datetime.strptime(final_date_str, "%Y-%m-%d")
                    days_passed = (datetime.now() - final_dt).days

                    # 根据文档类型选择半衰期
                    half_life = conf_cfg.date_decay.half_life_days  # 默认值
                    if doc_type in conf_cfg.date_decay.type_specific_decay:
                        half_life = conf_cfg.date_decay.type_specific_decay[doc_type]

                    # 指数衰减公式: 2^(-days/half_life)
                    decay = math.pow(0.5, days_passed / half_life)
                    date_w = max(conf_cfg.date_decay.min_weight, decay)
                except (ValueError, AttributeError):
                    date_w = conf_cfg.date_decay.min_weight

        # D. 对数平滑融合 (关键点)
        # 原始乘积
        raw_factor = dt_w * st_w * date_w
        # 对数化：使用 ln(1 + x) 保证非负且增长平滑
        conf_score = math.log1p(raw_factor)

        # E. 生成理由描述
        if time_match_mode:
            reason = f"Type:{doc_type}({dt_w}) | Status:{status}({st_w}) | TimeMatch:{date_w:.1f}"
        else:
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
        query_time: dict | None = None,
    ) -> list[RetrievalResult]:
        """内部检索逻辑：整合向量、FTS5 与 动态权重
        
        Args:
            query_vector: 查询向量
            keywords: 关键词字符串
            limit: 返回数量限制
            vault_filter: 仓库过滤列表
            alpha: 向量权重
            beta: 关键词权重
            query_time: 时间范围查询参数 {"year", "month", "day"}
        """

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
            conf_score, conf_reason = self._calculate_dynamic_confidence(
                row["confidence_json"], query_time
            )

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
