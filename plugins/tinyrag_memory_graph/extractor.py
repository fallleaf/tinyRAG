#!/usr/bin/env python3
"""
extractor.py - 双层实体/关系抽取管道

实现文档级（Frontmatter/Wikilinks）和 Chunk 级（NLP/LLM）的实体/关系抽取。
"""

import hashlib
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import yaml
from loguru import logger

from plugins.tinyrag_memory_graph.config import ExtractionConfig, LLMConfig, MemoryGraphConfig
from plugins.tinyrag_memory_graph.models import Entity, EntityType, Relation, RelationType


# ============================================================
# 默认词典定义（代码内置，作为后备）
# ============================================================

# SQL/编程保留字黑名单（应被过滤，不应作为实体）
_DEFAULT_SQL_KEYWORDS = {
    # SQL 关键字
    "SELECT", "FROM", "WHERE", "JOIN", "LEFT", "RIGHT", "INNER", "OUTER", "ON",
    "AND", "OR", "NOT", "NULL", "TRUE", "FALSE", "IS", "IN", "LIKE", "BETWEEN",
    "ORDER", "BY", "GROUP", "HAVING", "LIMIT", "OFFSET", "UNION", "DISTINCT",
    "CREATE", "TABLE", "INDEX", "VIEW", "DROP", "ALTER", "ADD", "COLUMN",
    "PRIMARY", "KEY", "FOREIGN", "REFERENCES", "UNIQUE", "CHECK", "DEFAULT",
    "INSERT", "INTO", "VALUES", "UPDATE", "SET", "DELETE", "TRUNCATE",
    "COMMIT", "ROLLBACK", "TRANSACTION", "BEGIN", "END",
    "INT", "INTEGER", "BIGINT", "SMALLINT", "TINYINT",
    "VARCHAR", "CHAR", "TEXT", "LONGTEXT", "MEDIUMTEXT",
    "DECIMAL", "NUMERIC", "FLOAT", "DOUBLE", "REAL",
    "DATE", "TIME", "DATETIME", "TIMESTAMP", "YEAR",
    "BOOLEAN", "BOOL", "BLOB", "JSON",
    # Dockerfile 关键字
    "FROM", "RUN", "CMD", "ENTRYPOINT", "COPY", "ADD", "WORKDIR", "EXPOSE",
    "ENV", "ARG", "LABEL", "USER", "VOLUME", "HEALTHCHECK", "SHELL",
    # 常见编程保留字/状态词
    "COPY", "MOVE", "DELETE", "UPDATE", "INSERT", "GET", "POST", "PUT", "PATCH",
    "pending", "processing", "done", "failed", "active", "inactive", "enabled", "disabled",
    "paid", "unpaid", "reserved", "available", "completed", "cancelled",
    "name", "value", "type", "status", "id", "key", "data", "result", "error",
    # spaCy 常见误识别的英文词（这些词被误识别为 PERSON/LOC/ORG）
    # 注意：技术术语由 rule 正确识别，spacy 提取的被过滤
    "gateway", "burstCapacity", "Reque", "ReL", "Learning", "Boosting",
    "Hugging", "Transformers", "order",  # spaCy 常见误识别
    # 中文常见误识别
    "模型", "上下文", "独立性强", "简化版", "第一", "第二", "第三",
    "下午", "晚上", "次日", "一周", "两周",
}

# 技术术语白名单（即使全大写也视为有效技术实体）
_DEFAULT_TECH_WHITELIST = {
    # AI/ML 框架
    "BERT", "GPT", "LLM", "NLP", "CNN", "RNN", "LSTM", "GRU", "GAN", "VAE",
    "PCA", "SNE", "UMAP", "TF-IDF", "BERT", "T5", "XLNet", "RoBERTa",
    # 数据库
    "MySQL", "PostgreSQL", "MongoDB", "Redis", "SQLite", "Neo4j", "Milvus",
    # 基础设施
    "Kubernetes", "Docker", "Nginx", "Apache", "Linux", "Unix", "Windows",
    # 云服务
    "AWS", "GCP", "Azure", "Aliyun",
    # 监控/日志
    "Prometheus", "Grafana", "ELK", "Jaeger", "Zipkin",
    # 消息队列
    "Kafka", "RabbitMQ", "RocketMQ", "ActiveMQ",
    # 框架/库
    "AlexNet", "VGG", "ResNet", "LeNet", "EfficientNet",
    # 其他技术术语
    "REST", "gRPC", "GraphQL", "JSON", "YAML", "XML", "HTTP", "HTTPS",
    "TCP", "UDP", "IP", "DNS", "CDN", "SSL", "TLS", "SSH",
    "QPS", "TPS", "SLA", "GMV", "DAU", "MAU", "ARPU",
    "CPU", "GPU", "RAM", "SSD", "HDD", "IOPS",
    "SDK", "API", "CLI", "GUI", "UI", "UX",
    # GPU/计算相关
    "CUDA", "cuDNN", "ROCm", "OpenCL", "MPI", "OpenMP",
}

# ============================================================
# 增强实体提取：中文技术领域词典（默认值）
# ============================================================

# 中文技术术语词典（用于规则匹配）
_DEFAULT_CHINESE_TECH_TERMS = {
    # AI/ML 领域
    "深度学习": EntityType.TECHNOLOGY,
    "机器学习": EntityType.TECHNOLOGY,
    "自然语言处理": EntityType.TECHNOLOGY,
    "计算机视觉": EntityType.TECHNOLOGY,
    "神经网络": EntityType.TECHNOLOGY,
    "卷积神经网络": EntityType.TECHNOLOGY,
    "循环神经网络": EntityType.TECHNOLOGY,
    "注意力机制": EntityType.TECHNOLOGY,
    "迁移学习": EntityType.TECHNOLOGY,
    "强化学习": EntityType.TECHNOLOGY,
    "知识图谱": EntityType.TECHNOLOGY,
    "推荐系统": EntityType.TECHNOLOGY,
    "搜索引擎": EntityType.TECHNOLOGY,
    "语音识别": EntityType.TECHNOLOGY,
    "图像识别": EntityType.TECHNOLOGY,
    "目标检测": EntityType.TECHNOLOGY,
    "语义分割": EntityType.TECHNOLOGY,
    "文本分类": EntityType.TECHNOLOGY,
    "情感分析": EntityType.TECHNOLOGY,
    "命名实体识别": EntityType.TECHNOLOGY,
    "分词": EntityType.TECHNOLOGY,
    "词向量": EntityType.TECHNOLOGY,
    "预训练模型": EntityType.TECHNOLOGY,
    "微调": EntityType.TECHNOLOGY,
    "提示工程": EntityType.TECHNOLOGY,
    
    # 数据处理
    "数据清洗": EntityType.TECHNOLOGY,
    "数据标注": EntityType.TECHNOLOGY,
    "特征工程": EntityType.TECHNOLOGY,
    "数据挖掘": EntityType.TECHNOLOGY,
    "数据分析": EntityType.TECHNOLOGY,
    "数据仓库": EntityType.TECHNOLOGY,
    "数据湖": EntityType.TECHNOLOGY,
    
    # 架构/系统
    "微服务": EntityType.TECHNOLOGY,
    "分布式系统": EntityType.TECHNOLOGY,
    "容器化": EntityType.TECHNOLOGY,
    "负载均衡": EntityType.TECHNOLOGY,
    "服务发现": EntityType.TECHNOLOGY,
    "配置中心": EntityType.TECHNOLOGY,
    "消息队列": EntityType.TECHNOLOGY,
    "缓存": EntityType.TECHNOLOGY,
    "数据库": EntityType.TECHNOLOGY,
    "关系型数据库": EntityType.TECHNOLOGY,
    "文档数据库": EntityType.TECHNOLOGY,
    "向量数据库": EntityType.TECHNOLOGY,
    "图数据库": EntityType.TECHNOLOGY,
    
    # 开发相关
    "前端开发": EntityType.TECHNOLOGY,
    "后端开发": EntityType.TECHNOLOGY,
    "全栈开发": EntityType.TECHNOLOGY,
    "接口": EntityType.TECHNOLOGY,
    "框架": EntityType.TECHNOLOGY,
    "中间件": EntityType.TECHNOLOGY,
    "持续集成": EntityType.TECHNOLOGY,
    "持续部署": EntityType.TECHNOLOGY,
    "敏捷开发": EntityType.TECHNOLOGY,
    
    # 职位/角色
    "算法工程师": EntityType.CONCEPT,
    "数据科学家": EntityType.CONCEPT,
    "架构师": EntityType.CONCEPT,
    "产品经理": EntityType.CONCEPT,
    "前端工程师": EntityType.CONCEPT,
    "后端工程师": EntityType.CONCEPT,
    "运维工程师": EntityType.CONCEPT,
    "测试工程师": EntityType.CONCEPT,
}

# 中文实体后缀模式（用于识别组织、地点等）
_DEFAULT_CHINESE_ENTITY_SUFFIXES = {
    # 组织后缀
    "公司": EntityType.ORG,
    "集团": EntityType.ORG,
    "科技": EntityType.ORG,
    "实验室": EntityType.ORG,
    "研究院": EntityType.ORG,
    "研究所": EntityType.ORG,
    "大学": EntityType.ORG,
    "学院": EntityType.ORG,
    "医院": EntityType.ORG,
    "银行": EntityType.ORG,
    "基金": EntityType.ORG,
    "协会": EntityType.ORG,
    
    # 地点后缀
    "省": EntityType.LOC,
    "市": EntityType.LOC,
    "区": EntityType.LOC,
    "县": EntityType.LOC,
    "镇": EntityType.LOC,
    "村": EntityType.LOC,
    "岛": EntityType.LOC,
    "江": EntityType.LOC,
    "河": EntityType.LOC,
    "山": EntityType.LOC,
    "湖": EntityType.LOC,
    "湾": EntityType.LOC,
}

# ============================================================
# 增强关系提取：关系模式规则
# ============================================================

# 中文关系模式（正则表达式）
RELATION_PATTERNS = [
    # "A 由 B 开发" 模式（优先级高，放在前面）
    {
        "pattern": r"(.{2,20})由(.{2,10})开发",
        "subject": 1,
        "object": 2,
        "rel_type": RelationType.DEVELOPED_BY,
        "confidence": 0.9,
    },
    # "A 由 B 创建" 模式
    {
        "pattern": r"(.{2,20})由(.{2,10})创建",
        "subject": 1,
        "object": 2,
        "rel_type": RelationType.CREATED_BY,
        "confidence": 0.9,
    },
    # "A 由 B 提出" 模式
    {
        "pattern": r"(.{2,20})由(.{2,10})提出",
        "subject": 1,
        "object": 2,
        "rel_type": RelationType.CREATED_BY,
        "confidence": 0.9,
    },
    # "A 在 B 工作" 模式
    {
        "pattern": r"(.{2,10})在(.{2,15})工作",
        "subject": 1,
        "object": 2,
        "rel_type": RelationType.WORKS_FOR,
        "confidence": 0.9,
    },
    # "A 就职于 B" 模式
    {
        "pattern": r"(.{2,10})就职于(.{2,15})",
        "subject": 1,
        "object": 2,
        "rel_type": RelationType.WORKS_FOR,
        "confidence": 0.9,
    },
    # "A 使用 B" 模式
    {
        "pattern": r"(.{2,20})使用(.{2,15})",
        "subject": 1,
        "object": 2,
        "rel_type": RelationType.USED_BY,
        "confidence": 0.8,
    },
    # "A 依赖 B" 模式
    {
        "pattern": r"(.{2,20})依赖(.{2,15})",
        "subject": 1,
        "object": 2,
        "rel_type": RelationType.DEPENDS_ON,
        "confidence": 0.9,
    },
    # "A 基于 B" 模式
    {
        "pattern": r"(.{2,20})基于(.{2,15})",
        "subject": 1,
        "object": 2,
        "rel_type": RelationType.DERIVED_FROM,
        "confidence": 0.85,
    },
    # "A 位于 B" 模式
    {
        "pattern": r"(.{2,20})位于(.{2,15})",
        "subject": 1,
        "object": 2,
        "rel_type": RelationType.LOCATED_AT,
        "confidence": 0.9,
    },
    # "A 包含 B" 模式
    {
        "pattern": r"(.{2,20})包含(.{2,15})",
        "subject": 1,
        "object": 2,
        "rel_type": RelationType.PART_OF,
        "confidence": 0.8,
    },
    # "A 负责 B" 模式
    {
        "pattern": r"(.{2,10})负责(.{2,15})",
        "subject": 1,
        "object": 2,
        "rel_type": RelationType.MANAGED_BY,
        "confidence": 0.8,
    },
    # "A 管理 B" 模式
    {
        "pattern": r"(.{2,10})管理(.{2,15})",
        "subject": 1,
        "object": 2,
        "rel_type": RelationType.MANAGED_BY,
        "confidence": 0.85,
    },
    # "A 是 B 的..." 模式（通用性较强，放在后面）
    {
        "pattern": r"(.{2,15})是(.{2,15})的",
        "subject": 1,
        "object": 2,
        "rel_type": RelationType.BELONGS_TO,
        "confidence": 0.7,
    },
    # 英文模式: "A was developed by B"
    {
        "pattern": r"([A-Za-z][A-Za-z0-9_\-\s]{0,25})\s+was\s+developed\s+by\s+([A-Za-z][A-Za-z0-9_\-\s]{0,25})",
        "subject": 1,
        "object": 2,
        "rel_type": RelationType.DEVELOPED_BY,
        "confidence": 0.9,
    },
    # 英文模式: "A is developed by B"
    {
        "pattern": r"([A-Za-z][A-Za-z0-9_\-]{0,25})\s+is\s+developed\s+by\s+([A-Za-z][A-Za-z0-9_\-\s]{0,25})",
        "subject": 1,
        "object": 2,
        "rel_type": RelationType.DEVELOPED_BY,
        "confidence": 0.9,
    },
    # 英文模式: "A was created by B"
    {
        "pattern": r"([A-Za-z][A-Za-z0-9_\-\s]{0,25})\s+was\s+created\s+by\s+([A-Za-z][A-Za-z0-9_\-\s]{0,25})",
        "subject": 1,
        "object": 2,
        "rel_type": RelationType.CREATED_BY,
        "confidence": 0.9,
    },
    # 英文模式: "A uses B"
    {
        "pattern": r"([A-Za-z][A-Za-z0-9_\-]{0,25})\s+uses\s+([A-Za-z][A-Za-z0-9_\-\s]{0,25})",
        "subject": 1,
        "object": 2,
        "rel_type": RelationType.USED_BY,
        "confidence": 0.8,
    },
    # 英文模式: "A depends on B"
    {
        "pattern": r"([A-Za-z][A-Za-z0-9_\-]{0,25})\s+depends\s+on\s+([A-Za-z][A-Za-z0-9_\-\s]{0,25})",
        "subject": 1,
        "object": 2,
        "rel_type": RelationType.DEPENDS_ON,
        "confidence": 0.9,
    },
    # 英文模式: "A is based on B"
    {
        "pattern": r"([A-Za-z][A-Za-z0-9_\-]{0,25})\s+is\s+based\s+on\s+([A-Za-z][A-Za-z0-9_\-\s]{0,25})",
        "subject": 1,
        "object": 2,
        "rel_type": RelationType.DERIVED_FROM,
        "confidence": 0.85,
    },
    # 英文模式: "A works for B"
    {
        "pattern": r"([A-Za-z][A-Za-z0-9_\-]{0,25})\s+works\s+for\s+([A-Za-z][A-Za-z0-9_\-\s]{0,25})",
        "subject": 1,
        "object": 2,
        "rel_type": RelationType.WORKS_FOR,
        "confidence": 0.9,
    },
    # 英文模式: "A is located in B"
    {
        "pattern": r"([A-Za-z][A-Za-z0-9_\-]{0,25})\s+is\s+located\s+(?:in|at)\s+([A-Za-z][A-Za-z0-9_\-\s]{0,25})",
        "subject": 1,
        "object": 2,
        "rel_type": RelationType.LOCATED_AT,
        "confidence": 0.9,
    },
]


# ============================================================
# 词典加载器
# ============================================================

class EntityDictLoader:
    """
    实体词典加载器

    支持从 YAML 配置文件加载实体词典，并与默认词典合并。
    """

    _instance = None
    _initialized = False

    # 运行时词典（可被配置文件覆盖/扩展）
    SQL_KEYWORDS: set = set()
    TECH_WHITELIST: set = set()
    CHINESE_TECH_TERMS: dict = {}
    CHINESE_ENTITY_SUFFIXES: dict = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def initialize(cls, config: ExtractionConfig | None = None) -> None:
        """
        初始化词典

        Args:
            config: 提取配置，包含 entity_dicts_path
        """
        if cls._initialized:
            return

        # 加载默认词典
        cls.SQL_KEYWORDS = _DEFAULT_SQL_KEYWORDS.copy()
        cls.TECH_WHITELIST = _DEFAULT_TECH_WHITELIST.copy()
        cls.CHINESE_TECH_TERMS = _DEFAULT_CHINESE_TECH_TERMS.copy()
        cls.CHINESE_ENTITY_SUFFIXES = _DEFAULT_CHINESE_ENTITY_SUFFIXES.copy()

        # 如果有配置文件，加载并合并
        if config and config.entity_dicts_path:
            cls._load_from_file(config.entity_dicts_path)

        cls._initialized = True
        logger.info(
            f"[EntityDictLoader] 词典已加载: "
            f"SQL黑名单={len(cls.SQL_KEYWORDS)}, "
            f"技术白名单={len(cls.TECH_WHITELIST)}, "
            f"中文术语={len(cls.CHINESE_TECH_TERMS)}, "
            f"中文后缀={len(cls.CHINESE_ENTITY_SUFFIXES)}"
        )

    @classmethod
    def _load_from_file(cls, filepath: str) -> None:
        """
        从 YAML 文件加载词典

        Args:
            filepath: 配置文件路径
        """
        path = Path(filepath)
        if not path.exists():
            logger.warning(f"[EntityDictLoader] 配置文件不存在: {filepath}")
            return

        try:
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            # 加载 SQL 关键字黑名单
            if "sql_keywords" in data:
                custom_keywords = data["sql_keywords"]
                if isinstance(custom_keywords, list):
                    cls.SQL_KEYWORDS.update(custom_keywords)
                    logger.debug(f"[EntityDictLoader] 加载 {len(custom_keywords)} 个 SQL 关键字")

            # 加载技术白名单
            if "tech_whitelist" in data:
                custom_whitelist = data["tech_whitelist"]
                if isinstance(custom_whitelist, list):
                    cls.TECH_WHITELIST.update(custom_whitelist)
                    logger.debug(f"[EntityDictLoader] 加载 {len(custom_whitelist)} 个技术白名单项")

            # 加载中文技术术语
            if "chinese_tech_terms" in data:
                custom_terms = data["chinese_tech_terms"]
                if isinstance(custom_terms, dict):
                    for term, entity_type_str in custom_terms.items():
                        # 将字符串类型转换为 EntityType 枚举
                        entity_type = cls._parse_entity_type(entity_type_str)
                        cls.CHINESE_TECH_TERMS[term] = entity_type
                    logger.debug(f"[EntityDictLoader] 加载 {len(custom_terms)} 个中文技术术语")

            # 加载中文实体后缀
            if "chinese_entity_suffixes" in data:
                custom_suffixes = data["chinese_entity_suffixes"]
                if isinstance(custom_suffixes, dict):
                    for suffix, entity_type_str in custom_suffixes.items():
                        entity_type = cls._parse_entity_type(entity_type_str)
                        cls.CHINESE_ENTITY_SUFFIXES[suffix] = entity_type
                    logger.debug(f"[EntityDictLoader] 加载 {len(custom_suffixes)} 个中文实体后缀")

            logger.info(f"[EntityDictLoader] 成功加载配置文件: {filepath}")

        except Exception as e:
            logger.warning(f"[EntityDictLoader] 加载配置文件失败: {e}")

    @staticmethod
    def _parse_entity_type(type_str: str) -> str:
        """
        解析实体类型字符串

        Args:
            type_str: 实体类型字符串（如 "TECHNOLOGY", "ORG"）

        Returns:
            EntityType 枚举值
        """
        type_mapping = {
            "TECHNOLOGY": EntityType.TECHNOLOGY,
            "PERSON": EntityType.PERSON,
            "ORG": EntityType.ORG,
            "LOC": EntityType.LOC,
            "LOCATION": EntityType.LOC,
            "DATE": EntityType.DATE,
            "EVENT": EntityType.EVENT,
            "CONCEPT": EntityType.CONCEPT,
            "MISC": EntityType.MISC,
        }
        return type_mapping.get(type_str.upper(), EntityType.MISC)


# 全局词典实例（延迟初始化）
_dict_loader = EntityDictLoader()


def get_sql_keywords() -> set:
    """获取 SQL 关键字黑名单"""
    if not _dict_loader._initialized:
        _dict_loader.initialize()
    return _dict_loader.SQL_KEYWORDS


def get_tech_whitelist() -> set:
    """获取技术术语白名单"""
    if not _dict_loader._initialized:
        _dict_loader.initialize()
    return _dict_loader.TECH_WHITELIST


def get_chinese_tech_terms() -> dict:
    """获取中文技术术语词典"""
    if not _dict_loader._initialized:
        _dict_loader.initialize()
    return _dict_loader.CHINESE_TECH_TERMS


def get_chinese_entity_suffixes() -> dict:
    """获取中文实体后缀模式"""
    if not _dict_loader._initialized:
        _dict_loader.initialize()
    return _dict_loader.CHINESE_ENTITY_SUFFIXES


def clean_entity_name(name: str) -> str | None:
    """
    清理实体名称，过滤无效实体

    Args:
        name: 原始实体名称

    Returns:
        清理后的实体名称，如果无效则返回 None
    """
    if not name:
        return None

    original_name = name

    # 1. 移除 Markdown 代码块标记
    name = re.sub(r"```.*?```", "", name, flags=re.DOTALL)

    # 2. 移除 Markdown 链接标记 [[...]]
    name = re.sub(r"\[\[(.*?)\]\]", r"\1", name)

    # 3. 移除首尾特殊字符
    name = re.sub(r"^[\W_]+|[\W_]+$", "", name)

    # 4. 过滤过长的随机字符串（长度>50 且全为字母数字）
    if len(name) > 50 and re.match(r"^[A-Za-z0-9]+$", name):
        logger.debug(f"[EntityCleaner] 过滤过长随机字符串：{original_name}")
        return None

    # 5. 过滤长度异常的实体（但允许版本号格式）
    if len(name) < 2:
        logger.debug(f"[EntityCleaner] 过滤长度异常实体：{original_name}")
        return None

    # 6. 过滤纯数字、纯符号的实体（但允许版本号格式）
    if re.match(r"^[\d\W]+$", name) and not re.match(r"^v?\d+\.\d+(\.\d+)*$", name):
        logger.debug(f"[EntityCleaner] 过滤纯数字/符号实体：{original_name}")
        return None

    # 7. 过滤包含过多特殊字符的实体
    special_char_count = len(re.findall(r"[\W_]", name))
    if special_char_count > len(name) * 0.5:
        logger.debug(f"[EntityCleaner] 过滤特殊字符过多实体：{original_name}")
        return None

    # 8. 【新增】过滤 SQL/编程保留字黑名单
    name_upper = name.upper()
    name_lower = name.lower()
    sql_keywords = get_sql_keywords()
    if name_upper in sql_keywords or name_lower in sql_keywords or name in sql_keywords:
        logger.debug(f"[EntityCleaner] 过滤保留字实体：{original_name}")
        return None

    # 9. 【新增】过滤表格中的纯小数数字（如 8.2, 4.5, 3.8）
    if re.match(r"^\d+\.\d+$", name) and len(name) <= 5:
        logger.debug(f"[EntityCleaner] 过滤表格数字实体：{original_name}")
        return None

    # 10. 【新增】过滤百分比数字（如 42%, 99.9%）
    if re.match(r"^\d+\.?\d*%$", name):
        logger.debug(f"[EntityCleaner] 过滤百分比实体：{original_name}")
        return None

    # 11. 【新增】过滤资源单位（如 500m, 1Gi, 512Mi）
    if re.match(r"^\d+(m|Mi|Gi|Ti|Ki|K|M|G|T)$", name):
        logger.debug(f"[EntityCleaner] 过滤资源单位实体：{original_name}")
        return None

    # 12. 【新增】对于全大写的英文短词，检查是否在技术白名单中
    # 注意：只对纯英文（不含中文）的单词进行检查
    tech_whitelist = get_tech_whitelist()
    if re.match(r"^[A-Z]+$", name) and 2 <= len(name) <= 10:
        if name not in tech_whitelist:
            logger.debug(f"[EntityCleaner] 过滤非白名单大写词：{original_name}")
            return None

    if original_name != name:
        logger.debug(f"[EntityCleaner] 清理实体名称：{original_name} -> {name}")

    return name if name.strip() else None


@dataclass
class ExtractionResult:
    """抽取结果"""

    entities: list[Entity] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    wikilinks: list[str] = field(default_factory=list)
    frontmatter: dict = field(default_factory=dict)
    processing_time_ms: float = 0.0
    source: str = "unknown"


class FrontmatterParser:
    """
    YAML Frontmatter 解析器

    提取文档级元数据，包括：
    - tags: 标签列表
    - related: 关联文档
    - author: 作者
    - project: 项目
    - status: 状态
    - aliases: 别名
    - 其他自定义字段
    """

    # YAML Frontmatter 边界正则
    FRONTMATTER_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL | re.MULTILINE)

    @classmethod
    def parse(cls, content: str) -> dict:
        """解析 Frontmatter"""
        match = cls.FRONTMATTER_PATTERN.match(content)
        if not match:
            return {}

        yaml_content = match.group(1)
        try:
            import yaml

            return yaml.safe_load(yaml_content) or {}
        except Exception as e:
            # YAML 解析失败，使用简单解析作为后备
            logger.debug(f"[FrontmatterParser] YAML parse failed, fallback to simple parse: {e}")
            return cls._simple_parse(yaml_content)

    @classmethod
    def _simple_parse(cls, yaml_content: str) -> dict:
        """简单 YAML 解析（处理常见格式）"""
        result = {}
        for line in yaml_content.split("\n"):
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()

            # 处理列表值
            if value.startswith("[") and value.endswith("]"):
                value = [v.strip() for v in value[1:-1].split(",") if v.strip()]
            elif (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
                value = value[1:-1]

            result[key] = value

        return result

    @classmethod
    def extract_frontmatter_fields(cls, frontmatter: dict) -> dict:
        """提取标准化的 Frontmatter 字段"""
        return {
            "tags": cls._ensure_list(frontmatter.get("tags", [])),
            "related": cls._ensure_list(frontmatter.get("related", [])),
            "author": frontmatter.get("author", ""),
            "project": frontmatter.get("project", ""),
            "status": frontmatter.get("status", ""),
            "aliases": cls._ensure_list(frontmatter.get("aliases", [])),
            "doc_type": frontmatter.get("doc_type", frontmatter.get("type", "")),
            "created": frontmatter.get("created", frontmatter.get("date", "")),
            "title": frontmatter.get("title", ""),
        }

    @classmethod
    def _ensure_list(cls, value) -> list:
        """确保值为列表"""
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            return [v.strip() for v in value.split(",") if v.strip()]
        return []


class WikilinkExtractor:
    """
    Obsidian Wikilink 提取器

    提取 [[Wikilink]] 格式的内部链接，支持：
    - [[链接文本]]
    - [[链接文本|显示文本]]
    - [[链接文本#标题]]
    - [[链接文本#标题|显示文本]]
    """

    # Wikilink 正则模式
    WIKILINK_PATTERN = re.compile(r"\[\[([^\]|#]+)(?:[#|]([^\]]*))?\]\]", re.MULTILINE)

    @classmethod
    def extract(cls, content: str) -> list[str]:
        """提取所有 Wikilink 目标"""
        links = []
        for match in cls.WIKILINK_PATTERN.finditer(content):
            target = match.group(1).strip()
            if target:
                links.append(target)
        return list(set(links))  # 去重

    @classmethod
    def extract_with_positions(cls, content: str) -> list[dict]:
        """提取 Wikilink 及其位置"""
        results = []
        for match in cls.WIKILINK_PATTERN.finditer(content):
            target = match.group(1).strip()
            if target:
                results.append(
                    {
                        "target": target,
                        "start": match.start(),
                        "end": match.end(),
                        "heading": match.group(2).strip() if match.group(2) else None,
                    }
                )
        return results


class NLPExtractor:
    """
    基于 spaCy 的 NLP 实体抽取器

    支持中文和英文实体识别。
    """

    def __init__(self, config: ExtractionConfig):
        self.config = config
        self._nlp = None
        self._initialized = False
        # 初始化词典加载器
        EntityDictLoader.initialize(config)

    def _ensure_nlp(self):
        """延迟加载 spaCy 模型"""
        if self._initialized:
            return

        try:
            import spacy

            model_name = self.config.spacy_model

            # 尝试加载模型
            try:
                self._nlp = spacy.load(model_name)
            except OSError:
                # 尝试下载模型
                from spacy.cli import download

                download(model_name)
                self._nlp = spacy.load(model_name)

            self._initialized = True
        except Exception as e:
            logger.warning(f"[NLPExtractor] Failed to load spaCy model: {e}")
            self._nlp = None
            self._initialized = True

    def extract_entities(self, text: str, chunk_id: str | None = None) -> list[Entity]:
        """
        从文本中抽取实体（增强版）
        
        提取策略：
        1. spaCy NER 实体识别
        2. 名词短语提取（noun_chunks）
        3. 中文实体后缀识别
        4. 基于词性的组合提取
        """
        self._ensure_nlp()

        if not self._nlp:
            return []

        entities = []
        try:
            doc = self._nlp(text[:5000])  # 限制长度防止超时
            
            # 已识别实体的位置集合（避免重复）
            seen_positions = set()

            # 1. spaCy NER 实体识别
            for ent in doc.ents:
                # 安全获取置信度，兼容不同 spaCy 版本
                confidence = 0.8  # 默认置信度
                try:
                    if hasattr(ent, "_") and hasattr(ent._, "confidence"):
                        confidence = getattr(ent._, "confidence", 0.8)
                except (TypeError, AttributeError):
                    pass

                # 清洗实体名称
                cleaned_name = clean_entity_name(ent.text)
                if cleaned_name is None:
                    continue

                entity = Entity(
                    id=self._generate_entity_id(cleaned_name, ent.label_),
                    canonical_name=cleaned_name,
                    type=self._map_entity_type(ent.label_),
                    confidence=min(1.0, confidence),
                    source="spacy",
                    chunk_id=chunk_id,
                )
                entities.append(entity)
                seen_positions.add((ent.start, ent.end))
            
            # 2. 名词短语提取（适用于英文）
            for chunk in doc.noun_chunks:
                # 跳过已识别的实体
                if (chunk.start, chunk.end) in seen_positions:
                    continue
                
                # 跳过过短或过长的短语
                if len(chunk.text) < 3 or len(chunk.text) > 50:
                    continue
                
                # 清洗名词短语
                cleaned_name = clean_entity_name(chunk.text)
                if cleaned_name is None:
                    continue
                
                # 基于中心词词性确定实体类型
                entity_type = self._infer_entity_type_from_chunk(chunk)
                
                entity = Entity(
                    id=self._generate_entity_id(cleaned_name, entity_type),
                    canonical_name=cleaned_name,
                    type=entity_type,
                    confidence=0.6,  # 名词短语置信度较低
                    source="noun_chunk",
                    chunk_id=chunk_id,
                )
                entities.append(entity)
                seen_positions.add((chunk.start, chunk.end))
            
            # 3. 中文实体后缀识别
            chinese_entity_suffixes = get_chinese_entity_suffixes()
            for token in doc:
                # 检查是否匹配中文实体后缀
                for suffix, entity_type in chinese_entity_suffixes.items():
                    if token.text.endswith(suffix) and len(token.text) > len(suffix):
                        cleaned_name = clean_entity_name(token.text)
                        if cleaned_name is None:
                            continue
                        
                        entity = Entity(
                            id=self._generate_entity_id(cleaned_name, entity_type),
                            canonical_name=cleaned_name,
                            type=entity_type,
                            confidence=0.85,
                            source="suffix_pattern",
                            chunk_id=chunk_id,
                        )
                        entities.append(entity)
                        break
            
            # 4. 基于词性的组合提取（识别技术术语组合）
            entities.extend(self._extract_tech_entities_by_pos(doc, chunk_id, seen_positions))
            
        except Exception as e:
            logger.warning(f"[NLPExtractor] Error extracting entities: {e}")
            # 返回已提取的实体，不中断流程

        return self._deduplicate_entities(entities)
    
    def _infer_entity_type_from_chunk(self, chunk) -> str:
        """
        从名词短语推断实体类型
        
        基于中心词的词性和修饰词进行推断
        """
        # 获取中心词（root）
        root = chunk.root
        
        # 基于词性推断
        if root.pos_ == "PROPN":  # 专有名词
            # 根据上下文判断是人名、组织还是地点
            if root.ent_type_:
                return self._map_entity_type(root.ent_type_)
            return EntityType.MISC
        
        if root.pos_ == "NOUN":  # 普通名词
            # 检查是否有技术术语修饰
            tech_whitelist = get_tech_whitelist()
            chinese_tech_terms = get_chinese_tech_terms()
            for child in chunk:
                if child.text in tech_whitelist or child.text in chinese_tech_terms:
                    return EntityType.TECHNOLOGY
                if child.pos_ == "ADJ":  # 形容词修饰
                    adj_text = child.text.lower()
                    # 技术相关形容词
                    if adj_text in ("neural", "deep", "machine", "learning", "artificial"):
                        return EntityType.TECHNOLOGY
        
        # 检查短语中是否包含技术关键词
        chunk_text = chunk.text.lower()
        tech_keywords = ["algorithm", "model", "system", "network", "framework", "api", "database"]
        for kw in tech_keywords:
            if kw in chunk_text:
                return EntityType.TECHNOLOGY
        
        return EntityType.CONCEPT
    
    def _extract_tech_entities_by_pos(self, doc, chunk_id: int | None, seen_positions: set) -> list[Entity]:
        """
        基于词性标注提取技术实体
        
        识别模式：
        1. 形容词 + 名词组合（如 "neural network"）
        2. 名词 + 名词组合（如 "data mining"）
        """
        entities = []
        
        for i, token in enumerate(doc):
            if (token.i, token.i + 1) in seen_positions:
                continue
            
            # 模式1: 形容词 + 名词（技术术语常见模式）
            if token.pos_ == "ADJ" and i + 1 < len(doc):
                next_token = doc[i + 1]
                if next_token.pos_ in ("NOUN", "PROPN"):
                    # 组合成技术术语
                    combined = f"{token.text} {next_token.text}"
                    cleaned_name = clean_entity_name(combined)
                    if cleaned_name:
                        entity = Entity(
                            id=self._generate_entity_id(cleaned_name, EntityType.TECHNOLOGY),
                            canonical_name=cleaned_name,
                            type=EntityType.TECHNOLOGY,
                            confidence=0.7,
                            source="pos_pattern",
                            chunk_id=chunk_id,
                        )
                        entities.append(entity)
                        seen_positions.add((token.i, next_token.i + 1))
            
            # 模式2: 名词 + 名词（技术术语常见模式）
            if token.pos_ == "NOUN" and i + 1 < len(doc):
                next_token = doc[i + 1]
                if next_token.pos_ == "NOUN":
                    # 检查是否是已知的技术术语组合
                    combined = f"{token.text} {next_token.text}"
                    combined_lower = combined.lower()
                    
                    # 常见技术术语组合
                    tech_combos = {
                        "data mining", "machine learning", "deep learning",
                        "neural network", "natural language", "computer vision",
                        "knowledge graph", "recommendation system", "search engine",
                    }
                    
                    if combined_lower in tech_combos:
                        cleaned_name = clean_entity_name(combined)
                        if cleaned_name:
                            entity = Entity(
                                id=self._generate_entity_id(cleaned_name, EntityType.TECHNOLOGY),
                                canonical_name=cleaned_name,
                                type=EntityType.TECHNOLOGY,
                                confidence=0.85,
                                source="pos_pattern",
                                chunk_id=chunk_id,
                            )
                            entities.append(entity)
                            seen_positions.add((token.i, next_token.i + 1))
        
        return entities

    def _map_entity_type(self, spacy_label: str) -> str:
        """映射 spaCy 实体类型到标准类型"""
        mapping = {
            # 英文
            "PERSON": EntityType.PERSON,
            "ORG": EntityType.ORG,
            "GPE": EntityType.LOC,
            "LOC": EntityType.LOC,
            "DATE": EntityType.DATE,
            "EVENT": EntityType.EVENT,
            "PRODUCT": EntityType.TECHNOLOGY,
            "WORK_OF_ART": EntityType.CONCEPT,
            # 中文 (ORG, LOC, DATE 与英文相同，不重复)
            "PER": EntityType.PERSON,
            "TIME": EntityType.DATE,
        }
        return mapping.get(spacy_label, EntityType.MISC)

    def _generate_entity_id(self, name: str, entity_type: str) -> str:
        """生成实体 ID"""
        normalized = f"{entity_type}:{name}".lower()
        return hashlib.md5(normalized.encode()).hexdigest()[:16]

    def _deduplicate_entities(self, entities: list[Entity]) -> list[Entity]:
        """去重实体"""
        seen = {}
        for e in entities:
            if e.id not in seen or e.confidence > seen[e.id].confidence:
                seen[e.id] = e
        return list(seen.values())


class RuleExtractor:
    """
    基于规则的实体抽取器

    使用预定义词典和规则进行实体识别。
    """

    # 默认规则词典
    # 注意：使用 (?<![a-zA-Z]) 和 (?![a-zA-Z]) 替代 \b，以支持中英文混合文本
    DEFAULT_PATTERNS: ClassVar[dict] = {
        # 技术术语：预定义技术名词（支持中英文混合）
        r"(?<![a-zA-Z])(Kubernetes|Docker|Nginx|Apache|TensorFlow|PyTorch|Keras|"
        r"Prometheus|Grafana|Jaeger|Zipkin|Kafka|RabbitMQ|Redis|MongoDB|MySQL|PostgreSQL|"
        r"Neo4j|Milvus|Elasticsearch|FastAPI|Flask|Django|Spring|Vue|React|Angular|"
        r"Python|Java|JavaScript|TypeScript|Rust|Go|Node\.js|Hadoop|Spark|Flink|"
        r"OpenAI|Anthropic|Gemini|Claude|ChatGPT)(?![a-zA-Z])": EntityType.TECHNOLOGY,
        # 技术缩写：全大写 2-10 字母
        r"(?<![a-zA-Z])([A-Z]{2,10})(?![a-zA-Z])": EntityType.TECHNOLOGY,
        # 版本号
        r"(?<![a-zA-Z0-9])(v?\d+\.\d+(?:\.\d+)?)(?![a-zA-Z0-9])": EntityType.CONCEPT,
        # 邮箱
        r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})": EntityType.PERSON,
        # URL
        r"(https?://[^\s]+)": EntityType.CONCEPT,
    }

    def __init__(self, config: ExtractionConfig):
        self.config = config
        self.patterns = self.DEFAULT_PATTERNS.copy()
        # 初始化词典加载器
        EntityDictLoader.initialize(config)
        self._load_custom_dict()

    def _load_custom_dict(self):
        """加载自定义词典"""
        if self.config.rule_dict_path:
            try:
                import json

                with open(self.config.rule_dict_path, encoding="utf-8") as f:
                    custom = json.load(f)
                    for pattern, entity_type in custom.get("patterns", {}).items():
                        self.patterns[pattern] = entity_type
            except FileNotFoundError:
                logger.warning(f"[RuleExtractor] Custom dict file not found: {self.config.rule_dict_path}")
            except Exception as e:
                logger.warning(f"[RuleExtractor] Failed to load custom dict: {e}")

    def extract_entities(self, text: str, chunk_id: str | None = None) -> list[Entity]:
        """
        从文本中抽取实体（增强版）
        
        提取策略：
        1. 正则规则匹配（技术术语、版本号、邮箱、URL等）
        2. 中文技术术语词典匹配
        """
        entities = []

        # 1. 正则规则匹配
        for pattern, entity_type in self.patterns.items():
            try:
                for match in re.finditer(pattern, text):
                    name = match.group(1) if match.lastindex else match.group(0)

                    # 清洗实体名称
                    cleaned_name = clean_entity_name(name)
                    if cleaned_name is None:
                        continue

                    entity = Entity(
                        id=hashlib.md5(f"{entity_type}:{cleaned_name}".encode()).hexdigest()[:16],
                        canonical_name=cleaned_name,
                        type=entity_type,
                        confidence=1.0,  # 规则匹配置信度高
                        source="rule",
                        chunk_id=chunk_id,
                    )
                    entities.append(entity)
            except re.error as e:
                logger.warning(f"[RuleExtractor] Invalid regex pattern '{pattern}': {e}")
                continue
            except Exception as e:
                logger.debug(f"[RuleExtractor] Pattern matching error for '{pattern}': {e}")
                continue
        
        # 2. 中文技术术语词典匹配
        entities.extend(self._extract_chinese_tech_terms(text, chunk_id))

        return self._deduplicate_entities(entities)
    
    def _extract_chinese_tech_terms(self, text: str, chunk_id: int | None) -> list[Entity]:
        """
        提取中文技术术语
        
        使用词典精确匹配，避免误识别
        """
        entities = []
        
        chinese_tech_terms = get_chinese_tech_terms()
        for term, entity_type in chinese_tech_terms.items():
            if term in text:
                # 使用正则确保完整匹配
                pattern = re.escape(term)
                for match in re.finditer(pattern, text):
                    cleaned_name = clean_entity_name(match.group())
                    if cleaned_name:
                        entity = Entity(
                            id=hashlib.md5(f"{entity_type}:{cleaned_name}".encode()).hexdigest()[:16],
                            canonical_name=cleaned_name,
                            type=entity_type,
                            confidence=1.0,
                            source="rule",
                            chunk_id=chunk_id,
                        )
                        entities.append(entity)
        
        return entities

    def _deduplicate_entities(self, entities: list[Entity]) -> list[Entity]:
        """去重实体"""
        seen = {}
        for e in entities:
            if e.id not in seen:
                seen[e.id] = e
        return list(seen.values())


class LLMExtractor:
    """
    基于 LLM 的实体/关系抽取器

    用于处理复杂文本或 NLP 失败时的后备方案。
    支持 llama-cpp-python 加载 GGUF 量化模型。
    """

    def __init__(self, config: ExtractionConfig, llm_config: LLMConfig | None = None):
        """
        初始化 LLM 抽取器

        Args:
            config: 抽取配置（包含超时等参数）
            llm_config: LLM 模型配置（包含模型路径、参数等）
        """
        self.config = config
        self.llm_config = llm_config or LLMConfig()
        self._llm = None
        self._model_loaded = False

    def _load_llm(self):
        """加载 LLM 模型"""
        if self._model_loaded:
            return

        try:
            import llama_cpp

            model_path = self._find_llm_model()
            if model_path:
                logger.info(f"[LLMExtractor] Loading model from: {model_path}")
                self._llm = llama_cpp.Llama(
                    model_path=str(model_path),
                    n_ctx=self.llm_config.n_ctx,
                    n_threads=self.llm_config.n_threads,
                    n_gpu_layers=self.llm_config.n_gpu_layers,
                    n_batch=self.llm_config.n_batch,
                    verbose=self.llm_config.verbose,
                )
                logger.info("[LLMExtractor] Model loaded successfully")
            else:
                logger.warning("[LLMExtractor] No LLM model found")
        except ImportError:
            logger.warning("[LLMExtractor] llama-cpp-python not installed, LLM extraction disabled")
        except Exception as e:
            logger.warning(f"[LLMExtractor] Failed to load LLM model: {e}")
        finally:
            self._model_loaded = True

    def extract_entities(self, text: str, chunk_id: str | None = None) -> tuple[list[Entity], list[dict]]:
        """
        使用 LLM 抽取实体和关系
        
        Args:
            text: 输入文本
            chunk_id: 关联的 Chunk ID
            
        Returns:
            (实体列表, 关系字典列表)
        """

        # 加载模型（延迟加载）
        self._load_llm()

        if not self._llm:
            return [], []

        # 使用 ChatML 格式 (Qwen/通义专用格式) + Few-shot 示例
        # 限制文本长度，避免超出上下文
        truncated_text = text[:1500] if len(text) > 1500 else text

        # 增强 prompt：同时提取实体和关系
        prompt = f"""<|im_start|>system
你是一个知识图谱构建助手。从文本中抽取实体和关系，返回JSON格式:
{{
  "entities": [{{"name": "实体名", "type": "类型"}}],
  "relations": [{{"subject": "主体", "predicate": "关系", "object": "客体"}}]
}}

实体类型: PERSON(人名), LOCATION(地点/城市), ORG(组织/公司), TECH(技术/概念), DATE(日期)
关系类型: WORKS_FOR(工作于), LOCATED_AT(位于), CREATED_BY(创建), MANAGED_BY(管理), USED_BY(使用), DEVELOPED_BY(开发), DEPENDS_ON(依赖), RELATED_TO(相关)

注意：
1. 只提取明确提到的关系，不要推断
2. 关系的 subject 和 object 必须是已提取的实体
3. 如果没有关系，relations 数组可以为空
<|im_end|>
<|im_start|>user
抽取：张明在字节跳动担任算法工程师，主要负责推荐系统的开发。
<|im_end|>
<|im_start|>assistant
{{
  "entities": [
    {{"name": "张明", "type": "PERSON"}},
    {{"name": "字节跳动", "type": "ORG"}},
    {{"name": "算法工程师", "type": "TECH"}},
    {{"name": "推荐系统", "type": "TECH"}}
  ],
  "relations": [
    {{"subject": "张明", "predicate": "WORKS_FOR", "object": "字节跳动"}},
    {{"subject": "张明", "predicate": "MANAGED_BY", "object": "推荐系统"}}
  ]
}}
<|im_end|>
<|im_start|>user
抽取：{truncated_text}
<|im_end|>
<|im_start|>assistant
"""

        try:
            import signal

            def timeout_handler(signum, frame):
                raise TimeoutError("LLM extraction timeout")

            # 仅在 Unix 系统上使用 signal
            if hasattr(signal, "SIGALRM"):
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(self.config.llm_max_latency_ms // 1000 + 1)

            try:
                response = self._llm(
                    prompt,
                    max_tokens=self.llm_config.max_tokens,
                    temperature=self.llm_config.temperature,
                    # 不使用 stop，让模型完整输出
                )
            finally:
                if hasattr(signal, "SIGALRM"):
                    signal.alarm(0)

            # 解析响应
            text_response = response["choices"][0]["text"]

            # 使用正则提取 JSON - 更健壮的解析方式
            data = self._parse_json_response(text_response)

            if data:
                entities = []
                relations = []
                
                # 解析实体
                for e in data.get("entities", []):
                    if isinstance(e, dict) and "name" in e:
                        # 清洗实体名称
                        cleaned_name = clean_entity_name(e["name"])
                        if cleaned_name is None:
                            continue

                        entities.append(
                            Entity(
                                id=hashlib.md5(f"{e.get('type', 'UNKNOWN')}:{cleaned_name}".encode()).hexdigest()[:16],
                                canonical_name=cleaned_name,
                                type=e.get("type", "UNKNOWN"),
                                confidence=0.7,
                                source="llm",
                                chunk_id=chunk_id,
                            )
                        )

                # 解析关系
                for r in data.get("relations", []):
                    if isinstance(r, dict) and all(k in r for k in ("subject", "predicate", "object")):
                        # 清洗关系中的实体名称
                        subj = clean_entity_name(r["subject"])
                        obj = clean_entity_name(r["object"])
                        
                        if subj and obj:
                            relations.append({
                                "subject": subj,
                                "subject_type": r.get("subject_type", "UNKNOWN"),
                                "predicate": r["predicate"],
                                "object": obj,
                                "object_type": r.get("object_type", "UNKNOWN"),
                                "rel_type": r["predicate"],  # 使用 predicate 作为关系类型
                                "chunk_id": chunk_id,
                                "confidence": 0.7,
                                "source": "llm",
                            })

                return entities, relations

        except TimeoutError:
            logger.warning("[LLMExtractor] Extraction timeout")
        except Exception as e:
            logger.warning(f"[LLMExtractor] Error: {e}")

        return [], []

    def _parse_json_response(self, text: str) -> dict | None:
        """解析 LLM 响应中的 JSON，使用多种策略提取"""
        import json
        import re

        # 策略1: 直接解析整个响应
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 策略2: 查找第一个完整的 JSON 对象
        # 匹配 {...} 模式，处理嵌套括号
        brace_count = 0
        start = text.find("{")
        if start >= 0:
            for i, c in enumerate(text[start:], start):
                if c == "{":
                    brace_count += 1
                elif c == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        try:
                            return json.loads(text[start : i + 1])
                        except json.JSONDecodeError:
                            break

        # 策略3: 使用正则提取 entities 数组
        entities_match = re.search(r'"entities"\s*:\s*\[(.*?)\]', text, re.DOTALL)
        if entities_match:
            try:
                # 尝试解析为数组
                entities_str = "[" + entities_match.group(1) + "]"
                entities = json.loads(entities_str)
                return {"entities": entities}
            except json.JSONDecodeError:
                pass

        # 策略4: 提取所有 name-type 对
        entities = []
        name_pattern = re.compile(r'"name"\s*:\s*"([^"]+)"')
        type_pattern = re.compile(r'"type"\s*:\s*"([^"]+)"')

        # 查找所有 {...} 块
        for match in re.finditer(r'\{[^{}]*"name"[^{}]*\}', text):
            block = match.group()
            name_match = name_pattern.search(block)
            type_match = type_pattern.search(block)
            if name_match:
                entities.append({"name": name_match.group(1), "type": type_match.group(1) if type_match else "UNKNOWN"})

        if entities:
            return {"entities": entities}

        return None

    def _find_llm_model(self):
        """查找 LLM 模型文件"""
        from pathlib import Path

        # 优先使用配置中的路径
        config_path = self.llm_config.get_model_full_path()
        if config_path.exists():
            return config_path

        # 后备候选路径
        candidates = [
            "~/.cache/llama.cpp/qwen1_5-0_5b-chat-q4_k_m.gguf",  # 新文件名格式
            "~/.cache/llama.cpp/qwen1.5-0.5b-chat-q4_k_m.gguf",  # 旧文件名格式
            "~/.cache/llama.cpp/qwen-1.8b-chat-q4_k_m.gguf",
        ]

        for pattern in candidates:
            expanded = Path(pattern).expanduser()
            if expanded.exists():
                return expanded

        # 尝试在缓存目录中搜索任何 .gguf 文件
        cache_dir = Path(self.llm_config.cache_dir).expanduser()
        if cache_dir.exists():
            gguf_files = list(cache_dir.glob("*.gguf"))
            if gguf_files:
                return gguf_files[0]

        return None


class DualLayerExtractor:
    """
    双层抽取管道

    协调文档级和 Chunk 级抽取，实现 FR-1.2/FR-1.3 需求。
    
    支持的抽取模式：
    - rule: 仅使用规则提取（最快）
    - spacy: 规则 + spaCy NER + 依存句法关系提取
    - llm_fallback: 规则 + spaCy + LLM 后备（完整流水线）
    """

    def __init__(self, config: MemoryGraphConfig):
        self.config = config
        self.extraction_config = config.extraction

        # 初始化抽取器
        # rule 模式：仅规则
        # spacy 模式：规则 + spaCy
        # llm_fallback 模式：规则 + spaCy + LLM 后备（完整流水线）
        self.nlp_extractor = (
            NLPExtractor(self.extraction_config)
            if self.extraction_config.chunk_mode in ("spacy", "llm_fallback")
            else None
        )
        self.rule_extractor = RuleExtractor(self.extraction_config)
        self.llm_extractor = (
            LLMExtractor(self.extraction_config, self.config.llm)
            if self.extraction_config.chunk_mode == "llm_fallback"
            else None
        )
        
        # 初始化依存句法关系提取器
        self.relation_extractor = (
            DependencyRelationExtractor(self.extraction_config)
            if self.extraction_config.chunk_mode in ("spacy", "llm_fallback")
            else None
        )
        
        # 初始化实体消歧器
        self.disambiguator = EntityDisambiguator()

    def _ensure_json_serializable(self, data: dict) -> dict:
        """确保数据可 JSON 序列化"""
        import datetime

        result = {}
        for k, v in data.items():
            if isinstance(v, (datetime.date, datetime.datetime)):
                result[k] = v.isoformat()
            elif isinstance(v, dict):
                result[k] = self._ensure_json_serializable(v)
            elif isinstance(v, list):
                result[k] = [
                    item.isoformat() if isinstance(item, (datetime.date, datetime.datetime)) else item for item in v
                ]
            else:
                result[k] = v
        return result

    def extract_document_level(self, content: str, note_id: str) -> ExtractionResult:
        """
        文档级抽取（FR-1.2）

        解析 YAML Frontmatter + [[Wikilink]]
        """
        start_time = time.time()

        result = ExtractionResult(source="document_level")

        # 1. 解析 Frontmatter
        frontmatter = FrontmatterParser.parse(content)
        result.frontmatter = self._ensure_json_serializable(FrontmatterParser.extract_frontmatter_fields(frontmatter))

        # 2. 提取 Wikilinks
        wikilinks = WikilinkExtractor.extract(content)
        result.wikilinks = wikilinks

        # 3. 从 Frontmatter 创建实体
        if result.frontmatter.get("author"):
            # ✅ 新增：清洗实体名称
            cleaned_name = clean_entity_name(result.frontmatter["author"])
            if cleaned_name:
                result.entities.append(
                    Entity(
                        id=hashlib.md5(f"person:{cleaned_name}".encode()).hexdigest()[:16],
                        canonical_name=cleaned_name,
                        type=EntityType.PERSON,
                        confidence=1.0,
                        source="frontmatter",
                        chunk_id=None,  # Frontmatter 实体不关联特定 Chunk
                    )
                )

        if result.frontmatter.get("project"):
            # ✅ 新增：清洗实体名称
            cleaned_name = clean_entity_name(result.frontmatter["project"])
            if cleaned_name:
                result.entities.append(
                    Entity(
                        id=hashlib.md5(f"project:{cleaned_name}".encode()).hexdigest()[:16],
                        canonical_name=cleaned_name,
                        type=EntityType.PROJECT,
                        confidence=1.0,
                        source="frontmatter",
                        chunk_id=None,  # Frontmatter 实体不关联特定 Chunk
                    )
                )

        # 4. 为标签创建实体
        for tag in result.frontmatter.get("tags", []):
            # ✅ 新增：清洗实体名称
            cleaned_name = clean_entity_name(tag)
            if cleaned_name:
                result.entities.append(
                    Entity(
                        id=hashlib.md5(f"tag:{cleaned_name}".encode()).hexdigest()[:16],
                        canonical_name=cleaned_name,
                        type=EntityType.CONCEPT,
                        confidence=1.0,
                        source="frontmatter",
                        chunk_id=None,  # Frontmatter 实体不关联特定 Chunk
                    )
                )

        result.processing_time_ms = (time.time() - start_time) * 1000
        return result

    def extract_chunk_level(self, chunk_content: str, chunk_id: int, note_id: str) -> ExtractionResult:
        """
        Chunk 级抽取（FR-1.3）

        使用 spacy + 规则词典，可选 LLM 后备。
        同时提取实体和关系。
        """
        import sys

        start_time = time.time()
        result = ExtractionResult(source="chunk_level")
        
        # 存储关系字典（后续由 graph_builder 处理）
        raw_relations = []

        # 1. 规则抽取（快速）
        rule_entities = self.rule_extractor.extract_entities(chunk_content, chunk_id)
        result.entities.extend(rule_entities)

        # 2. NLP 抽取（如果启用）
        if self.nlp_extractor:
            try:
                # 检查解释器是否正在关闭
                if hasattr(sys, "flags") and sys.flags is None:
                    # 解释器正在关闭，跳过 NLP 抽取
                    pass
                else:
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(self.nlp_extractor.extract_entities, chunk_content, chunk_id)
                        try:
                            nlp_entities = future.result(timeout=self.extraction_config.chunk_timeout_ms / 1000)
                            result.entities.extend(nlp_entities)
                        except FuturesTimeoutError:
                            pass
            except RuntimeError as e:
                # 解释器关闭时的错误，静默忽略
                if "interpreter shutdown" in str(e):
                    pass
                else:
                    logger.warning(f"[DualLayerExtractor] NLP extraction error: {e}")
            except Exception as e:
                logger.warning(f"[DualLayerExtractor] NLP extraction error: {e}")

        # 3. 去重实体
        result.entities = self._deduplicate_entities(result.entities)
        
        # 4. 实体消歧（规范化名称）
        for entity in result.entities:
            canonical = self.disambiguator.get_canonical_name(entity.canonical_name)
            if canonical != entity.canonical_name:
                entity.canonical_name = canonical

        # 5. 关系提取（如果启用 spaCy）
        if self.relation_extractor:
            try:
                dep_relations = self.relation_extractor.extract_relations(chunk_content, chunk_id)
                raw_relations.extend(dep_relations)
            except Exception as e:
                logger.debug(f"[DualLayerExtractor] Relation extraction error: {e}")

        # 6. LLM 后备（如果配置且召回不足）
        if self.llm_extractor and len(result.entities) < self.extraction_config.llm_fallback_threshold:
            try:
                llm_entities, llm_relations = self.llm_extractor.extract_entities(chunk_content, chunk_id)
                result.entities.extend(llm_entities)
                raw_relations.extend(llm_relations)
            except Exception as e:
                # LLM 后备失败不影响主流程，记录调试日志
                logger.debug(f"[DualLayerExtractor] LLM fallback extraction error: {e}")

        # 7. 最终去重
        result.entities = self._deduplicate_entities(result.entities)
        
        # 将关系存储到 frontmatter 中（作为临时存储）
        if raw_relations:
            result.frontmatter["_raw_relations"] = raw_relations

        result.processing_time_ms = (time.time() - start_time) * 1000
        return result

    def create_relations_from_wikilinks(
        self, src_chunk_id: int, wikilinks: list[str], chunk_map: dict[str, int]
    ) -> list[Relation]:
        """
        从 Wikilink 创建关系（FR-1.4 双层映射）

        Args:
            src_chunk_id: 源 Chunk ID
            wikilinks: Wikilink 目标列表
            chunk_map: {filepath/note_id: chunk_id} 映射

        Returns:
            关系列表
        """
        relations = []

        for link in wikilinks:
            # 查找目标 Chunk
            tgt_chunk_id = chunk_map.get(link) or chunk_map.get(f"{link}.md")

            if tgt_chunk_id:
                relation = Relation(
                    src_chunk_id=src_chunk_id,
                    tgt_chunk_id=tgt_chunk_id,
                    rel_type=RelationType.LINKS_TO,
                    scope="chunk",
                    weight=1.0,  # Wikilink 权重高
                    evidence_chunk_id=src_chunk_id,
                )
                relations.append(relation)

        return relations

    def _deduplicate_entities(self, entities: list[Entity]) -> list[Entity]:
        """
        去重实体，优先保留 rule 来源的正确识别
        
        去重策略：
        1. 同一名称（忽略大小写），优先保留 rule 来源（置信度高、类型正确）
        2. 如果都是 spacy，保留置信度最高的
        3. 最终再按 ID 去重（处理边界情况）
        """
        # 按名称分组
        by_name = {}
        for e in entities:
            name_key = e.canonical_name.lower()
            if name_key not in by_name:
                by_name[name_key] = []
            by_name[name_key].append(e)
        
        # 对每个名称组，选择最佳实体
        selected = []
        for name_key, group in by_name.items():
            if len(group) == 1:
                selected.append(group[0])
            else:
                # 优先选择 rule 来源（置信度高，类型正确）
                rule_entities = [e for e in group if e.source == 'rule']
                if rule_entities:
                    # 选择第一个 rule 实体（它们应该都一样）
                    selected.append(rule_entities[0])
                else:
                    # 没有 rule，选择置信度最高的
                    best = max(group, key=lambda x: x.confidence)
                    selected.append(best)
        
        # 最终按 ID 去重（处理边界情况）
        seen = {}
        for e in selected:
            if e.id not in seen:
                seen[e.id] = e
        return list(seen.values())


class DependencyRelationExtractor:
    """
    基于依存句法的关系提取器
    
    利用 spaCy 的依存解析功能，从文本中提取主谓宾三元组。
    支持中文和英文的依存关系分析。
    """
    
    # 中文动词到关系类型的映射
    VERB_TO_RELATION = {
        # 工作/职位关系
        "工作": RelationType.WORKS_FOR,
        "任职": RelationType.WORKS_FOR,
        "就职": RelationType.WORKS_FOR,
        "任职于": RelationType.WORKS_FOR,
        "服务于": RelationType.WORKS_FOR,
        
        # 创建/开发关系
        "创建": RelationType.CREATED_BY,
        "创建于": RelationType.CREATED_BY,
        "开发": RelationType.DEVELOPED_BY,
        "开发于": RelationType.DEVELOPED_BY,
        "设计": RelationType.CREATED_BY,
        "构建": RelationType.CREATED_BY,
        "实现": RelationType.CREATED_BY,
        
        # 管理/负责关系
        "管理": RelationType.MANAGED_BY,
        "负责": RelationType.MANAGED_BY,
        "领导": RelationType.MANAGED_BY,
        "主管": RelationType.MANAGED_BY,
        
        # 使用/依赖关系
        "使用": RelationType.USED_BY,
        "采用": RelationType.USED_BY,
        "依赖": RelationType.DEPENDS_ON,
        "基于": RelationType.DERIVED_FROM,
        
        # 位置关系
        "位于": RelationType.LOCATED_AT,
        "在": RelationType.LOCATED_AT,
        "坐落于": RelationType.LOCATED_AT,
        
        # 归属关系
        "属于": RelationType.BELONGS_TO,
        "归属于": RelationType.BELONGS_TO,
        "隶属于": RelationType.BELONGS_TO,
        
        # 包含关系
        "包含": RelationType.PART_OF,
        "包括": RelationType.PART_OF,
        "由...组成": RelationType.PART_OF,
        
        # 作者关系
        "著": RelationType.AUTHORED_BY,
        "编写": RelationType.AUTHORED_BY,
        "撰写": RelationType.AUTHORED_BY,
        "作者": RelationType.AUTHORED_BY,
    }
    
    # 英文动词到关系类型的映射
    VERB_TO_RELATION_EN = {
        "work": RelationType.WORKS_FOR,
        "works": RelationType.WORKS_FOR,
        "worked": RelationType.WORKS_FOR,
        "create": RelationType.CREATED_BY,
        "creates": RelationType.CREATED_BY,
        "created": RelationType.CREATED_BY,
        "develop": RelationType.DEVELOPED_BY,
        "develops": RelationType.DEVELOPED_BY,
        "developed": RelationType.DEVELOPED_BY,
        "manage": RelationType.MANAGED_BY,
        "manages": RelationType.MANAGED_BY,
        "managed": RelationType.MANAGED_BY,
        "use": RelationType.USED_BY,
        "uses": RelationType.USED_BY,
        "used": RelationType.USED_BY,
        "depend": RelationType.DEPENDS_ON,
        "depends": RelationType.DEPENDS_ON,
        "depended": RelationType.DEPENDS_ON,
        "locate": RelationType.LOCATED_AT,
        "located": RelationType.LOCATED_AT,
        "belong": RelationType.BELONGS_TO,
        "belongs": RelationType.BELONGS_TO,
        "author": RelationType.AUTHORED_BY,
        "authored": RelationType.AUTHORED_BY,
        "write": RelationType.AUTHORED_BY,
        "wrote": RelationType.AUTHORED_BY,
        "written": RelationType.AUTHORED_BY,
    }
    
    def __init__(self, config: ExtractionConfig):
        self.config = config
        self._nlp = None
        self._initialized = False
        
        # 合并动词映射
        self.verb_mapping = {**self.VERB_TO_RELATION, **self.VERB_TO_RELATION_EN}
    
    def _ensure_nlp(self):
        """延迟加载 spaCy 模型"""
        if self._initialized:
            return
        
        try:
            import spacy
            model_name = self.config.spacy_model
            
            try:
                self._nlp = spacy.load(model_name)
            except OSError:
                from spacy.cli import download
                download(model_name)
                self._nlp = spacy.load(model_name)
            
            self._initialized = True
        except Exception as e:
            logger.warning(f"[DependencyRelationExtractor] Failed to load spaCy model: {e}")
            self._nlp = None
            self._initialized = True
    
    def extract_relations(self, text: str, chunk_id: int | None = None) -> list[dict]:
        """
        从文本中提取关系三元组（增强版）
        
        提取策略：
        1. 基于正则模式的关系提取（高精度）
        2. 基于依存路径的主谓宾提取
        3. 基于实体邻近度的共现关系
        
        Args:
            text: 输入文本
            chunk_id: 关联的 Chunk ID
            
        Returns:
            关系字典列表，每个包含 subject, predicate, object, rel_type
        """
        relations = []
        
        # 1. 基于正则模式的关系提取（优先级最高）
        pattern_relations = self._extract_pattern_relations(text, chunk_id)
        relations.extend(pattern_relations)
        
        # 2. 使用 spaCy 依存解析（需要加载模型）
        self._ensure_nlp()
        
        if self._nlp:
            try:
                doc = self._nlp(text[:3000])  # 限制长度
                
                # 方法2: 基于依存路径的主谓宾提取
                for sent in doc.sents:
                    sent_relations = self._extract_svo_from_sentence(sent, chunk_id)
                    relations.extend(sent_relations)
                
                # 方法3: 基于实体邻近度的共现关系
                cooccurrence_relations = self._extract_cooccurrence_relations(doc, chunk_id)
                relations.extend(cooccurrence_relations)
                
            except Exception as e:
                logger.debug(f"[DependencyRelationExtractor] Error extracting relations: {e}")
        
        return self._deduplicate_relations(relations)
    
    def _extract_pattern_relations(self, text: str, chunk_id: int | None) -> list[dict]:
        """
        基于正则模式提取关系（高精度）
        
        使用预定义的关系模式匹配文本，提取实体对和关系类型
        """
        relations = []
        
        for pattern_config in RELATION_PATTERNS:
            pattern = pattern_config["pattern"]
            
            try:
                for match in re.finditer(pattern, text):
                    subject = match.group(pattern_config["subject"]).strip()
                    obj = match.group(pattern_config["object"]).strip()
                    
                    # 清理匹配结果中的多余字符
                    # 移除主语中的尾随动词/介词（如 "是"、"由"）
                    subject = re.sub(r'[是为由在]', '', subject).strip()
                    # 移除宾语中的首尾非字母数字字符
                    obj = re.sub(r'^[\s是为由在]+|[\s是的]+$', '', obj).strip()
                    
                    # 清洗实体名称
                    subj_cleaned = clean_entity_name(subject)
                    obj_cleaned = clean_entity_name(obj)
                    
                    if not subj_cleaned or not obj_cleaned:
                        continue
                    
                    # 跳过相同实体
                    if subj_cleaned == obj_cleaned:
                        continue
                    
                    # 跳过过长的实体（可能是匹配错误）
                    if len(subj_cleaned) > 30 or len(obj_cleaned) > 30:
                        continue
                    
                    relations.append({
                        "subject": subj_cleaned,
                        "subject_type": "UNKNOWN",
                        "predicate": pattern_config["rel_type"],
                        "object": obj_cleaned,
                        "object_type": "UNKNOWN",
                        "rel_type": pattern_config["rel_type"],
                        "chunk_id": chunk_id,
                        "confidence": pattern_config["confidence"],
                        "source": "pattern",
                    })
                    
            except re.error as e:
                logger.debug(f"[DependencyRelationExtractor] Invalid pattern: {e}")
            except Exception as e:
                logger.debug(f"[DependencyRelationExtractor] Pattern matching error: {e}")
        
        return relations
    
    def _extract_svo_from_sentence(self, sent, chunk_id: int | None) -> list[dict]:
        """
        从句子中提取主谓宾三元组
        
        基于 spaCy 依存解析：
        - nsubj/nsubjpass: 名词性主语
        - dobj: 直接宾语
        - iobj: 间接宾语
        - pobj: 介词宾语
        - ROOT: 谓语中心词
        """
        relations = []
        
        # 找到句子中的实体（已识别的命名实体）
        entities_in_sent = {ent.start: ent for ent in sent.doc.ents 
                           if ent.sent == sent}
        
        # 找到所有动词作为潜在谓语
        for token in sent:
            if token.pos_ not in ("VERB", "AUX"):
                continue
            
            # 查找主语
            subjects = self._find_subjects(token)
            if not subjects:
                continue
            
            # 查找宾语
            objects = self._find_objects(token)
            if not objects:
                continue
            
            # 获取关系类型
            rel_type = self._map_verb_to_relation(token.lemma_)
            
            # 构建三元组
            for subj in subjects:
                for obj in objects:
                    # 检查主语和宾语是否都是有效实体
                    subj_entity = self._get_entity_span(subj, entities_in_sent)
                    obj_entity = self._get_entity_span(obj, entities_in_sent)
                    
                    if subj_entity and obj_entity:
                        relations.append({
                            "subject": subj_entity,
                            "subject_type": self._get_entity_type(subj, entities_in_sent),
                            "predicate": token.lemma_,
                            "object": obj_entity,
                            "object_type": self._get_entity_type(obj, entities_in_sent),
                            "rel_type": rel_type,
                            "chunk_id": chunk_id,
                            "confidence": 0.7,
                            "source": "dependency",
                        })
        
        return relations
    
    def _find_subjects(self, verb_token) -> list:
        """查找动词的主语"""
        subjects = []
        for child in verb_token.children:
            if child.dep_ in ("nsubj", "nsubjpass", "csubj", "csubjpass"):
                # 获取完整的主语名词短语
                subject_span = self._get_noun_phrase(child)
                subjects.append(subject_span)
        return subjects
    
    def _find_objects(self, verb_token) -> list:
        """查找动词的宾语"""
        objects = []
        for child in verb_token.children:
            if child.dep_ in ("dobj", "iobj", "pobj", "attr"):
                obj_span = self._get_noun_phrase(child)
                objects.append(obj_span)
            elif child.dep_ == "prep":
                # 处理介词短语 "work for", "located at"
                for grandchild in child.children:
                    if grandchild.dep_ == "pobj":
                        obj_span = self._get_noun_phrase(grandchild)
                        objects.append(obj_span)
        return objects
    
    def _get_noun_phrase(self, token) -> str:
        """获取完整的名词短语"""
        # 包含修饰词（形容词、限定词等）
        parts = []
        for child in token.children:
            if child.dep_ in ("amod", "nummod", "det", "compound") and child.i < token.i:
                parts.append(child.text)
        parts.append(token.text)
        return "".join(parts) if parts else token.text
    
    def _get_entity_span(self, text: str, entities: dict) -> str | None:
        """获取实体文本"""
        # 简单检查：文本是否非空且有意义
        if text and len(text) >= 2:
            return text
        return None
    
    def _get_entity_type(self, text: str, entities: dict) -> str:
        """获取实体类型"""
        for start, ent in entities.items():
            if text in ent.text:
                return ent.label_
        return "UNKNOWN"
    
    def _map_verb_to_relation(self, verb: str) -> str:
        """将动词映射到关系类型"""
        verb_lower = verb.lower()
        return self.verb_mapping.get(verb_lower, RelationType.RELATED_TO)
    
    def _extract_cooccurrence_relations(self, doc, chunk_id: int | None) -> list[dict]:
        """
        基于实体邻近度提取共现关系
        
        如果两个实体在同一句子中出现，则认为它们有关联
        """
        relations = []
        
        for sent in doc.sents:
            # 获取句子中的所有实体
            sent_entities = [ent for ent in doc.ents if ent.sent == sent]
            
            if len(sent_entities) < 2:
                continue
            
            # 为每对实体创建共现关系
            for i, ent1 in enumerate(sent_entities):
                for ent2 in sent_entities[i+1:]:
                    # 跳过相同类型的实体（通常是并列关系）
                    if ent1.label_ == ent2.label_:
                        continue
                    
                    # 确保实体名称有效
                    name1 = clean_entity_name(ent1.text)
                    name2 = clean_entity_name(ent2.text)
                    
                    if not name1 or not name2:
                        continue
                    
                    relations.append({
                        "subject": name1,
                        "subject_type": ent1.label_,
                        "predicate": "RELATED_TO",
                        "object": name2,
                        "object_type": ent2.label_,
                        "rel_type": RelationType.RELATED_TO,
                        "chunk_id": chunk_id,
                        "confidence": 0.5,  # 共现关系置信度较低
                        "source": "cooccurrence",
                    })
        
        return relations
    
    def _deduplicate_relations(self, relations: list[dict]) -> list[dict]:
        """去重关系"""
        seen = set()
        unique = []
        for rel in relations:
            key = (rel["subject"], rel["rel_type"], rel["object"])
            if key not in seen:
                seen.add(key)
                unique.append(rel)
        return unique


class EntityDisambiguator:
    """
    实体消歧与别名管理
    
    实现功能：
    1. 实体别名管理（同一实体的不同称呼）
    2. 实体消歧（同名不同实体）
    3. 上下文感知消歧
    """
    
    # 预定义的别名映射
    ALIAS_MAPPING = {
        # 公司别名
        "阿里巴巴": "阿里巴巴集团",
        "阿里": "阿里巴巴集团",
        "蚂蚁金服": "蚂蚁集团",
        "字节跳动": "字节跳动",
        "字节": "字节跳动",
        "腾讯": "腾讯",
        "腾讯科技": "腾讯",
        
        # 技术别名
        "深度学习框架": "深度学习",
        "机器学习算法": "机器学习",
        "神经网络模型": "神经网络",
        
        # 人名别名
        "李开复": "李开复",
        "开复": "李开复",
    }
    
    # 实体消歧规则（基于上下文关键词）
    DISAMBIGUATION_RULES = {
        "苹果": {
            "fruit": ["水果", "吃", "甜", "种植", "果园"],
            "company": ["手机", "公司", "产品", "发布", "科技"],
        },
        "小米": {
            "grain": ["粮食", "种植", "农田", "粥"],
            "company": ["手机", "公司", "产品", "发布", "科技"],
        },
        "亚马逊": {
            "river": ["河流", "雨林", "南美"],
            "company": ["电商", "云服务", "AWS", "购物"],
        },
    }
    
    def __init__(self):
        self.alias_mapping = self.ALIAS_MAPPING.copy()
        self.disambiguation_rules = self.DISAMBIGUATION_RULES.copy()
    
    def get_canonical_name(self, mention: str) -> str:
        """
        获取实体的规范名称
        
        Args:
            mention: 实体提及
            
        Returns:
            规范化后的实体名称
        """
        return self.alias_mapping.get(mention, mention)
    
    def disambiguate(self, mention: str, context: str) -> tuple[str, str]:
        """
        基于上下文消歧实体
        
        Args:
            mention: 实体提及
            context: 上下文文本
            
        Returns:
            (消歧后的实体类型, 置信度)
        """
        if mention not in self.disambiguation_rules:
            return "UNKNOWN", 0.5
        
        rules = self.disambiguation_rules[mention]
        scores = {}
        
        for sense, keywords in rules.items():
            score = sum(1 for kw in keywords if kw in context)
            scores[sense] = score
        
        if not scores or max(scores.values()) == 0:
            return "UNKNOWN", 0.5
        
        best_sense = max(scores, key=scores.get)
        confidence = scores[best_sense] / sum(scores.values()) if sum(scores.values()) > 0 else 0.5
        
        return best_sense, confidence
    
    def add_alias(self, alias: str, canonical: str):
        """添加别名映射"""
        self.alias_mapping[alias] = canonical
    
    def get_aliases(self, canonical: str) -> list[str]:
        """获取实体的所有别名"""
        aliases = [canonical]
        for alias, canon in self.alias_mapping.items():
            if canon == canonical:
                aliases.append(alias)
        return aliases


__all__ = [
    "DependencyRelationExtractor",
    "DualLayerExtractor",
    "EntityDisambiguator",
    "ExtractionResult",
    "FrontmatterParser",
    "LLMExtractor",
    "NLPExtractor",
    "RuleExtractor",
    "WikilinkExtractor",
]
