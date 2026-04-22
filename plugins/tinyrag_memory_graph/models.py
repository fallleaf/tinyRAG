#!/usr/bin/env python3
"""
models.py - 数据模型定义

定义 Memory Graph 插件所需的数据表结构和 ORM 模型。
"""

import json
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any


def _json_serialize(obj: Any) -> str:
    """JSON 序列化辅助函数，处理 date/datetime。

    Args:
        obj: 需要序列化的对象

    Returns:
        ISO 格式的日期时间字符串

    Raises:
        TypeError: 如果对象类型不支持序列化
    """
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _ensure_json_serializable(data: Any) -> Any:
    """确保数据可 JSON 序列化。

    Args:
        data: 需要检查的数据（支持 dict, list, datetime, date）

    Returns:
        可 JSON 序列化的数据副本
    """
    if isinstance(data, dict):
        return {k: _ensure_json_serializable(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_ensure_json_serializable(item) for item in data]
    elif isinstance(data, (datetime, date)):
        return data.isoformat()
    return data


@dataclass
class Note:
    """
    文档级元数据模型

    对应 notes 表，存储文档的 Frontmatter 和基本信息。
    """

    note_id: str  # 文档唯一标识 (filepath hash)
    filepath: str  # 文件路径
    title: str = ""  # 文档标题
    frontmatter_json: dict = field(default_factory=dict)  # Frontmatter 元数据
    created_at: int = field(default_factory=lambda: int(time.time()))

    def to_db_row(self) -> tuple:
        """转换为数据库行"""
        # 确保 frontmatter_json 可序列化
        safe_frontmatter = _ensure_json_serializable(self.frontmatter_json)
        return (
            self.note_id,
            self.filepath,
            self.title,
            json.dumps(safe_frontmatter, ensure_ascii=False, default=_json_serialize),
            self.created_at,
        )

    @classmethod
    def from_db_row(cls, row: dict) -> "Note":
        """从数据库行创建"""
        fm_json = row.get("frontmatter_json", "{}")
        if isinstance(fm_json, str):
            fm_json = json.loads(fm_json) if fm_json else {}
        return cls(
            note_id=row["note_id"],
            filepath=row["filepath"],
            title=row.get("title", ""),
            frontmatter_json=fm_json,
            created_at=row.get("created_at", int(time.time())),
        )

    def get_tags(self) -> list[str]:
        """获取标签列表"""
        tags = self.frontmatter_json.get("tags", [])
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",") if t.strip()]
        return tags

    def get_related(self) -> list[str]:
        """获取关联文档列表"""
        related = self.frontmatter_json.get("related", [])
        if isinstance(related, str):
            related = [r.strip() for r in related.split(",") if r.strip()]
        return related

    def get_aliases(self) -> list[str]:
        """获取别名列表"""
        aliases = self.frontmatter_json.get("aliases", [])
        if isinstance(aliases, str):
            aliases = [a.strip() for a in aliases.split(",") if a.strip()]
        return aliases


@dataclass
class Entity:
    """
    实体模型

    对应 entities 表，存储从文档中抽取的实体。
    """

    id: str  # 实体唯一标识
    canonical_name: str  # 规范化名称
    type: str = "UNKNOWN"  # 实体类型 (PERSON, ORG, LOC, MISC, CONCEPT)
    confidence: float = 1.0  # 置信度
    source: str = "nlp"  # 来源 (nlp / llm / rule)
    chunk_id: str | None = None  # 所属 Chunk ID（用于图谱遍历）

    def to_db_row(self) -> tuple:
        """转换为数据库行"""
        return (
            self.id,
            self.canonical_name,
            self.type,
            self.confidence,
            self.source,
            self.chunk_id,
        )

    @classmethod
    def from_db_row(cls, row: dict) -> "Entity":
        """从数据库行创建"""
        return cls(
            id=row["id"],
            canonical_name=row["canonical_name"],
            type=row.get("type", "UNKNOWN"),
            confidence=row.get("confidence", 1.0),
            source=row.get("source", "nlp"),
            chunk_id=row.get("chunk_id"),
        )


@dataclass
class Relation:
    """
    关系模型

    对应 relations 表，存储实体间的关系边。
    """

    src_chunk_id: int  # 源 Chunk ID
    tgt_chunk_id: int  # 目标 Chunk ID
    rel_type: str  # 关系类型 (LINKS_TO, MENTIONS, RELATED_TO, etc.)
    scope: str = "chunk"  # 作用域 (doc / chunk)
    weight: float = 0.8  # 关系权重
    evidence_chunk_id: int | None = None  # 证据 Chunk ID
    last_hit: int | None = None  # 最后访问时间戳
    access_count: int = 0  # 访问次数

    def to_db_row(self) -> tuple:
        """转换为数据库行（用于 INSERT）"""
        return (
            self.src_chunk_id,
            self.tgt_chunk_id,
            self.rel_type,
            self.scope,
            self.weight,
            self.evidence_chunk_id,
            self.last_hit,
        )

    @classmethod
    def from_db_row(cls, row: dict) -> "Relation":
        """从数据库行创建"""
        return cls(
            src_chunk_id=row["src_chunk_id"],
            tgt_chunk_id=row["tgt_chunk_id"],
            rel_type=row["rel_type"],
            scope=row.get("scope", "chunk"),
            weight=row.get("weight", 0.8),
            evidence_chunk_id=row.get("evidence_chunk_id"),
            last_hit=row.get("last_hit"),
            access_count=row.get("access_count", 0),
        )

    def touch(self):
        """更新访问时间和计数"""
        self.last_hit = int(time.time())
        self.access_count += 1

    def boost(self, amount: float = 0.05, max_weight: float = 1.0):
        """强化权重"""
        self.weight = min(self.weight + amount, max_weight)

    def decay(self, factor: float = 0.9, min_weight: float = 0.1):
        """衰减权重"""
        self.weight = max(self.weight * factor, min_weight)


@dataclass
class GraphBuildJob:
    """
    异步建图任务模型

    对应 graph_build_jobs 表，追踪建图任务状态。
    """

    job_id: str  # 任务唯一标识
    note_id: str  # 文档 ID
    chunk_ids: list[int] = field(default_factory=list)  # Chunk ID 列表
    status: str = "pending"  # pending / processing / done / failed
    created_at: int = field(default_factory=lambda: int(time.time()))
    started_at: int | None = None
    finished_at: int | None = None
    error_msg: str | None = None

    def to_db_row(self) -> tuple:
        """转换为数据库行"""
        return (
            self.job_id,
            self.note_id,
            json.dumps(self.chunk_ids),
            self.status,
            self.created_at,
            self.error_msg,
        )

    @classmethod
    def from_db_row(cls, row: dict) -> "GraphBuildJob":
        """从数据库行创建"""
        chunk_ids_json = row.get("chunk_ids", "[]")
        if isinstance(chunk_ids_json, str):
            chunk_ids = json.loads(chunk_ids_json) if chunk_ids_json else []
        else:
            chunk_ids = chunk_ids_json
        return cls(
            job_id=row["job_id"],
            note_id=row["note_id"],
            chunk_ids=chunk_ids,
            status=row.get("status", "pending"),
            created_at=row.get("created_at", int(time.time())),
            error_msg=row.get("error_msg"),
        )

    def start(self):
        """标记任务开始"""
        self.status = "processing"
        self.started_at = int(time.time())

    def complete(self):
        """标记任务完成"""
        self.status = "done"
        self.finished_at = int(time.time())

    def fail(self, error: str):
        """标记任务失败"""
        self.status = "failed"
        self.finished_at = int(time.time())
        self.error_msg = error


@dataclass
class ChunkMeta:
    """
    Chunk 扩展元数据

    用于扩展 chunks 表的字段（通过 ALTER TABLE 添加）。
    """

    chunk_id: int
    note_id: str | None = None
    inherited_meta: dict = field(default_factory=dict)
    is_representative: bool = False

    def to_db_update(self) -> tuple:
        """转换为数据库更新参数"""
        # 确保 inherited_meta 可序列化
        safe_meta = _ensure_json_serializable(self.inherited_meta)
        return (
            self.note_id,
            json.dumps(safe_meta, ensure_ascii=False, default=_json_serialize),
            1 if self.is_representative else 0,
            self.chunk_id,
        )


# 关系类型常量
class RelationType:
    """预定义关系类型"""

    LINKS_TO = "LINKS_TO"  # [[Wikilink]] 链接
    MENTIONS = "MENTIONS"  # 实体提及
    RELATED_TO = "RELATED_TO"  # 相关关系
    SAME_AS = "SAME_AS"  # 同义关系
    PART_OF = "PART_OF"  # 部分关系
    DERIVED_FROM = "DERIVED_FROM"  # 派生关系
    TAGGED_WITH = "TAGGED_WITH"  # 标签关联
    AUTHORED_BY = "AUTHORED_BY"  # 作者关联
    BELONGS_TO = "BELONGS_TO"  # 归属关系
    DEVELOPED_BY = "DEVELOPED_BY"  # 开发关系
    CREATED_BY = "CREATED_BY"  # 创建关系
    MANAGED_BY = "MANAGED_BY"  # 管理关系
    WORKS_FOR = "WORKS_FOR"  # 工作关系
    LOCATED_AT = "LOCATED_AT"  # 位置关系
    DEPENDS_ON = "DEPENDS_ON"  # 依赖关系
    USED_BY = "USED_BY"  # 使用关系


# 实体类型常量
class EntityType:
    """预定义实体类型"""

    PERSON = "PERSON"  # 人物
    ORG = "ORG"  # 组织
    LOC = "LOC"  # 地点
    MISC = "MISC"  # 杂项
    CONCEPT = "CONCEPT"  # 概念
    PROJECT = "PROJECT"  # 项目
    TECHNOLOGY = "TECHNOLOGY"  # 技术
    DATE = "DATE"  # 日期
    EVENT = "EVENT"  # 事件


__all__ = [
    "ChunkMeta",
    "Entity",
    "EntityType",
    "GraphBuildJob",
    "Note",
    "Relation",
    "RelationType",
]
