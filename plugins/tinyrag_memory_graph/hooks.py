#!/usr/bin/env python3
"""
hooks.py - 钩子协议与上下文定义

定义插件与 tinyRAG 核心之间的通信协议，实现零侵入式集成。
"""

import sqlite3
from dataclasses import dataclass, field
from enum import Enum


class HookType(Enum):
    """钩子类型枚举"""

    ON_ADD_DOCUMENT = "on_add_document"  # 文档入库后触发
    ON_SEARCH_AFTER = "on_search_after"  # 检索融合后触发
    ON_RESPONSE = "on_response"  # LLM 输出后触发
    ON_DELETE_DOCUMENT = "on_delete_document"  # 文档删除后触发
    ON_REBUILD_INDEX = "on_rebuild_index"  # 重建索引时触发


@dataclass
class HookContext:
    """
    钩子上下文 - 插件与核心之间的数据传递载体

    Attributes:
        query: 用户查询文本
        query_vec: 查询向量
        results: 检索结果列表（可被插件修改）
        document: 文档内容
        metadata: 文档元数据
        db_conn: 数据库连接
        skip: 是否跳过后续处理
        note_id: 文档 ID
        chunk_ids: Chunk ID 列表
        frontmatter: YAML Frontmatter 元数据
        chunks_content: Chunk 内容列表
        response: LLM 响应内容
        base_alpha: 基础检索的语义权重
        base_beta: 基础检索的关键词权重
    """

    # 检索相关
    query: str | None = None
    query_vec: list[float] | None = None
    results: list[dict] = field(default_factory=list)

    # 文档相关
    document: str | None = None
    metadata: dict = field(default_factory=dict)
    note_id: str | None = None
    chunk_ids: list[int] = field(default_factory=list)
    frontmatter: dict = field(default_factory=dict)
    chunks_content: list[str] = field(default_factory=list)

    # 响应相关
    response: str | None = None

    # 系统资源
    db_conn: sqlite3.Connection | None = None

    # 控制标志
    skip: bool = False

    # 扩展数据
    extra: dict = field(default_factory=dict)

    # 基础检索权重参数（修复问题1：插件感知基础检索权重）
    base_alpha: float | None = None
    base_beta: float | None = None

    def get_result_chunk_ids(self) -> list[int]:
        """从结果中提取 chunk_id 列表"""
        return [r.get("chunk_id") or r.get("id") for r in self.results if r.get("chunk_id") or r.get("id")]

    def to_dict(self) -> dict:
        """转换为字典（用于序列化）"""
        return {
            "query": self.query,
            "results_count": len(self.results),
            "note_id": self.note_id,
            "chunk_ids": self.chunk_ids,
            "skip": self.skip,
        }


@dataclass
class HookResult:
    """
    钩子执行结果

    Attributes:
        success: 是否成功
        modified: 是否修改了上下文
        message: 结果消息
        metrics: 性能指标
    """

    success: bool = True
    modified: bool = False
    message: str = ""
    metrics: dict = field(default_factory=dict)

    @classmethod
    def ok(cls, message: str = "", modified: bool = False, metrics: dict | None = None) -> "HookResult":
        """创建成功结果"""
        return cls(success=True, modified=modified, message=message, metrics=metrics or {})

    @classmethod
    def fail(cls, message: str, metrics: dict | None = None) -> "HookResult":
        """创建失败结果"""
        return cls(success=False, modified=False, message=message, metrics=metrics or {})


class HookProtocol:
    """
    钩子协议基类

    所有插件钩子必须实现此协议。
    """

    @property
    def hook_type(self) -> HookType:
        """返回钩子类型"""
        raise NotImplementedError

    @property
    def priority(self) -> int:
        """返回执行优先级（数字越小越先执行）"""
        return 100

    async def execute(self, ctx: HookContext) -> HookResult:
        """执行钩子逻辑"""
        raise NotImplementedError


__all__ = ["HookContext", "HookProtocol", "HookResult", "HookType"]
