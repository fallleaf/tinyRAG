#!/usr/bin/env python3
"""
exceptions.py - tinyRAG 自定义异常类

提供项目特定的异常类，便于区分不同类型的错误。
"""

from typing import Optional, Any


class TinyRAGError(Exception):
    """tinyRAG 基础异常类"""
    
    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | details: {self.details}"
        return self.message


# ==================== 配置相关异常 ====================

class ConfigError(TinyRAGError):
    """配置错误"""
    pass


class ConfigValidationError(ConfigError):
    """配置验证错误"""
    pass


# ==================== 存储相关异常 ====================

class StorageError(TinyRAGError):
    """存储层错误"""
    pass


class DatabaseError(StorageError):
    """数据库操作错误"""
    pass


class DatabaseConnectionError(DatabaseError):
    """数据库连接错误"""
    pass


class VectorStoreError(StorageError):
    """向量存储错误"""
    pass


# ==================== 索引相关异常 ====================

class IndexError(TinyRAGError):
    """索引构建错误"""
    pass


class ChunkingError(IndexError):
    """文档分块错误"""
    pass


class EmbeddingError(IndexError):
    """向量化错误"""
    pass


# ==================== 检索相关异常 ====================

class RetrievalError(TinyRAGError):
    """检索错误"""
    pass


class VectorRecallError(RetrievalError):
    """向量召回错误"""
    pass


class GraphTraversalError(RetrievalError):
    """图遍历错误"""
    pass


# ==================== 插件相关异常 ====================

class PluginError(TinyRAGError):
    """插件错误"""
    pass


class PluginLoadError(PluginError):
    """插件加载错误"""
    pass


class PluginInitError(PluginError):
    """插件初始化错误"""
    pass


class HookExecutionError(PluginError):
    """钩子执行错误"""
    
    def __init__(
        self, 
        message: str, 
        plugin_name: Optional[str] = None,
        hook_name: Optional[str] = None,
        details: Optional[dict[str, Any]] = None
    ):
        super().__init__(message, details)
        self.plugin_name = plugin_name
        self.hook_name = hook_name
    
    def __str__(self) -> str:
        prefix = f"[{self.plugin_name}.{self.hook_name}] " if self.plugin_name and self.hook_name else ""
        if self.details:
            return f"{prefix}{self.message} | details: {self.details}"
        return f"{prefix}{self.message}"


# ==================== 提取相关异常 ====================

class ExtractionError(TinyRAGError):
    """实体/关系提取错误"""
    pass


class NLPExtractionError(ExtractionError):
    """NLP 提取错误"""
    pass


class LLMAExtractionError(ExtractionError):
    """LLM 提取错误"""
    pass


class FrontmatterParseError(ExtractionError):
    """Frontmatter 解析错误"""
    pass


# ==================== 文件操作异常 ====================

class FileOperationError(TinyRAGError):
    """文件操作错误"""
    pass


class FileNotFoundError(FileOperationError):
    """文件未找到错误"""
    pass


class FileReadError(FileOperationError):
    """文件读取错误"""
    pass


# ==================== 可恢复异常（用于重试） ====================

class RecoverableError(TinyRAGError):
    """可恢复错误（可重试）"""
    
    def __init__(
        self, 
        message: str, 
        retry_count: int = 0,
        max_retries: int = 3,
        details: Optional[dict[str, Any]] = None
    ):
        super().__init__(message, details)
        self.retry_count = retry_count
        self.max_retries = max_retries
    
    def should_retry(self) -> bool:
        """是否应该重试"""
        return self.retry_count < self.max_retries
    
    def with_retry(self) -> "RecoverableError":
        """返回增加重试次数的新异常"""
        return RecoverableError(
            self.message,
            retry_count=self.retry_count + 1,
            max_retries=self.max_retries,
            details=self.details
        )


class TransientError(RecoverableError):
    """临时性错误（网络抖动、资源暂时不可用等）"""
    pass


__all__ = [
    # 基础
    "TinyRAGError",
    "RecoverableError",
    "TransientError",
    # 配置
    "ConfigError",
    "ConfigValidationError",
    # 存储
    "StorageError",
    "DatabaseError",
    "DatabaseConnectionError",
    "VectorStoreError",
    # 索引
    "IndexError",
    "ChunkingError",
    "EmbeddingError",
    # 检索
    "RetrievalError",
    "VectorRecallError",
    "GraphTraversalError",
    # 插件
    "PluginError",
    "PluginLoadError",
    "PluginInitError",
    "HookExecutionError",
    # 提取
    "ExtractionError",
    "NLPExtractionError",
    "LLMAExtractionError",
    "FrontmatterParseError",
    # 文件
    "FileOperationError",
    "FileNotFoundError",
    "FileReadError",
]
