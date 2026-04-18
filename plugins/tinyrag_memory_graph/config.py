#!/usr/bin/env python3
"""
config.py - 插件配置定义

定义 Memory Graph 插件的所有可配置项，支持 YAML 配置文件加载。
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml


@dataclass
class ExtractionConfig:
    """实体/关系抽取配置"""
    # Chunk 级抽取模式: spacy / rule / llm_fallback
    chunk_mode: Literal["spacy", "rule", "llm_fallback"] = "spacy"
    # LLM 超时熔断阈值 (ms)
    llm_max_latency_ms: int = 500
    # spacy 模型名称
    spacy_model: str = "zh_core_web_sm"
    # 规则词典路径
    rule_dict_path: str | None = None
    # 单 Chunk 抽取超时 (ms)
    chunk_timeout_ms: int = 150
    # NLP 召回低于此值时触发 LLM
    llm_fallback_threshold: int = 2


@dataclass
class RetrievalConfig:
    """混合检索配置 - 统一评分版"""
    # 向量召回数量
    vector_top_k: int = 20
    # 图扩展最大跳数
    max_hops: int = 2
    # 关系权重阈值
    min_edge_weight: float = 0.4
    # 图遍历最大节点数
    max_traverse_nodes: int = 50
    # RRF 融合参数
    alpha: float = 1.0   # 基础分数保留系数（默认完全保留）
    beta: float = 0.15   # 图谱增强权重
    gamma: float = 0.1   # 偏好加成权重
    # 上下文 Token 预算
    max_context_tokens: int = 3500


@dataclass
class MemifyConfig:
    """记忆代谢配置"""
    # 代谢触发间隔 (秒)
    interval_sec: int = 1800
    # 权重衰减：超过 N 天未访问
    decay_days: int = 30
    # 衰减系数
    decay_factor: float = 0.9
    # 陈旧标记：访问次数 < N 且年龄 > M 天
    stale_access_threshold: int = 2
    stale_age_days: int = 60
    # 路径强化增量
    path_boost: float = 0.05
    # 原则提炼：标签共现阈值
    principle_co_occurrence: int = 3
    principle_min_access: int = 5


@dataclass
class QueueConfig:
    """异步队列配置"""
    # 最大工作线程数
    max_workers: int = 2
    # 队列最大容量
    max_queue_size: int = 1000
    # 任务超时 (秒)
    task_timeout: int = 300
    # 重试次数
    max_retries: int = 3


@dataclass
class MetricsConfig:
    """监控埋点配置"""
    # 是否启用监控
    enabled: bool = True
    # 指标输出目录
    output_dir: str = "./metrics"
    # 导出格式: csv / prometheus
    export_format: Literal["csv", "prometheus"] = "csv"
    # 采样间隔 (秒)
    sample_interval: int = 60


@dataclass
class MemoryGraphConfig:
    """
    Memory Graph 插件完整配置

    支持从 YAML 字典或文件加载。
    """
    # 全局开关
    enabled: bool = True
    # 插件名称
    name: str = "memory-graph"
    # 插件版本
    version: str = "1.0.0"

    # 子配置
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    memify: MemifyConfig = field(default_factory=MemifyConfig)
    queue: QueueConfig = field(default_factory=QueueConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryGraphConfig":
        """从字典创建配置"""
        if not data:
            return cls()

        extraction_data = data.get("extraction", {})
        retrieval_data = data.get("retrieval", {})
        memify_data = data.get("memify", {})
        queue_data = data.get("queue", {})
        metrics_data = data.get("metrics", {})

        # 兼容顶层配置（简化配置格式）
        top_level_overrides = {}
        simple_keys = {
            "extract_chunk_mode": ("extraction", "chunk_mode"),
            "llm_max_latency_ms": ("extraction", "llm_max_latency_ms"),
            "max_hops": ("retrieval", "max_hops"),
            "alpha": ("retrieval", "alpha"),
            "beta": ("retrieval", "beta"),
            "gamma": ("retrieval", "gamma"),
            "memify_interval_sec": ("memify", "interval_sec"),
            "max_context_tokens": ("retrieval", "max_context_tokens"),
        }

        for key, (section, subkey) in simple_keys.items():
            if key in data:
                if section == "extraction":
                    extraction_data[subkey] = data[key]
                elif section == "retrieval":
                    retrieval_data[subkey] = data[key]
                elif section == "memify":
                    memify_data[subkey] = data[key]

        return cls(
            enabled=data.get("enabled", True),
            name=data.get("name", "memory-graph"),
            version=data.get("version", "1.0.0"),
            extraction=ExtractionConfig(
                chunk_mode=extraction_data.get("chunk_mode", "spacy"),
                llm_max_latency_ms=extraction_data.get("llm_max_latency_ms", 500),
                spacy_model=extraction_data.get("spacy_model", "zh_core_web_sm"),
                rule_dict_path=extraction_data.get("rule_dict_path"),
                chunk_timeout_ms=extraction_data.get("chunk_timeout_ms", 150),
                llm_fallback_threshold=extraction_data.get("llm_fallback_threshold", 2),
            ),
            retrieval=RetrievalConfig(
                vector_top_k=retrieval_data.get("vector_top_k", 20),
                max_hops=retrieval_data.get("max_hops", 2),
                min_edge_weight=retrieval_data.get("min_edge_weight", 0.4),
                max_traverse_nodes=retrieval_data.get("max_traverse_nodes", 50),
                alpha=retrieval_data.get("alpha", 1.0),   # 基础分数保留系数
                beta=retrieval_data.get("beta", 0.15),    # 图谱增强权重
                gamma=retrieval_data.get("gamma", 0.1),   # 偏好加成权重
                max_context_tokens=retrieval_data.get("max_context_tokens", 3500),
            ),
            memify=MemifyConfig(
                interval_sec=memify_data.get("interval_sec", 1800),
                decay_days=memify_data.get("decay_days", 30),
                decay_factor=memify_data.get("decay_factor", 0.9),
                stale_access_threshold=memify_data.get("stale_access_threshold", 2),
                stale_age_days=memify_data.get("stale_age_days", 60),
                path_boost=memify_data.get("path_boost", 0.05),
                principle_co_occurrence=memify_data.get("principle_co_occurrence", 3),
                principle_min_access=memify_data.get("principle_min_access", 5),
            ),
            queue=QueueConfig(
                max_workers=queue_data.get("max_workers", 2),
                max_queue_size=queue_data.get("max_queue_size", 1000),
                task_timeout=queue_data.get("task_timeout", 300),
                max_retries=queue_data.get("max_retries", 3),
            ),
            metrics=MetricsConfig(
                enabled=metrics_data.get("enabled", True),
                output_dir=metrics_data.get("output_dir", "./metrics"),
                export_format=metrics_data.get("export_format", "csv"),
                sample_interval=metrics_data.get("sample_interval", 60),
            ),
        )

    @classmethod
    def from_yaml(cls, path: str) -> "MemoryGraphConfig":
        """从 YAML 文件加载配置"""
        p = Path(path)
        if not p.exists():
            return cls()

        with open(p, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        # 提取 plugins.memory-graph 配置
        if "plugins" in data and "memory-graph" in data["plugins"]:
            plugin_config = data["plugins"]["memory-graph"]
        else:
            plugin_config = data

        return cls.from_dict(plugin_config)

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "enabled": self.enabled,
            "name": self.name,
            "version": self.version,
            "extraction": {
                "chunk_mode": self.extraction.chunk_mode,
                "llm_max_latency_ms": self.extraction.llm_max_latency_ms,
                "spacy_model": self.extraction.spacy_model,
                "rule_dict_path": self.extraction.rule_dict_path,
                "chunk_timeout_ms": self.extraction.chunk_timeout_ms,
                "llm_fallback_threshold": self.extraction.llm_fallback_threshold,
            },
            "retrieval": {
                "vector_top_k": self.retrieval.vector_top_k,
                "max_hops": self.retrieval.max_hops,
                "min_edge_weight": self.retrieval.min_edge_weight,
                "max_traverse_nodes": self.retrieval.max_traverse_nodes,
                "alpha": self.retrieval.alpha,
                "beta": self.retrieval.beta,
                "gamma": self.retrieval.gamma,
                "max_context_tokens": self.retrieval.max_context_tokens,
            },
            "memify": {
                "interval_sec": self.memify.interval_sec,
                "decay_days": self.memify.decay_days,
                "decay_factor": self.memify.decay_factor,
                "stale_access_threshold": self.memify.stale_access_threshold,
                "stale_age_days": self.memify.stale_age_days,
                "path_boost": self.memify.path_boost,
                "principle_co_occurrence": self.memify.principle_co_occurrence,
                "principle_min_access": self.memify.principle_min_access,
            },
            "queue": {
                "max_workers": self.queue.max_workers,
                "max_queue_size": self.queue.max_queue_size,
                "task_timeout": self.queue.task_timeout,
                "max_retries": self.queue.max_retries,
            },
            "metrics": {
                "enabled": self.metrics.enabled,
                "output_dir": self.metrics.output_dir,
                "export_format": self.metrics.export_format,
                "sample_interval": self.metrics.sample_interval,
            },
        }


__all__ = [
    "ExtractionConfig",
    "MemifyConfig",
    "MemoryGraphConfig",
    "MetricsConfig",
    "QueueConfig",
    "RetrievalConfig",
]
