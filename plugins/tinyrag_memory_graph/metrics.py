#!/usr/bin/env python3
"""
metrics.py - 监控埋点模块

实现 FR-4.3 监控埋点需求，支持：
- 检索延迟追踪
- 图遍历统计
- 队列深度监控
- LLM 抽取失败计数
- CSV / Prometheus 导出
"""
import csv
import json
import threading
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class MetricRecord:
    """单条指标记录"""
    timestamp: float
    metric_name: str
    value: float
    labels: dict = field(default_factory=dict)


class MetricsCollector:
    """
    指标收集器

    线程安全的指标收集和导出。
    """

    def __init__(self, output_dir: str = "./metrics",
                 export_format: str = "csv",
                 sample_interval: int = 60):
        self.output_dir = Path(output_dir)
        self.export_format = export_format
        self.sample_interval = sample_interval

        # 指标存储
        self._counters: dict[str, float] = defaultdict(float)
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, list[float]] = defaultdict(list)

        # 记录缓冲
        self._records: list[MetricRecord] = []

        # 线程锁
        self._lock = threading.Lock()

        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ==================== 指标记录 ====================

    def counter(self, name: str, value: float = 1.0, labels: dict | None = None):
        """计数器指标"""
        with self._lock:
            key = self._make_key(name, labels)
            self._counters[key] += value
            self._records.append(MetricRecord(
                timestamp=time.time(),
                metric_name=name,
                value=value,
                labels=labels or {},
            ))

    def gauge(self, name: str, value: float, labels: dict | None = None):
        """仪表盘指标"""
        with self._lock:
            key = self._make_key(name, labels)
            self._gauges[key] = value
            self._records.append(MetricRecord(
                timestamp=time.time(),
                metric_name=name,
                value=value,
                labels=labels or {},
            ))

    def histogram(self, name: str, value: float, labels: dict | None = None):
        """直方图指标"""
        with self._lock:
            key = self._make_key(name, labels)
            self._histograms[key].append(value)
            self._records.append(MetricRecord(
                timestamp=time.time(),
                metric_name=name,
                value=value,
                labels=labels or {},
            ))

    def _make_key(self, name: str, labels: dict | None = None) -> str:
        """生成指标键"""
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    # ==================== 预定义指标 ====================

    def record_retrieval_latency(self, latency_ms: float, mode: str = "hybrid"):
        """记录检索延迟"""
        self.histogram("retrieval_latency_ms", latency_ms, {"mode": mode})

    def record_graph_traverse_nodes(self, nodes: int, hops: int):
        """记录图遍历节点数"""
        self.histogram("graph_traverse_nodes", nodes, {"max_hops": str(hops)})

    def record_queue_depth(self, pending: int, processing: int):
        """记录队列深度"""
        self.gauge("queue_depth_pending", pending)
        self.gauge("queue_depth_processing", processing)

    def record_llm_extract_failure(self, error_type: str):
        """记录 LLM 抽取失败"""
        self.counter("llm_extract_fail_count", 1.0, {"error_type": error_type})

    def record_entity_count(self, count: int, source: str):
        """记录实体数量"""
        self.counter("entities_extracted", count, {"source": source})

    def record_relation_count(self, count: int, rel_type: str):
        """记录关系数量"""
        self.counter("relations_created", count, {"type": rel_type})

    def record_memify_stats(self, decayed: int, boosted: int, principles: int):
        """记录代谢统计"""
        self.counter("relations_decayed", decayed)
        self.counter("relations_boosted", boosted)
        self.counter("principles_generated", principles)

    # ==================== 导出 ====================

    def export_csv(self, filename: str | None = None) -> str:
        """导出为 CSV"""
        if not filename:
            filename = f"metrics_{int(time.time())}.csv"

        filepath = self.output_dir / filename

        with self._lock, open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "metric_name", "value", "labels"])

            for record in self._records:
                writer.writerow([
                    record.timestamp,
                    record.metric_name,
                    record.value,
                    json.dumps(record.labels),
                ])

        return str(filepath)

    def export_prometheus(self) -> str:
        """导出为 Prometheus 格式"""
        lines = []

        with self._lock:
            # 计数器
            lines.append("# TYPE memory_graph_counter counter")
            for key, value in self._counters.items():
                lines.append(f"memory_graph_counter_{key} {value}")

            # 仪表盘
            lines.append("\n# TYPE memory_graph_gauge gauge")
            for key, value in self._gauges.items():
                lines.append(f"memory_graph_gauge_{key} {value}")

            # 直方图摘要
            lines.append("\n# TYPE memory_graph_histogram summary")
            for key, values in self._histograms.items():
                if values:
                    sorted_values = sorted(values)
                    count = len(sorted_values)
                    total = sum(sorted_values)

                    lines.append(f'memory_graph_histogram_{key}_count {count}')
                    lines.append(f'memory_graph_histogram_{key}_sum {total}')

                    # 分位数
                    for q in [0.5, 0.9, 0.95, 0.99]:
                        idx = int(count * q)
                        if idx < count:
                            lines.append(f'memory_graph_histogram_{key}{{quantile="{q}"}} {sorted_values[idx]}')

        return "\n".join(lines)

    def export_json(self) -> str:
        """导出为 JSON"""
        with self._lock:
            data = {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {k: {"values": v, "count": len(v), "sum": sum(v)}
                               for k, v in self._histograms.items()},
                "records": [asdict(r) for r in self._records[-1000:]],  # 最近 1000 条
            }
        return json.dumps(data, indent=2)

    # ==================== 查询 ====================

    def get_summary(self) -> dict:
        """获取指标摘要"""
        with self._lock:
            summary = {
                "total_records": len(self._records),
                "counters": len(self._counters),
                "gauges": len(self._gauges),
                "histograms": len(self._histograms),
            }

            # 直方图统计
            for key, values in self._histograms.items():
                if values:
                    sorted_values = sorted(values)
                    count = len(sorted_values)
                    summary[f"histogram_{key}"] = {
                        "count": count,
                        "min": sorted_values[0],
                        "max": sorted_values[-1],
                        "mean": sum(sorted_values) / count,
                        "p50": sorted_values[int(count * 0.5)],
                        "p95": sorted_values[int(count * 0.95)],
                        "p99": sorted_values[int(count * 0.99)] if count > 1 else sorted_values[0],
                    }

            return summary

    def clear(self):
        """清空指标"""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._records.clear()


# 全局指标收集器实例
_global_collector: MetricsCollector | None = None


def get_metrics_collector() -> MetricsCollector:
    """获取全局指标收集器"""
    global _global_collector
    if _global_collector is None:
        _global_collector = MetricsCollector()
    return _global_collector


def init_metrics(output_dir: str = "./metrics",
                 export_format: str = "csv",
                 sample_interval: int = 60) -> MetricsCollector:
    """初始化全局指标收集器"""
    global _global_collector
    _global_collector = MetricsCollector(
        output_dir=output_dir,
        export_format=export_format,
        sample_interval=sample_interval,
    )
    return _global_collector


__all__ = [
    "MetricRecord",
    "MetricsCollector",
    "get_metrics_collector",
    "init_metrics",
]
