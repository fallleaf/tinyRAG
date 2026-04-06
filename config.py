#!/usr/bin/env python3
"""
config.py - tinyRAG 配置契约与加载层
v2.0:
1. 置信度改为 Frontmatter 驱动: doc_type_rules / status_rules / date_decay
2. 移除 path_rules (文件路径) 和 type_rules (内容块类型)
3. Pydantic v2 强类型校验
"""

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class VaultConfig(BaseModel):
    """仓库配置单元"""

    path: str
    name: str
    enabled: bool = True

    @field_validator("path")
    @classmethod
    def expand_vault_path(cls, v: str) -> str:
        return str(Path(v).expanduser())


class DateDecayConfig(BaseModel):
    """日期衰减配置"""

    enabled: bool = True
    half_life_days: int = 365  # 半衰期(天): 文档超过此天数后权重降为 default * 0.5
    min_weight: float = 0.5   # 衰减后的权重下限


class ConfidenceConfig(BaseModel):
    """
    置信度与融合权重配置 (v2.0 - Frontmatter 驱动)
    
    权重计算公式:
        final_weight = doc_type_weight × status_weight × date_weight
    
    所有未匹配的字段使用 default_weight
    """

    # Frontmatter doc_type 权重映射
    doc_type_rules: dict[str, float] = Field(
        default_factory=lambda: {
            "technical": 1.2,
            "project": 1.15,
            "meeting": 1.1,
            "faq": 1.05,
            "blog": 1.0,
            "reflection": 0.95,
        }
    )
    # Frontmatter status 权重映射
    status_rules: dict[str, float] = Field(
        default_factory=lambda: {
            "进行中": 1.2,
            "已完成": 1.0,
            "待开始": 0.9,
            "已归档": 0.8,
        }
    )
    # 日期衰减配置
    date_decay: DateDecayConfig = Field(default_factory=DateDecayConfig)
    # 未匹配时的默认权重
    default_weight: float = 1.0
    # RRF 融合参数 (检索阶段使用, 与分块无关)
    fusion: dict[str, float] = {"alpha": 0.6, "beta": 0.2}


class ModelConfig(BaseModel):
    """嵌入模型配置"""

    name: str = "BAAI/bge-small-zh-v1.5"
    size: Literal["large", "small"] = "small"
    cache_dir: str = "~/.cache/fastembed"
    unload_after_seconds: int = 30
    dimensions: int = 512

    @field_validator("cache_dir")
    @classmethod
    def expand_cache_dir(cls, v: str) -> str:
        return str(Path(v).expanduser())


class Settings(BaseModel):
    """全局配置根模型"""

    vaults: list[VaultConfig] = Field(
        default_factory=lambda: [
            VaultConfig(path="~/NanobotMemory", name="personal", enabled=True),
            VaultConfig(path="~/Obsidian", name="work", enabled=True),
        ]
    )
    db_path: str = "./data/rag.db"
    embedding_model: ModelConfig = Field(default_factory=ModelConfig)
    confidence: ConfidenceConfig = Field(default_factory=ConfidenceConfig)
    chunking: dict[str, int] = {"max_tokens": 512, "overlap": 50}
    memory_limit_mb: int = 500
    log_level: str = "INFO"
    maintenance: dict[str, Any] = {"soft_delete_threshold": 0.2, "auto_vacuum": True}

    @field_validator("db_path")
    @classmethod
    def expand_db_path(cls, v: str) -> str:
        return str(Path(v).resolve())


def load_config(config_path: str = "config.yaml") -> Settings:
    """
    加载并校验 YAML 配置文件
    :param config_path: 配置文件路径
    :return: 校验通过的 Settings 实例
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"❌ 配置文件不存在: {path.absolute()}")

    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return Settings(**data)
    except yaml.YAMLError as e:
        raise ValueError(f"❌ YAML 解析失败: {e}") from e
    except Exception as e:
        raise ValueError(f"❌ 配置校验失败: {e}") from e


# 兼容旧版直接调用
if __name__ == "__main__":
    try:
        cfg = load_config()
        print("✅ 配置加载成功")
        print(f"📂 启用仓库: {[v.name for v in cfg.vaults if v.enabled]}")
        print(f"🗄️ 数据库路径: {cfg.db_path}")
        print(f"🤖 模型: {cfg.embedding_model.name} (dim={cfg.embedding_model.dimensions})")
        print(f"⚖️ 置信度: doc_type_rules={cfg.confidence.doc_type_rules}")
        print(f"   status_rules={cfg.confidence.status_rules}")
        print(f"   date_decay={cfg.confidence.date_decay}")
    except Exception as e:
        print(e)
