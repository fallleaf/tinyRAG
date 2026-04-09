#!/usr/bin/env python3
"""
config.py - tinyRAG 配置契约与加载层
特性:
1. Pydantic v2 强类型校验
2. 路径自动展开 (~ -> 绝对路径)
3. 结构化 Vault 配置 (path, name, enabled)
4. 安全默认值与缺失字段回退
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
        # 仅展开 ~/，不强制要求目录已存在（便于首次部署/动态创建）
        return str(Path(v).expanduser())


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


class DateDecayConfig(BaseModel):
    """日期衰减配置"""

    enabled: bool = True
    half_life_days: int = 180
    min_weight: float = 0.5


class CacheConfig(BaseModel):
    """缓存配置"""

    db_path: str = "./data/cache.db"
    ttl_seconds: int = 3600
    max_entries: int = 1000

    @field_validator("db_path")
    @classmethod
    def expand_cache_db_path(cls, v: str) -> str:
        return str(Path(v).expanduser().resolve())


class ConfidenceConfig(BaseModel):
    """置信度与融合权重配置"""

    # chunk 内容类型权重 (用于检索期动态计算)
    type_rules: dict[str, float] = {
        "code": 1.1,
        "table": 1.05,
        "header": 1.0,
        "text": 0.95,
        "list": 0.9,
    }
    # 文档类型权重 (Frontmatter doc_type 字段)
    doc_type_rules: dict[str, float] = {
        "blog": 1.0,
        "diary": 1.1,
        "project": 1.15,
        "note": 1.0,
    }
    # 文档状态权重 (Frontmatter status 字段)
    status_rules: dict[str, float] = {
        "已完成": 1.1,
        "进行中": 0.9,
        "草稿": 0.7,
    }
    # 日期衰减配置
    date_decay: DateDecayConfig = Field(default_factory=DateDecayConfig)
    # 未匹配字段时的默认权重
    default_weight: float = 1.0


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
    # 修复 L-new2: 删除死配置 memory_limit_mb（未被任何代码使用）
    # memory_limit_mb: int = 500
    log_level: str = "INFO"
    # 检索引擎融合权重 (HybridEngine 读取)
    retrieval: dict[str, Any] = {"alpha": 0.7, "beta": 0.3}
    maintenance: dict[str, Any] = {"soft_delete_threshold": 0.2, "auto_vacuum": True}
    cache: CacheConfig = Field(default_factory=CacheConfig)

    @field_validator("db_path")
    @classmethod
    def expand_db_path(cls, v: str) -> str:
        return str(Path(v).expanduser().resolve())


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
    except Exception as e:
        print(e)
