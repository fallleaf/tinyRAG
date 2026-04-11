#!/usr/bin/env python3
"""
config.py - tinyRAG 配置契约与加载层
特性:
1. Pydantic v2 强类型校验
2. 路径自动展开 (~ -> 绝对路径)
3. 结构化 Vault 配置 (path, name, enabled)
4. 安全默认值与缺失字段回退
5. 支持 per-vault 排除规则
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class ExcludeConfig(BaseModel):
    """扫描排除规则配置"""

    # 排除的目录名（不进入递归）
    dirs: list[str] = Field(
        default_factory=lambda: [
            ".git",
            ".obsidian",
            ".trash",
            ".Trash",
            "node_modules",
            "__pycache__",
            ".venv",
            "venv",
            ".idea",
            ".vscode",
        ]
    )
    # 排除的路径模式（glob 风格）
    patterns: list[str] = Field(default_factory=list)


class VaultConfig(BaseModel):
    """仓库配置单元"""

    path: str
    name: str
    enabled: bool = True
    # per-vault 排除规则（可选，与全局规则合并）
    exclude: ExcludeConfig | None = None

    @field_validator("path")
    @classmethod
    def expand_vault_path(cls, v: str) -> str:
        return str(Path(v).expanduser())


class ModelConfig(BaseModel):
    """嵌入模型配置"""

    name: str = "BAAI/bge-small-zh-v1.5"
    size: Literal["large", "small"] = "small"
    cache_dir: str = "~/.cache/fastembed"
    unload_after_seconds: int = 30
    dimensions: int = 512
    batch_size: int = 64

    @field_validator("cache_dir")
    @classmethod
    def expand_cache_dir(cls, v: str) -> str:
        return str(Path(v).expanduser())


class DateDecayConfig(BaseModel):
    """日期衰减配置"""

    enabled: bool = True
    half_life_days: int = 365
    min_weight: float = 0.5
    type_specific_decay: dict[str, int] = Field(default_factory=dict)


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

    type_rules: dict[str, float] = Field(
        default_factory=lambda: {
            "code": 1.1,
            "table": 1.05,
            "header": 1.0,
            "text": 0.95,
            "list": 0.9,
        }
    )
    doc_type_rules: dict[str, float] = Field(default_factory=dict)
    status_rules: dict[str, float] = Field(
        default_factory=lambda: {
            "published": 1.2,
            "completed": 1.1,
            "active": 0.9,
            "draft": 0.7,
            "archived": 0.5,
        }
    )
    date_decay: DateDecayConfig = Field(default_factory=DateDecayConfig)
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
    stream_batch_size: int = 1000
    max_concurrent_files: int = 2
    log_level: str = "INFO"
    retrieval: dict[str, Any] = {"alpha": 0.7, "beta": 0.3}
    maintenance: dict[str, Any] = {"soft_delete_threshold": 0.2, "auto_vacuum": True}
    cache: CacheConfig = Field(default_factory=CacheConfig)
    jieba_user_dict: str = ""
    # 全局扫描排除规则（作为默认值，与 per-vault 规则合并）
    exclude: ExcludeConfig = Field(default_factory=ExcludeConfig)

    @field_validator("db_path")
    @classmethod
    def expand_db_path(cls, v: str) -> str:
        return str(Path(v).expanduser().resolve())


def load_config(config_path: str = "config.yaml") -> Settings:
    """
    加载并校验 YAML 配置文件
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


def get_merged_exclude(vault: VaultConfig, global_exclude: ExcludeConfig) -> ExcludeConfig:
    """
    获取 vault 的合并排除规则。
    规则：全局规则 + vault 特定规则合并（去重）

    :param vault: vault 配置
    :param global_exclude: 全局排除规则
    :return: 合并后的排除规则
    """
    if vault.exclude is None:
        return global_exclude

    # 合并 dirs（去重）
    merged_dirs = list(set(global_exclude.dirs + vault.exclude.dirs))

    # 合并 patterns（去重）
    merged_patterns = list(set(global_exclude.patterns + vault.exclude.patterns))

    return ExcludeConfig(dirs=merged_dirs, patterns=merged_patterns)


# 兼容旧版直接调用
if __name__ == "__main__":
    try:
        cfg = load_config()
        print("✅ 配置加载成功")
        print(f"📂 启用仓库: {[v.name for v in cfg.vaults if v.enabled]}")
        print(f"🗄️ 数据库路径: {cfg.db_path}")
        print(f"🤖 模型: {cfg.embedding_model.name} (dim={cfg.embedding_model.dimensions})")
        print(f"🚫 全局排除目录: {cfg.exclude.dirs}")
        for v in cfg.vaults:
            if v.exclude:
                print(f"🚫 {v.name} 排除目录: {v.exclude.dirs}")
    except Exception as e:
        print(e)
