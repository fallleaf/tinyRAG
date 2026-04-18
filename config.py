#!/usr/bin/env python3
"""config.py - tinyRAG 配置契约与加载层 (v2.2 - 插件支持)"""

import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class ExcludeConfig(BaseModel):
    dirs: list[str] = Field(default_factory=list)
    patterns: list[str] = Field(default_factory=list)

    @field_validator("dirs", "patterns", mode="before")
    @classmethod
    def none_to_empty_list(cls, v):
        return v if v is not None else []


class VaultConfig(BaseModel):
    path: str
    name: str
    enabled: bool = True
    exclude: ExcludeConfig = Field(default_factory=ExcludeConfig)

    @field_validator("path")
    @classmethod
    def expand_vault_path(cls, v: str) -> str:
        return str(Path(v).expanduser())


class ModelConfig(BaseModel):
    name: str = "BAAI/bge-small-zh-v1.5"
    size: Literal["large", "small"] = "small"
    cache_dir: str = "~/.cache/fastembed"
    unload_after_seconds: int = 30
    dimensions: int = 512
    batch_size: int = 32

    @field_validator("cache_dir")
    @classmethod
    def expand_cache_dir(cls, v: str) -> str:
        return str(Path(v).expanduser())


class DateDecayConfig(BaseModel):
    enabled: bool = True
    half_life_days: int = 365
    min_weight: float = 0.5
    type_specific_decay: dict[str, int] = Field(default_factory=dict)


class CacheConfig(BaseModel):
    db_path: str = "./data/cache.db"
    ttl_seconds: int = 3600
    max_entries: int = 1000

    @field_validator("db_path")
    @classmethod
    def expand_cache_db_path(cls, v: str) -> str:
        return str(Path(v).expanduser().resolve())


class ConfidenceConfig(BaseModel):
    type_rules: dict[str, float] = Field(
        default_factory=lambda: {"code": 1.1, "table": 1.05, "header": 1.0, "text": 0.95, "list": 0.9}
    )
    doc_type_rules: dict[str, float] = Field(default_factory=dict)
    status_rules: dict[str, float] = Field(
        default_factory=lambda: {"published": 1.2, "completed": 1.1, "active": 0.9, "draft": 0.7, "archived": 0.5}
    )
    date_decay: DateDecayConfig = Field(default_factory=DateDecayConfig)
    default_weight: float = 1.0


class ChunkingConfig(BaseModel):
    max_tokens: int = 512
    overlap: int = 50
    token_mode: Literal["estimate", "tiktoken"] = "estimate"
    chinese_chars_per_token: float = 1.5
    english_chars_per_token: float = 4.0
    max_chars_multiplier: float = 2.5


class PluginConfig(BaseModel):
    """单个插件配置"""

    name: str
    enabled: bool = True
    priority: int = 100
    config: dict[str, Any] = Field(default_factory=dict)


class PluginsConfig(BaseModel):
    """插件系统配置"""

    enabled: bool = True
    auto_discover: bool = True
    plugins: list[PluginConfig] = Field(default_factory=list)


class Settings(BaseModel):
    vaults: list[VaultConfig] = Field(
        default_factory=lambda: [
            VaultConfig(path="~/NanobotMemory", name="personal", enabled=True),
            VaultConfig(path="~/Obsidian", name="work", enabled=True),
        ]
    )
    db_path: str = "./data/rag.db"
    embedding_model: ModelConfig = Field(default_factory=ModelConfig)
    confidence: ConfidenceConfig = Field(default_factory=ConfidenceConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    log_level: str = "INFO"
    retrieval: dict[str, Any] = {"alpha": 0.7, "beta": 0.3}
    maintenance: dict[str, Any] = {"soft_delete_threshold": 0.2, "auto_vacuum": True}
    cache: CacheConfig = Field(default_factory=CacheConfig)
    jieba_user_dict: str = ""
    exclude: ExcludeConfig = Field(default_factory=ExcludeConfig)
    stream_batch_size: int = 100
    max_concurrent_files: int = 4
    plugins: PluginsConfig = Field(default_factory=PluginsConfig)

    @field_validator("db_path")
    @classmethod
    def expand_db_path(cls, v: str) -> str:
        return str(Path(v).expanduser().resolve())


# P0 修复：递归清洗 YAML 键值首尾空格
def _strip_yaml_keys_values(d):
    if isinstance(d, dict):
        return {
            re.sub(r"\s+", "", k): _strip_yaml_keys_values(v)
            if isinstance(v, (dict, list))
            else re.sub(r"^\s+|\s+$", "", v)
            if isinstance(v, str)
            else v
            for k, v in d.items()
        }
    elif isinstance(d, list):
        return [_strip_yaml_keys_values(i) for i in d]
    return d


def load_config(config_path: str = "config.yaml") -> Settings:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"❌ 配置文件不存在: {path.absolute()}")
    try:
        with open(path, encoding="utf-8") as f:
            raw_data = yaml.safe_load(f) or {}
        clean_data = _strip_yaml_keys_values(raw_data)
        return Settings(**clean_data)
    except yaml.YAMLError as e:
        raise ValueError(f"❌ YAML 解析失败: {e}") from e
    except Exception as e:
        raise ValueError(f"❌ 配置校验失败: {e}") from e


def get_merged_exclude(vault: VaultConfig, global_exclude: ExcludeConfig) -> ExcludeConfig:
    if vault.exclude is None:
        return global_exclude
    merged_dirs = vault.exclude.dirs if vault.exclude.dirs else global_exclude.dirs
    merged_patterns = list(set(global_exclude.patterns + vault.exclude.patterns))
    return ExcludeConfig(dirs=merged_dirs, patterns=merged_patterns)


if __name__ == "__main__":
    try:
        cfg = load_config()
        print("✅ 配置加载成功")
    except Exception as e:
        print(e)
