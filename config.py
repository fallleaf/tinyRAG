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


class ExcludeConfig(BaseModel):
    """排除规则配置"""
    
    # 要跳过的目录名（不进入子目录扫描）
    dirs: list[str] = Field(default_factory=list)
    # 要排除的文件模式（glob 模式）
    patterns: list[str] = Field(default_factory=list)


class VaultConfig(BaseModel):
    """仓库配置单元"""

    path: str
    name: str
    enabled: bool = True
    # per-vault 排除规则（覆盖全局配置）
    exclude: ExcludeConfig | None = None

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
    batch_size: int = 32  # 向量化批大小

    @field_validator("cache_dir")
    @classmethod
    def expand_cache_dir(cls, v: str) -> str:
        return str(Path(v).expanduser())


class DateDecayConfig(BaseModel):
    """日期衰减配置"""

    enabled: bool = True
    half_life_days: int = 365  # 默认半衰期 1 年
    min_weight: float = 0.5
    # 按文档类型差异化衰减（天）
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

    # chunk 内容类型权重 (用于检索期动态计算)
    type_rules: dict[str, float] = Field(
        default_factory=lambda: {
            "code": 1.1,
            "table": 1.05,
            "header": 1.0,
            "text": 0.95,
            "list": 0.9,
        }
    )
    # 文档类型权重 (Frontmatter doc_type 字段)
    # 注意：doc_type_rules 和 status_rules 应在 config.yaml 中定义
    doc_type_rules: dict[str, float] = Field(default_factory=dict)
    # 文档状态权重 (Frontmatter status 字段)
    # 5 种基础类型：published, completed, active, draft, archived
    status_rules: dict[str, float] = Field(
        default_factory=lambda: {
            "published": 1.2,
            "completed": 1.1,
            "active": 0.9,
            "draft": 0.7,
            "archived": 0.5,
        }
    )
    # 日期衰减配置
    date_decay: DateDecayConfig = Field(default_factory=DateDecayConfig)
    # 未匹配字段时的默认权重
    default_weight: float = 1.0


class ChunkingConfig(BaseModel):
    """分块配置"""

    max_tokens: int = 512
    overlap: int = 50
    # Token 计算模式: "estimate" (估算) 或 "tiktoken" (精确)
    token_mode: Literal["estimate", "tiktoken"] = "estimate"
    # 估算模式参数 (仅在 token_mode="estimate" 时生效)
    # 中文约 1.5 字符/token (实测 GPT/Cl100k 编码)
    chinese_chars_per_token: float = 1.5
    # 英文/符号约 4.0 字符/token
    english_chars_per_token: float = 4.0
    # max_chars 保守系数 (用于快速预判断)
    max_chars_multiplier: float = 2.5


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
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    # 修复 L-new2: 删除死配置 memory_limit_mb（未被任何代码使用）
    # memory_limit_mb: int = 500
    log_level: str = "INFO"
    # 检索引擎融合权重 (HybridEngine 读取)
    retrieval: dict[str, Any] = {"alpha": 0.7, "beta": 0.3}
    maintenance: dict[str, Any] = {"soft_delete_threshold": 0.2, "auto_vacuum": True}
    cache: CacheConfig = Field(default_factory=CacheConfig)
    # jieba 用户自定义词典路径（留空则使用默认词典）
    jieba_user_dict: str = ""
    # 全局排除规则
    exclude: ExcludeConfig = Field(default_factory=ExcludeConfig)
    # 流式处理配置（内存优化）
    stream_batch_size: int = 100  # 每累积多少 chunks 进行一次向量化入库
    max_concurrent_files: int = 4  # 并行处理文件数的上限

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


def get_merged_exclude(vault: VaultConfig, global_exclude: ExcludeConfig) -> ExcludeConfig:
    """
    合并 vault 级排除规则与全局排除规则
    
    :param vault: 仓库配置
    :param global_exclude: 全局排除规则
    :return: 合并后的排除规则
    
    规则：
    - dirs: vault 级覆盖全局（如果 vault 有定义）
    - patterns: 合并全局 + vault 级
    """
    if vault.exclude is None:
        return global_exclude
    
    # 合并 dirs（vault 级优先）
    if vault.exclude.dirs:
        merged_dirs = vault.exclude.dirs
    else:
        merged_dirs = global_exclude.dirs
    
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
    except Exception as e:
        print(e)
