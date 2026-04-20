#!/usr/bin/env python3
"""init_env.py - 全新环境初始化与自检"""

import sys
from pathlib import Path
from loguru import logger


def init():
    logger.info("🔍 tinyRAG 环境初始化...")

    # 1. 检查核心目录
    for d in ["data", "models", "logs"]:
        Path(d).mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ 目录就绪: ./{d}")

    # 2. 检查依赖
    try:
        import importlib.util

        deps = ["fastembed", "loguru", "pydantic", "sqlite_vec"]
        for dep in deps:
            if importlib.util.find_spec(dep) is None:
                raise ImportError(f"Module '{dep}' not found")

        logger.info("✅ 核心依赖已安装")
    except ImportError as e:
        logger.info(f"❌ 缺少依赖: {e}\n请运行: pip install -r requirements.txt")
        sys.exit(1)

    # 3. 加载并验证配置
    try:
        from config import load_config

        cfg = load_config("config.yaml")
        logger.info(f"✅ 配置加载成功 | DB: {cfg.db_path} | Model: {cfg.embedding_model.cache_dir}")

        # 检查 vaults
        valid_vaults = [v for v in cfg.vaults if Path(v).expanduser().exists()]
        if not valid_vaults:
            logger.info("⚠️  警告: 未找到任何有效的 Vault 路径，请检查 config.yaml")
        else:
            logger.info(f"✅ 有效 Vault 数量: {len(valid_vaults)}")
    except Exception as e:
        logger.info(f"❌ 配置校验失败: {e}")
        sys.exit(1)

    # 4. 模型预热 (触发 FastEmbed 首次下载)
    logger.info("📦 预热嵌入模型 (首次运行将下载约 100MB)...")
    try:
        from fastembed import TextEmbedding

        TextEmbedding(model_name=cfg.embedding_model.name, cache_dir=cfg.embedding_model.cache_dir)
        logger.info("✅ 模型缓存就绪")
    except Exception as e:
        logger.info(f"❌ 模型下载失败: {e}\n请检查网络或手动设置 HTTP_PROXY")
        sys.exit(1)

    logger.info("\n🎉 初始化完成！请执行: python build_index.py --force")


if __name__ == "__main__":
    init()
