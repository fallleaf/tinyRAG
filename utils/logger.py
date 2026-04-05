#!/usr/bin/env python3
# utils/logger.py
import sys


def setup_logger(level: str = "INFO", log_file: str = "logs/app.log"):
    """
    配置 Loguru 日志：
    1. 移除默认处理器
    2. 添加 stderr 处理器 (MCP 协议要求)
    3. 添加文件处理器 (详细调试)
    """
    from loguru import logger as loguru_logger

    loguru_logger.remove()

    # 1. 输出到 stderr (标准错误，不影响 stdout 的 JSON)
    loguru_logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True,
    )

    # 2. 输出到文件 (详细记录)
    loguru_logger.add(
        log_file,
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}",
    )

    return loguru_logger


# 全局日志实例
logger = setup_logger()
