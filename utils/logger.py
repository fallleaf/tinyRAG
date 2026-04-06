#!/usr/bin/env python3
# utils/logger.py - 懒加载日志工具
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


# ── 懒加载代理：避免模块导入时立即创建日志文件和目录 ──
_logger_instance = None


def _get_logger():
    """延迟初始化日志实例，首次使用时才创建"""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = setup_logger()
    return _logger_instance


class _LazyLogger:
    """代理类：将所有属性访问转发到延迟初始化的 logger 实例"""

    def __getattr__(self, name):
        return getattr(_get_logger(), name)

    def __setattr__(self, name, value):
        if name == "_initialized":
            super().__setattr__(name, value)
        else:
            setattr(_get_logger(), name, value)


# 模块级日志实例 (导入时不创建文件，首次调用 logger.info() 等时才初始化)
logger = _LazyLogger()
