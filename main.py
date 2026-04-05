#!/usr/bin/env python3
"""
Main entry point for the RAG System.
Launches the MCP server with proper error handling and configuration checks.
"""

import asyncio
import os
import sys
from pathlib import Path

# 确保工作目录正确
script_dir = Path(__file__).parent.resolve()
os.chdir(script_dir)
sys.path.insert(0, str(script_dir))

# 导入必须在 sys.path 修改之后
from mcp_server.server import RagServer  # noqa: E402
from utils.logger import logger  # noqa: E402


async def main():
    """Main entry point with robust error handling."""
    # 1. 配置检查
    config_path = script_dir / "config.yaml"
    if not config_path.exists():
        logger.critical(f"❌ 配置文件缺失：{config_path}")
        logger.critical("请确保 config.yaml 存在于 rag_system 根目录")
        sys.exit(1)

    logger.info("🌟 轻量级中文 RAG 系统启动")
    logger.info(f"   工作目录：{os.getcwd()}")
    logger.info(f"   配置文件：{config_path}")

    server = None
    try:
        # 2. 初始化服务器
        logger.info("🔧 初始化服务器组件...")
        server = RagServer()

        # 3. 运行服务器
        logger.info("🚀 启动 MCP 服务...")
        await server.run()

    except KeyboardInterrupt:
        logger.info("👋 收到中断信号，正在优雅关闭...")
    except Exception as e:
        logger.critical(f"❌ 服务器运行异常：{e}", exc_info=True)
        sys.exit(1)
    finally:
        # 4. 清理资源
        if server:
            try:
                await server.ctx.shutdown()
                logger.info("✅ 资源释放完成")
            except Exception as e:
                logger.error(f"⚠️ 关闭资源时出错：{e}")
        logger.info("🛑 系统已关闭")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 用户中断，退出程序")
        sys.exit(0)
