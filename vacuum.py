#!/usr/bin/env python3
"""
vacuum.py - RAG 数据库 VACUUM 工具
用于回收软删除文件占用的 SQLite 空间
"""

import argparse
import os
import sys

from loguru import logger

# 设置工作目录
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.insert(0, script_dir)

# 导入 RAG 系统模块（必须在设置 path 之后）
from config import load_config  # noqa: E402
from storage.database import DatabaseManager  # noqa: E402


def check_vacuum_needed(db: DatabaseManager) -> dict:
    """检查是否需要执行 VACUUM"""
    # 检查 files 表
    cursor = db.conn.execute("SELECT COUNT(*) FROM files WHERE is_deleted = 1")
    files_deleted = cursor.fetchone()[0]
    cursor = db.conn.execute("SELECT COUNT(*) FROM files WHERE is_deleted = 0")
    files_active = cursor.fetchone()[0]
    files_total = files_deleted + files_active
    files_ratio = (files_deleted / files_total * 100) if files_total > 0 else 0

    # 检查 chunks 表（主要问题所在）
    cursor = db.conn.execute("SELECT COUNT(*) FROM chunks WHERE is_deleted = 1")
    chunks_deleted = cursor.fetchone()[0]
    cursor = db.conn.execute("SELECT COUNT(*) FROM chunks WHERE is_deleted = 0")
    chunks_active = cursor.fetchone()[0]
    chunks_total = chunks_deleted + chunks_active
    chunks_ratio = (chunks_deleted / chunks_total * 100) if chunks_total > 0 else 0

    # 获取数据库文件大小
    db_path = os.path.expanduser("~/rag_system/data/rag.db")
    file_size_mb = os.path.getsize(db_path) / (1024 * 1024)

    # 取较高的比例作为判断依据
    max_ratio = max(files_ratio, chunks_ratio)
    needs_vacuum = max_ratio > 20

    return {
        "files_deleted": files_deleted,
        "files_active": files_active,
        "files_ratio": files_ratio,
        "chunks_deleted": chunks_deleted,
        "chunks_active": chunks_active,
        "chunks_ratio": chunks_ratio,
        "max_ratio": max_ratio,
        "file_size_mb": file_size_mb,
        "needs_vacuum": needs_vacuum,
    }


def clean_deleted_records(db: DatabaseManager, dry_run: bool = False) -> dict:
    """
    删除软删除记录（is_deleted = 1）
    返回删除统计信息
    """
    stats = {"files_deleted": 0, "chunks_deleted": 0, "vectors_deleted": 0}

    if dry_run:
        # 仅统计，不删除
        cursor = db.conn.execute("SELECT COUNT(*) FROM files WHERE is_deleted = 1")
        stats["files_deleted"] = cursor.fetchone()[0]
        cursor = db.conn.execute("SELECT COUNT(*) FROM chunks WHERE is_deleted = 1")
        stats["chunks_deleted"] = cursor.fetchone()[0]
        cursor = db.conn.execute(
            "SELECT COUNT(*) FROM vectors WHERE chunk_id IN (SELECT id FROM chunks WHERE is_deleted = 1)"
        )
        stats["vectors_deleted"] = cursor.fetchone()[0]
        return stats

    try:
        # 1. 删除 chunks 表中软删除的记录（先删除子表）
        cursor = db.conn.execute("DELETE FROM chunks WHERE is_deleted = 1")
        stats["chunks_deleted"] = cursor.rowcount
        logger.info(f"  已删除 chunks 表软删除记录：{stats['chunks_deleted']} 条")

        # 2. 删除 files 表中软删除的记录
        cursor = db.conn.execute("DELETE FROM files WHERE is_deleted = 1")
        stats["files_deleted"] = cursor.rowcount
        logger.info(f"  已删除 files 表软删除记录：{stats['files_deleted']} 条")

        # 3. 清理 FTS5 索引（自动同步，但显式重建更保险）
        logger.info("  重建 FTS5 索引...")
        db.conn.execute("INSERT INTO fts5_index(fts5_index) VALUES('rebuild')")

        # 4. 提交事务
        db.conn.commit()
        logger.success("  ✅ 软删除记录清理完成")

        return stats

    except Exception as e:
        db.conn.rollback()
        logger.error(f"  ❌ 清理失败：{e}")
        raise


def execute_vacuum(db: DatabaseManager, dry_run: bool = False) -> bool:
    """执行 VACUUM 命令"""
    if dry_run:
        logger.info("🔍 仅检查模式，不执行 VACUUM")
        return True

    try:
        logger.info("⚙️ 开始执行 VACUUM (这可能需要几分钟)...")
        db.conn.execute("VACUUM")
        db.conn.commit()
        logger.success("✅ VACUUM 完成！")
        return True
    except Exception as e:
        logger.error(f"❌ VACUUM 失败：{e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="RAG 数据库 VACUUM 工具")
    parser.add_argument(
        "--dry-run", action="store_true", help="仅检查，不执行清理和 VACUUM"
    )
    parser.add_argument(
        "--clean-only", action="store_true", help="仅清理软删除记录，不执行 VACUUM"
    )
    parser.add_argument(
        "--vacuum-only", action="store_true", help="仅执行 VACUUM，不清理记录"
    )
    parser.add_argument("--force", action="store_true", help="即使软删除比例低也执行")
    args = parser.parse_args()

    # 加载配置
    try:
        config = load_config("config.yaml")
        db = DatabaseManager(config.db_path)
        logger.info("✅ 数据库连接成功")
    except Exception as e:
        logger.critical(f"❌ 数据库连接失败：{e}")
        sys.exit(1)

    # 检查状态
    stats = check_vacuum_needed(db)

    logger.info("📊 数据库状态:")
    logger.info(
        f"  files 表：已删除 {stats['files_deleted']}, 活跃 {stats['files_active']}, 比例 {stats['files_ratio']:.1f}%"
    )
    logger.info(
        f"  chunks 表：已删除 {stats['chunks_deleted']}, 活跃 {stats['chunks_active']}, 比例 {stats['chunks_ratio']:.1f}%"
    )
    logger.info(f"  数据库大小：{stats['file_size_mb']:.2f} MB")

    if stats["max_ratio"] > 20 or args.force:
        if args.dry_run:
            logger.info("💡 建议执行清理和 VACUUM (软删除比例 > 20%)")
            logger.info(
                f"   预计清理：{stats['chunks_deleted']} chunks + {stats['files_deleted']} files"
            )
        else:
            # 1. 先清理软删除记录
            if not args.vacuum_only:
                logger.info("\n🧹 步骤 1: 清理软删除记录...")
                clean_stats = clean_deleted_records(db, dry_run=False)
                logger.info(
                    f"   共删除 {clean_stats['chunks_deleted'] + clean_stats['files_deleted']} 条记录"
                )

            # 2. 执行 VACUUM
            if not args.clean_only:
                logger.info("\n🗜️  步骤 2: 执行 VACUUM...")
                if execute_vacuum(db, dry_run=False):
                    # 验证结果
                    new_stats = check_vacuum_needed(db)

                    # 检查文件大小变化
                    new_size = os.path.getsize(
                        os.path.expanduser("~/rag_system/data/rag.db")
                    ) / (1024 * 1024)
                    saved = stats["file_size_mb"] - new_size
                    if saved > 0:
                        logger.success(
                            f"💾 节省空间：{saved:.2f} MB (从 {stats['file_size_mb']:.2f} MB 到 {new_size:.2f} MB)"
                        )
                    else:
                        logger.info(f"ℹ️  文件大小变化：{new_size:.2f} MB")

                    logger.info(
                        f"   清理后 chunks 软删除比例：{new_stats['chunks_ratio']:.1f}%"
                    )
                else:
                    sys.exit(1)
            else:
                logger.info("ℹ️  仅清理模式，跳过 VACUUM")
    else:
        logger.info("ℹ️  软删除比例较低，无需执行清理和 VACUUM")

    db.conn.close()


if __name__ == "__main__":
    main()
