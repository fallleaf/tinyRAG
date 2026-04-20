#!/usr/bin/env python3
"""
vacuum.py - RAG 数据库 VACUUM 工具
用于回收软删除文件占用的 SQLite 空间
支持清理关联的图谱数据（relations, principles, notes, graph_build_jobs）
"""

import argparse
import os
import sys

# 设置工作目录
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.insert(0, script_dir)

# 导入 RAG 系统模块（必须在设置 path 之后）
from config import load_config
from storage.database import DatabaseManager
from utils.logger import logger, setup_logger

# 初始化日志
_script_dir = os.path.dirname(os.path.abspath(__file__))
setup_logger(level="INFO")


def check_vacuum_needed(db: DatabaseManager, config) -> dict:
    """检查是否需要执行 VACUUM，包含图谱数据统计"""
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

    # 检查图谱相关数据（与软删除 chunks/files 关联的）
    # 先检查表是否存在
    def _table_exists(conn, table_name):
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        return cursor.fetchone() is not None

    relations_exist = _table_exists(db.conn, "relations")
    principles_exist = _table_exists(db.conn, "principles")
    notes_exist = _table_exists(db.conn, "notes")
    jobs_exist = _table_exists(db.conn, "graph_build_jobs")

    # 1. 与软删除 chunks 关联的 relations
    if relations_exist:
        cursor = db.conn.execute("""
            SELECT COUNT(*) FROM relations
            WHERE src_chunk_id IN (SELECT id FROM chunks WHERE is_deleted = 1)
               OR tgt_chunk_id IN (SELECT id FROM chunks WHERE is_deleted = 1)
        """)
        relations_to_delete = cursor.fetchone()[0]
    else:
        relations_to_delete = 0

    # 2. 与软删除 chunks 关联的 principles
    if principles_exist:
        cursor = db.conn.execute("""
            SELECT COUNT(*) FROM principles
            WHERE chunk_id IN (SELECT id FROM chunks WHERE is_deleted = 1)
        """)
        principles_to_delete = cursor.fetchone()[0]
    else:
        principles_to_delete = 0

    # 3. 与软删除 files 关联的 notes（通过 filepath）
    if notes_exist:
        cursor = db.conn.execute("""
            SELECT COUNT(*) FROM notes
            WHERE filepath IN (
                SELECT file_path FROM files WHERE is_deleted = 1
                UNION
                SELECT absolute_path FROM files WHERE is_deleted = 1
            )
        """)
        notes_to_delete = cursor.fetchone()[0]
    else:
        notes_to_delete = 0

    # 4. 与这些 notes 关联的 graph_build_jobs
    if jobs_exist:
        cursor = db.conn.execute("""
            SELECT COUNT(*) FROM graph_build_jobs
            WHERE note_id IN (
                SELECT note_id FROM notes
                WHERE filepath IN (
                    SELECT file_path FROM files WHERE is_deleted = 1
                    UNION
                    SELECT absolute_path FROM files WHERE is_deleted = 1
                )
            )
        """)
        jobs_to_delete = cursor.fetchone()[0]
    else:
        jobs_to_delete = 0

    # 获取数据库文件大小
    # 修复 C2: 使用传入的 config.db_path
    db_path = config.db_path
    file_size_mb = os.path.getsize(db_path) / (1024 * 1024)

    # 修复 L-new3: 从 config.maintenance 读取阈值，兼容 dict 和对象访问
    maintenance = getattr(config, "maintenance", {}) or {}
    if isinstance(maintenance, dict):
        threshold_pct = maintenance.get("soft_delete_threshold", 0.2) * 100
    else:
        threshold_pct = getattr(maintenance, "soft_delete_threshold", 0.2) * 100

    # 取较高的比例作为判断依据
    max_ratio = max(files_ratio, chunks_ratio)
    needs_vacuum = max_ratio > threshold_pct

    return {
        "files_deleted": files_deleted,
        "files_active": files_active,
        "files_ratio": files_ratio,
        "chunks_deleted": chunks_deleted,
        "chunks_active": chunks_active,
        "chunks_ratio": chunks_ratio,
        "max_ratio": max_ratio,
        "threshold_pct": threshold_pct,  # 新增：记录实际使用的阈值
        "file_size_mb": file_size_mb,
        "needs_vacuum": needs_vacuum,
        # 图谱相关统计
        "relations_to_delete": relations_to_delete,
        "principles_to_delete": principles_to_delete,
        "notes_to_delete": notes_to_delete,
        "jobs_to_delete": jobs_to_delete,
    }


def clean_deleted_records(db: DatabaseManager, dry_run: bool = False) -> dict:
    """
    删除软删除记录（is_deleted = 1）
    同时清理关联的图谱数据
    返回删除统计信息
    """
    stats = {
        "files_deleted": 0,
        "chunks_deleted": 0,
        "vectors_deleted": 0,
        "relations_deleted": 0,
        "principles_deleted": 0,
        "notes_deleted": 0,
        "jobs_deleted": 0,
    }

    # 检查表是否存在
    def _table_exists(conn, table_name):
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        return cursor.fetchone() is not None

    relations_exist = _table_exists(db.conn, "relations")
    principles_exist = _table_exists(db.conn, "principles")
    notes_exist = _table_exists(db.conn, "notes")
    jobs_exist = _table_exists(db.conn, "graph_build_jobs")
    vectors_exist = _table_exists(db.conn, "vectors")

    if dry_run:
        # 仅统计，不删除
        cursor = db.conn.execute("SELECT COUNT(*) FROM files WHERE is_deleted = 1")
        stats["files_deleted"] = cursor.fetchone()[0]
        cursor = db.conn.execute("SELECT COUNT(*) FROM chunks WHERE is_deleted = 1")
        stats["chunks_deleted"] = cursor.fetchone()[0]
        if vectors_exist:
            cursor = db.conn.execute(
                "SELECT COUNT(*) FROM vectors WHERE chunk_id IN (SELECT id FROM chunks WHERE is_deleted = 1)"
            )
            stats["vectors_deleted"] = cursor.fetchone()[0]

        # 统计图谱相关数据
        if relations_exist:
            cursor = db.conn.execute("""
                SELECT COUNT(*) FROM relations
                WHERE src_chunk_id IN (SELECT id FROM chunks WHERE is_deleted = 1)
                   OR tgt_chunk_id IN (SELECT id FROM chunks WHERE is_deleted = 1)
            """)
            stats["relations_deleted"] = cursor.fetchone()[0]

        if principles_exist:
            cursor = db.conn.execute("""
                SELECT COUNT(*) FROM principles
                WHERE chunk_id IN (SELECT id FROM chunks WHERE is_deleted = 1)
            """)
            stats["principles_deleted"] = cursor.fetchone()[0]

        if notes_exist:
            cursor = db.conn.execute("""
                SELECT COUNT(*) FROM notes
                WHERE filepath IN (
                    SELECT file_path FROM files WHERE is_deleted = 1
                    UNION
                    SELECT absolute_path FROM files WHERE is_deleted = 1
                )
            """)
            stats["notes_deleted"] = cursor.fetchone()[0]

        if jobs_exist:
            cursor = db.conn.execute("""
                SELECT COUNT(*) FROM graph_build_jobs
                WHERE note_id IN (
                    SELECT note_id FROM notes
                    WHERE filepath IN (
                        SELECT file_path FROM files WHERE is_deleted = 1
                        UNION
                        SELECT absolute_path FROM files WHERE is_deleted = 1
                    )
                )
            """)
            stats["jobs_deleted"] = cursor.fetchone()[0]

        return stats

    try:
        # =====================================================
        # 清理顺序：先清理依赖关系（子表 -> 父表）
        # =====================================================

        # 1. 删除与软删除 chunks 关联的 relations（图谱关系）
        if relations_exist:
            cursor = db.conn.execute("""
                DELETE FROM relations
                WHERE src_chunk_id IN (SELECT id FROM chunks WHERE is_deleted = 1)
                   OR tgt_chunk_id IN (SELECT id FROM chunks WHERE is_deleted = 1)
            """)
            stats["relations_deleted"] = cursor.rowcount
            if stats["relations_deleted"] > 0:
                logger.info(f"  已删除 relations 表记录：{stats['relations_deleted']} 条（关联软删除 chunks）")

        # 2. 删除与软删除 chunks 关联的 principles（原则）
        if principles_exist:
            cursor = db.conn.execute("""
                DELETE FROM principles
                WHERE chunk_id IN (SELECT id FROM chunks WHERE is_deleted = 1)
            """)
            stats["principles_deleted"] = cursor.rowcount
            if stats["principles_deleted"] > 0:
                logger.info(f"  已删除 principles 表记录：{stats['principles_deleted']} 条（关联软删除 chunks）")

        # 3. 删除与软删除 files 关联的 graph_build_jobs（建图任务）
        if jobs_exist:
            cursor = db.conn.execute("""
                DELETE FROM graph_build_jobs
                WHERE note_id IN (
                    SELECT note_id FROM notes
                    WHERE filepath IN (
                        SELECT file_path FROM files WHERE is_deleted = 1
                        UNION
                        SELECT absolute_path FROM files WHERE is_deleted = 1
                    )
                )
            """)
            stats["jobs_deleted"] = cursor.rowcount
            if stats["jobs_deleted"] > 0:
                logger.info(f"  已删除 graph_build_jobs 表记录：{stats['jobs_deleted']} 条（关联软删除 files）")

        # 4. 删除与软删除 files 关联的 notes（文档记录）
        if notes_exist:
            cursor = db.conn.execute("""
                DELETE FROM notes
                WHERE filepath IN (
                    SELECT file_path FROM files WHERE is_deleted = 1
                    UNION
                    SELECT absolute_path FROM files WHERE is_deleted = 1
                )
            """)
            stats["notes_deleted"] = cursor.rowcount
            if stats["notes_deleted"] > 0:
                logger.info(f"  已删除 notes 表记录：{stats['notes_deleted']} 条（关联软删除 files）")

        # 5. 删除 chunks 表中软删除的记录
        cursor = db.conn.execute("DELETE FROM chunks WHERE is_deleted = 1")
        stats["chunks_deleted"] = cursor.rowcount
        logger.info(f"  已删除 chunks 表软删除记录：{stats['chunks_deleted']} 条")

        # 6. 删除 files 表中软删除的记录
        cursor = db.conn.execute("DELETE FROM files WHERE is_deleted = 1")
        stats["files_deleted"] = cursor.rowcount
        logger.info(f"  已删除 files 表软删除记录：{stats['files_deleted']} 条")

        # 7. 清理 FTS5 索引（自动同步，但显式重建更保险）
        logger.info("  重建 FTS5 索引...")
        db.conn.execute("INSERT INTO fts5_index(fts5_index) VALUES('rebuild')")

        # 8. 提交事务
        db.conn.commit()

        # 汇总统计
        total_deleted = (
            stats["chunks_deleted"]
            + stats["files_deleted"]
            + stats["relations_deleted"]
            + stats["principles_deleted"]
            + stats["notes_deleted"]
            + stats["jobs_deleted"]
        )
        logger.success(f"  ✅ 软删除记录清理完成，共删除 {total_deleted} 条记录")

        # 输出图谱清理详情
        graph_deleted = (
            stats["relations_deleted"] + stats["principles_deleted"] + stats["notes_deleted"] + stats["jobs_deleted"]
        )
        if graph_deleted > 0:
            logger.info(
                f"  📊 图谱相关：relations={stats['relations_deleted']}, "
                f"principles={stats['principles_deleted']}, "
                f"notes={stats['notes_deleted']}, jobs={stats['jobs_deleted']}"
            )

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
    parser = argparse.ArgumentParser(description="RAG 数据库 VACUUM 工具（含图谱数据清理）")
    parser.add_argument("--dry-run", action="store_true", help="仅检查，不执行清理和 VACUUM")
    parser.add_argument("--clean-only", action="store_true", help="仅清理软删除记录，不执行 VACUUM")
    parser.add_argument("--vacuum-only", action="store_true", help="仅执行 VACUUM，不清理记录")
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
    # 修复 C2: 传入 config 参数
    stats = check_vacuum_needed(db, config)

    logger.info("📊 数据库状态:")
    logger.info(
        f"  files 表：已删除 {stats['files_deleted']}, 活跃 {stats['files_active']}, 比例 {stats['files_ratio']:.1f}%"
    )
    logger.info(
        f"  chunks 表：已删除 {stats['chunks_deleted']}, 活跃 {stats['chunks_active']}, 比例 {stats['chunks_ratio']:.1f}%"
    )
    logger.info(f"  数据库大小：{stats['file_size_mb']:.2f} MB")

    # 显示图谱相关统计
    graph_total = (
        stats["relations_to_delete"]
        + stats["principles_to_delete"]
        + stats["notes_to_delete"]
        + stats["jobs_to_delete"]
    )
    if graph_total > 0:
        logger.info("📊 图谱关联数据待清理:")
        logger.info(f"  relations: {stats['relations_to_delete']} 条")
        logger.info(f"  principles: {stats['principles_to_delete']} 条")
        logger.info(f"  notes: {stats['notes_to_delete']} 条")
        logger.info(f"  graph_build_jobs: {stats['jobs_to_delete']} 条")

    if stats["max_ratio"] > 20 or args.force:
        if args.dry_run:
            logger.info("💡 建议执行清理和 VACUUM (软删除比例 > 20%)")
            logger.info(
                f" 预计清理：{stats['chunks_deleted']} chunks + {stats['files_deleted']} files + {graph_total} 图谱记录"
            )
        else:
            # 1. 先清理软删除记录
            if not args.vacuum_only:
                logger.info("\n🧹 步骤 1: 清理软删除记录（含图谱数据）...")
                clean_stats = clean_deleted_records(db, dry_run=False)
                total = (
                    clean_stats["chunks_deleted"]
                    + clean_stats["files_deleted"]
                    + clean_stats.get("relations_deleted", 0)
                    + clean_stats.get("principles_deleted", 0)
                    + clean_stats.get("notes_deleted", 0)
                    + clean_stats.get("jobs_deleted", 0)
                )
                logger.info(f" 共删除 {total} 条记录")

            # 2. 执行 VACUUM
            if not args.clean_only:
                logger.info("\n🗜️ 步骤 2: 执行 VACUUM...")
                if execute_vacuum(db, dry_run=False):
                    # 验证结果
                    # 修复 C2: 传入 config 参数
                    new_stats = check_vacuum_needed(db, config)

                    # 检查文件大小变化
                    # 修复 H1: 使用 config.db_path 而非硬编码
                    new_size = os.path.getsize(config.db_path) / (1024 * 1024)
                    saved = stats["file_size_mb"] - new_size
                    if saved > 0:
                        logger.success(
                            f"💾 节省空间：{saved:.2f} MB (从 {stats['file_size_mb']:.2f} MB 到 {new_size:.2f} MB)"
                        )
                    else:
                        logger.info(f"ℹ️ 文件大小变化：{new_size:.2f} MB")

                    logger.info(f" 清理后 chunks 软删除比例：{new_stats['chunks_ratio']:.1f}%")
                else:
                    sys.exit(1)
            else:
                logger.info("ℹ️ 仅清理模式，跳过 VACUUM")
    else:
        logger.info("ℹ️ 软删除比例较低，无需执行清理和 VACUUM")

    db.conn.close()


if __name__ == "__main__":
    main()
