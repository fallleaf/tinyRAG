#!/usr/bin/env python3
"""
迁移脚本：移除 files 表的 file_hash UNIQUE 约束

背景：
- 旧版 schema 中 file_hash 有 UNIQUE 约束
- 新版 schema (v0.3.3) 移除了该约束，允许同一 vault 内不同路径的相同内容文件

使用方法：
  python scripts/migrate_remove_file_hash_unique.py

注意：
- 此迁移会重建 files 表，但保留所有数据
- 如果有大量数据，可能需要几分钟
"""

import sqlite3
import sys
import time
from pathlib import Path

# 添加项目根目录到 sys.path
script_dir = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(script_dir))

from config import load_config
from utils.logger import setup_logger

logger = setup_logger(level="INFO")


def check_old_schema(conn: sqlite3.Connection) -> bool:
    """检查是否存在旧的 file_hash UNIQUE 约束"""
    cursor = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='files'")
    row = cursor.fetchone()
    if not row:
        return False

    schema = row[0]
    # 检查是否有 file_hash UNIQUE 约束
    return "file_hash TEXT NOT NULL UNIQUE" in schema or "file_hash UNIQUE" in schema


def migrate(conn: sqlite3.Connection):
    """执行迁移：重建 files 表"""
    logger.info("开始迁移...")

    # 1. 创建临时表（新版 schema，无 file_hash UNIQUE）
    logger.info("步骤 1/5: 创建临时表...")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS files_new (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vault_name TEXT NOT NULL,
            file_path TEXT NOT NULL,
            absolute_path TEXT NOT NULL,
            file_hash TEXT NOT NULL,
            file_size INTEGER,
            mtime INTEGER,
            is_deleted INTEGER DEFAULT 0,
            created_at INTEGER DEFAULT (strftime('%s', 'now')),
            updated_at INTEGER DEFAULT (strftime('%s', 'now')),
            UNIQUE(vault_name, file_path)
        )
    """)

    # 2. 复制数据到临时表
    logger.info("步骤 2/5: 复制数据到临时表...")
    conn.execute("""
        INSERT INTO files_new (id, vault_name, file_path, absolute_path, file_hash, file_size, mtime, is_deleted, created_at, updated_at)
        SELECT id, vault_name, file_path, absolute_path, file_hash, file_size, mtime, is_deleted, created_at, updated_at
        FROM files
    """)

    # 3. 删除旧表
    logger.info("步骤 3/5: 删除旧表...")
    conn.execute("DROP TABLE files")

    # 4. 重命名临时表
    logger.info("步骤 4/5: 重命名临时表...")
    conn.execute("ALTER TABLE files_new RENAME TO files")

    # 5. 重建索引
    logger.info("步骤 5/5: 重建索引...")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_files_hash ON files(file_hash)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_files_vault ON files(vault_name)")

    conn.commit()
    logger.success("✅ 迁移完成！")


def main():
    try:
        config = load_config(str(script_dir / "config.yaml"))
        db_path = config.db_path

        if not Path(db_path).exists():
            logger.error(f"❌ 数据库文件不存在: {db_path}")
            return 1

        logger.info(f"📂 数据库路径: {db_path}")

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row

        # 检查是否需要迁移
        if not check_old_schema(conn):
            logger.info("✅ 数据库已是新版 schema，无需迁移")
            conn.close()
            return 0

        logger.warning("⚠️ 检测到旧版 schema，需要迁移")

        # 统计数据量
        cursor = conn.execute("SELECT COUNT(*) FROM files")
        file_count = cursor.fetchone()[0]
        logger.info(f"📊 当前文件记录数: {file_count}")

        if file_count > 0:
            # 执行迁移
            start_time = time.time()
            migrate(conn)
            elapsed = time.time() - start_time
            logger.info(f"⏱️ 迁移耗时: {elapsed:.2f}s")

            # 验证迁移结果
            cursor = conn.execute("SELECT COUNT(*) FROM files")
            new_count = cursor.fetchone()[0]
            logger.info(f"✅ 迁移后文件记录数: {new_count}")

            if new_count != file_count:
                logger.error(f"❌ 数据丢失！原 {file_count} 条，现 {new_count} 条")
                return 1

        conn.close()
        logger.success("🎉 迁移成功！现在可以重新运行 build_index.py")
        return 0

    except Exception as e:
        logger.error(f"❌ 迁移失败: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
