#!/usr/bin/env python3
"""
数据库迁移脚本：移除 files.file_hash 的 UNIQUE 约束

SQLite 不支持 ALTER TABLE DROP CONSTRAINT，需要通过重建表的方式迁移。

使用方法:
    python scripts/migrate_remove_file_hash_unique.py /path/to/tinyrag.db

注意：
1. 执行前请备份数据库
2. 执行时确保没有其他进程在使用数据库
"""

import argparse
import os
import shutil
import sqlite3
import time
from datetime import datetime


def check_has_unique_constraint(conn: sqlite3.Connection) -> bool:
    """检查 files 表是否有 file_hash UNIQUE 约束"""
    cursor = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='files'")
    row = cursor.fetchone()
    if not row:
        return False
    
    sql = row[0].upper()
    # 检查是否有 file_hash UNIQUE 约束
    return "FILE_HASH TEXT UNIQUE" in sql or "UNIQUE (FILE_HASH)" in sql


def migrate_database(db_path: str, backup: bool = True) -> bool:
    """
    迁移数据库，移除 file_hash UNIQUE 约束
    
    :param db_path: 数据库文件路径
    :param backup: 是否创建备份
    :return: 迁移是否成功
    """
    print(f"📂 数据库路径: {db_path}")
    
    if not os.path.exists(db_path):
        print(f"❌ 数据库文件不存在: {db_path}")
        return False
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    # 检查是否需要迁移
    if not check_has_unique_constraint(conn):
        print("✅ 数据库已经是新版 schema，无需迁移")
        conn.close()
        return True
    
    print("⚠️ 检测到旧版 schema，file_hash 有 UNIQUE 约束")
    
    # 创建备份
    if backup:
        backup_path = f"{db_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(db_path, backup_path)
        print(f"📦 已创建备份: {backup_path}")
    
    try:
        print("🚀 开始迁移...")
        
        # 在事务中执行迁移
        cursor = conn.cursor()
        
        # 1. 创建新表（无 UNIQUE 约束）
        print("  1️⃣ 创建新表结构...")
        cursor.execute("""
            CREATE TABLE files_new (
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
        
        # 2. 复制数据
        print("  2️⃣ 复制数据...")
        cursor.execute("""
            INSERT INTO files_new 
            SELECT id, vault_name, file_path, absolute_path, file_hash, 
                   file_size, mtime, is_deleted, created_at, updated_at
            FROM files
        """)
        
        # 3. 删除旧表
        print("  3️⃣ 删除旧表...")
        cursor.execute("DROP TABLE files")
        
        # 4. 重命名新表
        print("  4️⃣ 重命名新表...")
        cursor.execute("ALTER TABLE files_new RENAME TO files")
        
        # 5. 重建索引
        print("  5️⃣ 重建索引...")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_files_hash ON files(file_hash)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_files_vault ON files(vault_name)")
        
        # 提交事务
        conn.commit()
        
        print("✅ 迁移完成！file_hash UNIQUE 约束已移除")
        
        # 验证
        if check_has_unique_constraint(conn):
            print("❌ 验证失败：约束仍然存在")
            return False
        
        print("✅ 验证通过：约束已成功移除")
        return True
        
    except Exception as e:
        conn.rollback()
        print(f"❌ 迁移失败: {e}")
        print("💡 请从备份恢复数据库")
        return False
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="移除 files.file_hash UNIQUE 约束的数据库迁移脚本"
    )
    parser.add_argument(
        "db_path",
        help="SQLite 数据库文件路径"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="跳过备份步骤（不推荐）"
    )
    
    args = parser.parse_args()
    
    success = migrate_database(args.db_path, backup=not args.no_backup)
    
    if success:
        print("\n🎉 迁移成功！现在可以使用 tinyRAG v1.1.2 了")
    else:
        print("\n💥 迁移失败，请检查错误信息")
        exit(1)


if __name__ == "__main__":
    main()
