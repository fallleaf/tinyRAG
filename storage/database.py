#!/usr/bin/env python3
"""
storage/database.py - SQLite 核心数据库管理器
修复清单:
1. ✅ 修复 def init -> def __init__ (实例化崩溃)
2. ✅ 移除 executescript 的无效 BEGIN/COMMIT 包裹 (解决 cannot commit 报错)
3. ✅ upsert_file 移除自提交 commit，将事务控制权交还调用方 (支持 scan_engine 批量事务)
4. ✅ 向量维度参数化，不再硬编码 512
5. ✅ 开启外键约束 (ON DELETE CASCADE)
"""

import os
import sqlite3

from utils.logger import logger


class DatabaseManager:
    def __init__(self, db_path: str, vec_dimension: int = 512):
        """
        :param db_path: SQLite 数据库文件路径
        :param vec_dimension: 向量模型维度 (默认 512，由 config.py 动态传入)
        """
        self.db_path = db_path
        self.vec_dimension = vec_dimension
        self.conn: sqlite3.Connection | None = None
        self.vec_support = False
        self._init_db()

    def _init_db(self):
        """初始化连接、PRAGMA、加载 Schema 与扩展"""
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            self.conn.execute("PRAGMA encoding = 'UTF-8'")
            self.conn.execute("PRAGMA journal_mode = WAL")
            self.conn.execute("PRAGMA busy_timeout = 5000")
            self.conn.execute("PRAGMA foreign_keys = ON")  # 启用外键级联删除

            # 1. 加载 Schema
            schema_path = os.path.join(
                os.path.dirname(__file__), "..", "schema_v0.3.2.sql"
            )
            if os.path.exists(schema_path):
                with open(schema_path, encoding="utf-8") as f:
                    self.conn.executescript(f.read())
                logger.info("✅ 数据库 Schema 加载成功 (v0.3.2)")
            else:
                logger.warning("⚠️ Schema 文件未找到，尝试创建基础表")
                self._create_basic_tables()

            # 2. 初始化 sqlite-vec 扩展
            try:
                import sqlite_vec

                sqlite_vec.load(self.conn)
                self.vec_support = True
                logger.info("✅ sqlite-vec 扩展加载成功")

                self.conn.execute(f"""
                    CREATE VIRTUAL TABLE IF NOT EXISTS vectors USING vec0(
                        chunk_id INTEGER PRIMARY KEY,
                        embedding float[{self.vec_dimension}]
                    )
                """)
                logger.info(f"✅ 向量表 (vectors) 创建成功 (dim={self.vec_dimension})")
                self.conn.commit()  # 提交向量表创建
            except Exception as e:
                self.vec_support = False
                logger.warning(f"⚠️ sqlite-vec 加载失败，系统将降级为 FTS5 模式: {e}")

        except Exception as e:
            if self.conn:
                import contextlib

                with contextlib.suppress(Exception):
                    self.conn.rollback()
            logger.critical(f"❌ 数据库初始化失败：{e}")
            raise

    def _create_basic_tables(self):
        """降级方案：无 Schema 文件时手动建表"""
        self.conn.executescript("""
            PRAGMA encoding = "UTF-8";
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vault_name TEXT NOT NULL,
                file_path TEXT NOT NULL,
                absolute_path TEXT NOT NULL,
                file_hash TEXT NOT NULL UNIQUE,
                file_size INTEGER,
                mtime INTEGER,
                is_deleted INTEGER DEFAULT 0,
                created_at INTEGER DEFAULT (strftime('%s', 'now')),
                updated_at INTEGER DEFAULT (strftime('%s', 'now')),
                UNIQUE(vault_name, file_path)
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                content_type TEXT NOT NULL,
                section_title TEXT,
                section_path TEXT,
                start_pos INTEGER NOT NULL,
                end_pos INTEGER NOT NULL,
                confidence_path_weight REAL DEFAULT 1.0,
                confidence_type_weight REAL DEFAULT 1.0,
                confidence_final_weight REAL DEFAULT 1.0,
                metadata TEXT,
                is_deleted INTEGER DEFAULT 0,
                created_at INTEGER DEFAULT (strftime('%s', 'now')),
                updated_at INTEGER DEFAULT (strftime('%s', 'now')),
                FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS fts5_index USING fts5(content);
            CREATE TABLE IF NOT EXISTS index_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at INTEGER DEFAULT (strftime('%s', 'now'))
            );
        """)
        logger.info("✅ 基础表创建成功 (降级模式)")

    def find_file_by_hash(
        self, file_hash: str, include_deleted: bool = False
    ) -> dict | None:
        """根据文件哈希查找文件记录"""
        try:
            sql = "SELECT id, vault_name, file_path, absolute_path, file_hash, is_deleted FROM files WHERE file_hash = ?"
            if not include_deleted:
                sql += " AND is_deleted = 0"
            cursor = self.conn.execute(sql, (file_hash,))
            row = cursor.fetchone()
            return dict(row) if row else None
        except Exception as e:
            logger.error(f"❌ 查找文件失败：{e}")
            return None

    def upsert_file(self, file_meta: dict) -> int:
        """
        插入或更新文件记录 (UPSERT)
        ⚠️ 重要：此方法不执行 commit，事务控制权交由调用方 (如 scan_engine.process_report)
        """
        try:
            cursor = self.conn.execute(
                """
                INSERT INTO files (vault_name, file_path, absolute_path, file_hash, file_size, mtime, is_deleted)
                VALUES (?, ?, ?, ?, ?, ?, 0)
                ON CONFLICT(vault_name, file_path) DO UPDATE SET
                    file_hash = excluded.file_hash,
                    file_size = excluded.file_size,
                    mtime = excluded.mtime,
                    updated_at = strftime('%s', 'now')
                RETURNING id
            """,
                (
                    file_meta["vault_name"],
                    file_meta["file_path"],
                    file_meta["absolute_path"],
                    file_meta["file_hash"],
                    file_meta["file_size"],
                    file_meta["mtime"],
                ),
            )
            row = cursor.fetchone()
            return row["id"] if row else -1
        except Exception as e:
            logger.error(f"❌ 插入/更新文件失败：{e}")
            return -1

    def close(self):
        """安全关闭数据库连接"""
        if self.conn:
            self.conn.close()
            self.conn = None
            logger.info("✅ 数据库连接已关闭")


__all__ = ["DatabaseManager"]
