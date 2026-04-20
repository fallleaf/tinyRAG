#!/usr/bin/env python3
"""storage/database.py - SQLite 核心数据库管理器 (v2.1)"""

import os
import sqlite3
from utils.logger import logger

_FALLBACK_SCHEMA = """
PRAGMA encoding = "UTF-8";
PRAGMA journal_mode = WAL;
PRAGMA busy_timeout = 5000;
CREATE TABLE IF NOT EXISTS files (id INTEGER PRIMARY KEY AUTOINCREMENT, vault_name TEXT NOT NULL, file_path TEXT NOT NULL, absolute_path TEXT NOT NULL, file_hash TEXT NOT NULL, file_size INTEGER, mtime INTEGER, is_deleted INTEGER DEFAULT 0, created_at INTEGER DEFAULT (strftime('%s', 'now')), updated_at INTEGER DEFAULT (strftime('%s', 'now')), UNIQUE(vault_name, file_path));
CREATE TABLE IF NOT EXISTS chunks (id INTEGER PRIMARY KEY AUTOINCREMENT, file_id INTEGER NOT NULL, chunk_index INTEGER NOT NULL, content TEXT NOT NULL, content_type TEXT NOT NULL, section_title TEXT, section_path TEXT, start_pos INTEGER NOT NULL, end_pos INTEGER NOT NULL, confidence_path_weight REAL DEFAULT 1.0, confidence_type_weight REAL DEFAULT 1.0, confidence_final_weight REAL DEFAULT 1.0, metadata TEXT, confidence_json TEXT, is_deleted INTEGER DEFAULT 0, created_at INTEGER DEFAULT (strftime('%s', 'now')), updated_at INTEGER DEFAULT (strftime('%s', 'now')), FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE);
CREATE VIRTUAL TABLE IF NOT EXISTS fts5_index USING fts5(content);
CREATE TABLE IF NOT EXISTS index_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at INTEGER DEFAULT (strftime('%s', 'now')));
"""


class DatabaseManager:
    def __init__(self, db_path: str, vec_dimension: int = 512):
        self.db_path = db_path
        self.vec_dimension = vec_dimension
        self.conn: sqlite3.Connection | None = None
        self.vec_support = False
        self._init_db()

    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            self.conn.execute("PRAGMA encoding = 'UTF-8'")
            self.conn.execute("PRAGMA journal_mode = WAL")
            self.conn.execute("PRAGMA busy_timeout = 5000")
            self.conn.execute("PRAGMA foreign_keys = ON")
            self.conn.enable_load_extension(True)
            schema_path = os.path.join(os.path.dirname(__file__), "..", "schema_v0.3.3.sql")
            if os.path.exists(schema_path):
                with open(schema_path, encoding="utf-8") as f:
                    self.conn.executescript(f.read())
                logger.info("✅ 数据库 Schema 加载成功 (v0.3.3)")
            else:
                self.conn.executescript(_FALLBACK_SCHEMA)
                logger.info("✅ 基础表创建成功 (降级模式)")
            try:
                import sqlite_vec

                self.conn.load_extension(sqlite_vec.loadable_path())
                self.conn.enable_load_extension(False)
                self.vec_support = True
                self.conn.execute(
                    f"CREATE VIRTUAL TABLE IF NOT EXISTS vectors USING vec0(chunk_id INTEGER PRIMARY KEY, embedding float[{self.vec_dimension}])"
                )
                logger.info(f"✅ 向量表创建成功 (dim={self.vec_dimension})")
                self.conn.commit()
            except Exception as e:
                self.vec_support = False
                logger.warning(f"⚠️ sqlite-vec 加载失败，降级为 FTS5 模式: {e}")
        except Exception as e:
            if self.conn:
                import contextlib

                with contextlib.suppress(Exception):
                    self.conn.rollback()
            logger.critical(f"❌ 数据库初始化失败：{e}")
            raise

    # P2 优化：批量插入/重建索引时临时调整 PRAGMA
    def begin_bulk_insert(self):
        self.conn.execute("PRAGMA synchronous = OFF")
        self.conn.execute("PRAGMA journal_size_limit = 16777216")
        self.conn.execute("PRAGMA cache_size = -64000")
        self.conn.execute("BEGIN TRANSACTION")

    def end_bulk_insert(self, commit: bool = True):
        try:
            if commit:
                self.conn.commit()
            else:
                self.conn.rollback()
        except Exception as e:
            logger.error(f"事务提交失败: {e}")
            if not commit:
                self.conn.rollback()
        self.conn.execute("PRAGMA synchronous = NORMAL")
        self.conn.execute("PRAGMA cache_size = -16000")

    def find_file_by_hash(
        self, file_hash: str, include_deleted: bool = False, vault_name: str | None = None
    ) -> dict | None:
        try:
            sql = (
                "SELECT id, vault_name, file_path, absolute_path, file_hash, is_deleted FROM files WHERE file_hash = ?"
            )
            params: list = [file_hash]
            if vault_name:
                sql += " AND vault_name = ?"
                params.append(vault_name)
            if not include_deleted:
                sql += " AND is_deleted = 0"
            row = self.conn.execute(sql, params).fetchone()
            return dict(row) if row else None
        except Exception as e:
            logger.error(f"❌ 查找文件失败：{e}")
            return None

    def upsert_file(self, file_meta: dict) -> int:
        try:
            cursor = self.conn.execute(
                "INSERT INTO files (vault_name, file_path, absolute_path, file_hash, file_size, mtime, is_deleted) VALUES (?, ?, ?, ?, ?, ?, 0) ON CONFLICT(vault_name, file_path) DO UPDATE SET file_hash = excluded.file_hash, file_size = excluded.file_size, mtime = excluded.mtime, updated_at = strftime('%s', 'now') RETURNING id",
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
        except sqlite3.IntegrityError as e:
            if "file_hash" in str(e):
                logger.warning(f"⚠️ 数据库 schema 版本过旧，请运行：python scripts/migrate_remove_file_hash_unique.py")
                return -1
            raise
        except Exception as e:
            logger.error(f"❌ 插入/更新文件失败：{e}")
            return -1

    def search_vectors(self, query_vector: list[float], limit: int = 10) -> list[tuple[int, float]]:
        if not self.vec_support or not query_vector:
            return []
        try:
            import array

            query_blob = array.array("f", query_vector).tobytes()
            cursor = self.conn.execute(
                "SELECT chunk_id, distance FROM vectors WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
                (query_blob, limit),
            )
            return [(row[0], 1.0 / (1.0 + row[1])) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"❌ 向量搜索失败: {e}")
            return []

    def escape_fts5_query(self, query: str) -> str:
        """转义 FTS5 查询，对中文使用 jieba 分词"""
        import re
        import jieba

        # 检测是否包含中文字符
        has_chinese = bool(re.search(r"[\u4e00-\u9fff]", query))

        if has_chinese:
            # 中文：使用 jieba 分词
            terms = list(jieba.cut(query))
            escaped_terms = [term.replace('"', '""') for term in terms if term.strip() and len(term) > 1]
            return " OR ".join(escaped_terms) if escaped_terms else query.replace('"', '""')
        else:
            # 英文：使用前缀搜索
            terms = query.split()
            escaped_terms = [term.replace('"', '""') + "*" for term in terms if term.strip()]
            return " OR ".join(escaped_terms)

    def search_fts(self, keywords: str, limit: int = 10) -> list[tuple[int, float]]:
        if not keywords or not keywords.strip():
            return []
        try:
            escaped_query = self.escape_fts5_query(keywords)
            cursor = self.conn.execute(
                "SELECT rowid, bm25(fts5_index) as score FROM fts5_index WHERE fts5_index MATCH ? ORDER BY score LIMIT ?",
                (escaped_query, limit),
            )
            # FTS5 BM25 返回负数分数，取绝对值并放大 100000 倍以匹配向量分数范围
            return [(row[0], abs(row[1]) * 100000) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"❌ FTS5 搜索失败: {e}")
            return []

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None
            logger.info("✅ 数据库连接已关闭")


__all__ = ["DatabaseManager"]
