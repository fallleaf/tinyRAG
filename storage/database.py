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

# 修复 L5: 将 Schema SQL 以字符串常量内嵌，避免双源维护风险
_FALLBACK_SCHEMA = """
PRAGMA encoding = "UTF-8";
PRAGMA journal_mode = WAL;
PRAGMA busy_timeout = 5000;

CREATE TABLE IF NOT EXISTS files (
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
    confidence_json TEXT,
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
"""


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

            # 启用扩展加载（sqlite-vec 需要）
            self.conn.enable_load_extension(True)

            # 1. 加载 Schema
            # 修复 L5: 优先使用 Schema 文件，否则使用内嵌降级 Schema
            schema_path = os.path.join(os.path.dirname(__file__), "..", "schema_v0.3.3.sql")
            if os.path.exists(schema_path):
                with open(schema_path, encoding="utf-8") as f:
                    self.conn.executescript(f.read())
                logger.info("✅ 数据库 Schema 加载成功 (v0.3.3)")
            else:
                logger.warning("⚠️ Schema 文件未找到，使用内嵌降级 Schema")
                self.conn.executescript(_FALLBACK_SCHEMA)
                logger.info("✅ 基础表创建成功 (降级模式)")

            # 2. 初始化 sqlite-vec 扩展
            try:
                import sqlite_vec

                self.conn.load_extension(sqlite_vec.loadable_path())
                self.conn.enable_load_extension(False)  # 加载完成后关闭，提升安全性
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

    def find_file_by_hash(
        self, file_hash: str, include_deleted: bool = False, vault_name: str | None = None
    ) -> dict | None:
        """根据文件哈希查找文件记录"""
        try:
            sql = (
                "SELECT id, vault_name, file_path, absolute_path, file_hash, is_deleted FROM files WHERE file_hash = ?"
            )
            params: list = [file_hash]
            # 修复 L4: 增加 vault_name 过滤，避免跨 vault 恢复错绑
            if vault_name:
                sql += " AND vault_name = ?"
                params.append(vault_name)
            if not include_deleted:
                sql += " AND is_deleted = 0"
            cursor = self.conn.execute(sql, params)
            row = cursor.fetchone()
            return dict(row) if row else None
        except Exception as e:
            logger.error(f"❌ 查找文件失败：{e}")
            return None

    def upsert_file(self, file_meta: dict) -> int:
        """
        插入或更新文件记录 (UPSERT)
        ⚠️ 重要：此方法不执行 commit，事务控制权交由调用方 (如 scan_engine.process_report)

        修复：处理旧版 schema 中 file_hash UNIQUE 约束冲突
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
        except sqlite3.IntegrityError as e:
            # 处理旧版 schema 的 file_hash UNIQUE 约束冲突
            if "file_hash" in str(e):
                logger.warning(f"⚠️ 跳过重复 Hash 文件（旧 schema 约束）：{file_meta['file_path']}")
                return -1
            raise
        except Exception as e:
            logger.error(f"❌ 插入/更新文件失败：{e}")
            return -1

    def search_vectors(self, query_vector: list[float], limit: int = 10) -> list[tuple[int, float]]:
        """
        向量近邻搜索 (sqlite-vec KNN)
        :return: [(chunk_id, similarity_score), ...]  similarity ∈ [0, 1]
        """
        if not self.vec_support or not query_vector:
            return []
        try:
            import array

            query_blob = array.array("f", query_vector).tobytes()
            cursor = self.conn.execute(
                """
                SELECT chunk_id, distance
                FROM vectors
                WHERE embedding MATCH ?
                ORDER BY distance
                LIMIT ?
                """,
                (query_blob, limit),
            )
            # vec0 返回的是 L2 距离，转换为余弦相似度近似: score = 1 / (1 + distance)
            return [(row[0], 1.0 / (1.0 + row[1])) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"❌ 向量搜索失败: {e}")
            return []

    def escape_fts5_query(self, query: str) -> str:
        """转义 FTS5 特殊字符，使用 OR 查询提高召回率。

        Args:
            query: 查询文本（可能包含多个关键字）

        Returns:
            转义后的 OR 查询字符串

        Example:
            输入: "极简 网络"
            输出: "\"极简\" OR \"网络\""
        """
        # FTS5 特殊字符: * ^ " ( )
        # 修复 H3: 使用短语包裹策略，彻底消除运算符注入风险
        # 改为 OR 查询：提高召回率，返回包含任一关键字的文档
        terms = query.split()
        escaped_terms = ['"' + term.replace('"', '""') + '"' for term in terms if term.strip()]
        return " OR ".join(escaped_terms)

    def search_fts(self, keywords: str, limit: int = 10) -> list[tuple[int, float]]:
        """
        FTS5 全文搜索 (BM25)
        :return: [(chunk_id, bm25_score), ...]  越大越相关
        """
        if not keywords or not keywords.strip():
            return []
        try:
            # 修复 H3: 转义 FTS5 特殊字符
            escaped_query = self.escape_fts5_query(keywords)
            cursor = self.conn.execute(
                """
                SELECT rowid, rank
                FROM fts5_index
                WHERE fts5_index MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (escaped_query, limit),
            )
            # FTS5 rank 为负数（绝对值越大越相关），取反使其正向
            return [(row[0], -row[1]) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"❌ FTS5 搜索失败: {e}")
            return []

    def close(self):
        """安全关闭数据库连接"""
        if self.conn:
            self.conn.close()
            self.conn = None
            logger.info("✅ 数据库连接已关闭")


__all__ = ["DatabaseManager"]
