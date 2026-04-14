#!/usr/bin/env python3
"""
storage/cache.py - 持久化查询缓存 (原子 LRU + TTL)
修复: 竞态条件、频繁 Commit、被动清理、死锁问题
"""

import json
import os
import random
import sqlite3
import threading
import time
from typing import Any

from utils.logger import logger


class QueryCache:
    def __init__(
        self,
        db_path: str = "./data/cache.db",
        ttl_seconds: int = 3600,
        max_entries: int = 1000,
    ):
        self.db_path = db_path
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS query_cache (
                cache_key TEXT PRIMARY KEY, result_data TEXT NOT NULL,
                created_at REAL NOT NULL, last_accessed REAL NOT NULL, hit_count INTEGER DEFAULT 1
            )
        """)
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_last_accessed ON query_cache(last_accessed)")
        self._conn.commit()
        logger.info(f"✅ 查询缓存初始化：{self.db_path} (TTL={self.ttl_seconds}s, Max={self.max_entries})")

    def get(self, cache_key: str) -> Any | None:
        with self._lock:
            if not self._conn:
                return None
            cursor = self._conn.execute(
                "SELECT result_data, created_at FROM query_cache WHERE cache_key = ?",
                (cache_key,),
            )
            row = cursor.fetchone()
            if not row:
                return None

            if time.time() - row["created_at"] > self.ttl_seconds:
                self._conn.execute("DELETE FROM query_cache WHERE cache_key = ?", (cache_key,))
                self._conn.commit()
                return None

            self._conn.execute(
                "UPDATE query_cache SET last_accessed = ?, hit_count = hit_count + 1 WHERE cache_key = ?",
                (time.time(), cache_key),
            )
            self._conn.commit()

            # ✅ 概率性清理过期数据 (~10% 触发，基于随机数避免时间戳聚集)
            # 修复 C1：调用无锁私有方法，避免死锁
            if random.random() < 0.1:
                self._cleanup_expired_unlocked()

            try:
                return json.loads(row["result_data"])
            except json.JSONDecodeError:
                # 修复 C3：内联 SQL，避免调用 delete() 造成死锁
                self._conn.execute("DELETE FROM query_cache WHERE cache_key = ?", (cache_key,))
                self._conn.commit()
                return None

    def set(self, cache_key: str, data: Any) -> bool:
        with self._lock:
            if not self._conn:
                return False
            try:
                result_data = json.dumps(data, ensure_ascii=False)
            except TypeError:
                return False

            now = time.time()
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                cursor = self._conn.execute("SELECT COUNT(*) as cnt FROM query_cache")
                if cursor.fetchone()["cnt"] >= self.max_entries:
                    self._conn.execute(
                        "DELETE FROM query_cache WHERE cache_key = (SELECT cache_key FROM query_cache ORDER BY last_accessed ASC LIMIT 1)"
                    )

                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO query_cache (cache_key, result_data, created_at, last_accessed, hit_count)
                    VALUES (?, ?, ?, ?, 1)
                    """,
                    (cache_key, result_data, now, now),
                )
                self._conn.commit()
                return True
            except Exception as e:
                self._conn.rollback()
                logger.error(f"❌ 缓存写入失败: {e}")
                return False

    def delete(self, cache_key: str) -> bool:
        with self._lock:
            if not self._conn:
                return False
            self._conn.execute("DELETE FROM query_cache WHERE cache_key = ?", (cache_key,))
            self._conn.commit()
            return True

    def clear(self) -> int:
        with self._lock:
            if not self._conn:
                return 0
            cursor = self._conn.execute("SELECT COUNT(*) FROM query_cache")
            count = cursor.fetchone()[0]
            self._conn.execute("DELETE FROM query_cache")
            self._conn.commit()
            return count

    def _cleanup_expired_unlocked(self) -> int:
        """
        不加锁的内部清理方法，由调用方负责持锁

        修复 C1 死锁：此方法不获取锁，供 get() 等已在锁内的方法调用
        """
        if not self._conn:
            return 0
        cutoff = time.time() - self.ttl_seconds
        self._conn.execute("DELETE FROM query_cache WHERE created_at < ?", (cutoff,))
        cursor = self._conn.execute("SELECT changes()")
        count = cursor.fetchone()[0]
        self._conn.commit()
        if count:
            logger.debug(f"🗑️ 清理过期缓存: {count} 条")
        return count

    def cleanup_expired(self) -> int:
        """
        公开方法，自动加锁（供外部调用）

        保持原有 API 不变，外部调用者无需修改
        """
        with self._lock:
            return self._cleanup_expired_unlocked()

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None


_cache_instance: QueryCache | None = None
_cache_lock = threading.Lock()


def get_cache(db_path: str = "./data/cache.db", ttl_seconds: int = 3600, max_entries: int = 1000) -> QueryCache:
    """获取全局查询缓存实例

    注意：单例模式，第一次调用后参数被缓存，后续调用参数将被忽略。

    Args:
        db_path: 缓存数据库路径
        ttl_seconds: 默认过期时间（秒）
        max_entries: 最大缓存条目数

    Returns:
        QueryCache 实例
    """
    global _cache_instance
    # 修复 M3: 使用锁保护单例创建
    with _cache_lock:
        if _cache_instance is None:
            _cache_instance = QueryCache(db_path, ttl_seconds, max_entries)
        else:
            # 修复 M3: 添加警告日志，提示参数被忽略
            logger.warning(
                f"⚠️ get_cache() 已存在实例，参数将被忽略。"
                f"当前实例: db_path={_cache_instance.db_path}, "
                f"ttl_seconds={_cache_instance.ttl_seconds}, "
                f"max_entries={_cache_instance.max_entries}"
            )
    return _cache_instance


__all__ = ["QueryCache", "get_cache"]
