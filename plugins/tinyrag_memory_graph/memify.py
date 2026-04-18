#!/usr/bin/env python3
"""
memify.py - 记忆代谢模块

实现 FR-3 记忆代谢与自优化需求，包括：
- 权重衰减
- 路径强化
- 原则提炼
- 异步调度
"""
import asyncio
import contextlib
import sqlite3
import time
from dataclasses import dataclass

from plugins.tinyrag_memory_graph.config import MemoryGraphConfig
from plugins.tinyrag_memory_graph.storage import GraphStorage


@dataclass
class MemifyStats:
    """代谢统计"""
    relations_decayed: int = 0
    relations_marked_stale: int = 0
    relations_boosted: int = 0
    principles_generated: int = 0
    principles_approved: int = 0
    processing_time_ms: float = 0.0


class MemifyEngine:
    """
    记忆代谢引擎（FR-3）

    负责知识图谱的自我优化和代谢清理。
    """

    def __init__(self, db_conn: sqlite3.Connection, config: MemoryGraphConfig):
        self.db = db_conn
        self.config = config
        self.memify_config = config.memify
        self.storage = GraphStorage(db_conn, config)

        self._running = False
        self._task: asyncio.Task | None = None
        self._last_run: int = 0

    async def start(self):
        """启动定时代谢任务"""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self):
        """停止定时任务"""
        self._running = False
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def _run_loop(self):
        """定时运行循环"""
        while self._running:
            try:
                await asyncio.sleep(self.memify_config.interval_sec)
                if self._running:
                    await self.run_memify()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[MemifyEngine] Error in run loop: {e}")
                await asyncio.sleep(60)  # 错误后等待

    async def run_memify(self) -> MemifyStats:
        """
        执行一轮记忆代谢

        Returns:
            代谢统计
        """
        start_time = time.time()
        stats = MemifyStats()

        try:
            # 1. 权重衰减（FR-3.1）
            stats.relations_decayed = await self._decay_weights()
            self.db.commit()

            # 2. 标记陈旧关系（FR-3.1）
            stats.relations_marked_stale = await self._mark_stale()
            self.db.commit()

            # 3. 路径强化（FR-3.2）- 基于最近的访问记录
            stats.relations_boosted = await self._boost_recent_paths()
            self.db.commit()

            # 4. 原则提炼（FR-3.3）
            stats.principles_generated = await self._extract_principles()
            self.db.commit()

            # 更新元数据
            self._last_run = int(time.time())
            self._update_metadata()

        except Exception as e:
            print(f"[MemifyEngine] Memify error: {e}")
            self.db.rollback()

        stats.processing_time_ms = (time.time() - start_time) * 1000
        return stats

    async def _decay_weights(self) -> int:
        """
        权重衰减（FR-3.1）

        last_hit > 30天 → weight *= 0.9
        """
        return self.storage.decay_old_relations(
            days=self.memify_config.decay_days,
            factor=self.memify_config.decay_factor,
            min_weight=0.1,
        )

    async def _mark_stale(self) -> int:
        """
        标记陈旧关系（FR-3.1）

        access_count < 2 且 age > 60天 标记 stale
        """
        cutoff_time = int(time.time()) - self.memify_config.stale_age_days * 86400

        try:
            # 查找陈旧关系
            cursor = self.db.execute(
                """SELECT src_chunk_id, tgt_chunk_id, rel_type
                   FROM relations
                   WHERE access_count < ?
                     AND created_at < ?
                     AND weight < 0.3""",
                (self.memify_config.stale_access_threshold, cutoff_time)
            )

            stale_relations = cursor.fetchall()

            # 标记为陈旧（通过降低权重）
            for rel in stale_relations:
                self.db.execute(
                    """UPDATE relations SET weight = 0.1
                       WHERE src_chunk_id = ? AND tgt_chunk_id = ? AND rel_type = ?""",
                    (rel[0], rel[1], rel[2])
                )

            return len(stale_relations)

        except Exception as e:
            print(f"[MemifyEngine] Mark stale error: {e}")
            return 0

    async def _boost_recent_paths(self) -> int:
        """
        路径强化（FR-3.2）

        检索命中边 → weight = MIN(weight + 0.05, 1.0)
        """
        try:
            # 查找最近被访问的关系
            recent_threshold = int(time.time()) - 3600  # 最近 1 小时

            cursor = self.db.execute(
                """SELECT src_chunk_id, tgt_chunk_id, rel_type
                   FROM relations
                   WHERE last_hit > ?""",
                (recent_threshold,)
            )

            recent_relations = cursor.fetchall()

            for rel in recent_relations:
                self.storage.boost_relation(
                    rel[0], rel[1], rel[2],
                    amount=self.memify_config.path_boost,
                    max_weight=1.0,
                )

            return len(recent_relations)

        except Exception as e:
            print(f"[MemifyEngine] Boost paths error: {e}")
            return 0

    async def _extract_principles(self) -> int:
        """
        原则提炼（FR-3.3）

        多标签共现 ≥3 次 + access_count ≥5 → 自动生成原则
        """
        try:
            # 获取高频标签共现
            co_occurrences = self.storage.get_tag_co_occurrences(
                min_count=self.memify_config.principle_co_occurrence
            )

            principles_created = 0

            for co in co_occurrences:
                tag1, tag2 = co["tag1"], co["tag2"]

                # 查找具有这两个标签的高访问 Chunk
                cursor = self.db.execute(
                    """SELECT c.id, c.content, c.inherited_meta
                       FROM chunks c
                       WHERE c.access_count >= ?
                         AND c.inherited_meta LIKE ?
                         AND c.inherited_meta LIKE ?
                       LIMIT 5""",
                    (self.memify_config.principle_min_access,
                     f'%{tag1}%', f'%{tag2}%')
                )

                matching_chunks = cursor.fetchall()

                for chunk in matching_chunks:
                    chunk_id = chunk[0]
                    content = chunk[1]

                    # 检查是否已存在原则
                    existing = self.db.execute(
                        "SELECT id FROM principles WHERE chunk_id = ?",
                        (chunk_id,)
                    ).fetchone()

                    if existing:
                        continue

                    # 生成原则
                    principle_text = self._generate_principle_text(
                        content, tag1, tag2
                    )

                    if principle_text:
                        self.db.execute(
                            """INSERT INTO principles
                               (chunk_id, principle_text, tags, is_approved)
                               VALUES (?, ?, ?, 0)""",
                            (chunk_id, principle_text,
                             f'["{tag1}", "{tag2}"]')
                        )
                        principles_created += 1

            return principles_created

        except Exception as e:
            print(f"[MemifyEngine] Extract principles error: {e}")
            return 0

    def _generate_principle_text(self, content: str, tag1: str, tag2: str) -> str:
        """
        从内容生成原则文本

        简单实现：提取关键句子。
        完整实现可调用 LLM。
        """
        # 分句
        sentences = content.replace('。', '。\n').replace('！', '！\n').replace('？', '？\n').split('\n')
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

        if not sentences:
            return ""

        # 选择包含标签相关词的句子
        for sent in sentences:
            if tag1 in sent or tag2 in sent:
                return sent[:200]

        # 否则返回第一句
        return sentences[0][:200] if sentences else ""

    def _update_metadata(self):
        """更新元数据"""
        try:
            self.db.execute(
                """INSERT OR REPLACE INTO index_metadata (key, value, updated_at)
                   VALUES ('memory_graph_last_memify', ?, strftime('%s', 'now'))""",
                (str(self._last_run),)
            )
            self.db.commit()
        except Exception:
            pass

    def touch_relation(self, src_chunk_id: int, tgt_chunk_id: int, rel_type: str):
        """
        更新关系访问记录（用于 on_response 钩子）
        """
        self.storage.touch_relation(src_chunk_id, tgt_chunk_id, rel_type)
        self.db.commit()

    def record_search_hit(self, chunk_ids: list[int]):
        """
        记录检索命中（用于统计标签共现）
        """
        # 获取这些 Chunk 的标签
        for cid in chunk_ids:
            try:
                row = self.db.execute(
                    "SELECT inherited_meta FROM chunks WHERE id = ?",
                    (cid,)
                ).fetchone()

                if row and row[0]:
                    import json
                    meta = json.loads(row[0])
                    tags = meta.get("tags", [])

                    # 记录标签共现
                    for i, t1 in enumerate(tags):
                        for t2 in tags[i + 1:]:
                            self.storage.record_tag_co_occurrence(t1, t2)

            except Exception:
                pass

        self.db.commit()

    def get_stats(self) -> dict:
        """获取代谢统计"""
        return {
            "last_run": self._last_run,
            "running": self._running,
            "interval_sec": self.memify_config.interval_sec,
        }


class PrincipleManager:
    """
    原则管理器

    管理自动生成的原则，支持人工审核。
    """

    def __init__(self, db_conn: sqlite3.Connection):
        self.db = db_conn

    def get_pending_principles(self, limit: int = 20) -> list[dict]:
        """获取待审核原则"""
        cursor = self.db.execute(
            """SELECT p.*, c.content as chunk_content
               FROM principles p
               JOIN chunks c ON p.chunk_id = c.id
               WHERE p.is_approved = 0
               ORDER BY p.created_at DESC
               LIMIT ?""",
            (limit,)
        )
        return [dict(r) for r in cursor.fetchall()]

    def approve_principle(self, principle_id: int) -> bool:
        """批准原则"""
        try:
            self.db.execute(
                "UPDATE principles SET is_approved = 1 WHERE id = ?",
                (principle_id,)
            )
            self.db.commit()
            return True
        except Exception:
            return False

    def reject_principle(self, principle_id: int) -> bool:
        """拒绝原则"""
        try:
            self.db.execute(
                "DELETE FROM principles WHERE id = ?",
                (principle_id,)
            )
            self.db.commit()
            return True
        except Exception:
            return False

    def get_approved_principles(self, tags: list[str] | None = None) -> list[dict]:
        """获取已批准原则（可按标签过滤）"""
        if tags:
            # 构建查询条件
            conditions = " OR ".join([f"tags LIKE '%{t}%'" for t in tags])
            query = f"""SELECT * FROM principles
                        WHERE is_approved = 1 AND ({conditions})
                        ORDER BY access_count DESC"""
        else:
            query = "SELECT * FROM principles WHERE is_approved = 1 ORDER BY access_count DESC"

        cursor = self.db.execute(query)
        return [dict(r) for r in cursor.fetchall()]


__all__ = ["MemifyEngine", "MemifyStats", "PrincipleManager"]
