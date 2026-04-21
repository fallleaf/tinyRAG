#!/usr/bin/env python3
"""
storage.py - 图谱存储层

封装 Memory Graph 的所有数据库操作，包括：
- Schema 迁移
- 实体/关系 CRUD
- 异步任务管理
- 图遍历查询
"""

import array
import sqlite3
import time
from pathlib import Path

from loguru import logger

from plugins.tinyrag_memory_graph.config import MemoryGraphConfig
from plugins.tinyrag_memory_graph.models import ChunkMeta, Entity, GraphBuildJob, Note, Relation


def _row_to_dict(row) -> dict:
    """将 sqlite3.Row 或 tuple 转换为 dict"""
    if row is None:
        return {}
    # 尝试 sqlite3.Row 接口
    if hasattr(row, "keys"):
        return {key: row[key] for key in row.keys()}
    # 如果是 tuple，无法转换（需要列名），返回空
    if isinstance(row, tuple):
        raise ValueError("Cannot convert tuple to dict without column names. Use conn.row_factory = sqlite3.Row")
    # 尝试字典接口
    return dict(row)


class GraphStorage:
    """
    图谱存储管理器

    负责所有与图谱相关的数据库操作。
    """

    def __init__(self, db_conn: sqlite3.Connection, config: MemoryGraphConfig):
        self.conn = db_conn
        self.config = config
        self._initialized = False
        # 设置 row_factory 以便正确获取列名
        self._original_row_factory = self.conn.row_factory
        self.conn.row_factory = sqlite3.Row

    def initialize(self):
        """初始化 Schema（执行迁移）"""
        if self._initialized:
            return

        # 1. 执行 Schema 迁移脚本
        schema_path = Path(__file__).parent / "schema_v0.3.4_plus_memory.sql"
        if schema_path.exists():
            try:
                with open(schema_path, encoding="utf-8") as f:
                    sql_script = f.read()
                self.conn.executescript(sql_script)
                logger.info(f"[GraphStorage] ✅ Schema 脚本执行成功: {schema_path}")
            except Exception as e:
                logger.error(f"[GraphStorage] ❌ Schema 脚本执行失败: {e}")
                import traceback

                traceback.print_exc()
        else:
            logger.warning(f"[GraphStorage] ⚠️ Schema 文件不存在: {schema_path}")

        # 2. 检查并添加 chunks 表扩展列
        self._ensure_chunk_columns()

        # 3. 检查并添加 entities 表 chunk_id 列
        self._ensure_entity_columns()

        self.conn.commit()
        self._initialized = True

        # 验证表是否创建成功
        cursor = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('notes', 'entities', 'relations', 'graph_build_jobs')"
        )
        tables = [row[0] for row in cursor.fetchall()]
        logger.info(f"[GraphStorage] 📊 已创建的图谱表: {tables}")

    def _ensure_chunk_columns(self):
        """确保 chunks 表有所需的扩展列"""
        # 获取现有列
        cursor = self.conn.execute("PRAGMA table_info(chunks)")
        existing_columns = {row[1] for row in cursor.fetchall()}

        # 需要添加的列
        columns_to_add = [
            ("note_id", "TEXT REFERENCES notes(note_id)"),
            ("inherited_meta", "TEXT DEFAULT '{}'"),
            ("is_representative", "INTEGER DEFAULT 0"),
            ("access_count", "INTEGER DEFAULT 0"),
            ("last_accessed", "INTEGER"),
        ]

        added_any = False
        for col_name, col_def in columns_to_add:
            if col_name not in existing_columns:
                try:
                    self.conn.execute(f"ALTER TABLE chunks ADD COLUMN {col_name} {col_def}")
                    added_any = True
                    logger.info(f"[GraphStorage] ✅ 添加列: chunks.{col_name}")
                except sqlite3.OperationalError as e:
                    # 列可能已存在，静默忽略
                    if "duplicate column" not in str(e).lower():
                        logger.warning(f"[GraphStorage] ⚠️ 添加列失败 {col_name}: {e}")
                    # else: 列已存在，忽略
                except Exception as e:
                    if "duplicate column" not in str(e).lower():
                        logger.warning(f"[GraphStorage] ⚠️ 添加列异常 {col_name}: {e}")

        if added_any:
            self.conn.commit()

    def _ensure_entity_columns(self):
        """确保 entities 表有所需的扩展列"""
        # 获取现有列
        cursor = self.conn.execute("PRAGMA table_info(entities)")
        existing_columns = {row[1] for row in cursor.fetchall()}

        # 需要添加的列
        columns_to_add = [
            ("chunk_id", "TEXT"),
        ]

        added_any = False
        for col_name, col_def in columns_to_add:
            if col_name not in existing_columns:
                try:
                    self.conn.execute(f"ALTER TABLE entities ADD COLUMN {col_name} {col_def}")
                    added_any = True
                    logger.info(f"[GraphStorage] ✅ 添加列：entities.{col_name}")
                except sqlite3.OperationalError as e:
                    # 列可能已存在，静默忽略
                    if "duplicate column" not in str(e).lower():
                        logger.warning(f"[GraphStorage] ⚠️ 添加列失败 {col_name}: {e}")
                    # else: 列已存在，忽略
                except Exception as e:
                    if "duplicate column" not in str(e).lower():
                        logger.warning(f"[GraphStorage] ⚠️ 添加列异常 {col_name}: {e}")

        if added_any:
            self.conn.commit()

    # ==================== Note 操作 ====================

    def upsert_note(self, note: Note) -> bool:
        """插入或更新文档记录"""
        try:
            self.conn.execute(
                """INSERT INTO notes (note_id, filepath, title, frontmatter_json, created_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(note_id) DO UPDATE SET
                       filepath = excluded.filepath,
                       title = excluded.title,
                       frontmatter_json = excluded.frontmatter_json,
                       updated_at = strftime('%s', 'now')""",
                note.to_db_row(),
            )
            return True
        except Exception as e:
            logger.error(f"[GraphStorage] upsert_note error: {e}")
            return False

    def get_note(self, note_id: str) -> Note | None:
        """获取文档记录"""
        row = self.conn.execute("SELECT * FROM notes WHERE note_id = ?", (note_id,)).fetchone()
        return Note.from_db_row(_row_to_dict(row)) if row else None

    def get_note_by_filepath(self, filepath: str) -> Note | None:
        """通过文件路径获取文档记录"""
        row = self.conn.execute("SELECT * FROM notes WHERE filepath = ?", (filepath,)).fetchone()
        return Note.from_db_row(_row_to_dict(row)) if row else None

    def delete_note(self, note_id: str) -> bool:
        """删除文档记录（级联删除关系）"""
        try:
            # 先删除相关关系
            self.conn.execute(
                "DELETE FROM relations WHERE src_chunk_id IN (SELECT id FROM chunks WHERE note_id = ?)", (note_id,)
            )
            self.conn.execute(
                "DELETE FROM relations WHERE tgt_chunk_id IN (SELECT id FROM chunks WHERE note_id = ?)", (note_id,)
            )
            # 删除任务
            self.conn.execute("DELETE FROM graph_build_jobs WHERE note_id = ?", (note_id,))
            # 删除文档记录
            self.conn.execute("DELETE FROM notes WHERE note_id = ?", (note_id,))
            return True
        except Exception as e:
            logger.error(f"[GraphStorage] delete_note error: {e}")
            return False

    # ==================== Entity 操作 ====================

    def upsert_entity(self, entity: Entity) -> bool:
        """插入或更新实体"""
        try:
            self.conn.execute(
                """INSERT INTO entities (id, canonical_name, type, confidence, source, chunk_id)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                       canonical_name = excluded.canonical_name,
                       type = excluded.type,
                       confidence = MAX(confidence, excluded.confidence),
                       source = excluded.source,
                       chunk_id = COALESCE(excluded.chunk_id, chunk_id),
                       updated_at = strftime('%s', 'now')""",
                entity.to_db_row(),
            )
            return True
        except Exception as e:
            # 如果是数据库关闭导致的错误，静默返回
            if "closed database" in str(e) or "Cannot operate" in str(e):
                return False
            logger.error(f"[GraphStorage] upsert_entity error: {e}")
            return False

    def upsert_entities_batch(self, entities: list[Entity]) -> int:
        """批量插入或更新实体（性能优化版）

        Args:
            entities: 实体列表

        Returns:
            成功写入的数量
        """
        if not entities:
            return 0

        success_count = 0
        try:
            # 使用 executemany 批量操作
            rows = [e.to_db_row() for e in entities]
            self.conn.executemany(
                """INSERT INTO entities (id, canonical_name, type, confidence, source, chunk_id)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                       canonical_name = excluded.canonical_name,
                       type = excluded.type,
                       confidence = MAX(confidence, excluded.confidence),
                       source = excluded.source,
                       chunk_id = COALESCE(excluded.chunk_id, chunk_id),
                       updated_at = strftime('%s', 'now')""",
                rows,
            )
            success_count = len(entities)
        except Exception as e:
            if "closed database" not in str(e) and "Cannot operate" not in str(e):
                logger.error(f"[GraphStorage] upsert_entities_batch error: {e}")
            # 降级为逐条插入
            for entity in entities:
                if self.upsert_entity(entity):
                    success_count += 1

        return success_count

    def get_entity(self, entity_id: str) -> Entity | None:
        """获取实体"""
        row = self.conn.execute("SELECT * FROM entities WHERE id = ?", (entity_id,)).fetchone()
        return Entity.from_db_row(_row_to_dict(row)) if row else None

    def get_entities_by_name(self, name: str) -> list[Entity]:
        """通过名称查找实体（返回所有匹配，包括不同 chunk_id 的）"""
        rows = self.conn.execute(
            "SELECT * FROM entities WHERE canonical_name = ?",
            (name,),
        ).fetchall()
        return [Entity.from_db_row(_row_to_dict(r)) for r in rows]

    def get_entities_by_name_with_chunk(self, name: str) -> list[Entity]:
        """通过名称查找实体（仅返回有 chunk_id 的）"""
        rows = self.conn.execute(
            """SELECT * FROM entities
               WHERE canonical_name = ? AND chunk_id IS NOT NULL""",
            (name,),
        ).fetchall()
        return [Entity.from_db_row(_row_to_dict(r)) for r in rows]

    # ==================== Relation 操作 ====================

    def upsert_relation(self, relation: Relation) -> bool:
        """插入或更新关系"""
        try:
            self.conn.execute(
                """INSERT INTO relations (src_chunk_id, tgt_chunk_id, rel_type, scope, weight, evidence_chunk_id, last_hit)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(src_chunk_id, tgt_chunk_id, rel_type) DO UPDATE SET
                       scope = excluded.scope,
                       weight = MAX(weight, excluded.weight),
                       evidence_chunk_id = COALESCE(excluded.evidence_chunk_id, evidence_chunk_id),
                       updated_at = strftime('%s', 'now')""",
                relation.to_db_row(),
            )
            return True
        except Exception as e:
            logger.error(f"[GraphStorage] upsert_relation error: {e}")
            return False

    def upsert_relations_batch(self, relations: list[Relation]) -> int:
        """批量插入或更新关系（性能优化版）

        Args:
            relations: 关系列表

        Returns:
            成功写入的数量
        """
        if not relations:
            return 0

        success_count = 0
        try:
            # 使用 executemany 批量操作
            rows = [r.to_db_row() for r in relations]
            self.conn.executemany(
                """INSERT INTO relations (src_chunk_id, tgt_chunk_id, rel_type, scope, weight, evidence_chunk_id, last_hit)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(src_chunk_id, tgt_chunk_id, rel_type) DO UPDATE SET
                       scope = excluded.scope,
                       weight = MAX(weight, excluded.weight),
                       evidence_chunk_id = COALESCE(excluded.evidence_chunk_id, evidence_chunk_id),
                       updated_at = strftime('%s', 'now')""",
                rows,
            )
            success_count = len(relations)
        except Exception as e:
            if "closed database" not in str(e) and "Cannot operate" not in str(e):
                logger.error(f"[GraphStorage] upsert_relations_batch error: {e}")
            # 降级为逐条插入
            for relation in relations:
                if self.upsert_relation(relation):
                    success_count += 1

        return success_count

    def get_relations_from(self, chunk_id: int, min_weight: float = 0.0) -> list[Relation]:
        """获取从某 Chunk 出发的关系"""
        rows = self.conn.execute(
            "SELECT * FROM relations WHERE src_chunk_id = ? AND weight >= ? ORDER BY weight DESC",
            (chunk_id, min_weight),
        ).fetchall()
        return [Relation.from_db_row(_row_to_dict(r)) for r in rows]

    def get_relations_to(self, chunk_id: int, min_weight: float = 0.0) -> list[Relation]:
        """获取指向某 Chunk 的关系"""
        rows = self.conn.execute(
            "SELECT * FROM relations WHERE tgt_chunk_id = ? AND weight >= ? ORDER BY weight DESC",
            (chunk_id, min_weight),
        ).fetchall()
        return [Relation.from_db_row(_row_to_dict(r)) for r in rows]

    def touch_relation(self, src_chunk_id: int, tgt_chunk_id: int, rel_type: str) -> bool:
        """更新关系访问时间和计数"""
        try:
            self.conn.execute(
                """UPDATE relations SET
                       last_hit = strftime('%s', 'now'),
                       access_count = access_count + 1,
                       updated_at = strftime('%s', 'now')
                   WHERE src_chunk_id = ? AND tgt_chunk_id = ? AND rel_type = ?""",
                (src_chunk_id, tgt_chunk_id, rel_type),
            )
            return True
        except Exception as e:
            logger.error(f"[GraphStorage] touch_relation error: {e}")
            return False

    def boost_relation(
        self, src_chunk_id: int, tgt_chunk_id: int, rel_type: str, amount: float = 0.05, max_weight: float = 1.0
    ) -> bool:
        """强化关系权重"""
        try:
            self.conn.execute(
                """UPDATE relations SET
                       weight = MIN(weight + ?, ?),
                       updated_at = strftime('%s', 'now')
                   WHERE src_chunk_id = ? AND tgt_chunk_id = ? AND rel_type = ?""",
                (amount, max_weight, src_chunk_id, tgt_chunk_id, rel_type),
            )
            return True
        except Exception as e:
            logger.error(f"[GraphStorage] boost_relation error: {e}")
            return False

    def decay_old_relations(self, days: int, factor: float, min_weight: float = 0.1) -> int:
        """衰减长期未访问的关系"""
        cutoff_time = int(time.time()) - days * 86400
        try:
            cursor = self.conn.execute(
                """UPDATE relations SET
                       weight = MAX(weight * ?, ?),
                       updated_at = strftime('%s', 'now')
                   WHERE last_hit IS NOT NULL AND last_hit < ? AND weight > ?""",
                (factor, min_weight, cutoff_time, min_weight),
            )
            return cursor.rowcount
        except Exception as e:
            logger.error(f"[GraphStorage] decay_old_relations error: {e}")
            return 0

    # ==================== 图遍历 ====================

    def traverse_graph(
        self, seed_chunk_ids: list[int], max_hops: int = 2, min_weight: float = 0.4, max_nodes: int = 50
    ) -> list[dict]:
        """
        递归 CTE 图遍历

        返回: [(chunk_id, hop_distance, path_weight, path, scope)]
        """
        if not seed_chunk_ids:
            return []

        # 使用递归 CTE 进行图遍历
        seeds = ",".join(map(str, seed_chunk_ids))
        query = f"""
            WITH RECURSIVE graph_traverse AS (
                -- 基础：种子节点
                SELECT
                    tgt_chunk_id AS chunk_id,
                    1 AS hop,
                    weight AS path_weight,
                    src_chunk_id || '->' || tgt_chunk_id AS path,
                    scope
                FROM relations
                WHERE src_chunk_id IN ({seeds}) AND weight >= ?

                UNION ALL

                -- 递归：扩展邻居
                SELECT
                    r.tgt_chunk_id,
                    gt.hop + 1,
                    MIN(gt.path_weight, r.weight),
                    gt.path || '->' || r.tgt_chunk_id,
                    r.scope
                FROM relations r
                JOIN graph_traverse gt ON r.src_chunk_id = gt.chunk_id
                WHERE gt.hop < ? AND r.weight >= ?
            )
            SELECT chunk_id, hop, path_weight, path, scope
            FROM graph_traverse
            WHERE chunk_id NOT IN ({seeds})
            GROUP BY chunk_id
            ORDER BY path_weight DESC, hop ASC
            LIMIT ?
        """

        try:
            rows = self.conn.execute(query, (min_weight, max_hops, min_weight, max_nodes)).fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"[GraphStorage] traverse_graph error: {e}")
            return []

    # ==================== Chunk 扩展操作 ====================

    def update_chunk_meta(self, chunk_meta: ChunkMeta) -> bool:
        """更新 Chunk 的扩展元数据"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.conn.execute(
                    """UPDATE chunks SET
                           note_id = ?,
                           inherited_meta = ?,
                           is_representative = ?
                       WHERE id = ?""",
                    chunk_meta.to_db_update(),
                )
                return True
            except sqlite3.OperationalError as e:
                error_msg = str(e).lower()
                # 如果是列不存在的错误，尝试添加列后重试
                if "no column" in error_msg or "table has no column" in error_msg:
                    logger.info("[GraphStorage] 检测到缺失列，尝试添加...")
                    self._ensure_chunk_columns()
                    continue
                # 如果是数据库锁定，等待重试
                elif "locked" in error_msg and attempt < max_retries - 1:
                    import time as _time

                    _time.sleep(0.1 * (attempt + 1))
                    continue
                else:
                    if "closed database" not in error_msg and "cannot operate" not in error_msg:
                        logger.error(f"[GraphStorage] update_chunk_meta error: {e}")
                    return False
            except Exception as e:
                # 静默处理数据库关闭错误
                error_msg = str(e).lower()
                if "closed database" in error_msg or "cannot operate" in error_msg:
                    return False
                # 重试其他错误
                if attempt < max_retries - 1:
                    import time as _time

                    _time.sleep(0.1 * (attempt + 1))
                    continue
                logger.error(f"[GraphStorage] update_chunk_meta error: {e}")
                return False
        return False

    def get_representative_chunk(self, note_id: str) -> int | None:
        """获取文档的代表 Chunk ID"""
        row = self.conn.execute(
            "SELECT id FROM chunks WHERE note_id = ? AND is_representative = 1 LIMIT 1", (note_id,)
        ).fetchone()
        return row[0] if row else None

    def set_representative_chunk(self, chunk_id: int, note_id: str) -> bool:
        """设置代表 Chunk（先清除其他代表标记）"""
        try:
            self.conn.execute("UPDATE chunks SET is_representative = 0 WHERE note_id = ?", (note_id,))
            self.conn.execute("UPDATE chunks SET is_representative = 1, note_id = ? WHERE id = ?", (note_id, chunk_id))
            return True
        except Exception as e:
            logger.error(f"[GraphStorage] set_representative_chunk error: {e}")
            return False

    def get_note_id_by_chunk(self, chunk_id: int) -> str | None:
        """获取 Chunk 所属的 Note ID"""
        try:
            row = self.conn.execute("SELECT note_id FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
            return row[0] if row else None
        except Exception as e:
            logger.error(f"[GraphStorage] get_note_id_by_chunk error: {e}")
            return None

    def get_chunks_by_note(self, note_id: str) -> list[dict]:
        """获取文档的所有 Chunk（包含 embedding）"""
        try:
            # JOIN chunks 和 vectors 表获取 embedding
            # vectors 是 sqlite-vec 的虚拟表，embedding 以 blob 格式存储
            rows = self.conn.execute(
                """SELECT c.id, c.content, v.embedding
                   FROM chunks c
                   LEFT JOIN vectors v ON c.id = v.chunk_id
                   WHERE c.note_id = ?""",
                (note_id,),
            ).fetchall()

            chunks = []
            for row in rows:
                # sqlite-vec 存储的 embedding 是二进制 blob 格式，需用 array 解析
                embedding = list(array.array("f", row[2])) if row[2] else []
                chunks.append({"id": row[0], "content": row[1], "embedding": embedding})
            return chunks
        except Exception as e:
            logger.error(f"[GraphStorage] get_chunks_by_note error: {e}")
            return []

    # ==================== 任务管理 ====================

    def create_job(self, job: GraphBuildJob) -> bool:
        """创建建图任务"""
        try:
            self.conn.execute(
                """INSERT INTO graph_build_jobs (job_id, note_id, chunk_ids, status, created_at, error_msg)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                job.to_db_row(),
            )
            self.conn.commit()  # 确保提交
            return True
        except sqlite3.IntegrityError as e:
            error_msg = str(e).lower()
            # 区分错误类型：外键约束违反 vs 唯一约束违反
            if "foreign key" in error_msg:
                # 外键约束违反：note_id 不存在于 notes 表
                logger.error(f"[GraphStorage] create_job 外键约束失败，note_id={job.note_id} 不存在于 notes 表: {e}")
                return False
            else:
                # 唯一约束违反：任务已存在，更新状态
                self.conn.execute(
                    "UPDATE graph_build_jobs SET status = ?, error_msg = NULL WHERE note_id = ?",
                    ("pending", job.note_id),
                )
                self.conn.commit()  # 确保提交
                return True
        except Exception as e:
            logger.error(f"[GraphStorage] create_job error: {e}")
            return False

    def get_pending_job(self) -> GraphBuildJob | None:
        """获取一个待处理任务"""
        row = self.conn.execute(
            """SELECT * FROM graph_build_jobs
               WHERE status = 'pending'
               ORDER BY created_at ASC
               LIMIT 1"""
        ).fetchone()
        return GraphBuildJob.from_db_row(_row_to_dict(row)) if row else None

    def get_job_by_note_id(self, note_id: str) -> GraphBuildJob | None:
        """通过 note_id 获取任务"""
        row = self.conn.execute(
            """SELECT * FROM graph_build_jobs
               WHERE note_id = ?
               ORDER BY created_at DESC
               LIMIT 1""",
            (note_id,),
        ).fetchone()
        return GraphBuildJob.from_db_row(_row_to_dict(row)) if row else None

    def update_job_status(self, job: GraphBuildJob) -> bool:
        """更新任务状态"""
        try:
            self.conn.execute(
                """UPDATE graph_build_jobs SET
                       status = ?,
                       started_at = ?,
                       finished_at = ?,
                       error_msg = ?,
                       retry_count = retry_count + 1
                   WHERE job_id = ?""",
                (job.status, job.started_at, job.finished_at, job.error_msg, job.job_id),
            )
            self.conn.commit()  # ✅ 修复：确保状态更新立即提交
            return True
        except Exception as e:
            # 静默处理数据库关闭错误
            if "closed database" in str(e) or "Cannot operate" in str(e):
                return False
            logger.error(f"[GraphStorage] update_job_status error: {e}")
            return False

    def get_job_stats(self) -> dict:
        """获取任务统计"""
        rows = self.conn.execute("SELECT status, COUNT(*) as cnt FROM graph_build_jobs GROUP BY status").fetchall()
        return {r["status"]: r["cnt"] for r in rows}

    def get_entities_with_chunk_ids(self, names: list[str]) -> list[Entity]:
        """批量获取有 chunk_id 的实体"""
        if not names:
            return []

        placeholders = ",".join(["?" for _ in names])
        rows = self.conn.execute(
            f"""SELECT * FROM entities
                WHERE canonical_name IN ({placeholders})
                AND chunk_id IS NOT NULL""",
            names,
        ).fetchall()
        return [Entity.from_db_row(_row_to_dict(r)) for r in rows]

    # ==================== 标签共现 ====================

    def record_tag_co_occurrence(self, tag1: str, tag2: str) -> bool:
        """记录标签共现"""
        # 保证顺序一致性
        t1, t2 = sorted([tag1, tag2])
        try:
            self.conn.execute(
                """INSERT INTO tag_co_occurrence (tag1, tag2, co_occurrence_count, last_updated)
                   VALUES (?, ?, 1, strftime('%s', 'now'))
                   ON CONFLICT(tag1, tag2) DO UPDATE SET
                       co_occurrence_count = co_occurrence_count + 1,
                       last_updated = strftime('%s', 'now')""",
                (t1, t2),
            )
            return True
        except Exception as e:
            logger.error(f"[GraphStorage] record_tag_co_occurrence error: {e}")
            return False

    def get_tag_co_occurrences(self, min_count: int = 3) -> list[dict]:
        """获取高频标签共现"""
        rows = self.conn.execute(
            "SELECT * FROM tag_co_occurrence WHERE co_occurrence_count >= ? ORDER BY co_occurrence_count DESC",
            (min_count,),
        ).fetchall()
        return [dict(r) for r in rows]


__all__ = ["GraphStorage"]
