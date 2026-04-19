#!/usr/bin/env python3
"""
graph_builder.py - 异步建图管线

实现 FR-1 异步建图需求，包括：
- 任务队列管理
- 双层抽取管道调用
- 代表 Chunk 映射
- 关系写入
"""
import asyncio
import hashlib
import sqlite3
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

from plugins.tinyrag_memory_graph.config import MemoryGraphConfig
from plugins.tinyrag_memory_graph.extractor import DualLayerExtractor, WikilinkExtractor
from plugins.tinyrag_memory_graph.models import ChunkMeta, GraphBuildJob, Note, Relation, RelationType
from plugins.tinyrag_memory_graph.storage import GraphStorage


@dataclass
class BuildTask:
    """建图任务"""
    note_id: str
    filepath: str
    content: str
    chunk_ids: list[int]
    chunks_content: list[str]
    priority: int = 0


class GraphBuildQueue:
    """
    异步建图队列（FR-1.5）

    管理建图任务的生命周期，支持：
    - 异步任务提交
    - 后台处理
    - 状态追踪
    - 错误处理与重试
    """

    def __init__(self, storage: GraphStorage, config: MemoryGraphConfig):
        self.storage = storage
        self.config = config
        self.queue_config = config.queue

        self._executor: Optional[ThreadPoolExecutor] = None
        self._running = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # 统计
        self._stats = {
            "total_processed": 0,
            "total_failed": 0,
            "total_entities": 0,
            "total_relations": 0,
        }

    def _extract_title_from_content(self, content: str) -> str:
        """从文档内容提取标题（回退策略）"""
        import re
        # 尝试匹配第一个 # 标题
        match = re.search(r'^#+\s+(.+?)(?:\s*\n|$)', content, re.MULTILINE)
        if match:
            return match.group(1).strip()
        return ""

    def _extract_frontmatter(self, content: str) -> dict:
        """从文档内容提取 frontmatter"""
        from plugins.tinyrag_memory_graph.extractor import FrontmatterParser
        return FrontmatterParser.parse(content)

    def start(self):
        """启动后台处理"""
        if self._running:
            return

        # ✅ 恢复机制：重置遗留的 running 状态任务为 pending
        # 这些任务可能是上次程序异常退出时遗留的
        try:
            cursor = self.storage.conn.execute(
                """UPDATE graph_build_jobs 
                   SET status = 'pending', error_msg = 'Recovered from interrupted run'
                   WHERE status = 'running'"""
            )
            if cursor.rowcount > 0:
                self.storage.conn.commit()
                print(f"[GraphBuildQueue] 🔄 已恢复 {cursor.rowcount} 个中断的任务")
        except Exception as e:
            print(f"[GraphBuildQueue] ⚠️ 恢复任务失败: {e}")

        self._executor = ThreadPoolExecutor(
            max_workers=self.queue_config.max_workers,
            thread_name_prefix="graph-builder"
        )
        self._running = True

    def stop(self):
        """停止后台处理"""
        self._running = False
        if self._executor:
            try:
                # 先尝试正常关闭，等待任务完成
                self._executor.shutdown(wait=True, cancel_futures=False)
            except TypeError:
                # Python < 3.9 不支持 cancel_futures 参数
                self._executor.shutdown(wait=True)
            except Exception as e:
                print(f"[GraphBuildQueue] 停止时出错: {e}")
            finally:
                self._executor = None

    def submit(self, task: BuildTask, sync: bool = False) -> str:
        """
        提交建图任务（FR-1.1）

        将任务推入异步队列，立即返回。

        Args:
            task: 建图任务
            sync: 是否同步执行（CLI 模式下应设为 True）
        """
        job = GraphBuildJob(
            job_id=str(uuid.uuid4()),
            note_id=task.note_id,
            chunk_ids=task.chunk_ids,
            status="pending",
        )

        # 立即提取 title 和 frontmatter，不等待后台任务
        title = self._extract_title_from_content(task.content)
        frontmatter = self._extract_frontmatter(task.content)

        # 先创建 note 记录（因为 graph_build_jobs 有外键约束）
        from plugins.tinyrag_memory_graph.models import Note
        note = Note(
            note_id=task.note_id,
            filepath=task.filepath,
            title=title or frontmatter.get("title", ""),
            frontmatter_json=frontmatter if frontmatter else None,
        )
        self.storage.upsert_note(note)
        self.storage.conn.commit()

        # 写入任务表
        self.storage.create_job(job)

        # 判断执行模式：sync 参数优先，否则检查 executor 状态
        if sync or not (self._executor and self._running):
            # 同步执行（CLI 模式或线程池未启动时）
            print(f"[GraphBuildQueue] 同步执行任务: {task.note_id}")
            self._process_task(task, is_sync=True)
        else:
            # 后台执行（服务器模式）
            print(f"[GraphBuildQueue] 提交后台任务: {task.note_id}")
            self._executor.submit(self._process_task, task, False)

        return job.job_id

    def _process_task(self, task: BuildTask, is_sync: bool = False):
        """后台处理任务"""
        # 异步模式下检查运行状态
        if not is_sync and not self._running:
            return

        job = None
        try:
            # 通过 note_id 获取特定任务（而不是任意 pending 任务）
            job = self.storage.get_job_by_note_id(task.note_id)
            if not job:
                print(f"[GraphBuildQueue] 未找到任务: {task.note_id}")
                return

            # 检查任务状态
            if job.status != "pending":
                print(f"[GraphBuildQueue] 任务状态不是 pending: {job.status}")
                return

            # 异步模式下再次检查运行状态
            if not is_sync and not self._running:
                return

            job.start()
            self.storage.update_job_status(job)

            # 执行建图
            stats = self._build_graph(task)

            # 异步模式下检查运行状态
            if not is_sync and not self._running:
                return

            # 标记完成
            job.complete()
            self.storage.update_job_status(job)

            # 更新统计
            self._stats["total_processed"] += 1
            self._stats["total_entities"] += stats.get("entities", 0)
            self._stats["total_relations"] += stats.get("relations", 0)

        except Exception as e:
            # 如果是解释器关闭导致的错误，静默忽略
            if "interpreter shutdown" in str(e) or "closed database" in str(e):
                return
            if job:
                try:
                    job.fail(str(e))
                    self.storage.update_job_status(job)
                except Exception:
                    pass  # 数据库可能已关闭
            self._stats["total_failed"] += 1
            print(f"[GraphBuildQueue] Task failed: {e}")

    def _build_graph(self, task: BuildTask) -> dict:
        """
        执行实际的图构建（FR-1.2/1.3/1.4）

        Returns:
            统计信息 {entities: N, relations: M}
        """
        stats = {"entities": 0, "relations": 0}

        # 初始化抽取器
        extractor = DualLayerExtractor(self.config)

        # 1. 文档级抽取
        doc_result = extractor.extract_document_level(task.content, task.note_id)

        # 2. 创建 Note 记录
        note = Note(
            note_id=task.note_id,
            filepath=task.filepath,
            title=doc_result.frontmatter.get("title", ""),
            frontmatter_json=doc_result.frontmatter,
        )
        self.storage.upsert_note(note)

        # 3. 写入文档级实体
        for entity in doc_result.entities:
            self.storage.upsert_entity(entity)
            stats["entities"] += 1

        # 4. 确定代表 Chunk（FR-1.4）
        representative_chunk_id = self._find_representative_chunk(
            task.chunk_ids, task.chunks_content, doc_result.wikilinks
        )

        # 5. 更新 Chunk 元数据
        inherited_meta = {
            "author": doc_result.frontmatter.get("author", ""),
            "project": doc_result.frontmatter.get("project", ""),
            "status": doc_result.frontmatter.get("status", ""),
            "tags": doc_result.frontmatter.get("tags", []),
        }

        for i, chunk_id in enumerate(task.chunk_ids):
            chunk_meta = ChunkMeta(
                chunk_id=chunk_id,
                note_id=task.note_id,
                inherited_meta=inherited_meta,
                is_representative=(chunk_id == representative_chunk_id),
            )
            self.storage.update_chunk_meta(chunk_meta)

        # 6. Chunk 级抽取
        for i, (chunk_id, chunk_content) in enumerate(zip(task.chunk_ids, task.chunks_content)):
            chunk_result = extractor.extract_chunk_level(
                chunk_content, chunk_id, task.note_id
            )

            # 写入实体
            for entity in chunk_result.entities:
                self.storage.upsert_entity(entity)
                stats["entities"] += 1

        # 7. 创建 Wikilink 关系（需要全局 chunk_map）
        # 这部分在批量处理后单独执行
        wikilink_relations = self._create_wikilink_relations(
            task, doc_result.wikilinks, representative_chunk_id
        )
        for rel in wikilink_relations:
            self.storage.upsert_relation(rel)
            stats["relations"] += 1

        # 8. 提交事务
        self.storage.conn.commit()

        return stats

    def _find_representative_chunk(self, chunk_ids: list[int],
                                    chunks_content: list[str],
                                    wikilinks: list[str]) -> int:
        """
        查找代表 Chunk（FR-1.4）

        策略：
        1. 优先选择包含最多 Wikilink 的 Chunk
        2. 否则选择第一个 Chunk
        """
        if not chunk_ids:
            return -1

        best_chunk_id = chunk_ids[0]
        best_score = -1

        for chunk_id, content in zip(chunk_ids, chunks_content):
            # 计算该 Chunk 包含的 Wikilink 数量
            chunk_wikilinks = WikilinkExtractor.extract(content)
            common_links = set(chunk_wikilinks) & set(wikilinks)
            score = len(common_links)

            if score > best_score:
                best_score = score
                best_chunk_id = chunk_id

        return best_chunk_id

    def _create_wikilink_relations(self, task: BuildTask,
                                    wikilinks: list[str],
                                    representative_chunk_id: int) -> list[Relation]:
        """创建 Wikilink 关系"""
        relations = []

        # 尝试解析每个 Wikilink 目标对应的 Chunk
        for link in wikilinks:
            # 查找目标文档
            target_note = self.storage.get_note_by_filepath(f"{link}.md")
            if not target_note:
                # 尝试不带扩展名
                target_note = self.storage.get_note_by_filepath(link)

            if target_note:
                # 查找目标的代表 Chunk
                target_chunk_id = self.storage.get_representative_chunk(target_note.note_id)
                if target_chunk_id and representative_chunk_id:
                    relation = Relation(
                        src_chunk_id=representative_chunk_id,
                        tgt_chunk_id=target_chunk_id,
                        rel_type=RelationType.LINKS_TO,
                        scope="doc",
                        weight=1.0,
                        evidence_chunk_id=representative_chunk_id,
                    )
                    relations.append(relation)

        return relations

    def get_stats(self) -> dict:
        """获取队列统计"""
        queue_stats = self.storage.get_job_stats()
        return {
            **self._stats,
            "queue": queue_stats,
            "running": self._running,
        }


class GraphBuilder:
    """
    图构建器 - 高层 API

    提供简单的接口用于构建知识图谱。
    """

    def __init__(self, db_conn: sqlite3.Connection, config: MemoryGraphConfig):
        self.storage = GraphStorage(db_conn, config)
        self.config = config
        self.queue = GraphBuildQueue(self.storage, config)

        # 初始化 Schema
        self.storage.initialize()

    def start(self):
        """启动后台处理"""
        self.queue.start()

    def stop(self):
        """停止后台处理"""
        self.queue.stop()

    def build_for_document(self, filepath: str, content: str,
                           chunk_ids: list[int],
                           chunks_content: list[str],
                           sync: bool = False) -> str:
        """
        为文档构建图谱

        Args:
            filepath: 文件路径
            content: 完整文档内容
            chunk_ids: Chunk ID 列表
            chunks_content: Chunk 内容列表
            sync: 是否同步执行（CLI 模式应设为 True）

        Returns:
            任务 ID
        """
        note_id = self._generate_note_id(filepath)

        task = BuildTask(
            note_id=note_id,
            filepath=filepath,
            content=content,
            chunk_ids=chunk_ids,
            chunks_content=chunks_content,
        )

        return self.queue.submit(task, sync=sync)

    def build_batch(self, documents: list[dict]) -> list[str]:
        """
        批量构建图谱

        Args:
            documents: [{filepath, content, chunk_ids, chunks_content}, ...]

        Returns:
            任务 ID 列表
        """
        job_ids = []
        for doc in documents:
            job_id = self.build_for_document(
                filepath=doc["filepath"],
                content=doc["content"],
                chunk_ids=doc["chunk_ids"],
                chunks_content=doc["chunks_content"],
            )
            job_ids.append(job_id)
        return job_ids

    def delete_document(self, filepath: str):
        """删除文档的图谱数据"""
        note_id = self._generate_note_id(filepath)
        self.storage.delete_note(note_id)
        self.storage.conn.commit()

    def _generate_note_id(self, filepath: str) -> str:
        """生成 Note ID"""
        return hashlib.md5(filepath.encode()).hexdigest()[:16]

    def get_stats(self) -> dict:
        """获取构建统计"""
        return self.queue.get_stats()


__all__ = ["GraphBuildQueue", "GraphBuilder", "BuildTask"]
