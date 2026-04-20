#!/usr/bin/env python3
"""build_index.py - tinyRAG 高性能索引构建器 (v2.2 - 插件支持)"""

import argparse
import array
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from chunker.markdown_splitter import MarkdownSplitter
from config import load_config
from embedder.embed_engine import EmbeddingEngine

# 插件支持
from plugins.bootstrap import PluginLoader
from scanner.scan_engine import DEFAULT_SKIP_DIRS, Scanner
from storage.database import DatabaseManager
from utils.jieba_helper import jieba_segment, load_jieba_user_dict
from utils.logger import logger, setup_logger

setup_logger(level="INFO")


def prepare_fts_content(chunk, file_path: str) -> str:
    metadata = chunk.metadata or {}
    tags = metadata.get("tags", [])
    if tags is None:
        tags = []
    if isinstance(tags, str):
        tags = [tags]
    tag_str = " ".join([f"#{t.strip()}" for t in tags if t])
    doc_type = metadata.get("doc_type") or ""
    filename = os.path.basename(file_path)
    section_title = chunk.section_title or ""

    # 预计算分词结果，避免重复调用
    seg_filename = jieba_segment(filename)
    seg_section = jieba_segment(section_title)
    seg_path = jieba_segment(chunk.section_path or "")
    seg_tags = jieba_segment(tag_str)
    seg_type = jieba_segment(doc_type)
    seg_content = jieba_segment(chunk.content)

    # 文件名和章节标题权重加倍（重复一次）
    parts = [
        seg_filename,
        seg_filename,  # 文件名权重 x2
        seg_path,
        seg_section,
        seg_section,  # 章节标题权重 x2
        seg_tags,
        seg_type,
        seg_content,
    ]
    return " ".join(filter(None, parts)).strip()


def json_serialize(obj):
    from datetime import date, datetime

    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def process_file_worker(file_item: dict, splitter: MarkdownSplitter) -> tuple[int, list[Any], str, str]:
    """处理单个文件，返回 (file_id, chunks, file_path, absolute_path)"""
    abs_path = Path(file_item["absolute_path"])
    try:
        content = abs_path.read_text(encoding="utf-8")
        mtime = file_item.get("mtime")
        chunks = splitter.split(content, mtime)
        return file_item["id"], chunks, file_item["file_path"], file_item["absolute_path"]
    except Exception as e:
        logger.error(f"❌ 读取/分块失败：{abs_path} - {e}")
        return file_item["id"], [], file_item["file_path"], file_item["absolute_path"]


def process_and_commit_batch(
    chunks: list[tuple[int, Any, str, str]],  # 新增: (file_id, chunk, file_path, absolute_path)
    embedder: EmbeddingEngine,
    db: DatabaseManager,
    start_idx: int,
    commit: bool = True,
    plugin_loader: PluginLoader = None,
    file_chunks_collector: dict = None,  # 新增：收集文件级别的 chunks
) -> int:
    if not chunks:
        return start_idx
    texts = [c[1].content for c in chunks]
    embeddings = embedder.embed(texts)
    inserted_chunk_ids = []  # 收集新插入的 chunk_id
    for idx, ((file_id, chunk, f_path, abs_path), emb) in enumerate(zip(chunks, embeddings, strict=False)):
        chunk_idx = start_idx + idx
        metadata_json = json.dumps(chunk.metadata or {}, ensure_ascii=False, default=json_serialize)
        confidence_json = json.dumps(chunk.confidence_metadata or {}, ensure_ascii=False, default=json_serialize)
        cursor = db.conn.execute(
            """INSERT INTO chunks (file_id, chunk_index, content, content_type, section_title, section_path, start_pos, end_pos, confidence_final_weight, metadata, confidence_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                file_id,
                chunk_idx,
                chunk.content,
                chunk.content_type.value,
                chunk.section_title,
                chunk.section_path,
                chunk.start_pos,
                chunk.end_pos,
                1.0,
                metadata_json,
                confidence_json,
            ),
        )
        new_chunk_id = cursor.lastrowid
        inserted_chunk_ids.append((new_chunk_id, file_id, chunk, f_path))

        # 收集文件级别的 chunks（用于触发 on_file_indexed 钩子）
        if file_chunks_collector is not None:
            if file_id not in file_chunks_collector:
                file_chunks_collector[file_id] = {
                    "chunks": [],
                    "file_path": f_path,
                    "absolute_path": abs_path,  # 新增绝对路径
                }
            file_chunks_collector[file_id]["chunks"].append(
                {
                    "id": new_chunk_id,
                    "content": chunk.content,
                    "metadata": chunk.metadata,
                }
            )

        if db.vec_support:
            db.conn.execute(
                "INSERT INTO vectors (chunk_id, embedding) VALUES (?, ?)",
                (new_chunk_id, array.array("f", emb).tobytes()),
            )
        db.conn.execute(
            "INSERT INTO fts5_index (rowid, content) VALUES (?, ?)", (new_chunk_id, prepare_fts_content(chunk, f_path))
        )

    # 触发插件钩子: on_chunks_indexed
    if plugin_loader and inserted_chunk_ids:
        try:
            for chunk_id, file_id, chunk, f_path in inserted_chunk_ids:
                plugin_loader.invoke_hook(
                    "on_chunks_indexed",
                    chunk_id=chunk_id,
                    file_id=file_id,
                    content=chunk.content,
                    metadata=chunk.metadata,
                )
        except Exception as e:
            logger.warning(f"⚠️ 插件钩子执行失败: {e}")

    if commit:
        db.conn.commit()
    return start_idx + len(chunks)


def main():
    parser = argparse.ArgumentParser(description="tinyRAG 高性能索引构建器")
    parser.add_argument("--force", action="store_true", help="重建所有索引")
    parser.add_argument("--batch-size", type=int, default=128, help="向量化批大小")
    args = parser.parse_args()
    try:
        config = load_config("config.yaml")
    except Exception as e:
        logger.critical(f"❌ 配置加载失败：{e}")
        sys.exit(1)

    load_jieba_user_dict(config)
    db = DatabaseManager(config.db_path, vec_dimension=config.embedding_model.dimensions)
    global_skip_dirs = DEFAULT_SKIP_DIRS | frozenset(config.exclude.dirs)
    scanner = Scanner(db, skip_dirs=global_skip_dirs, global_patterns=config.exclude.patterns)
    splitter = MarkdownSplitter(config)
    embedder = EmbeddingEngine(
        model_name=config.embedding_model.name,
        cache_dir=config.embedding_model.cache_dir,
        batch_size=config.embedding_model.batch_size,
        unload_after_seconds=config.embedding_model.unload_after_seconds,
    )

    # 初始化插件系统
    plugin_loader = None
    if config.plugins.enabled:
        try:
            plugin_loader = PluginLoader(config, None)  # CLI 模式无 context
            plugin_loader.load_all()
            # 为插件设置数据库连接
            for plugin in plugin_loader.get_all_plugins().values():
                if hasattr(plugin, "set_db_connection"):
                    plugin.set_db_connection(db.conn)
            logger.info(f"✅ 插件系统已加载 {len(plugin_loader.get_all_plugins())} 个插件")
        except Exception as e:
            logger.warning(f"⚠️ 插件加载失败: {e}")
            plugin_loader = None

    vault_configs = [(v.name, v.path) for v in config.vaults if v.enabled]
    if not vault_configs:
        logger.warning("⚠️ 未启用任何仓库，跳过扫描")
        db.close()
        return

    vault_excludes: dict[str, tuple[frozenset[str], list[str]]] = {}
    for v in config.vaults:
        if v.enabled:
            vault_excludes[v.name] = (frozenset(v.exclude.dirs), v.exclude.patterns) if v.exclude else (frozenset(), [])

    files_to_index = []
    if args.force:
        logger.info("🔄 模式：强制重建所有索引")

        # 先触发插件钩子，让插件清理它们的数据
        if plugin_loader:
            try:
                logger.info("🧹 通知插件清理数据...")
                plugin_loader.invoke_hook("on_index_rebuild", force=True)
            except Exception as e:
                logger.warning(f"⚠️ 插件 on_index_rebuild 钩子执行失败: {e}")

        # 清理核心表数据
        db.conn.execute("DELETE FROM fts5_index")
        cursor = db.conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='vectors'")
        if cursor.fetchone():
            db.conn.execute("DELETE FROM vectors")
        db.conn.execute("DELETE FROM chunks")
        db.conn.execute("DELETE FROM files")
        db.conn.commit()
        report = scanner.scan_vaults(vault_configs, vault_excludes)
        for meta in report.new_files:
            db.upsert_file(meta.to_dict())
        db.conn.commit()
        files_to_index = [
            dict(row)
            for row in db.conn.execute(
                "SELECT id, absolute_path, file_path, mtime FROM files WHERE is_deleted = 0"
            ).fetchall()
        ]
    else:
        # ✅ 增量模式逻辑重构
        report = scanner.scan_vaults(vault_configs, vault_excludes)
        scanner.process_report(report)

        changed_paths = [f.absolute_path for f in report.new_files + report.modified_files]
        changed_paths.extend([f.new_absolute_path for f in report.moved_files])

        if changed_paths:
            placeholders = ",".join(["?"] * len(changed_paths))
            cursor = db.conn.execute(
                f"SELECT id, absolute_path, file_path, mtime FROM files WHERE absolute_path IN ({placeholders})",
                changed_paths,
            )
            files_to_index = [dict(row) for row in cursor.fetchall()]

        # 🔧 精准自愈：仅在无变更时检查缺失 chunks，并过滤空文件
        missing = []  # 初始化避免未定义错误
        if not files_to_index:
            cursor = db.conn.execute(
                """
                SELECT f.id, f.vault_name, f.absolute_path, f.file_path, f.mtime
                FROM files f
                WHERE f.is_deleted = 0
                  AND NOT EXISTS (SELECT 1 FROM chunks c WHERE c.file_id = f.id)
                LIMIT 1000
            """
            )
            missing = [dict(row) for row in cursor.fetchall()]
        if missing:
            # 格式化文件标识：vault名称 / 相对路径
            missing_display = [f"{m['vault_name']}/{m['file_path']}" for m in missing]

            # 智能日志：数量少时全列，数量多时截断防刷屏
            if len(missing) <= 5:
                logger.info(f"🔧 发现 {len(missing)} 个已注册但无索引的文件，准备补充: {', '.join(missing_display)}")
            else:
                logger.info(
                    f"🔧 发现 {len(missing)} 个已注册但无索引的文件，准备补充 (前5个): {', '.join(missing_display[:5])} ..."
                )

            files_to_index.extend(missing)
        if not files_to_index:
            logger.info("✨ 索引已是最新，无需更新。")
            db.close()
            return

    stream_batch_size = getattr(config, "stream_batch_size", 100)
    max_concurrent_files = getattr(config, "max_concurrent_files", os.cpu_count() or 4)
    logger.info(f"🚀 开始处理 {len(files_to_index)} 个文件...")
    logger.info(
        f"⚙️ 配置: batch={config.embedding_model.batch_size}, stream_batch={stream_batch_size}, workers={max_concurrent_files}"
    )

    pending_chunks: list[tuple[int, Any, str, str]] = []  # (file_id, chunk, file_path, absolute_path)
    total_processed = 0
    global_chunk_idx = 0
    total_files_with_chunks = 0
    start_time = time.time()
    file_chunks_collector: dict = {}  # 收集文件级别的 chunks（用于触发 on_file_indexed 钩子）

    try:
        from tqdm import tqdm

        pbar = tqdm(total=len(files_to_index), desc="文件处理", unit="文件", file=sys.stdout, leave=True)
    except ImportError:
        pbar = None

    try:
        with ThreadPoolExecutor(max_workers=max_concurrent_files) as executor:
            for f_id, chunks, f_path, abs_path in executor.map(
                lambda f: process_file_worker(f, splitter), files_to_index
            ):
                for c in chunks:
                    pending_chunks.append((f_id, c, f_path, abs_path))
                if chunks:
                    total_files_with_chunks += 1
                if len(pending_chunks) >= stream_batch_size:
                    global_chunk_idx = process_and_commit_batch(
                        pending_chunks,
                        embedder,
                        db,
                        global_chunk_idx,
                        plugin_loader=plugin_loader,
                        file_chunks_collector=file_chunks_collector,
                    )
                    total_processed += len(pending_chunks)
                    pending_chunks.clear()
                if pbar:
                    pbar.update(1)
        if pending_chunks:
            global_chunk_idx = process_and_commit_batch(
                pending_chunks,
                embedder,
                db,
                global_chunk_idx,
                plugin_loader=plugin_loader,
                file_chunks_collector=file_chunks_collector,
            )
            total_processed += len(pending_chunks)
            pending_chunks.clear()
    except Exception as e:
        logger.error(f"❌ 索引构建失败: {e}", exc_info=True)
        if pbar:
            pbar.close()
        if plugin_loader:
            plugin_loader.shutdown()
        db.close()
        sys.exit(1)

    if pbar:
        pbar.close()
    elapsed = time.time() - start_time

    # ✅ 结果校验：避免“处理了文件但实际无内容可索引”的歧义日志
    if total_processed == 0 and total_files_with_chunks == 0:
        logger.warning("⚠️ 已扫描指定文件，但内容为空或全部被排除规则过滤，未生成新索引。")
    else:
        logger.success(
            f"🎉 构建完成！处理 {total_files_with_chunks} 个有效文件，{total_processed} 个 chunks，耗时：{elapsed:.2f}s"
        )

    # 触发文件级别的 on_file_indexed 钩子（用于图谱构建等）
    if plugin_loader and file_chunks_collector:
        try:
            logger.info(f"🔧 触发插件 on_file_indexed 钩子处理 {len(file_chunks_collector)} 个文件...")
            for file_id, data in file_chunks_collector.items():
                plugin_loader.invoke_hook(
                    "on_file_indexed",
                    file_id=file_id,
                    chunks=data["chunks"],
                    filepath=data["file_path"],
                    absolute_path=data.get("absolute_path", ""),  # 新增绝对路径
                )
            logger.info("✅ 插件钩子处理完成")
        except Exception as e:
            logger.warning(f"⚠️ on_file_indexed 钩子执行失败: {e}")

    # 关闭插件系统
    if plugin_loader:
        plugin_loader.shutdown()
    db.close()


if __name__ == "__main__":
    main()
