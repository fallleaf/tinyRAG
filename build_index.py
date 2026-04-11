#!/usr/bin/env python3
"""
build_index.py - tinyRAG 高性能索引构建器 (v2.0 流式处理版)
"""
import argparse
import array
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

_script_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(_script_dir))

import jieba

from chunker.markdown_splitter import MarkdownSplitter
from config import get_merged_exclude, load_config
from embedder.embed_engine import EmbeddingEngine
from scanner.scan_engine import Scanner
from storage.database import DatabaseManager
from utils.logger import logger, setup_logger

setup_logger(level="INFO", log_file=str(_script_dir / "logs" / "build_index.log"))

_jieba_dict_loaded = False


def _ensure_jieba_user_dict(config):
    """加载 jieba 用户自定义词典"""
    global _jieba_dict_loaded
    if _jieba_dict_loaded:
        return

    if hasattr(config, "jieba_user_dict") and config.jieba_user_dict:
        dict_path = Path(config.jieba_user_dict).expanduser()
        if not dict_path.is_absolute():
            dict_path = _script_dir / dict_path
        if dict_path.exists():
            jieba.load_userdict(str(dict_path))
            logger.info(f"📚 已加载 jieba 用户字典: {dict_path}")
        else:
            logger.warning(f"⚠️ jieba 用户字典不存在: {dict_path}")

    _jieba_dict_loaded = True


def _jieba_segment(text: str) -> str:
    """对中文文本进行 jieba 分词"""
    if not text or not text.strip():
        return ""
    return " ".join(jieba.cut_for_search(text))


def prepare_fts_content(chunk, file_path: str) -> str:
    """构建复合检索字符串"""
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

    parts = [
        _jieba_segment(filename), _jieba_segment(filename),
        _jieba_segment(chunk.section_path or ""),
        _jieba_segment(section_title), _jieba_segment(section_title),
        _jieba_segment(tag_str), _jieba_segment(doc_type),
        _jieba_segment(chunk.content),
    ]
    return " ".join(filter(None, parts)).strip()


def json_serialize(obj):
    from datetime import date, datetime
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def process_file_worker(file_item: dict, splitter: MarkdownSplitter) -> tuple[int, list[Any], str]:
    """并行分块任务单元"""
    abs_path = Path(file_item["absolute_path"])
    try:
        content = abs_path.read_text(encoding="utf-8")
        mtime = file_item.get("mtime")
        chunks = splitter.split(content, mtime)
        return file_item["id"], chunks, file_item["file_path"]
    except Exception as e:
        logger.error(f"❌ 读取/分块失败：{abs_path} - {e}")
        return file_item["id"], [], file_item["file_path"]


def process_and_commit_batch(
    chunks: list[tuple[int, Any, str]],
    embedder: EmbeddingEngine,
    db: DatabaseManager,
    start_idx: int,
) -> int:
    """处理一批 chunks：向量化 + 入库"""
    if not chunks:
        return start_idx

    texts = [c[1].content for c in chunks]
    embeddings = embedder.embed(texts)

    for idx, ((file_id, chunk, f_path), emb) in enumerate(zip(chunks, embeddings)):
        chunk_idx = start_idx + idx
        metadata_json = json.dumps(chunk.metadata or {}, ensure_ascii=False, default=json_serialize)
        confidence_json = json.dumps(chunk.confidence_metadata or {}, ensure_ascii=False, default=json_serialize)

        cursor = db.conn.execute(
            """INSERT INTO chunks (file_id, chunk_index, content, content_type, section_title, section_path,
               start_pos, end_pos, confidence_final_weight, metadata, confidence_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                file_id, chunk_idx, chunk.content, chunk.content_type.value,
                chunk.section_title, chunk.section_path, chunk.start_pos, chunk.end_pos,
                1.0, metadata_json, confidence_json
            )
        )
        new_chunk_id = cursor.lastrowid

        if db.vec_support:
            db.conn.execute(
                "INSERT INTO vectors (chunk_id, embedding) VALUES (?, ?)",
                (new_chunk_id, array.array("f", emb).tobytes())
            )

        db.conn.execute(
            "INSERT INTO fts5_index (rowid, content) VALUES (?, ?)",
            (new_chunk_id, prepare_fts_content(chunk, f_path))
        )

    db.conn.commit()
    return start_idx + len(chunks)


def main():
    parser = argparse.ArgumentParser(description="tinyRAG 高性能索引构建器")
    parser.add_argument("--force", action="store_true", help="重建所有索引")
    args = parser.parse_args()

    try:
        config = load_config("config.yaml")
    except Exception as e:
        logger.critical(f"❌ 配置加载失败：{e}")
        sys.exit(1)

    _ensure_jieba_user_dict(config)

    db = DatabaseManager(config.db_path, vec_dimension=config.embedding_model.dimensions)

    # 从配置读取全局排除规则
    global_skip_dirs = frozenset(config.exclude.dirs) if hasattr(config, "exclude") else frozenset()
    global_exclude_patterns = config.exclude.patterns if hasattr(config, "exclude") else []
    scanner = Scanner(db, skip_dirs=global_skip_dirs, exclude_patterns=global_exclude_patterns)
    logger.info(f"🚫 全局排除目录: {list(global_skip_dirs)}")
    if global_exclude_patterns:
        logger.info(f"🚫 全局排除模式: {global_exclude_patterns}")

    splitter = MarkdownSplitter(config)

    embedder = EmbeddingEngine(
        model_name=config.embedding_model.name,
        cache_dir=config.embedding_model.cache_dir,
        batch_size=config.embedding_model.batch_size,
        unload_after_seconds=config.embedding_model.unload_after_seconds,
    )

    vault_configs = [(v.name, v.path) for v in config.vaults if v.enabled]
    if not vault_configs:
        logger.warning("⚠️ 未启用任何仓库，跳过扫描")
        db.close()
        return

    # 构建 per-vault 排除规则（合并全局规则 + vault 特定规则）
    vault_excludes: dict[str, tuple[frozenset[str], list[str]]] = {}
    for v in config.vaults:
        if v.enabled:
            merged = get_merged_exclude(v, config.exclude)
            vault_excludes[v.name] = (frozenset(merged.dirs), merged.patterns)
            if v.exclude:
                logger.info(f"🚫 {v.name} 特定排除目录: {v.exclude.dirs}")
                if v.exclude.patterns:
                    logger.info(f"🚫 {v.name} 特定排除模式: {v.exclude.patterns}")

    logger.info(f"📂 将索引以下仓库: {[v[0] for v in vault_configs]}")

    files_to_index = []
    if args.force:
        logger.info("🔄 模式：强制重建所有索引")
        db.conn.execute("DELETE FROM fts5_index")
        db.conn.execute("DELETE FROM vectors")
        db.conn.execute("DELETE FROM chunks")
        db.conn.execute("DELETE FROM files")
        db.conn.commit()

        report = scanner.scan_vaults(vault_configs, vault_excludes)

        for meta in report.new_files:
            db.upsert_file(meta.to_dict())
        db.conn.commit()

        cursor = db.conn.execute("SELECT id, absolute_path, file_path, mtime FROM files WHERE is_deleted = 0")
        files_to_index = [dict(row) for row in cursor.fetchall()]
        logger.info(f"📊 强制重建：发现 {len(files_to_index)} 个文件")
    else:
        report = scanner.scan_vaults(vault_configs, vault_excludes)
        scanner.process_report(report)
        logger.info("🔄 模式：增量更新")
        changed_paths = [f.absolute_path for f in report.new_files + report.modified_files]
        changed_paths.extend([f.new_absolute_path for f in report.moved_files])
        if changed_paths:
            placeholders = ",".join(["?"] * len(changed_paths))
            cursor = db.conn.execute(
                f"SELECT id, absolute_path, file_path, mtime FROM files WHERE absolute_path IN ({placeholders})",
                changed_paths,
            )
            files_to_index = [dict(row) for row in cursor.fetchall()]

        cursor = db.conn.execute("""
            SELECT f.id, f.absolute_path, f.file_path, f.mtime FROM files f
            WHERE f.is_deleted = 0 AND NOT EXISTS (SELECT 1 FROM chunks c WHERE c.file_id = f.id) LIMIT 1000
        """)
        missing = [dict(row) for row in cursor.fetchall()]
        if missing:
            logger.info(f"⚠️ 发现 {len(missing)} 个文件缺少 chunks，补充索引中...")
            files_to_index.extend(missing)

    if not files_to_index:
        logger.info("✨ 没有检测到变更，索引已是最新。")
        db.close()
        return

    stream_batch_size = config.stream_batch_size
    max_concurrent_files = config.max_concurrent_files

    logger.info(f"🚀 开始处理 {len(files_to_index)} 个文件（流式模式）...")
    logger.info(f"⚙️ 配置: batch_size={config.embedding_model.batch_size}, "
                f"stream_batch_size={stream_batch_size}, max_concurrent_files={max_concurrent_files}")

    pending_chunks: list[tuple[int, Any, str]] = []
    total_processed = 0
    global_chunk_idx = 0
    total_files_with_chunks = 0

    try:
        from tqdm import tqdm
        pbar = tqdm(total=len(files_to_index), desc="文件处理", unit="文件", file=sys.stdout, leave=True)
    except ImportError:
        pbar = None

    start_time = time.time()

    try:
        with ThreadPoolExecutor(max_workers=max_concurrent_files) as executor:
            for f_id, chunks, f_path in executor.map(
                lambda f: process_file_worker(f, splitter),
                files_to_index
            ):
                for c in chunks:
                    pending_chunks.append((f_id, c, f_path))

                if chunks:
                    total_files_with_chunks += 1

                if len(pending_chunks) >= stream_batch_size:
                    global_chunk_idx = process_and_commit_batch(
                        pending_chunks, embedder, db, global_chunk_idx
                    )
                    total_processed += len(pending_chunks)
                    logger.debug(f"✅ 已处理 {total_processed} 个 chunks")
                    pending_chunks.clear()

                if pbar:
                    pbar.update(1)

        if pending_chunks:
            global_chunk_idx = process_and_commit_batch(
                pending_chunks, embedder, db, global_chunk_idx
            )
            total_processed += len(pending_chunks)
            pending_chunks.clear()

    except Exception as e:
        logger.error(f"❌ 索引构建失败: {e}", exc_info=True)
        if pbar:
            pbar.close()
        db.close()
        sys.exit(1)

    if pbar:
        pbar.close()

    elapsed = time.time() - start_time
    logger.success(
        f"🎉 索引构建完成！共处理 {total_files_with_chunks} 个文件，"
        f"{total_processed} 个 chunks，耗时：{elapsed:.2f}s"
    )
    db.close()


if __name__ == "__main__":
    main()
