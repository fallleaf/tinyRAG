#!/usr/bin/env python3
"""build_index.py - tinyRAG 高性能索引构建器 (v2.1)"""
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
from scanner.scan_engine import DEFAULT_SKIP_DIRS, Scanner
from storage.database import DatabaseManager
from utils.logger import logger, setup_logger
from utils.jieba_helper import jieba_segment, load_jieba_user_dict

setup_logger(level="INFO")

def prepare_fts_content(chunk, file_path: str) -> str:
    metadata = chunk.metadata or {}
    tags = metadata.get("tags", [])
    if tags is None: tags = []
    if isinstance(tags, str): tags = [tags]
    tag_str = " ".join([f"#{t.strip()}" for t in tags if t])
    doc_type = metadata.get("doc_type") or ""
    filename = os.path.basename(file_path)
    section_title = chunk.section_title or ""
    parts = [
        jieba_segment(filename), jieba_segment(filename),
        jieba_segment(chunk.section_path or ""), jieba_segment(section_title),
        jieba_segment(section_title), jieba_segment(tag_str),
        jieba_segment(doc_type), jieba_segment(chunk.content)
    ]
    return " ".join(filter(None, parts)).strip()

def json_serialize(obj):
    from datetime import date, datetime
    if isinstance(obj, (datetime, date)): return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

def process_file_worker(file_item: dict, splitter: MarkdownSplitter) -> tuple[int, list[Any], str]:
    abs_path = Path(file_item["absolute_path"])
    try:
        content = abs_path.read_text(encoding="utf-8")
        mtime = file_item.get("mtime")
        chunks = splitter.split(content, mtime)
        return file_item["id"], chunks, file_item["file_path"]
    except Exception as e:
        logger.error(f"❌ 读取/分块失败：{abs_path} - {e}")
        return file_item["id"], [], file_item["file_path"]

def process_and_commit_batch(chunks: list[tuple[int, Any, str]], embedder: EmbeddingEngine, db: DatabaseManager, start_idx: int, commit: bool = True) -> int:
    if not chunks: return start_idx
    texts = [c[1].content for c in chunks]
    embeddings = embedder.embed(texts)
    for idx, ((file_id, chunk, f_path), emb) in enumerate(zip(chunks, embeddings, strict=False)):
        chunk_idx = start_idx + idx
        metadata_json = json.dumps(chunk.metadata or {}, ensure_ascii=False, default=json_serialize)
        confidence_json = json.dumps(chunk.confidence_metadata or {}, ensure_ascii=False, default=json_serialize)
        cursor = db.conn.execute(
            """INSERT INTO chunks (file_id, chunk_index, content, content_type, section_title, section_path, start_pos, end_pos, confidence_final_weight, metadata, confidence_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (file_id, chunk_idx, chunk.content, chunk.content_type.value, chunk.section_title, chunk.section_path, chunk.start_pos, chunk.end_pos, 1.0, metadata_json, confidence_json)
        )
        new_chunk_id = cursor.lastrowid
        if db.vec_support:
            db.conn.execute("INSERT INTO vectors (chunk_id, embedding) VALUES (?, ?)", (new_chunk_id, array.array("f", emb).tobytes()))
        db.conn.execute("INSERT INTO fts5_index (rowid, content) VALUES (?, ?)", (new_chunk_id, prepare_fts_content(chunk, f_path)))
    if commit: db.conn.commit()
    return start_idx + len(chunks)

def main():
    parser = argparse.ArgumentParser(description="tinyRAG 高性能索引构建器")
    parser.add_argument("--force", action="store_true", help="重建所有索引")
    parser.add_argument("--batch-size", type=int, default=128, help="向量化批大小")
    args = parser.parse_args()
    try: config = load_config("config.yaml")
    except Exception as e: logger.critical(f"❌ 配置加载失败：{e}"); sys.exit(1)
    
    load_jieba_user_dict(config)
    db = DatabaseManager(config.db_path, vec_dimension=config.embedding_model.dimensions)
    global_skip_dirs = DEFAULT_SKIP_DIRS | frozenset(config.exclude.dirs)
    scanner = Scanner(db, skip_dirs=global_skip_dirs, global_patterns=config.exclude.patterns)
    splitter = MarkdownSplitter(config)
    embedder = EmbeddingEngine(model_name=config.embedding_model.name, cache_dir=config.embedding_model.cache_dir, batch_size=config.embedding_model.batch_size, unload_after_seconds=config.embedding_model.unload_after_seconds)
    
    vault_configs = [(v.name, v.path) for v in config.vaults if v.enabled]
    if not vault_configs: logger.warning("⚠️ 未启用任何仓库"); db.close(); return
    vault_excludes: dict[str, tuple[frozenset[str], list[str]]] = {}
    for v in config.vaults:
        if v.enabled:
            if v.exclude: vault_excludes[v.name] = (frozenset(v.exclude.dirs), v.exclude.patterns)
            else: vault_excludes[v.name] = (frozenset(), [])

    files_to_index = []
    if args.force:
        logger.info("🔄 模式：强制重建所有索引")
        db.conn.execute("DELETE FROM fts5_index")
        if db.conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='vectors'").fetchone(): db.conn.execute("DELETE FROM vectors")
        db.conn.execute("DELETE FROM chunks"); db.conn.execute("DELETE FROM files")
        db.conn.commit()
        report = scanner.scan_vaults(vault_configs, vault_excludes)
        for meta in report.new_files: db.upsert_file(meta.to_dict())
        db.conn.commit()
        files_to_index = [dict(row) for row in db.conn.execute("SELECT id, absolute_path, file_path, mtime FROM files WHERE is_deleted = 0").fetchall()]
    else:
        report = scanner.scan_vaults(vault_configs, vault_excludes)
        scanner.process_report(report)
        changed_paths = [f.absolute_path for f in report.new_files + report.modified_files]
        changed_paths.extend([f.new_absolute_path for f in report.moved_files])
        if changed_paths:
            placeholders = ",".join(["?"] * len(changed_paths))
            files_to_index = [dict(row) for row in db.conn.execute(f"SELECT id, absolute_path, file_path, mtime FROM files WHERE absolute_path IN ({placeholders})", changed_paths).fetchall()]
        missing = [dict(row) for row in db.conn.execute("SELECT f.id, f.absolute_path, f.file_path, f.mtime FROM files f WHERE f.is_deleted = 0 AND NOT EXISTS (SELECT 1 FROM chunks c WHERE c.file_id = f.id) LIMIT 1000").fetchall()]
        if missing: logger.info(f"⚠️ 补充 {len(missing)} 个缺失 chunks"); files_to_index.extend(missing)
        if not files_to_index: logger.info("✨ 无变更，跳过构建"); db.close(); return

    stream_batch_size = getattr(config, "stream_batch_size", 100)
    max_concurrent_files = getattr(config, "max_concurrent_files", os.cpu_count() or 4)
    logger.info(f"🚀 开始处理 {len(files_to_index)} 个文件...")
    
    pending_chunks: list[tuple[int, Any, str]] = []
    total_processed = 0
    global_chunk_idx = 0
    total_files_with_chunks = 0
    start_time = time.time()
    
    try:
        from tqdm import tqdm
        pbar = tqdm(total=len(files_to_index), desc="文件处理", unit="文件", file=sys.stdout, leave=True)
    except ImportError:
        pbar = None

    db.begin_bulk_insert()
    try:
        with ThreadPoolExecutor(max_workers=max_concurrent_files) as executor:
            for f_id, chunks, f_path in executor.map(lambda f: process_file_worker(f, splitter), files_to_index):
                for c in chunks: pending_chunks.append((f_id, c, f_path))
                if chunks: total_files_with_chunks += 1
                if len(pending_chunks) >= stream_batch_size:
                    global_chunk_idx = process_and_commit_batch(pending_chunks, embedder, db, global_chunk_idx, commit=False)
                    total_processed += len(pending_chunks)
                    pending_chunks.clear()
                if pbar: pbar.update(1)
        if pending_chunks:
            global_chunk_idx = process_and_commit_batch(pending_chunks, embedder, db, global_chunk_idx, commit=False)
            total_processed += len(pending_chunks)
        db.end_bulk_insert(commit=True)
    except Exception as e:
        db.end_bulk_insert(commit=False)
        logger.error(f"❌ 索引构建失败: {e}", exc_info=True)
        if pbar: pbar.close()
        db.close()
        sys.exit(1)
    if pbar: pbar.close()
    logger.success(f"🎉 构建完成！处理 {total_files_with_chunks} 文件，{total_processed} chunks，耗时：{time.time()-start_time:.2f}s")
    db.close()

if __name__ == "__main__":
    main()