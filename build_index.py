#!/usr/bin/env python3
"""
build_index.py - RAG System 高性能索引构建器
修复说明:
1. ✅ 修复 vault_configs 元组访问错误 (v[0] 替代 v.name)
2. ✅ Pydantic v2 兼容: .dict() → .model_dump()
3. ✅ 保持语义化 vault_name 与 enabled 过滤逻辑
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from loguru import logger

from chunker.markdown_splitter import MarkdownSplitter
from config import load_config
from embedder.embed_engine import EmbeddingEngine
from scanner.scan_engine import Scanner
from storage.database import DatabaseManager
from utils.logger import setup_logger

# 初始化日志
setup_logger(level="INFO", log_file="logs/build_index.log")


def prepare_fts_content(chunk, file_path: str) -> str:
    """构建复合检索字符串，增强关键词命中率"""
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
        filename,
        filename,  # 文件名重复加权
        chunk.section_path or "",
        section_title,
        section_title,
        tag_str,
        doc_type,
        chunk.content,
    ]
    return " ".join(filter(None, parts)).strip()


def json_serialize(obj):
    """自定义 JSON 序列化器"""
    from datetime import date, datetime

    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def process_file_worker(
    file_item: dict, splitter: MarkdownSplitter
) -> tuple[int, list[Any], str]:
    """并行分块任务单元"""
    abs_path = Path(file_item["absolute_path"])
    try:
        content = abs_path.read_text(encoding="utf-8")
        chunks = splitter.split(content, file_item["file_path"])
        return file_item["id"], chunks, file_item["file_path"]
    except Exception as e:
        logger.error(f"❌ 读取/分块失败：{abs_path} - {e}")
        return file_item["id"], [], file_item["file_path"]


def main():
    parser = argparse.ArgumentParser(description="RAG System 高性能索引构建器")
    parser.add_argument("--force", action="store_true", help="重建所有索引")
    parser.add_argument("--batch-size", type=int, default=128, help="向量化批大小")
    args = parser.parse_args()

    # 1. 初始化核心组件
    try:
        config = load_config("config.yaml")
    except Exception as e:
        logger.critical(f"❌ 配置加载失败：{e}")
        return 1

    db = DatabaseManager(config.db_path)
    scanner = Scanner(db)

    # ✅ 修复：Pydantic v2 使用 model_dump()
    splitter = MarkdownSplitter(
        max_tokens=config.chunking["max_tokens"],
        overlap=config.chunking["overlap"],
        confidence_config=config.confidence.model_dump(),
    )

    embedder = EmbeddingEngine(
        model_name=config.embedding_model.name,
        cache_dir=config.embedding_model.cache_dir,
        batch_size=32,
        unload_after_seconds=config.embedding_model.unload_after_seconds,
    )

    # 2.✅ 核心：严格使用 config 中的 name 字段，禁用 v_0/v_1
    vault_configs = [(v.name, v.path) for v in config.vaults if v.enabled]

    if not vault_configs:
        logger.warning("⚠️ 未启用任何仓库，跳过扫描")
        return

    # 📢 明确告知将写入 DB 的 vault_name
    target_names = [v[0] for v in vault_configs]
    logger.info(f"📂 将索引以下仓库 (DB vault_name): {target_names}")
    logger.info(f"📁 对应物理路径: {[v[1] for v in vault_configs]}")

    # 执行扫描
    report = scanner.scan_vaults(vault_configs)
    scanner.process_report(report)

    # 3. 筛选待处理文件
    files_to_index = []
    if args.force:
        logger.info("🔄 模式：强制重建所有索引")
        cursor = db.conn.execute(
            "SELECT id, absolute_path, file_path FROM files WHERE is_deleted = 0"
        )
        files_to_index = [dict(row) for row in cursor.fetchall()]
        db.conn.execute("DELETE FROM fts5_index")
        db.conn.execute("DELETE FROM vectors")
        db.conn.execute("DELETE FROM chunks")
    else:
        logger.info("🔄 模式：增量更新")
        changed_paths = [
            f.absolute_path for f in report.new_files + report.modified_files
        ]

        if changed_paths:
            placeholders = ",".join(["?"] * len(changed_paths))
            cursor = db.conn.execute(
                f"SELECT id, absolute_path, file_path FROM files WHERE absolute_path IN ({placeholders})",
                changed_paths,
            )
            files_to_index = [dict(row) for row in cursor.fetchall()]

        # 检查缺失 chunks 的文件
        cursor = db.conn.execute("""
            SELECT f.id, f.absolute_path, f.file_path
            FROM files f
            WHERE f.is_deleted = 0
            AND NOT EXISTS (SELECT 1 FROM chunks c WHERE c.file_id = f.id)
            LIMIT 1000
        """)
        missing_chunks_files = [dict(row) for row in cursor.fetchall()]
        if missing_chunks_files:
            logger.info(
                f"⚠️ 发现 {len(missing_chunks_files)} 个文件缺少 chunks，补充索引中..."
            )
            files_to_index.extend(missing_chunks_files)

    if not files_to_index:
        logger.info("✨ 没有检测到变更，索引已是最新。")
        return

    # 4. 跨文件分块收集
    logger.info(f"🚀 开始处理 {len(files_to_index)} 个文件...")
    all_pending_chunks = []

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = executor.map(
            lambda f: process_file_worker(f, splitter), files_to_index
        )
        for f_id, chunks, f_path in results:
            for c in chunks:
                all_pending_chunks.append((f_id, c, f_path))

    # 5. 分批向量化与入库
    total_chunks = len(all_pending_chunks)
    logger.info(
        f"🧩 待向量化块总数：{total_chunks}，采用 Batch Size: {args.batch_size}"
    )

    try:
        from tqdm import tqdm

        pbar = tqdm(total=total_chunks, desc="向量化进度", unit="块")
    except ImportError:
        pbar = None

    start_time = time.time()
    processed = 0

    for i in range(0, total_chunks, args.batch_size):
        batch = all_pending_chunks[i : i + args.batch_size]
        texts = [item[1].content for item in batch]

        try:
            embeddings = embedder.embed(texts)
        except Exception as e:
            logger.error(f"❌ 批次向量化失败: {e}")
            continue

        try:
            for idx, ((file_id, chunk, f_path), emb) in enumerate(
                zip(batch, embeddings)
            ):
                try:
                    metadata_json = json.dumps(
                        chunk.metadata or {}, ensure_ascii=False, default=json_serialize
                    )
                except Exception as e:
                    logger.warning(f"⚠️ metadata 序列化失败，使用空对象：{e}")
                    metadata_json = "{}"

                cursor = db.conn.execute(
                    """
                    INSERT INTO chunks (file_id, chunk_index, content, content_type, section_title, section_path,
                    start_pos, end_pos, confidence_final_weight, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        file_id,
                        chunk.chunk_index,
                        chunk.content,
                        chunk.content_type.value,
                        chunk.section_title,
                        chunk.section_path,
                        chunk.start_pos,
                        chunk.end_pos,
                        chunk.confidence_final_weight,
                        metadata_json,
                    ),
                )
                new_chunk_id = cursor.lastrowid

                if db.vec_support:
                    import array

                    db.conn.execute(
                        "INSERT INTO vectors (chunk_id, embedding) VALUES (?, ?)",
                        (new_chunk_id, array.array("f", emb).tobytes()),
                    )

                fts_text = prepare_fts_content(chunk, f_path)
                db.conn.execute(
                    "INSERT INTO fts5_index (rowid, content) VALUES (?, ?)",
                    (new_chunk_id, fts_text),
                )

            db.conn.commit()
            processed += len(batch)

            if pbar:
                pbar.update(len(batch))
            else:
                logger.info(f" ✅ 已完成：{processed} / {total_chunks}")

        except Exception as e:
            db.conn.rollback()
            import traceback

            logger.error(f"❌ 批次提交失败：{e}")
            logger.error(traceback.format_exc())
            if pbar:
                pbar.close()
            raise

    if pbar:
        pbar.close()

    logger.success(f"🎉 索引构建完成！耗时：{time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()
