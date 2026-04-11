#!/usr/bin/env python3
"""
build_index.py - tinyRAG 高性能索引构建器 (v2.0 流式处理版)

v2.0 优化内容:
1. ✅ 流式处理: 分批 chunks -> 向量化 -> 入库 -> 释放内存，避免全量内存占用
2. ✅ 统一 batch_size: 使用 config.embedding_model.batch_size 作为模型推理批大小
3. ✅ 内存优化: 限制并发文件数，适合低内存设备（如 4G RAM）
4. ✅ 进度追踪: 实时显示处理进度，支持断点续传

历史修复:
1. ✅ P0: 向量化失败不再 continue，改为阻断并明确报错，防止数据错位
2. ✅ P1: tqdm 与 loguru 终端冲突修复 (tqdm 输出至 stdout)
3. ✅ 依赖对齐: MarkdownSplitter(config), DatabaseManager(vec_dimension)
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

# 确保项目根目录在 sys.path
_script_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(_script_dir))

import jieba

from chunker.markdown_splitter import MarkdownSplitter
from config import load_config
from embedder.embed_engine import EmbeddingEngine
from scanner.scan_engine import Scanner
from storage.database import DatabaseManager
from utils.logger import logger, setup_logger

# 初始化日志 (默认输出到 stderr，与 tqdm 的 stdout 互不干扰)
setup_logger(level="INFO", log_file=str(_script_dir / "logs" / "build_index.log"))

# jieba 用户字典加载状态（避免重复加载）
_jieba_dict_loaded = False


def _ensure_jieba_user_dict(config):
    """加载 jieba 用户自定义词典（仅首次调用时加载）"""
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
    # 调试：验证分词效果
    test_text = "极简网络方案"
    result = list(jieba.cut_for_search(test_text))
    logger.info(f"🔍 分词测试: '{test_text}' -> {result}")

def _jieba_segment(text: str) -> str:
    """对中文文本进行 jieba 分词，返回空格拼接的词串"""
    if not text or not text.strip():
        return ""
    return " ".join(jieba.cut_for_search(text))


def prepare_fts_content(chunk, file_path: str) -> str:
    """构建复合检索字符串，所有中文文本字段经 jieba 分词后写入 FTS5"""
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
        _jieba_segment(filename), _jieba_segment(filename),  # 文件名重复加权
        _jieba_segment(chunk.section_path or ""),
        _jieba_segment(section_title), _jieba_segment(section_title),  # 标题重复加权
        _jieba_segment(tag_str), _jieba_segment(doc_type),
        _jieba_segment(chunk.content),  # 正文分词
    ]
    return " ".join(filter(None, parts)).strip()


def json_serialize(obj):
    """自定义 JSON 序列化器，处理 datetime/date 对象"""
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
    """
    处理一批 chunks：向量化 + 入库
    :param chunks: 待处理的 chunks 列表，格式 [(file_id, chunk, file_path), ...]
    :param embedder: 向量化引擎
    :param db: 数据库管理器
    :param start_idx: 起始 chunk 索引
    :return: 处理后的下一个 chunk 索引
    """
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

    # 1. 初始化核心组件
    try:
        config = load_config("config.yaml")
    except Exception as e:
        logger.critical(f"❌ 配置加载失败：{e}")
        sys.exit(1)

    # 加载 jieba 用户字典（必须在分词前加载）
    _ensure_jieba_user_dict(config)

    db = DatabaseManager(config.db_path, vec_dimension=config.embedding_model.dimensions)
    scanner = Scanner(db)
    splitter = MarkdownSplitter(config)
    
    # 使用配置中的 batch_size（模型推理批大小）
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

    logger.info(f"📂 将索引以下仓库: {[v[0] for v in vault_configs]}")

    # 2. 根据模式决定处理方式
    files_to_index = []
    if args.force:
        # 强制重建：清空所有表，重新扫描文件系统
        logger.info("🔄 模式：强制重建所有索引")
        db.conn.execute("DELETE FROM fts5_index")
        db.conn.execute("DELETE FROM vectors")
        db.conn.execute("DELETE FROM chunks")
        db.conn.execute("DELETE FROM files")  # 清空 files 表，避免幽灵文件
        db.conn.commit()

        # 重新扫描文件系统
        report = scanner.scan_vaults(vault_configs)

        # 将扫描结果插入 files 表
        for meta in report.new_files:
            db.upsert_file(meta.to_dict())
        db.conn.commit()

        # 从新插入的记录中获取待处理文件
        cursor = db.conn.execute("SELECT id, absolute_path, file_path, mtime FROM files WHERE is_deleted = 0")
        files_to_index = [dict(row) for row in cursor.fetchall()]
        logger.info(f"📊 强制重建：发现 {len(files_to_index)} 个文件")
    else:
        # 增量更新：先扫描，再处理变更
        report = scanner.scan_vaults(vault_configs)
        scanner.process_report(report)
        logger.info("🔄 模式：增量更新")
        # new_files 和 modified_files 使用 absolute_path，moved_files 使用 new_absolute_path
        changed_paths = [f.absolute_path for f in report.new_files + report.modified_files]
        changed_paths.extend([f.new_absolute_path for f in report.moved_files])
        if changed_paths:
            placeholders = ",".join(["?"] * len(changed_paths))
            cursor = db.conn.execute(
                f"SELECT id, absolute_path, file_path, mtime FROM files WHERE absolute_path IN ({placeholders})",
                changed_paths,
            )
            files_to_index = [dict(row) for row in cursor.fetchall()]

        # 防御性回滚：补充缺失 chunks 的文件
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

    # 3. 流式分块与向量化（内存优化）
    stream_batch_size = config.stream_batch_size  # 每累积多少 chunks 进行一次入库
    max_concurrent_files = config.max_concurrent_files  # 并行处理文件数的上限
    
    logger.info(f"🚀 开始处理 {len(files_to_index)} 个文件（流式模式）...")
    logger.info(f"⚙️ 配置: batch_size={config.embedding_model.batch_size}, "
                f"stream_batch_size={stream_batch_size}, max_concurrent_files={max_concurrent_files}")

    pending_chunks: list[tuple[int, Any, str]] = []
    total_processed = 0
    global_chunk_idx = 0
    total_files_with_chunks = 0

    # 进度条
    try:
        from tqdm import tqdm
        pbar = tqdm(total=len(files_to_index), desc="文件处理", unit="文件", file=sys.stdout, leave=True)
    except ImportError:
        pbar = None

    start_time = time.time()

    try:
        # 限制并发数，减少内存峰值
        with ThreadPoolExecutor(max_workers=max_concurrent_files) as executor:
            for f_id, chunks, f_path in executor.map(
                lambda f: process_file_worker(f, splitter),
                files_to_index
            ):
                for c in chunks:
                    pending_chunks.append((f_id, c, f_path))

                if chunks:
                    total_files_with_chunks += 1

                # 达到阈值立即处理，释放内存
                if len(pending_chunks) >= stream_batch_size:
                    global_chunk_idx = process_and_commit_batch(
                        pending_chunks, embedder, db, global_chunk_idx
                    )
                    total_processed += len(pending_chunks)
                    logger.debug(f"✅ 已处理 {total_processed} 个 chunks")
                    pending_chunks.clear()  # 🔑 释放内存！

                if pbar:
                    pbar.update(1)

        # 处理剩余的 chunks
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
