#!/usr/bin/env python3
"""
build_index.py - tinyRAG 高性能索引构建器 (P0/P1 修复版)
修复清单:
1. ✅ P0: 向量化失败不再 continue，改为阻断并明确报错，防止数据错位
2. ✅ P1: tqdm 与 loguru 终端冲突修复 (tqdm 输出至 stdout)
3. ✅ 依赖对齐: MarkdownSplitter(config), DatabaseManager(vec_dimension)
4. ✅ 进度追踪: 失败批次安全回滚，最终统一输出统计
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
from config import get_merged_exclude, load_config
from embedder.embed_engine import EmbeddingEngine
from scanner.scan_engine import DEFAULT_SKIP_DIRS, Scanner
from storage.database import DatabaseManager
from utils.logger import logger, setup_logger

# 初始化日志 (默认输出到 stderr，与 tqdm 的 stdout 互不干扰)
setup_logger(level="INFO", log_file=str(_script_dir / "logs" / "build_index.log"))

def _jieba_segment(text: str) -> str:
    """对中文文本进行 jieba 分词，返回空格拼接的词串"""
    if not text or not text.strip():
        return ""
    return " ".join(jieba.cut_for_search(text))

def prepare_fts_content(chunk, file_path: str) -> str:
    """构建复合检索字符串，所有中文文本字段经 jieba 分词后写入 FTS5"""
    metadata = chunk.metadata or {}
    tags = metadata.get("tags", [])
    if tags is None: tags = []
    if isinstance(tags, str): tags = [tags]
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
    file_id = file_item["id"]
    file_path = file_item["file_path"]
    
    # 防御性检查：文件是否存在
    if not abs_path.exists():
        logger.warning(f"⚠️ 文件不存在，跳过：{abs_path}")
        return file_id, [], file_path
    
    try:
        content = abs_path.read_text(encoding="utf-8")
        mtime = file_item.get("mtime")
        chunks = splitter.split(content, mtime)
        
        # 详细日志：分块结果
        if not chunks:
            logger.warning(f"⚠️ 文件分块为空：{file_path} (内容长度: {len(content)} 字符)")
        
        return file_id, chunks, file_path
    except Exception as e:
        logger.error(f"❌ 读取/分块失败：{abs_path} - {type(e).__name__}: {e}")
        return file_id, [], file_path

def main():
    parser = argparse.ArgumentParser(description="tinyRAG 高性能索引构建器")
    parser.add_argument("--force", action="store_true", help="重建所有索引")
    parser.add_argument("--batch-size", type=int, default=128, help="向量化批大小")
    args = parser.parse_args()

    # 1. 初始化核心组件
    try:
        config = load_config("config.yaml")
    except Exception as e:
        logger.critical(f"❌ 配置加载失败：{e}")
        sys.exit(1)

    # 构建全局排除规则（默认 + 全局配置）
    global_skip_dirs = DEFAULT_SKIP_DIRS | frozenset(config.exclude.dirs)
    
    # 构建 vault 自身的排除规则（不包含全局配置）
    vault_excludes: dict[str, tuple[frozenset[str], list[str]]] = {}
    for v in config.vaults:
        if v.enabled:
            if v.exclude:
                # 只取 vault 自身的配置
                vault_excludes[v.name] = (frozenset(v.exclude.dirs), v.exclude.patterns)
            else:
                # vault 没有配置时为空
                vault_excludes[v.name] = (frozenset(), [])

    db = DatabaseManager(config.db_path, vec_dimension=config.embedding_model.dimensions)
    scanner = Scanner(db, skip_dirs=global_skip_dirs, global_patterns=config.exclude.patterns)
    splitter = MarkdownSplitter(config)
    embedder = EmbeddingEngine(
        model_name=config.embedding_model.name,
        cache_dir=config.embedding_model.cache_dir,
        batch_size=32,
        unload_after_seconds=config.embedding_model.unload_after_seconds,
    )

    vault_configs = [(v.name, v.path) for v in config.vaults if v.enabled]
    if not vault_configs:
        logger.warning("⚠️ 未启用任何仓库，跳过扫描")
        db.close()
        return

    logger.info(f"📂 将索引以下仓库: {[v[0] for v in vault_configs]}")
    if config.exclude.dirs:
        logger.info(f"🚫 全局排除目录: {config.exclude.dirs}")
    if config.exclude.patterns:
        logger.info(f"🚫 全局排除模式: {config.exclude.patterns}")
    
    report = scanner.scan_vaults(vault_configs, vault_excludes)
    scanner.process_report(report)

    # 2. 筛选待处理文件
    files_to_index = []
    if args.force:
        logger.info("🔄 模式：强制重建所有索引")
        db.conn.execute("DELETE FROM fts5_index")
        db.conn.execute("DELETE FROM vectors")
        db.conn.execute("DELETE FROM chunks")
        db.conn.execute("UPDATE files SET is_deleted = 0")
        db.conn.commit()
        cursor = db.conn.execute("SELECT id, absolute_path, file_path, mtime FROM files WHERE is_deleted = 0")
        files_to_index = [dict(row) for row in cursor.fetchall()]
    else:
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

        # 防御性回滚：补充缺失 chunks 的文件
        cursor = db.conn.execute("""
            SELECT f.id, f.absolute_path, f.file_path, f.mtime, f.vault_name FROM files f
            WHERE f.is_deleted = 0 AND NOT EXISTS (SELECT 1 FROM chunks c WHERE c.file_id = f.id) LIMIT 1000
        """)
        missing = [dict(row) for row in cursor.fetchall()]
        if missing:
            # 检查文件是否存在，过滤掉已被物理删除的文件
            existing_files = []
            deleted_file_ids = []
            excluded_files = []
            
            for f in missing:
                abs_path = Path(f["absolute_path"])
                
                # 检查文件是否存在
                if not abs_path.exists():
                    deleted_file_ids.append(f["id"])
                    logger.warning(f"⚠️ 文件已物理删除，将标记为删除：{f['file_path']}")
                    continue
                
                # 检查文件是否被排除规则排除
                vault_name = f["vault_name"]
                if vault_name in vault_excludes:
                    skip_dirs, patterns = vault_excludes[vault_name]
                    rel_path = f["file_path"]
                    
                    # 检查目录排除
                    excluded = False
                    for part in Path(rel_path).parts[:-1]:  # 排除文件名本身
                        if part in skip_dirs:
                            excluded = True
                            break
                    
                    # 检查模式排除
                    if not excluded and patterns:
                        import fnmatch
                        for pattern in patterns:
                            if fnmatch.fnmatch(rel_path, pattern):
                                excluded = True
                                break
                            for part in rel_path.split(os.sep):
                                if fnmatch.fnmatch(part, pattern):
                                    excluded = True
                                    break
                            if excluded:
                                break
                    
                    if excluded:
                        excluded_files.append(f["id"])
                        logger.info(f"⏭️ 文件被排除规则排除，跳过：{rel_path}")
                        continue
                
                existing_files.append(f)
            
            # 将物理删除的文件标记为删除
            if deleted_file_ids:
                placeholders = ",".join(["?"] * len(deleted_file_ids))
                db.conn.execute(f"UPDATE files SET is_deleted=1 WHERE id IN ({placeholders})", deleted_file_ids)
                db.conn.commit()
                logger.info(f"🗑️ 已标记 {len(deleted_file_ids)} 个不存在的文件为删除")
            
            # 将被排除规则排除的文件也标记为删除
            if excluded_files:
                placeholders = ",".join(["?"] * len(excluded_files))
                db.conn.execute(f"UPDATE files SET is_deleted=1 WHERE id IN ({placeholders})", excluded_files)
                db.conn.commit()
                logger.info(f"⏭️ 已标记 {len(excluded_files)} 个被排除的文件为删除")
            
            if existing_files:
                logger.info(f"⚠️ 发现 {len(existing_files)} 个文件缺少 chunks，补充索引中...")
                files_to_index.extend(existing_files)

    if not files_to_index:
        logger.info("✨ 没有检测到变更，索引已是最新。")
        db.close()
        return

    # 3. 跨文件分块收集
    logger.info(f"🚀 开始处理 {len(files_to_index)} 个文件...")
    all_pending_chunks = []
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = executor.map(lambda f: process_file_worker(f, splitter), files_to_index)
        for f_id, chunks, f_path in results:
            for c in chunks:
                all_pending_chunks.append((f_id, c, f_path))

    total_chunks = len(all_pending_chunks)
    if total_chunks == 0:
        logger.info("✨ 未生成任何有效 chunks。")
        db.close()
        return

    # 4. 分批向量化与入库 (🔧 tqdm 与 loguru 冲突修复)
    logger.info(f"🧩 待向量化块总数：{total_chunks}，Batch Size: {args.batch_size}")
    try:
        from tqdm import tqdm
        # ✅ 关键：指定 file=sys.stdout，与 loguru 的 stderr 物理隔离
        pbar = tqdm(total=total_chunks, desc="向量化进度", unit="块", file=sys.stdout, leave=True)
    except ImportError:
        pbar = None

    start_time = time.time()
    processed = 0

    for i in range(0, total_chunks, args.batch_size):
        batch = all_pending_chunks[i : i + args.batch_size]
        texts = [item[1].content for item in batch]

        try:
            # 🔴 P0 修复：向量化失败直接抛出，阻断流程防止数据错位
            embeddings = embedder.embed(texts)
        except Exception as e:
            logger.error(f"❌ 批次向量化失败: {e}，中断索引任务以保护数据一致性")
            if pbar: pbar.close()
            db.close()
            sys.exit(1)

        try:
            for idx, ((file_id, chunk, f_path), emb) in enumerate(zip(batch, embeddings)):
                global_idx = processed + idx
                metadata_json = json.dumps(chunk.metadata or {}, ensure_ascii=False, default=json_serialize)
                confidence_json = json.dumps(chunk.confidence_metadata or {}, ensure_ascii=False, default=json_serialize)

                cursor = db.conn.execute(
                    """INSERT INTO chunks (file_id, chunk_index, content, content_type, section_title, section_path,
                       start_pos, end_pos, confidence_final_weight, metadata, confidence_json)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (file_id, global_idx, chunk.content, chunk.content_type.value,
                     chunk.section_title, chunk.section_path, chunk.start_pos, chunk.end_pos,
                     1.0, metadata_json, confidence_json)
                )
                new_chunk_id = cursor.lastrowid

                if db.vec_support:
                    db.conn.execute("INSERT INTO vectors (chunk_id, embedding) VALUES (?, ?)",
                                    (new_chunk_id, array.array("f", emb).tobytes()))

                db.conn.execute("INSERT INTO fts5_index (rowid, content) VALUES (?, ?)",
                                (new_chunk_id, prepare_fts_content(chunk, f_path)))

            db.conn.commit()
            processed += len(batch)
            if pbar:
                pbar.update(len(batch))
        except Exception as e:
            db.conn.rollback()
            logger.error(f"❌ 批次提交失败：{e}", exc_info=True)
            if pbar: pbar.close()
            db.close()
            sys.exit(1)

    if pbar:
        pbar.close()

    elapsed = time.time() - start_time
    logger.success(f"🎉 索引构建完成！成功处理 {processed}/{total_chunks} 个 Chunk，耗时：{elapsed:.2f}s")
    db.close()

if __name__ == "__main__":
    main()
