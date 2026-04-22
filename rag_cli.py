#!/usr/bin/env python3
"""
rag_cli.py - tinyRAG 命令行检索与运维工具 (v2.2 - 插件支持)
功能:
- search: 执行混合检索 (支持 console/json/csv 导出、Verbose调试、插件增强)
- status: 查看数据库和索引状态 (DB直查，毫秒级响应，插件状态)
- config: 查看/编辑/验证配置
- index: 索引管理 (build/scan)
- maintenance: 数据库运维 (vacuum/soft-delete清理)
用法:
python rag_cli.py search "极简网络" --top-k 5 --output json
python rag_cli.py status
python rag_cli.py index build --force
python rag_cli.py maintenance --dry-run
python rag_cli.py config --validate
"""

import argparse

# ✅ 在 rag_cli.py 顶部导入区添加以下 4 行
import array
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from io import StringIO
from pathlib import Path

from chunker.markdown_splitter import MarkdownSplitter

# 确保项目根目录在 sys.path
script_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(script_dir))

# 导入底层模块供 CLI 调用
import build_index as build_mod
import vacuum as vac_mod
from config import load_config
from embedder.embed_engine import EmbeddingEngine

# 插件支持
from plugins.bootstrap import PluginLoader
from retriever.hybrid_engine import HybridEngine
from scanner.scan_engine import DEFAULT_SKIP_DIRS, Scanner
from storage.database import DatabaseManager
from utils.jieba_helper import load_jieba_user_dict
from utils.logger import logger


def cmd_status(args):
    """显示系统状态 (优化版：DB直查 + 全维度配置摘要)"""
    logger.info("\n📊 tinyRAG 系统状态\n" + "=" * 50)
    try:
        cfg = load_config()
        db_path = Path(cfg.db_path).resolve()
        logger.info(f"🗄️ 数据库路径: {db_path}")

        if db_path.exists():
            db_size = db_path.stat().st_size / 1024 / 1024
            logger.info(f"   大小: {db_size:.2f} MB")

            db = DatabaseManager(str(db_path), vec_dimension=cfg.embedding_model.dimensions)
            try:
                # 1. 活跃数据
                files_active = db.conn.execute("SELECT COUNT(*) FROM files WHERE is_deleted=0").fetchone()[0]
                chunks_active = db.conn.execute("SELECT COUNT(*) FROM chunks WHERE is_deleted=0").fetchone()[0]
                logger.info(f"   活跃 Files: {files_active} | 活跃 Chunks: {chunks_active}")

                # 2. 软删除统计 (运维关键指标)
                files_deleted = db.conn.execute("SELECT COUNT(*) FROM files WHERE is_deleted=1").fetchone()[0]
                chunks_deleted = db.conn.execute("SELECT COUNT(*) FROM chunks WHERE is_deleted=1").fetchone()[0]
                logger.info(f"   软删除 Files: {files_deleted} | 软删除 Chunks: {chunks_deleted}")

                # 3. 向量引擎状态
                logger.info(f"   向量引擎: {'✅ sqlite-vec 已启用' if db.vec_support else '⚠️ 已降级 (FTS5)'}")

                # 4. 按 Vault 统计 (直接查 DB，替代慢速 os.walk)
                logger.info("\n📂 Vault 索引状态:")
                vault_stats = db.conn.execute(
                    "SELECT vault_name, COUNT(*) as cnt FROM files WHERE is_deleted=0 GROUP BY vault_name"
                ).fetchall()
                total_db_files = 0
                for row in vault_stats:
                    logger.info(f"   ✅ {row['vault_name']}:  {row['cnt']} 个已索引文件")
                    total_db_files += row["cnt"]
                if total_db_files == 0:
                    logger.info("   ⚠️ 暂无已索引文件，请运行 `python rag_cli.py index build --force`")
                else:
                    logger.info(f"   📈 总计: {total_db_files} 个已索引文件")

                # 5. 缓存状态
                cache_path = Path(cfg.cache.db_path).resolve()
                logger.info(
                    f"\n💾 查询缓存: {cfg.cache.db_path} (TTL={cfg.cache.ttl_seconds}s, Max={cfg.cache.max_entries})"
                )
                if cache_path.exists():
                    cache_size = cache_path.stat().st_size / 1024 / 1024
                    logger.info(f"   状态: ✅ 已存在 ({cache_size:.2f} MB)")
                else:
                    logger.info("   状态: ⏳ 未初始化 (首次检索时自动创建)")

            finally:
                db.close()
        else:
            logger.info("❌ 数据库未初始化，请运行: `python rag_cli.py index build --force`")

        # 核心配置摘要 (便于快速排查参数)
        logger.info("\n⚙️ 核心配置摘要")
        logger.info(
            f"   🤖 模型: {cfg.embedding_model.name} (dim={cfg.embedding_model.dimensions}, batch={cfg.embedding_model.batch_size})"
        )
        logger.info(f"   🔍 检索: alpha={cfg.retrieval.get('alpha', 0.7)}, beta={cfg.retrieval.get('beta', 0.3)}")
        logger.info(f"   ✂️ 分块: max_tokens={cfg.chunking.max_tokens}, overlap={cfg.chunking.overlap}")
        logger.info(
            f"   🕒 衰减: {'✅ 启用' if cfg.confidence.date_decay.enabled else '❌ 禁用'} (半衰期={cfg.confidence.date_decay.half_life_days}天)"
        )

        # 插件状态
        logger.info("\n🔌 插件系统")
        logger.info(f"   状态: {'✅ 已启用' if cfg.plugins.enabled else '❌ 已禁用'}")
        if cfg.plugins.enabled and cfg.plugins.plugins:
            logger.info("   已配置插件:")
            for p in cfg.plugins.plugins:
                status = "✅" if p.enabled else "⏸️"
                logger.info(f"      {status} {p.name} (priority={p.priority})")

        logger.info("\n" + "=" * 50)
        return 0
    except Exception as e:
        logger.error(f"❌ 状态检查失败: {e}")
        return 1


def cmd_search(args):
    """执行混合检索 (v2.2: 支持导出/Verbose/安全事务/插件增强)"""
    db = None
    plugin_loader = None
    try:
        cfg = load_config()
        db_path = Path(cfg.db_path).resolve()
        if not db_path.exists():
            logger.error("❌ 数据库不存在,请先运行索引构建.")
            return 1

        # 局部权重覆盖 (不污染全局配置)
        # 修复：--mode 设置默认权重，用户指定的 --alpha/--beta 优先级最高
        alpha = cfg.retrieval.get("alpha", 0.7)
        beta = cfg.retrieval.get("beta", 0.3)

        # 根据模式设置默认权重
        if args.mode == "keyword":
            alpha, beta = 0.0, 1.0
        elif args.mode == "semantic":
            alpha, beta = 1.0, 0.0
        # hybrid 模式使用配置文件的默认值（已在上面设置）

        # 用户显式指定的参数优先级最高（覆盖 mode 的默认值）
        if args.alpha is not None:
            alpha = args.alpha
        if args.beta is not None:
            beta = args.beta

        vaults = args.vaults if args.vaults else [v.name for v in cfg.vaults if v.enabled]
        vault_filter = vaults if vaults else None

        if args.verbose:
            import logging

            logging.getLogger().setLevel(logging.DEBUG)

        db = DatabaseManager(str(db_path), vec_dimension=cfg.embedding_model.dimensions)
        embed_engine = EmbeddingEngine(
            model_name=cfg.embedding_model.name,
            cache_dir=cfg.embedding_model.cache_dir,
            batch_size=cfg.embedding_model.batch_size,
            unload_after_seconds=cfg.embedding_model.unload_after_seconds,
        )
        retriever = HybridEngine(config=cfg, db=db, embed_engine=embed_engine)

        # 初始化插件系统 (用于增强检索)
        plugin_enabled = cfg.plugins.enabled  # 记录插件是否启用
        if plugin_enabled:
            try:
                plugin_loader = PluginLoader(cfg, None)
                plugin_loader.load_all()
                for plugin in plugin_loader.get_all_plugins().values():
                    if hasattr(plugin, "set_db_connection"):
                        plugin.set_db_connection(db.conn)
                    # 初始化插件组件
                    if hasattr(plugin, "_initialize_sync") and not getattr(plugin, "_initialized", True):
                        try:
                            plugin._initialize_sync()
                        except Exception as init_err:
                            logger.warning(f"⚠️ 插件初始化失败: {init_err}")
            except Exception as e:
                logger.warning(f"⚠️ 插件加载失败: {e}")
                plugin_loader = None
                plugin_enabled = False

        start = time.time()
        results = retriever.search(
            query=args.query, limit=args.top_k, vault_filter=vault_filter, alpha=alpha, beta=beta
        )

        # 插件增强检索（统一评分版）
        # 不启动插件时：使用基础检索分数
        # 启动插件时：保留基础分数，增加图谱分数作为增强
        query_vec = embed_engine.embed([args.query])[0] if plugin_loader else None
        if plugin_loader:
            try:
                # 修复：传递基础检索的权重参数给插件，使图谱增强能够感知用户指定的权重
                enhanced_results = plugin_loader.invoke_hook(
                    "on_search",
                    query=args.query,
                    results=[r.__dict__ for r in results],
                    query_vec=query_vec,
                    base_alpha=alpha,  # 传递基础检索的 alpha 权重
                    base_beta=beta,  # 传递基础检索的 beta 权重
                )
                # 如果插件返回了增强结果，使用增强结果替换原始结果
                if enhanced_results and isinstance(enhanced_results, list) and len(enhanced_results) > 0:
                    # enhanced_results 是钩子返回值列表，取第一个（插件的返回）
                    first_result = enhanced_results[0]
                    if isinstance(first_result, list) and len(first_result) > 0:
                        # 将插件增强结果转换为 RetrievalResult 兼容格式
                        from retriever.hybrid_engine import RetrievalResult

                        plugin_results = []
                        for r in first_result:
                            # 检查是否已经是 RetrievalResult
                            if isinstance(r, RetrievalResult):
                                plugin_results.append(r)
                            elif isinstance(r, dict):
                                # 从插件结果构建 RetrievalResult（统一评分，保留基础分数）
                                plugin_results.append(
                                    RetrievalResult(
                                        chunk_id=r.get("chunk_id", 0),
                                        content=r.get("content", ""),
                                        file_path=r.get("file_path", ""),
                                        absolute_path=r.get("absolute_path", r.get("file_path", "")),
                                        section=r.get("section", ""),
                                        start_pos=r.get("start_pos", 0),
                                        end_pos=r.get("end_pos", 0),
                                        vault_name=r.get("vault_name", ""),
                                        chunk_type=r.get("chunk_type", ""),
                                        # 保留基础检索分数
                                        semantic_score=r.get("semantic_score", 0.0),
                                        keyword_score=r.get("keyword_score", 0.0),
                                        confidence_score=r.get("confidence_score", 1.0),
                                        final_score=r.get("final_score", 0.0),
                                        confidence_reason=r.get("confidence_reason", ""),
                                        file_hash=r.get("file_hash", ""),
                                        # 图谱增强分值（插件启用时才有）
                                        graph_score=r.get("graph_score", 0.0),
                                        preference_score=r.get("preference_score", 0.0),
                                        hop_distance=r.get("hop_distance", 0),
                                        # 基础检索分数（修复问题3）
                                        base_final_score=r.get("base_final_score", r.get("final_score", 0.0)),
                                    )
                                )
                        if plugin_results:
                            results = plugin_results
                            if args.verbose:
                                logger.info(f"✅ 插件增强了 {len(results)} 条结果（统一评分 + 图谱分值）")
            except Exception as e:
                logger.warning(f"⚠️ 插件增强失败: {e}")

        elapsed = time.time() - start

        # 📦 导出支持
        if args.output == "json":
            # 构建统一的 JSON 输出格式
            output_data = {
                "query": args.query,
                "total": len(results),
                "plugin_enabled": plugin_enabled,
                "elapsed_ms": round(elapsed * 1000, 2),
                "results": [],
            }
            for i, r in enumerate(results):
                result_item = {
                    "rank": i + 1,
                    "file": r.file_path,
                    "abs_path": r.absolute_path,
                    "content": r.content[:500],
                    "score": round(r.final_score, 4),
                    "semantic_score": round(r.semantic_score, 4),
                    "keyword_score": round(r.keyword_score, 4),
                    "confidence": round(r.confidence_score, 4),
                }
                # 插件启用时，添加图谱增强字段
                if plugin_enabled:
                    result_item["graph_score"] = round(getattr(r, "graph_score", 0.0), 4)
                    result_item["preference_score"] = round(getattr(r, "preference_score", 0.0), 4)
                    result_item["hop_distance"] = getattr(r, "hop_distance", 0)
                    # 修复问题3：添加 base_final_score 字段便于调试和验证
                    result_item["base_final_score"] = round(getattr(r, "base_final_score", r.final_score), 4)
                output_data["results"].append(result_item)
            logger.info(json.dumps(output_data, indent=2, ensure_ascii=False))
        elif args.output == "csv":
            output = StringIO()
            writer = csv.DictWriter(
                output,
                fieldnames=[
                    "rank",
                    "file_path",
                    "absolute_path",
                    "section",
                    "vault_name",
                    "chunk_type",
                    "content",
                    "final_score",
                    "confidence_score",
                    "confidence_reason",
                ],
            )
            writer.writeheader()
            for i, r in enumerate(results, 1):
                writer.writerow(
                    {
                        "rank": i,
                        "file_path": r.file_path,
                        "absolute_path": r.absolute_path,
                        "section": r.section,
                        "vault_name": r.vault_name,
                        "chunk_type": r.chunk_type,
                        "content": r.content[:200].replace("\n", " "),
                        "final_score": r.final_score,
                        "confidence_score": r.confidence_score,
                        "confidence_reason": r.confidence_reason,
                    }
                )
            logger.info(output.getvalue())
        else:
            # 🖥️ 终端输出
            plugin_tag = " [插件增强]" if plugin_enabled else ""
            logger.info(f"\n📊 检索结果 ({len(results)} 条, 耗时 {elapsed:.2f}s){plugin_tag}:\n")
            if not results:
                logger.info("   未找到相关结果.")
                return 0
            for i, r in enumerate(results, 1):
                content_preview = r.content[:200] + "..." if len(r.content) > 200 else r.content
                # 基础分值（始终显示）
                scores = f"最终={r.final_score:.3f} | 语义={r.semantic_score:.3f} | 关键词={r.keyword_score:.3f}"
                # 插件启用时，显示图谱增强分值
                if plugin_enabled:
                    graph_score = getattr(r, "graph_score", 0.0)
                    hop_distance = getattr(r, "hop_distance", 0)
                    scores += f" | 图={graph_score:.3f}"
                    if hop_distance > 0:
                        scores += f" (跳数={hop_distance})"
                scores += f" | 置信度={r.confidence_score:.2f}"
                logger.info(f"{i}. [{scores}]")
                logger.info(f"   来源:{r.absolute_path}")
                logger.info(f"   类型:{r.vault_name} / {r.chunk_type} | 章节:{r.section}")
                logger.info(f"   内容:{content_preview}\n")
        return 0
    except Exception as e:
        logger.error(f"❌ 检索失败: {e}", exc_info=True)
        return 1
    finally:
        if plugin_loader:
            plugin_loader.shutdown()
        if db:
            db.close()


def cmd_config(args):
    """显示或编辑配置"""
    config_path = script_dir / "config.yaml"
    if not any([args.show, args.edit, args.validate, args.parsed]):
        args.show = True  # 默认行为

    if args.show:
        if config_path.exists():
            logger.info("\n📄 配置文件 (config.yaml):\n")
            logger.info(config_path.read_text(encoding="utf-8"))
        else:
            logger.info(f"❌ 配置文件不存在: {config_path}")
        return 0

    if args.parsed:
        try:
            cfg = load_config(str(config_path))
            import json

            from pydantic import TypeAdapter

            adapter = TypeAdapter(type(cfg))
            data = adapter.dump_python(cfg, mode="json")
            logger.info("\n📋 解析后的配置 (JSON):\n")
            logger.info(json.dumps(data, indent=2, ensure_ascii=False))
        except Exception as e:
            logger.error(f"❌ 配置解析失败: {e}")
            return 1
        return 0

    if args.validate:
        try:
            cfg = load_config(str(config_path))
            logger.info("\n✅ 配置验证通过!\n")
            logger.info("📂 仓库路径检查:")
            for v in cfg.vaults:
                v_path = Path(v.path).expanduser()
                if v_path.exists():
                    logger.info(f"   ✅ {v.name}: {v.path}")
                else:
                    logger.info(f"   ⚠️ {v.name}: {v.path} (路径不存在)")
            logger.info("\n🗄️ 数据库路径检查:")
            db_path = Path(cfg.db_path).expanduser()
            if db_path.parent.exists():
                logger.info(f"   ✅ 数据库目录存在: {db_path.parent}")
            else:
                logger.info(f"   ⚠️ 数据库目录不存在: {db_path.parent} (首次运行将自动创建)")
            if cfg.jieba_user_dict:
                dict_path = Path(cfg.jieba_user_dict).expanduser()
                if dict_path.exists():
                    logger.info(f"   ✅ jieba 词典: {dict_path}")
                else:
                    logger.info(f"   ⚠️ jieba 词典不存在: {dict_path}")
            return 0
        except Exception as e:
            logger.info(f"\n❌ 配置验证失败: {e}\n")
            return 1

    if args.edit:
        if not config_path.exists():
            logger.info(f"❌ 配置文件不存在: {config_path}")
            return 1
        import shutil
        import subprocess

        editors = [os.environ.get("EDITOR"), os.environ.get("VISUAL"), "nano", "vim", "vi", "code", "gedit"]
        editor = next((e for e in editors if e and shutil.which(e)), None)
        if not editor:
            logger.info("❌ 未找到可用编辑器！请设置 EDITOR 环境变量")
            return 1
        logger.info(f"📝 使用编辑器: {editor}")
        subprocess.run([editor, str(config_path)])
        return 0
    return 0


def cmd_index(args):
    """索引管理 (build / scan) - 修复计数器与空文件诊断"""
    cfg = load_config()
    load_jieba_user_dict(cfg)

    # 初始化插件系统
    plugin_loader = None
    if cfg.plugins.enabled:
        try:
            plugin_loader = PluginLoader(cfg, None)
            plugin_loader.load_all()
        except Exception as e:
            logger.warning(f"⚠️ 插件加载失败: {e}")
            plugin_loader = None

    if args.action == "build":
        logger.info("🔄 触发全量索引重建...")
        old_argv = sys.argv
        try:
            sys.argv = ["build_index", "--force"]
            build_mod.main()
        except SystemExit as e:
            return e.code
        finally:
            sys.argv = old_argv
            if plugin_loader:
                plugin_loader.shutdown()
        return 0

    # =========================
    # 修改点 1：cmd_index 内 scan 分支
    # =========================

    if args.action == "scan":
        logger.info("🔍 执行增量扫描与索引更新...")
        db = DatabaseManager(cfg.db_path, vec_dimension=cfg.embedding_model.dimensions)

        if plugin_loader:
            for plugin in plugin_loader.get_all_plugins().values():
                if hasattr(plugin, "set_db_connection"):
                    plugin.set_db_connection(db.conn)

        try:
            global_skip = DEFAULT_SKIP_DIRS | frozenset(cfg.exclude.dirs)
            scanner = Scanner(db, skip_dirs=global_skip, global_patterns=cfg.exclude.patterns)

            vault_configs = [(v.name, v.path) for v in cfg.vaults if v.enabled]

            vault_excludes = {}
            for v in cfg.vaults:
                if v.enabled:
                    vault_excludes[v.name] = (
                        (frozenset(v.exclude.dirs), v.exclude.patterns) if v.exclude else (frozenset(), [])
                    )

            report = scanner.scan_vaults(vault_configs, vault_excludes)
            scanner.process_report(report)

            changed_paths = [f.absolute_path for f in report.new_files + report.modified_files]
            changed_paths.extend([f.new_absolute_path for f in report.moved_files])

            if not changed_paths:
                logger.info("   ℹ️ 扫描报告为空，无需更新。")
                return 0

            placeholders = ",".join(["?"] * len(changed_paths))
            cursor = db.conn.execute(
                f"SELECT id, absolute_path, file_path, mtime FROM files WHERE absolute_path IN ({placeholders})",
                changed_paths,
            )
            files_to_index = [dict(row) for row in cursor.fetchall()]

            if not files_to_index:
                logger.info("   ℹ️ 无待索引文件。")
                return 0

            logger.info(f"   📦 发现 {len(files_to_index)} 个变更文件，开始重建索引...")

            splitter = MarkdownSplitter(cfg)

            embed_engine = EmbeddingEngine(
                model_name=cfg.embedding_model.name,
                cache_dir=cfg.embedding_model.cache_dir,
                batch_size=cfg.embedding_model.batch_size,
                unload_after_seconds=cfg.embedding_model.unload_after_seconds,
            )

            batch_size = cfg.embedding_model.batch_size
            pending = []
            processed = 0
            empty_files = 0
            failed_files = 0
            file_chunks_collector: dict = {}

            # ✅ 修复点：函数移到 with 外部
            def _split_file(f):
                p = Path(f["absolute_path"])
                if not p.exists():
                    return f["id"], [], f["file_path"], f["absolute_path"], "文件不存在"
                try:
                    content = p.read_text("utf-8")
                    chunks = splitter.split(content, f.get("mtime"))
                    return f["id"], chunks, f["file_path"], f["absolute_path"], "OK"
                except Exception as e:
                    return f["id"], [], f["file_path"], f["absolute_path"], f"异常:{e}"

            # 并发处理
            with ThreadPoolExecutor(max_workers=cfg.max_concurrent_files) as ex:
                for fid, chunks, fp, fabs, status in ex.map(_split_file, files_to_index):
                    if not chunks:
                        if status != "OK":
                            logger.warning(f"⚠️ 分块失败 {fp}: {status}")
                            failed_files += 1
                        else:
                            empty_files += 1
                        continue

                    for c in chunks:
                        # ✅ 修复：传 fabs
                        pending.append((fid, c, fp, fabs))

                    if len(pending) >= batch_size:
                        processed += _commit_batch(
                            pending,
                            embed_engine,
                            db,
                            plugin_loader=plugin_loader,
                            file_chunks_collector=file_chunks_collector,
                        )
                        pending.clear()

            # 尾批
            if pending:
                processed += _commit_batch(
                    pending,
                    embed_engine,
                    db,
                    plugin_loader=plugin_loader,
                    file_chunks_collector=file_chunks_collector,
                )
                pending.clear()

            # 插件钩子
            if plugin_loader and file_chunks_collector:
                try:
                    logger.info(f"   🔧 触发插件 on_file_indexed 钩子处理 {len(file_chunks_collector)} 个文件...")
                    for file_id, data in file_chunks_collector.items():
                        plugin_loader.invoke_hook(
                            "on_file_indexed",
                            file_id=file_id,
                            chunks=data["chunks"],
                            filepath=data["file_path"],
                            absolute_path=data.get("absolute_path", ""),
                        )
                    logger.info("   ✅ 插件钩子处理完成")
                except Exception as e:
                    logger.warning(f"⚠️ on_file_indexed 钩子执行失败: {e}")

            if processed == 0:
                if empty_files == len(files_to_index):
                    logger.info("   ⚠️ 所有文件为空或被过滤")
                else:
                    logger.info("   ℹ️ 未生成新 chunks")
            else:
                logger.info(f"   ✅ 增量重建完成：{processed} chunks")

            return 0

        except Exception as e:
            logger.error(f"❌ 扫描失败: {e}", exc_info=True)
            return 1
        finally:
            if plugin_loader:
                plugin_loader.shutdown()
            db.close()


# =========================
# 修改点 2：_commit_batch 修复 fabs
# =========================


def _commit_batch(pending, embed_engine, db, plugin_loader=None, file_chunks_collector=None):
    import json
    from datetime import date, datetime

    if not pending:
        return 0

    texts = [p[1].content for p in pending]

    try:
        embs = embed_engine.embed(texts)
    except Exception as e:
        logger.error(f"❌ 批次向量化失败: {e}")
        return 0

    inserted_chunk_ids = []

    try:
        db.conn.execute("PRAGMA synchronous = OFF;")

        # ✅ 修复：解包 fabs
        for (fid, chunk, fp, fabs), emb in zip(pending, embs):
            meta_json = json.dumps(
                chunk.metadata or {}, default=lambda o: o.isoformat() if isinstance(o, (date, datetime)) else str(o)
            )

            conf_json = json.dumps(
                chunk.confidence_metadata or {},
                default=lambda o: o.isoformat() if isinstance(o, (date, datetime)) else str(o),
            )

            cur = db.conn.execute(
                """INSERT INTO chunks (file_id, chunk_index, content, content_type, section_title, section_path, start_pos, end_pos, confidence_final_weight, metadata, confidence_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    fid,
                    0,
                    chunk.content,
                    chunk.content_type.value,
                    chunk.section_title,
                    chunk.section_path,
                    chunk.start_pos,
                    chunk.end_pos,
                    1.0,
                    meta_json,
                    conf_json,
                ),
            )

            cid = cur.lastrowid
            inserted_chunk_ids.append((cid, fid, chunk, fp))

            # ✅ 修复 fabs
            if file_chunks_collector is not None:
                if fid not in file_chunks_collector:
                    file_chunks_collector[fid] = {"chunks": [], "file_path": fp, "absolute_path": fabs}

                file_chunks_collector[fid]["chunks"].append(
                    {
                        "id": cid,
                        "content": chunk.content,
                        "metadata": chunk.metadata,
                    }
                )

            if db.vec_support:
                db.conn.execute(
                    "INSERT INTO vectors (chunk_id, embedding) VALUES (?,?)", (cid, array.array("f", emb).tobytes())
                )

            db.conn.execute(
                "INSERT INTO fts5_index (rowid, content) VALUES (?,?)", (cid, build_mod.prepare_fts_content(chunk, fp))
            )

        db.conn.commit()

        return len(pending)

    except Exception as e:
        db.conn.rollback()
        logger.error(f"❌ 批次提交失败：{e}")
        return 0

    finally:
        db.conn.execute("PRAGMA synchronous = NORMAL;")


def cmd_maintenance(args):
    """数据库运维 (vacuum / soft-delete 清理，含图谱数据)"""
    cfg = load_config()
    db = DatabaseManager(cfg.db_path, vec_dimension=cfg.embedding_model.dimensions)
    try:
        stats = vac_mod.check_vacuum_needed(db, cfg)
        logger.info(f"📊 当前软删除比例: files={stats['files_ratio']:.1f}%, chunks={stats['chunks_ratio']:.1f}%")
        logger.info(f"   数据库大小: {stats['file_size_mb']:.2f} MB")

        # 显示图谱关联数据统计
        graph_total = (
            stats.get("relations_to_delete", 0)
            + stats.get("principles_to_delete", 0)
            + stats.get("notes_to_delete", 0)
            + stats.get("jobs_to_delete", 0)
        )
        if graph_total > 0:
            logger.info(
                f"📊 图谱关联数据: relations={stats.get('relations_to_delete', 0)}, "
                f"principles={stats.get('principles_to_delete', 0)}, "
                f"notes={stats.get('notes_to_delete', 0)}, "
                f"jobs={stats.get('jobs_to_delete', 0)}"
            )

        if args.dry_run:
            logger.info(
                f"🔍 预计清理: {stats['chunks_deleted']} chunks + {stats['files_deleted']} files + {graph_total} 图谱记录"
            )
            return 0

        logger.info("🧹 开始清理软删除记录（含图谱数据）...")
        clean_stats = vac_mod.clean_deleted_records(db, dry_run=False)
        total_deleted = (
            clean_stats["chunks_deleted"]
            + clean_stats["files_deleted"]
            + clean_stats.get("relations_deleted", 0)
            + clean_stats.get("principles_deleted", 0)
            + clean_stats.get("notes_deleted", 0)
            + clean_stats.get("jobs_deleted", 0)
        )
        logger.info(f"   共删除 {total_deleted} 条记录")

        if not args.clean_only:
            logger.info("🗜️ 执行 VACUUM 空间回收...")
            vac_mod.execute_vacuum(db, dry_run=False)
            new_stats = vac_mod.check_vacuum_needed(db, cfg)
            new_size = os.path.getsize(cfg.db_path) / (1024 * 1024)
            saved = stats["file_size_mb"] - new_size
            logger.info(f"✅ 运维完成 | 节省空间: {saved:.2f} MB" if saved > 0 else "✅ 运维完成 | 空间未明显变化")
        else:
            logger.info("✅ 清理完成 (跳过 VACUUM)")
        return 0
    except Exception as e:
        logger.error(f"❌ 运维失败: {e}")
        return 1
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(
        description="tinyRAG CLI 工具 (v2.1 - 检索/状态/配置/索引/运维)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python rag_cli.py search "极简网络" --top-k 5 --output json
  python rag_cli.py search "2024年项目" --alpha 0.8 --beta 0.2
  python rag_cli.py status
  python rag_cli.py index build --force
  python rag_cli.py index scan
  python rag_cli.py maintenance --dry-run
  python rag_cli.py config --validate
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # search
    sp_search = subparsers.add_parser("search", help="执行混合检索")
    sp_search.add_argument("query", help="查询文本")
    sp_search.add_argument("--top-k", type=int, default=10, help="返回结果数量")
    sp_search.add_argument("--mode", choices=["hybrid", "keyword", "semantic"], default="hybrid", help="检索模式")
    sp_search.add_argument("--alpha", type=float, default=None, help="语义权重 (0.0-1.0)")
    sp_search.add_argument("--beta", type=float, default=None, help="关键词权重 (0.0-1.0)")
    sp_search.add_argument("--vaults", nargs="+", help="指定检索的仓库名称")
    sp_search.add_argument("--output", choices=["console", "json", "csv"], default="console", help="输出格式")
    sp_search.add_argument("--verbose", action="store_true", help="输出详细调试日志")
    sp_search.set_defaults(func=cmd_search)

    # status
    sp_status = subparsers.add_parser("status", help="查看系统状态 (DB直查)")
    sp_status.set_defaults(func=cmd_status)

    # config
    sp_config = subparsers.add_parser("config", help="管理配置")
    sp_config.add_argument("--show", action="store_true", help="显示原始 YAML")
    sp_config.add_argument("--parsed", action="store_true", help="显示解析后的 JSON")
    sp_config.add_argument("--validate", action="store_true", help="验证配置并检查路径")
    sp_config.add_argument("--edit", action="store_true", help="编辑配置文件")
    sp_config.set_defaults(func=cmd_config)

    # index
    sp_index = subparsers.add_parser("index", help="索引管理")
    sp_index_sub = sp_index.add_subparsers(dest="action")
    sp_index_build = sp_index_sub.add_parser("build", help="全量重建索引")
    sp_index_build.add_argument("--force", action="store_true", help="强制清空后重建")
    sp_index_scan = sp_index_sub.add_parser("scan", help="增量扫描更新")
    sp_index.set_defaults(func=cmd_index)

    # maintenance
    sp_maint = subparsers.add_parser("maintenance", help="数据库运维 (清理+VACUUM)")
    sp_maint.add_argument("--dry-run", action="store_true", help="仅检查，不执行")
    sp_maint.add_argument("--clean-only", action="store_true", help="仅清理软删除，不VACUUM")
    sp_maint.set_defaults(func=cmd_maintenance)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 0

    # 索引/运维需要 action
    if args.command == "index" and not getattr(args, "action", None):
        sp_index.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
