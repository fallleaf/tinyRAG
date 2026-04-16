#!/usr/bin/env python3
"""
rag_cli.py - tinyRAG 命令行检索与运维工具 (v2.1)
功能:
- search: 执行混合检索 (支持 console/json/csv 导出、Verbose调试)
- status: 查看数据库和索引状态 (DB直查，毫秒级响应)
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
import csv
import fnmatch
import json
import os
import sys
import time
from pathlib import Path
from io import StringIO

# ✅ 在 rag_cli.py 顶部导入区添加以下 4 行
import array
import json
from concurrent.futures import ThreadPoolExecutor
from chunker.markdown_splitter import MarkdownSplitter  # noqa: E402


# 确保项目根目录在 sys.path
script_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(script_dir))

from config import load_config
from embedder.embed_engine import EmbeddingEngine
from retriever.hybrid_engine import HybridEngine
from scanner.scan_engine import DEFAULT_SKIP_DIRS, Scanner
from storage.database import DatabaseManager
from utils.logger import setup_logger
from utils.jieba_helper import load_jieba_user_dict

# 导入底层模块供 CLI 调用
import build_index as build_mod
import vacuum as vac_mod

logger = setup_logger(level="INFO")


def cmd_status(args):
    """显示系统状态 (优化版：DB直查 + 全维度配置摘要)"""
    print("\n📊 tinyRAG 系统状态\n" + "=" * 50)
    try:
        cfg = load_config()
        db_path = Path(cfg.db_path).resolve()
        print(f"🗄️ 数据库路径: {db_path}")

        if db_path.exists():
            db_size = db_path.stat().st_size / 1024 / 1024
            print(f"   大小: {db_size:.2f} MB")

            db = DatabaseManager(str(db_path), vec_dimension=cfg.embedding_model.dimensions)
            try:
                # 1. 活跃数据
                files_active = db.conn.execute("SELECT COUNT(*) FROM files WHERE is_deleted=0").fetchone()[0]
                chunks_active = db.conn.execute("SELECT COUNT(*) FROM chunks WHERE is_deleted=0").fetchone()[0]
                print(f"   活跃 Files: {files_active} | 活跃 Chunks: {chunks_active}")

                # 2. 软删除统计 (运维关键指标)
                files_deleted = db.conn.execute("SELECT COUNT(*) FROM files WHERE is_deleted=1").fetchone()[0]
                chunks_deleted = db.conn.execute("SELECT COUNT(*) FROM chunks WHERE is_deleted=1").fetchone()[0]
                print(f"   软删除 Files: {files_deleted} | 软删除 Chunks: {chunks_deleted}")

                # 3. 向量引擎状态
                print(f"   向量引擎: {'✅ sqlite-vec 已启用' if db.vec_support else '⚠️ 已降级 (FTS5)'}")

                # 4. 按 Vault 统计 (直接查 DB，替代慢速 os.walk)
                print("\n📂 Vault 索引状态:")
                vault_stats = db.conn.execute(
                    "SELECT vault_name, COUNT(*) as cnt FROM files WHERE is_deleted=0 GROUP BY vault_name"
                ).fetchall()
                total_db_files = 0
                for row in vault_stats:
                    print(f"   ✅ {row['vault_name']}: {row['cnt']} 个已索引文件")
                    total_db_files += row['cnt']
                if total_db_files == 0:
                    print("   ⚠️ 暂无已索引文件，请运行 `python rag_cli.py index build --force`")
                else:
                    print(f"   📈 总计: {total_db_files} 个已索引文件")

                # 5. 缓存状态
                cache_path = Path(cfg.cache.db_path).resolve()
                print(f"\n💾 查询缓存: {cfg.cache.db_path} (TTL={cfg.cache.ttl_seconds}s, Max={cfg.cache.max_entries})")
                if cache_path.exists():
                    cache_size = cache_path.stat().st_size / 1024 / 1024
                    print(f"   状态: ✅ 已存在 ({cache_size:.2f} MB)")
                else:
                    print(f"   状态: ⏳ 未初始化 (首次检索时自动创建)")

            finally:
                db.close()
        else:
            print("❌ 数据库未初始化，请运行: `python rag_cli.py index build --force`")

        # 核心配置摘要 (便于快速排查参数)
        print(f"\n⚙️ 核心配置摘要")
        print(f"   🤖 模型: {cfg.embedding_model.name} (dim={cfg.embedding_model.dimensions}, batch={cfg.embedding_model.batch_size})")
        print(f"   🔍 检索: alpha={cfg.retrieval.get('alpha', 0.7)}, beta={cfg.retrieval.get('beta', 0.3)}")
        print(f"   ✂️ 分块: max_tokens={cfg.chunking.max_tokens}, overlap={cfg.chunking.overlap}")
        print(f"   🕒 衰减: {'✅ 启用' if cfg.confidence.date_decay.enabled else '❌ 禁用'} (半衰期={cfg.confidence.date_decay.half_life_days}天)")

        print("\n" + "=" * 50)
        return 0
    except Exception as e:
        logger.error(f"❌ 状态检查失败: {e}")
        return 1


def cmd_search(args):
    """执行混合检索 (v2.1: 支持导出/Verbose/安全事务)"""
    db = None
    try:
        cfg = load_config()
        db_path = Path(cfg.db_path).resolve()
        if not db_path.exists():
            logger.error("❌ 数据库不存在,请先运行索引构建.")
            return 1

        # 局部权重覆盖 (不污染全局配置)
        alpha = cfg.retrieval.get("alpha", 0.7)
        beta = cfg.retrieval.get("beta", 0.3)
        if args.alpha is not None: alpha = args.alpha
        if args.beta is not None: beta = args.beta
        if args.mode == "keyword": alpha, beta = 0.0, 1.0
        elif args.mode == "semantic": alpha, beta = 1.0, 0.0

        vaults = args.vaults if args.vaults else [v.name for v in cfg.vaults if v.enabled]
        vault_filter = vaults if vaults else None

        if args.verbose:
            logger.setLevel("DEBUG")

        db = DatabaseManager(str(db_path), vec_dimension=cfg.embedding_model.dimensions)
        embed_engine = EmbeddingEngine(
            model_name=cfg.embedding_model.name,
            cache_dir=cfg.embedding_model.cache_dir,
            batch_size=cfg.embedding_model.batch_size,
            unload_after_seconds=cfg.embedding_model.unload_after_seconds,
        )
        retriever = HybridEngine(config=cfg, db=db, embed_engine=embed_engine)

        start = time.time()
        results = retriever.search(
            query=args.query, limit=args.top_k, vault_filter=vault_filter, alpha=alpha, beta=beta
        )
        elapsed = time.time() - start

        # 📦 导出支持
        if args.output == "json":
            print(json.dumps(
                [{"rank": i+1, **r.__dict__} for i, r in enumerate(results)],
                indent=2, ensure_ascii=False, default=str
            ))
        elif args.output == "csv":
            output = StringIO()
            writer = csv.DictWriter(output, fieldnames=[
                "rank", "file_path", "absolute_path", "section", "vault_name",
                "chunk_type", "content", "final_score", "confidence_score", "confidence_reason"
            ])
            writer.writeheader()
            for i, r in enumerate(results, 1):
                writer.writerow({
                    "rank": i, "file_path": r.file_path, "absolute_path": r.absolute_path,
                    "section": r.section, "vault_name": r.vault_name, "chunk_type": r.chunk_type,
                    "content": r.content[:200].replace("\n", " "), "final_score": r.final_score,
                    "confidence_score": r.confidence_score, "confidence_reason": r.confidence_reason
                })
            print(output.getvalue())
        else:
            # 🖥️ 终端输出
            print(f"\n📊 检索结果 ({len(results)} 条, 耗时 {elapsed:.2f}s):\n")
            if not results:
                print("   未找到相关结果.")
                return 0
            for i, r in enumerate(results, 1):
                content_preview = r.content[:200] + "..." if len(r.content) > 200 else r.content
                scores = f"最终={r.final_score:.3f} | 语义={r.semantic_score:.3f} | 关键词={r.keyword_score:.3f} | 置信度={r.confidence_score:.2f}"
                print(f"{i}. [{scores}]")
                print(f"   来源:{r.absolute_path}")
                print(f"   类型:{r.vault_name} / {r.chunk_type} | 章节:{r.section}")
                print(f"   内容:{content_preview}\n")
        return 0
    except Exception as e:
        logger.error(f"❌ 检索失败: {e}", exc_info=True)
        return 1
    finally:
        if db:
            db.close()


def cmd_config(args):
    """显示或编辑配置"""
    config_path = script_dir / "config.yaml"
    if not any([args.show, args.edit, args.validate, args.parsed]):
        args.show = True  # 默认行为

    if args.show:
        if config_path.exists():
            print("\n📄 配置文件 (config.yaml):\n")
            print(config_path.read_text(encoding="utf-8"))
        else:
            print(f"❌ 配置文件不存在: {config_path}")
        return 0

    if args.parsed:
        try:
            cfg = load_config(str(config_path))
            import json
            from pydantic import TypeAdapter
            adapter = TypeAdapter(type(cfg))
            data = adapter.dump_python(cfg, mode='json')
            print("\n📋 解析后的配置 (JSON):\n")
            print(json.dumps(data, indent=2, ensure_ascii=False))
        except Exception as e:
            logger.error(f"❌ 配置解析失败: {e}")
            return 1
        return 0

    if args.validate:
        try:
            cfg = load_config(str(config_path))
            print("\n✅ 配置验证通过!\n")
            print("📂 仓库路径检查:")
            for v in cfg.vaults:
                v_path = Path(v.path).expanduser()
                if v_path.exists():
                    print(f"   ✅ {v.name}: {v.path}")
                else:
                    print(f"   ⚠️ {v.name}: {v.path} (路径不存在)")
            print("\n🗄️ 数据库路径检查:")
            db_path = Path(cfg.db_path).expanduser()
            if db_path.parent.exists():
                print(f"   ✅ 数据库目录存在: {db_path.parent}")
            else:
                print(f"   ⚠️ 数据库目录不存在: {db_path.parent} (首次运行将自动创建)")
            if cfg.jieba_user_dict:
                dict_path = Path(cfg.jieba_user_dict).expanduser()
                if dict_path.exists():
                    print(f"   ✅ jieba 词典: {dict_path}")
                else:
                    print(f"   ⚠️ jieba 词典不存在: {dict_path}")
            return 0
        except Exception as e:
            print(f"\n❌ 配置验证失败: {e}\n")
            return 1

    if args.edit:
        if not config_path.exists():
            print(f"❌ 配置文件不存在: {config_path}")
            return 1
        import shutil
        import subprocess
        editors = [os.environ.get("EDITOR"), os.environ.get("VISUAL"), "nano", "vim", "vi", "code", "gedit"]
        editor = next((e for e in editors if e and shutil.which(e)), None)
        if not editor:
            print("❌ 未找到可用编辑器！请设置 EDITOR 环境变量")
            return 1
        print(f"📝 使用编辑器: {editor}")
        subprocess.run([editor, str(config_path)])
        return 0
    return 0

def cmd_index(args):
    """索引管理 (build / scan) - 修复计数器与空文件诊断"""
    cfg = load_config()
    load_jieba_user_dict(cfg)

    if args.action == "build":
        print("🔄 触发全量索引重建...")
        old_argv = sys.argv
        try:
            sys.argv = ["build_index", "--force"]
            build_mod.main()
        except SystemExit as e:
            return e.code
        finally:
            sys.argv = old_argv
        return 0

    elif args.action == "scan":
        print("🔍 执行增量扫描与索引更新...")
        db = DatabaseManager(cfg.db_path, vec_dimension=cfg.embedding_model.dimensions)
        try:
            global_skip = DEFAULT_SKIP_DIRS | frozenset(cfg.exclude.dirs)
            scanner = Scanner(db, skip_dirs=global_skip, global_patterns=cfg.exclude.patterns)
            vault_configs = [(v.name, v.path) for v in cfg.vaults if v.enabled]
            vault_excludes = {}
            for v in cfg.vaults:
                if v.enabled:
                    vault_excludes[v.name] = (frozenset(v.exclude.dirs), v.exclude.patterns) if v.exclude else (frozenset(), [])

            # 1. 扫描与元数据同步
            report = scanner.scan_vaults(vault_configs, vault_excludes)
            scanner.process_report(report)

            changed_paths = [f.absolute_path for f in report.new_files + report.modified_files]
            changed_paths.extend([f.new_absolute_path for f in report.moved_files])

            if not changed_paths:
                print("   ℹ️ 扫描报告为空，无需更新。")
                return 0

            # 2. 查询待索引文件
            placeholders = ",".join(["?"] * len(changed_paths))
            cursor = db.conn.execute(
                f"SELECT id, absolute_path, file_path, mtime FROM files WHERE absolute_path IN ({placeholders})",
                changed_paths
            )
            files_to_index = [dict(row) for row in cursor.fetchall()]
            if not files_to_index:
                print("   ℹ️ 无待索引文件。")
                return 0

            print(f"   📦 发现 {len(files_to_index)} 个变更文件，开始重建索引...")
            
            splitter = MarkdownSplitter(cfg)
            embed_engine = EmbeddingEngine(
                model_name=cfg.embedding_model.name, cache_dir=cfg.embedding_model.cache_dir,
                batch_size=cfg.embedding_model.batch_size, unload_after_seconds=cfg.embedding_model.unload_after_seconds,
            )
            batch_size = cfg.embedding_model.batch_size
            pending = []
            processed = 0
            empty_files = 0
            failed_files = 0

            def _split_file(f):
                p = Path(f["absolute_path"])
                if not p.exists():
                    return f["id"], [], f["file_path"], "文件不存在"
                try:
                    content = p.read_text("utf-8")
                    chunks = splitter.split(content, f.get("mtime"))
                    return f["id"], chunks, f["file_path"], "OK"
                except Exception as e:
                    return f["id"], [], f["file_path"], f"异常:{e}"

            with ThreadPoolExecutor(max_workers=cfg.max_concurrent_files) as ex:
                for fid, chunks, fp, status in ex.map(_split_file, files_to_index):
                    if not chunks:
                        if status != "OK":
                            logger.warning(f"⚠️ 分块失败 {fp}: {status}")
                            failed_files += 1
                        empty_files += 1
                        continue
                        
                    for c in chunks:
                        pending.append((fid, c, fp))
                    if len(pending) >= batch_size:
                        processed += _commit_batch(pending, embed_engine, db)
                        pending.clear()

            # ✅ 修复：尾批处理计数器累加
            if pending:
                processed += _commit_batch(pending, embed_engine, db)
                pending.clear()

            # 📊 智能结果反馈
            if processed == 0:
                if empty_files == len(files_to_index):
                    print(f"   ⚠️ 所有文件内容为空、仅含 Frontmatter 或全被过滤，未生成有效 Chunks。")
                else:
                    print(f"   ℹ️ 文件已处理，但未生成新 Chunks（可能内容被排除规则过滤）。")
                if failed_files: print(f"   ❌ {failed_files} 个文件分块异常，详见日志。")
            else:
                print(f"   ✅ 增量重建完成：共处理 {processed} 个 chunks ({empty_files} 个空文件已跳过)")
            return 0
        except Exception as e:
            logger.error(f"❌ 扫描失败: {e}", exc_info=True)
            return 1
        finally:
            db.close()
    return 0

def _commit_batch(pending, embed_engine, db):
    """执行单批向量化与入库，返回成功处理的 chunks 数量"""
    from datetime import date, datetime
    import json, array
    if not pending: return 0
    
    texts = [p[1].content for p in pending]
    try:
        embs = embed_engine.embed(texts)
    except Exception as e:
        logger.error(f"❌ 批次向量化失败: {e}")
        return 0

    try:
        db.conn.execute("PRAGMA synchronous = OFF;")
        for (fid, chunk, fp), emb in zip(pending, embs):
            meta_json = json.dumps(chunk.metadata or {}, default=lambda o: o.isoformat() if isinstance(o, (date,datetime)) else str(o))
            conf_json = json.dumps(chunk.confidence_metadata or {}, default=lambda o: o.isoformat() if isinstance(o, (date,datetime)) else str(o))
            cur = db.conn.execute(
                "INSERT INTO chunks (file_id, chunk_index, content, content_type, section_title, section_path, start_pos, end_pos, confidence_final_weight, metadata, confidence_json) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (fid, 0, chunk.content, chunk.content_type.value, chunk.section_title, chunk.section_path, chunk.start_pos, chunk.end_pos, 1.0, meta_json, conf_json)
            )
            cid = cur.lastrowid
            if db.vec_support:
                db.conn.execute("INSERT INTO vectors (chunk_id, embedding) VALUES (?,?)", (cid, array.array("f", emb).tobytes()))
            db.conn.execute("INSERT INTO fts5_index (rowid, content) VALUES (?,?)", (cid, build_mod.prepare_fts_content(chunk, fp)))
        db.conn.commit()
        return len(pending)
    except Exception as e:
        db.conn.rollback()
        logger.error(f"❌ 批次提交失败：{e}")
        return 0
    finally:
        db.conn.execute("PRAGMA synchronous = NORMAL;")

def cmd_maintenance(args):
    """数据库运维 (vacuum / soft-delete 清理)"""
    cfg = load_config()
    db = DatabaseManager(cfg.db_path, vec_dimension=cfg.embedding_model.dimensions)
    try:
        stats = vac_mod.check_vacuum_needed(db, cfg)
        print(f"📊 当前软删除比例: files={stats['files_ratio']:.1f}%, chunks={stats['chunks_ratio']:.1f}%")
        print(f"   数据库大小: {stats['file_size_mb']:.2f} MB")

        if args.dry_run:
            print(f"🔍 预计清理: {stats['chunks_deleted']} chunks + {stats['files_deleted']} files")
            return 0

        print("🧹 开始清理软删除记录...")
        vac_mod.clean_deleted_records(db, dry_run=False)
        if not args.clean_only:
            print("🗜️ 执行 VACUUM 空间回收...")
            vac_mod.execute_vacuum(db, dry_run=False)
            new_stats = vac_mod.check_vacuum_needed(db, cfg)
            new_size = os.path.getsize(cfg.db_path) / (1024 * 1024)
            saved = stats["file_size_mb"] - new_size
            print(f"✅ 运维完成 | 节省空间: {saved:.2f} MB" if saved > 0 else "✅ 运维完成 | 空间未明显变化")
        else:
            print("✅ 清理完成 (跳过 VACUUM)")
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
