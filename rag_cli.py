#!/usr/bin/env python3
"""
rag_cli.py - tinyRAG 命令行检索与运维工具
基于 retriever/hybrid_engine.py 的 HybridEngine 实现
功能:
- search: 执行混合检索
- status: 查看数据库和索引状态
- config: 查看/编辑配置
用法:
  python rag_cli.py search "极简网络" --top-k 5
  python rag_cli.py search "网络优化" --alpha 0.8 --beta 0.1
  python rag_cli.py status
  python rag_cli.py config --show
"""

import argparse
import fnmatch
import os
import sys
import time
from pathlib import Path

# 确保项目根目录在 sys.path
script_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(script_dir))

from config import load_config  # noqa: E402
from embedder.embed_engine import EmbeddingEngine  # noqa: E402
from retriever.hybrid_engine import HybridEngine  # noqa: E402
from scanner.scan_engine import DEFAULT_SKIP_DIRS  # noqa: E402
from storage.database import DatabaseManager  # noqa: E402
from utils.logger import setup_logger  # noqa: E402

# 修复 L6: 使用绝对路径，避免依赖 CWD
logger = setup_logger(level="INFO")


def cmd_status(args):
    """显示系统状态"""
    print("\n📊 tinyRAG 状态\n")
    try:
        cfg = load_config()
        db_path = Path(cfg.db_path).resolve()

        print(f"✅ 数据库:{db_path}")
        if db_path.exists():
            db_size = db_path.stat().st_size / 1024 / 1024
            print(f"   大小:{db_size:.2f} MB")

            db = DatabaseManager(str(db_path))
            try:
                chunk_count = db.conn.execute("SELECT COUNT(*) FROM chunks WHERE is_deleted=0").fetchone()[0]
                file_count = db.conn.execute("SELECT COUNT(*) FROM files WHERE is_deleted=0").fetchone()[0]
                print(f"   活跃 Chunk:{chunk_count} | 活跃 Files:{file_count}")
                print(f"   向量支持:{'✅ 已启用' if db.vec_support else '⚠️ 已降级(FTS5)'}")
            finally:
                db.close()
        else:
            print("❌ 数据库未初始化,请运行: python build_index.py --force")

        print("\n📂 仓库状态:")
        total_files = 0

        # 全局跳过目录 = 内置默认 + 配置文件全局排除
        global_skip_dirs = DEFAULT_SKIP_DIRS | frozenset(cfg.exclude.dirs)
        global_patterns = cfg.exclude.patterns

        for vault in cfg.vaults:
            # ✅ 修复:支持 VaultConfig 对象或字符串路径
            if hasattr(vault, "path"):
                v_path = Path(vault.path).expanduser().resolve()
                v_name = vault.name if hasattr(vault, "name") else str(v_path)
            else:
                v_path = Path(vault).expanduser().resolve()
                v_name = str(v_path)

            if v_path.exists():
                # 获取 vault 自身的排除规则（不包含全局配置）
                vault_skip_dirs = frozenset()
                vault_patterns = []
                if vault.exclude:
                    if vault.exclude.dirs:
                        vault_skip_dirs = frozenset(vault.exclude.dirs)
                    if vault.exclude.patterns:
                        vault_patterns = vault.exclude.patterns

                # 合并跳过目录：全局 + vault 级（与 scan_engine 逻辑一致）
                all_skip_dirs = global_skip_dirs | vault_skip_dirs

                # 合并模式：全局 + vault 级（与 scan_engine 逻辑一致）
                all_patterns = list(set(global_patterns + vault_patterns))

                # 统计 Markdown 文件（应用排除规则）
                count = 0
                for root, dirs, files in os.walk(v_path):
                    # 排除指定目录（修改 dirs 列表会影响 os.walk 的遍历）
                    dirs[:] = [d for d in dirs if d not in all_skip_dirs]

                    for fname in files:
                        if not fname.endswith(".md"):
                            continue

                        # 检查文件模式排除规则（与 scan_engine._match_patterns 逻辑一致）
                        rel_path = os.path.relpath(os.path.join(root, fname), v_path)
                        excluded = False
                        for pattern in all_patterns:
                            if fnmatch.fnmatch(rel_path, pattern):
                                excluded = True
                                break
                            # 也匹配路径的各部分（与 scan_engine._match_patterns 一致）
                            for part in rel_path.split(os.sep):
                                if fnmatch.fnmatch(part, pattern):
                                    excluded = True
                                    break
                            if excluded:
                                break

                        if not excluded:
                            count += 1

                total_files += count
                print(f"   ✅ {v_name} ({count} 个 Markdown 文件)")
            else:
                print(f"   ❌ {v_name} (路径不存在)")
        print(f"\n总计:{total_files} 个 Markdown 文件")
        return 0
    except Exception as e:
        logger.error(f"❌ 状态检查失败:{e}")
        return 1


def cmd_search(args):
    """执行混合检索 (复用 HybridEngine)"""
    try:
        cfg = load_config()
        db_path = Path(cfg.db_path).resolve()
        if not db_path.exists():
            logger.error("❌ 数据库不存在,请先运行索引构建.")
            return 1

        # 修复 M5: 使用局部变量，不修改共享 config
        alpha = cfg.retrieval.get("alpha", 0.7)
        beta = cfg.retrieval.get("beta", 0.3)

        # CLI 权重覆盖
        if args.alpha is not None:
            alpha = args.alpha
        if args.beta is not None:
            beta = args.beta

        # 通过 alpha/beta 比值模拟检索模式
        if args.mode == "keyword":
            alpha = 0.0
            beta = 1.0
        elif args.mode == "semantic":
            alpha = 1.0
            beta = 0.0

        # 构建 vault 过滤列表
        if args.vaults:
            vaults = args.vaults
        else:
            vaults = [v.name for v in cfg.vaults if v.enabled]
            vaults = vaults if vaults else None

        logger.info(f"🔍 检索参数: alpha={alpha}, beta={beta}, vaults={vaults}")

        # 修复 M4: 从 config 读取维度
        db = DatabaseManager(str(db_path), vec_dimension=cfg.embedding_model.dimensions)

        # 初始化嵌入引擎
        embed_engine = EmbeddingEngine(
            model_name=cfg.embedding_model.name,
            cache_dir=cfg.embedding_model.cache_dir,
            batch_size=cfg.embedding_model.batch_size,
            unload_after_seconds=cfg.embedding_model.unload_after_seconds,
        )

        # 构造混合检索引擎
        retriever = HybridEngine(config=cfg, db=db, embed_engine=embed_engine)

        start = time.time()
        # 修复 M5: 传入 alpha/beta 参数
        results = retriever.search(query=args.query, limit=args.top_k, vault_filter=vaults, alpha=alpha, beta=beta)
        elapsed = time.time() - start

        print(f"\n📊 检索结果 ({len(results)} 条,耗时 {elapsed:.2f}s):\n")
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
        logger.error(f"❌ 检索失败:{e}", exc_info=True)
        return 1
    finally:
        # 修复 L1: 确保 db 连接被关闭
        if "db" in locals():
            db.close()


def cmd_config(args):
    """显示或编辑配置"""
    config_path = script_dir / "config.yaml"

    # 默认行为：显示配置概览
    if not args.show and not args.edit and not args.validate and not args.parsed:
        try:
            cfg = load_config(str(config_path))
            print("\n⚙️ tinyRAG 配置概览\n")
            print("=" * 60)

            # 仓库配置
            print("\n📂 仓库配置:")
            for v in cfg.vaults:
                status = "✅ 启用" if v.enabled else "⏸️ 禁用"
                exclude_info = ""
                if v.exclude:
                    dirs_count = len(v.exclude.dirs)
                    patterns_count = len(v.exclude.patterns)
                    if dirs_count or patterns_count:
                        exclude_info = f" (排除: {dirs_count} 目录, {patterns_count} 模式)"
                print(f"   {status} {v.name}: {v.path}{exclude_info}")

            # 模型配置
            print("\n🤖 嵌入模型:")
            print(f"   名称: {cfg.embedding_model.name}")
            print(f"   维度: {cfg.embedding_model.dimensions}")
            print(f"   批大小: {cfg.embedding_model.batch_size}")
            print(f"   缓存目录: {cfg.embedding_model.cache_dir}")

            # 分块配置
            print("\n✂️ 分块配置:")
            print(f"   最大 Token: {cfg.chunking.max_tokens}")
            print(f"   重叠 Token: {cfg.chunking.overlap}")
            print(f"   Token 模式: {cfg.chunking.token_mode}")

            # 检索配置
            print("\n🔍 检索配置:")
            print(f"   Alpha (语义权重): {cfg.retrieval.get('alpha', 0.7)}")
            print(f"   Beta (关键词权重): {cfg.retrieval.get('beta', 0.3)}")

            # 置信度配置
            print("\n📊 置信度权重:")
            print(f"   文档类型: {list(cfg.confidence.doc_type_rules.keys()) or '(默认)'}")
            print(f"   状态类型: {list(cfg.confidence.status_rules.keys())}")
            print(f"   日期衰减: {'启用' if cfg.confidence.date_decay.enabled else '禁用'}")
            if cfg.confidence.date_decay.enabled:
                print(f"   半衰期: {cfg.confidence.date_decay.half_life_days} 天")

            # 全局排除规则
            print("\n🚫 全局排除规则:")
            print(f"   目录: {cfg.exclude.dirs[:5]}{'...' if len(cfg.exclude.dirs) > 5 else ''}")
            print(f"   模式: {cfg.exclude.patterns[:5]}{'...' if len(cfg.exclude.patterns) > 5 else ''}")

            # 数据库
            print("\n🗄️ 数据库:")
            print(f"   路径: {cfg.db_path}")
            print(f"   缓存: {cfg.cache.db_path}")

            print("\n" + "=" * 60)
            print("💡 使用 --show 查看原始 YAML, --parsed 查看解析后配置, --validate 验证配置")
            return 0
        except Exception as e:
            logger.error(f"❌ 配置加载失败: {e}")
            return 1

    # 显示原始 YAML
    if args.show:
        if config_path.exists():
            print("\n📄 配置文件 (config.yaml):\n")
            print(config_path.read_text(encoding="utf-8"))
        else:
            print(f"❌ 配置文件不存在: {config_path}")
        return 0

    # 显示解析后的配置
    if args.parsed:
        try:
            cfg = load_config(str(config_path))
            import json

            from pydantic import TypeAdapter

            # 使用 Pydantic 序列化
            adapter = TypeAdapter(type(cfg))
            data = adapter.dump_python(cfg, mode='json')
            print("\n📋 解析后的配置 (JSON):\n")
            print(json.dumps(data, indent=2, ensure_ascii=False))
        except Exception as e:
            logger.error(f"❌ 配置解析失败: {e}")
            return 1
        return 0

    # 验证配置
    if args.validate:
        try:
            cfg = load_config(str(config_path))
            print("\n✅ 配置验证通过!\n")

            # 检查仓库路径
            print("📂 仓库路径检查:")
            for v in cfg.vaults:
                v_path = Path(v.path).expanduser()
                if v_path.exists():
                    print(f"   ✅ {v.name}: {v.path}")
                else:
                    print(f"   ⚠️ {v.name}: {v.path} (路径不存在)")

            # 检查数据库路径
            print("\n🗄️ 数据库路径检查:")
            db_path = Path(cfg.db_path).expanduser()
            db_dir = db_path.parent
            if db_dir.exists():
                print(f"   ✅ 数据库目录存在: {db_dir}")
            else:
                print(f"   ⚠️ 数据库目录不存在: {db_dir} (首次运行将自动创建)")

            # 检查 jieba 词典
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

    # 编辑配置
    if args.edit:
        if not config_path.exists():
            print(f"❌ 配置文件不存在: {config_path}")
            return 1
        import shutil
        import subprocess

        # 尝试多个编辑器（按优先级）
        editors = [
            os.environ.get("EDITOR"),  # 环境变量优先
            os.environ.get("VISUAL"),  # VISUAL 也是标准环境变量
            "nano",                    # 最常见的简单编辑器
            "vim",                     # Vim
            "vi",                      # 基础 vi
            "code",                    # VS Code
            "gedit",                   # GNOME 编辑器
        ]

        editor = None
        for e in editors:
            if e and shutil.which(e):
                editor = e
                break

        if not editor:
            print("❌ 未找到可用编辑器！")
            print("💡 请设置 EDITOR 环境变量或安装编辑器:")
            print("   export EDITOR=nano  # 或 vim, code 等")
            print("   sudo apt install nano  # 安装 nano")
            print(f"\n📄 或者直接编辑: {config_path}")
            return 1

        print(f"📝 使用编辑器: {editor}")
        subprocess.run([editor, str(config_path)])
        return 0

    return 1


def main():
    parser = argparse.ArgumentParser(
        description="tinyRAG CLI 工具 (检索 / 状态检查 / 配置管理)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本检索
  python rag_cli.py search "极简网络" --top-k 5
  # 调整权重
  python rag_cli.py search "网络优化" --alpha 0.8 --beta 0.1
  # 指定模式与仓库
  python rag_cli.py search "日记" --mode keyword --vaults personal
  # 查看系统状态
  python rag_cli.py status
  # 配置管理
  python rag_cli.py config                # 显示配置概览
  python rag_cli.py config --show         # 显示原始 YAML
  python rag_cli.py config --parsed       # 显示解析后的 JSON
  python rag_cli.py config --validate     # 验证配置
  python rag_cli.py config --edit         # 编辑配置文件
""",
    )
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    sp_search = subparsers.add_parser("search", help="执行混合检索")
    sp_search.add_argument("query", help="查询文本")
    sp_search.add_argument("--top-k", type=int, default=10, help="返回结果数量")
    sp_search.add_argument(
        "--mode",
        choices=["hybrid", "keyword", "semantic"],
        default="hybrid",
        help="检索模式",
    )
    sp_search.add_argument("--alpha", type=float, default=None, help="语义/RRF 权重")
    sp_search.add_argument("--beta", type=float, default=None, help="置信度权重")
    sp_search.add_argument("--vaults", nargs="+", help="指定检索的仓库名称")
    sp_search.set_defaults(func=cmd_search)

    sp_status = subparsers.add_parser("status", help="查看系统状态")
    sp_status.set_defaults(func=cmd_status)

    sp_config = subparsers.add_parser("config", help="管理配置")
    sp_config.add_argument("--show", action="store_true", help="显示原始 YAML 配置")
    sp_config.add_argument("--parsed", action="store_true", help="显示解析后的配置 (JSON)")
    sp_config.add_argument("--validate", action="store_true", help="验证配置并检查路径")
    sp_config.add_argument("--edit", action="store_true", help="编辑配置文件")
    sp_config.set_defaults(func=cmd_config)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
