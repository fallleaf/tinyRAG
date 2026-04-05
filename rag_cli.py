#!/usr/bin/env python3
"""
rag_cli.py - RAG System 命令行检索与运维工具
基于 retriever/hybrid_engine.py 的 HybridRetriever 实现
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
import os
import sys
import time
from pathlib import Path

# 确保项目根目录在 sys.path
script_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(script_dir))

from config import load_config  # noqa: E402
from retriever.hybrid_engine import HybridRetriever  # noqa: E402
from storage.database import DatabaseManager  # noqa: E402
from utils.logger import setup_logger  # noqa: E402

logger = setup_logger(level="INFO", log_file="logs/cli.log")


def cmd_status(args):
    """显示系统状态"""
    print("\n📊 RAG System 状态\n")
    try:
        cfg = load_config()
        db_path = Path(cfg.db_path).resolve()

        print(f"✅ 数据库:{db_path}")
        if db_path.exists():
            db_size = db_path.stat().st_size / 1024 / 1024
            print(f"   大小:{db_size:.2f} MB")

            db = DatabaseManager(str(db_path))
            try:
                chunk_count = db.conn.execute(
                    "SELECT COUNT(*) FROM chunks WHERE is_deleted=0"
                ).fetchone()[0]
                file_count = db.conn.execute(
                    "SELECT COUNT(*) FROM files WHERE is_deleted=0"
                ).fetchone()[0]
                print(f"   活跃 Chunk:{chunk_count} | 活跃 Files:{file_count}")
                print(
                    f"   向量支持:{'✅ 已启用' if db.vec_support else '⚠️ 已降级(FTS5)'}"
                )
            finally:
                db.close()
        else:
            print("❌ 数据库未初始化,请运行: python build_index.py --force")

        print("\n📂 仓库状态:")
        total_files = 0
        for vault in cfg.vaults:
            # ✅ 修复:支持 VaultConfig 对象或字符串路径
            if hasattr(vault, "path"):
                v_path = Path(vault.path).expanduser().resolve()
                v_name = vault.name if hasattr(vault, "name") else str(v_path)
            else:
                v_path = Path(vault).expanduser().resolve()
                v_name = str(v_path)

            if v_path.exists():
                count = len(list(v_path.rglob("*.md")))
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
    """执行混合检索 (完全复用 HybridRetriever)"""
    try:
        cfg = load_config()
        db_path = Path(cfg.db_path).resolve()
        if not db_path.exists():
            logger.error("❌ 数据库不存在,请先运行索引构建.")
            return 1

        # 权重配置
        alpha = (
            args.alpha
            if args.alpha is not None
            else cfg.confidence.fusion.get("alpha", 0.6)
        )
        beta = (
            args.beta
            if args.beta is not None
            else cfg.confidence.fusion.get("beta", 0.2)
        )

        # ✅ 核心修复:强制提取 vault name 字符串,兼容新旧配置结构
        if args.vaults:
            vaults = args.vaults
        else:
            vaults = []
            for v in cfg.vaults:
                enabled = getattr(v, "enabled", True)
                if enabled:
                    vaults.append(getattr(v, "name", str(v)))
            vaults = (
                vaults if vaults else None
            )  # ⚠️ 必须转为 None,否则 _fetch_results 会拦截全库

        logger.info(f"🔍 检索参数: alpha={alpha}, beta={beta}, vaults={vaults}")

        db = DatabaseManager(str(db_path))
        retriever = HybridRetriever(
            db=db,
            alpha=alpha,
            beta=beta,
            model_name=cfg.embedding_model.name,
            cache_dir=cfg.embedding_model.cache_dir,
        )

        start = time.time()
        results = retriever.search(
            query=args.query, mode=args.mode, top_k=args.top_k, vaults=vaults
        )
        elapsed = time.time() - start

        print(f"\n📊 检索结果 ({len(results)} 条,耗时 {elapsed:.2f}s):\n")
        if not results:
            print("   未找到相关结果.")
            return 0

        for i, r in enumerate(results, 1):
            content_preview = (
                r.content[:200] + "..." if len(r.content) > 200 else r.content
            )
            scores = f"最终={r.final_score:.3f} | 语义={r.semantic_score:.3f} | 关键词={r.keyword_score:.3f} | 置信度={r.confidence_score:.2f}"
            print(f"{i}. [{scores}]")
            print(f"   来源:{r.absolute_path}")
            print(f"   类型:{r.vault_name} / {r.chunk_type} | 章节:{r.section}")
            print(f"   内容:{content_preview}\n")
        return 0
    except Exception as e:
        logger.error(f"❌ 检索失败:{e}", exc_info=True)
        return 1


def cmd_config(args):
    """显示或编辑配置"""
    config_path = script_dir / "config.yaml"
    if args.show:
        if config_path.exists():
            print("\n📄 配置文件 (config.yaml):\n")
            print(config_path.read_text(encoding="utf-8"))
        else:
            print(f"❌ 配置文件不存在: {config_path}")
        return 0
    elif args.edit:
        import subprocess

        editor = os.environ.get("EDITOR", "vim")
        subprocess.run([editor, str(config_path)])
        return 0
    return 1


def main():
    parser = argparse.ArgumentParser(
        description="RAG System CLI 工具 (检索 / 状态检查 / 配置管理)",
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
  # 查看/编辑配置
  python rag_cli.py config --show
  python rag_cli.py config --edit
""",
    )
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    sp_search = subparsers.add_parser("search", help="执行混合检索")
    sp_search.add_argument("query", help="查询文本")
    sp_search.add_argument("--top-k", type=int, default=5, help="返回结果数量")
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
    sp_config.add_argument("--show", action="store_true", help="显示配置")
    sp_config.add_argument("--edit", action="store_true", help="编辑配置")
    sp_config.set_defaults(func=cmd_config)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
