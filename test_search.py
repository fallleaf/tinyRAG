#!/usr/bin/env python3
"""
test_search.py - 测试 tinyRAG 搜索功能
用法: python test_search.py "查询内容" [--top-k 10] [--mode hybrid]
"""

import argparse
import sys
from pathlib import Path

# 确保项目根目录在 sys.path
script_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(script_dir))

from config import load_config  # noqa: E402
from embedder.embed_engine import EmbeddingEngine  # noqa: E402
from retriever.hybrid_engine import HybridEngine  # noqa: E402
from storage.database import DatabaseManager  # noqa: E402


def test_search(query: str, top_k: int = 10, mode: str = "hybrid"):
    """模拟 MCP SearchTool 的行为"""
    print(f"\n🔍 测试搜索: '{query}'")
    print(f"   top_k={top_k}, mode={mode}")
    print("-" * 60)

    # 加载配置
    cfg = load_config()
    db_path = Path(cfg.db_path).resolve()

    if not db_path.exists():
        print(f"❌ 数据库不存在: {db_path}")
        return

    # 初始化组件
    db = DatabaseManager(str(db_path), vec_dimension=cfg.embedding_model.dimensions)
    embed_engine = EmbeddingEngine(
        model_name=cfg.embedding_model.name,
        cache_dir=cfg.embedding_model.cache_dir,
        batch_size=cfg.embedding_model.batch_size,
        unload_after_seconds=cfg.embedding_model.unload_after_seconds,
    )
    retriever = HybridEngine(config=cfg, db=db, embed_engine=embed_engine)

    # 模拟 MCP SearchTool 的参数
    if mode == "keyword":
        alpha, beta = 0.0, 1.0
    elif mode == "semantic":
        alpha, beta = 1.0, 0.0
    else:
        alpha, beta = None, None

    # 构建 vault_filter（修复后）
    vaults = [v.name for v in cfg.vaults if v.enabled]
    vault_filter = vaults if vaults else None

    print(f"   vault_filter={vault_filter}")
    print(f"   alpha={alpha}, beta={beta}")
    print("-" * 60)

    # 执行搜索
    results = retriever.search(query, limit=top_k, vault_filter=vault_filter, alpha=alpha, beta=beta)

    print(f"\n📊 搜索结果 ({len(results)} 条):\n")

    for i, r in enumerate(results, 1):
        content_preview = r.content[:150] + "..." if len(r.content) > 150 else r.content
        print(f"{i}. [{r.vault_name}] {r.file_path}")
        print(f"   分数: final={r.final_score:.4f} | 语义={r.semantic_score:.4f} | 关键词={r.keyword_score:.4f}")
        print(f"   置信度: {r.confidence_score:.4f} ({r.confidence_reason})")
        print(f"   内容: {content_preview}")
        print()

    db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="测试 tinyRAG 搜索")
    parser.add_argument("query", help="查询内容")
    parser.add_argument("--top-k", type=int, default=10, help="返回结果数量")
    parser.add_argument("--mode", choices=["hybrid", "keyword", "semantic"], default="hybrid", help="搜索模式")
    args = parser.parse_args()

    test_search(args.query, args.top_k, args.mode)
