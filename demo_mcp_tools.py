#!/usr/bin/env python3
"""
演示 MCP 工具调用 - 用户视角
展示如何通过 tinyRAG 的 MCP 工具获取知识库检索和文档总结
"""

from loguru import logger
import json
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from config import load_config
from storage.database import DatabaseManager
from retriever.hybrid_engine import HybridEngine
from embedder.embed_engine import EmbeddingEngine


class MCPSimulator:
    """模拟 MCP 客户端调用"""

    def __init__(self):
        self.config = load_config("config.yaml")
        self.db = DatabaseManager(self.config.db_path, vec_dimension=self.config.embedding_model.dimensions)
        self.embed_engine = EmbeddingEngine(
            model_name=self.config.embedding_model.name,
            cache_dir=self.config.embedding_model.cache_dir,
            batch_size=self.config.embedding_model.batch_size,
        )
        self.retriever = HybridEngine(config=self.config, db=self.db, embed_engine=self.embed_engine)

    def close(self):
        self.db.close()

    # ==================== Tools ====================

    def tool_stats(self) -> dict:
        """MCP Tool: stats - 获取知识库统计"""
        files_total = self.db.conn.execute("SELECT COUNT(*) FROM files WHERE is_deleted = 0").fetchone()[0]
        files_by_vault = self.db.conn.execute(
            "SELECT vault_name, COUNT(*) as cnt FROM files WHERE is_deleted = 0 GROUP BY vault_name"
        ).fetchall()
        chunks_total = self.db.conn.execute("SELECT COUNT(*) FROM chunks WHERE is_deleted = 0").fetchone()[0]
        try:
            vectors_total = self.db.conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
        except Exception:
            vectors_total = 0

        return {
            "files": {"total": files_total, "by_vault": {row["vault_name"]: row["cnt"] for row in files_by_vault}},
            "chunks": {"total": chunks_total, "avg_per_file": round(chunks_total / max(files_total, 1), 1)},
            "vectors": {"total": vectors_total, "dimensions": self.config.embedding_model.dimensions},
            "model": {"name": self.config.embedding_model.name},
        }

    def tool_search(self, query: str, mode: str = "hybrid", top_k: int = 5) -> dict:
        """MCP Tool: search - 混合检索"""
        if mode == "keyword":
            alpha, beta = 0.0, 1.0
        elif mode == "semantic":
            alpha, beta = 1.0, 0.0
        else:
            alpha, beta = None, None

        vaults = [v.name for v in self.config.vaults if v.enabled]
        results = self.retriever.search(
            query, limit=top_k, vault_filter=vaults if vaults else None, alpha=alpha, beta=beta
        )

        return {
            "query": query,
            "mode": mode,
            "total": len(results),
            "results": [
                {
                    "rank": i + 1,
                    "file": r.file_path,
                    "section": r.section,
                    "content": r.content[:200] + "..." if len(r.content) > 200 else r.content,
                    "score": round(r.final_score, 4),
                    "semantic_score": round(r.semantic_score, 4),
                    "keyword_score": round(r.keyword_score, 4),
                    "confidence": round(r.confidence_score, 4),
                }
                for i, r in enumerate(results)
            ],
        }

    # ==================== Prompts ====================

    def prompt_summarize_document(self, file_path: str) -> str:
        """MCP Prompt: summarize_document - 生成文档摘要提示词"""
        # 查找文件
        row = self.db.conn.execute(
            "SELECT id, file_path FROM files WHERE file_path = ? AND is_deleted = 0", (file_path,)
        ).fetchone()

        if not row:
            # 尝试模糊匹配
            row = self.db.conn.execute(
                "SELECT id, file_path FROM files WHERE file_path LIKE ? AND is_deleted = 0 LIMIT 1",
                (f"%{Path(file_path).name}%",),
            ).fetchone()

        if not row:
            return f"❌ 未找到文档: {file_path}"

        # 获取所有 chunks
        chunks = self.db.conn.execute(
            "SELECT content, section_title, confidence_json FROM chunks WHERE file_id = ? AND is_deleted = 0 ORDER BY chunk_index",
            (row["id"],),
        ).fetchall()

        if not chunks:
            return f"❌ 文档无内容: {file_path}"

        # 构建内容
        content_parts = []
        for c in chunks:
            section = c["section_title"] or "正文"
            content_parts.append(f"### {section}\n{c['content']}")

        full_content = "\n\n".join(content_parts)

        # 解析元数据
        conf = json.loads(chunks[0]["confidence_json"]) if chunks and chunks[0]["confidence_json"] else {}

        # 生成提示词（用户可以直接发给 LLM）
        prompt = f"""请总结以下文档的核心内容，提取关键信息点。

## 文档信息
- 路径: {row["file_path"]}
- 类型: {conf.get("doc_type", "未知")}
- 状态: {conf.get("status", "未知")}
- 日期: {conf.get("final_date", "未知")}

## 文档内容
{full_content[:6000]}

## 请输出
1. **文档摘要** (100字以内)
2. **关键要点** (3-5个要点)
3. **核心关键词** (5-10个关键词)
"""
        return prompt

    def prompt_search_with_context(self, query: str, top_k: int = 5) -> str:
        """MCP Prompt: search_with_context - 检索增强回答提示词"""
        # 先检索
        results = self.tool_search(query, top_k=top_k)

        if not results["results"]:
            return f"❌ 未找到相关内容: {query}"

        # 构建上下文
        context_parts = []
        for r in results["results"]:
            context_parts.append(
                f"【文档 {r['rank']}】{r['file']}\n章节: {r['section']}\n内容: {r['content']}\n相关度: {r['score']}"
            )

        context = "\n\n".join(context_parts)

        prompt = f"""基于以下知识库检索结果回答用户问题。

## 用户问题
{query}

## 知识库检索结果 ({results["total"]} 条)
{context}

## 请回答
请综合以上检索结果，用中文回答用户的问题。如果检索结果不足以回答问题，请诚实说明。
"""
        return prompt


def main():
    logger.info("=" * 60)
    logger.info("🚀 tinyRAG MCP 工具演示 - 用户视角")
    logger.info("=" * 60)

    sim = MCPSimulator()

    try:
        # 1. 获取知识库统计
        logger.info("\n📌 [Tool: stats] 获取知识库统计")
        logger.info("-" * 40)
        stats = sim.tool_stats()
        logger.info(f"  📁 文件总数: {stats['files']['total']}")
        logger.info(f"  📝 分块总数: {stats['chunks']['total']}")
        logger.info(f"  🔢 向量总数: {stats['vectors']['total']}")
        logger.info(f"  🤖 嵌入模型: {stats['model']['name']}")

        # 2. 搜索演示
        logger.info("\n📌 [Tool: search] 混合检索演示")
        logger.info("-" * 40)
        query = "人工智能"
        search_result = sim.tool_search(query, top_k=3)
        logger.info(f"  查询: '{query}'")
        logger.info(f"  结果数: {search_result['total']}")
        for r in search_result["results"]:
            logger.info(f"  [{r['rank']}] {r['file']} (分数: {r['score']})")

        # 3. 文档摘要提示词
        logger.info("\n📌 [Prompt: summarize_document] 文档摘要")
        logger.info("-" * 40)
        doc_path = "RAG检索增强生成技术.md"
        prompt = sim.prompt_summarize_document(doc_path)
        logger.info(f"  文档: {doc_path}")
        logger.info("\n生成的提示词 (可直接发送给 LLM):")
        logger.info("  " + "-" * 36)
        # 只显示提示词的前 800 字符
        preview = prompt[:800] + "...\n[内容已截断，完整提示词可发给 LLM]" if len(prompt) > 800 else prompt
        for line in preview.split("\n"):
            logger.info(f"  {line}")

        # 4. 检索增强回答提示词
        logger.info("\n📌 [Prompt: search_with_context] 检索增强回答")
        logger.info("-" * 40)
        query = "什么是RAG技术"
        prompt2 = sim.prompt_search_with_context(query, top_k=3)
        logger.info(f"  问题: '{query}'")
        logger.info("\n生成的提示词 (可直接发送给 LLM):")
        logger.info("  " + "-" * 36)
        preview2 = prompt2[:800] + "...\n[内容已截断]" if len(prompt2) > 800 else prompt2
        for line in preview2.split("\n"):
            logger.info(f"  {line}")

        logger.info("\n" + "=" * 60)
        logger.info("✅ MCP 工具演示完成")
        logger.info("=" * 60)

    finally:
        sim.close()


if __name__ == "__main__":
    main()
