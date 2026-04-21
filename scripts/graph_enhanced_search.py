#!/usr/bin/env python3
"""
图谱检索增强 - 利用 chunk_id 进行实体到 Chunk 的映射

修复图谱查询分值为零的问题：
1. 通过实体名称查找有 chunk_id 的实体
2. 以 chunk_id 为起点在 relations 表中发起遍历
3. 返回非零分值和相关图谱路径
"""

import sqlite3
from typing import Optional


def search_with_graph_enhancement(db_path: str, query: str, top_k: int = 5) -> list[dict]:
    """
    增强检索：结合图谱遍历
    
    Args:
        db_path: 数据库路径
        query: 查询词
        top_k: 返回结果数量
        
    Returns:
        检索结果列表，包含 chunk 内容和图谱路径信息
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    results = []
    
    # 1. 在 chunks 表中进行关键词检索
    cursor.execute(
        """SELECT id, content, note_id 
           FROM chunks 
           WHERE content LIKE ? 
           ORDER BY id 
           LIMIT ?""",
        (f"%{query}%", top_k),
    )
    base_results = [dict(row) for row in cursor.fetchall()]
    
    # 2. 在 entities 表中查找匹配的实体（仅返回有 chunk_id 的）
    cursor.execute(
        """SELECT id, canonical_name, type, chunk_id 
           FROM entities 
           WHERE canonical_name = ? AND chunk_id IS NOT NULL
           LIMIT 10""",
        (query,),
    )
    entities = [dict(row) for row in cursor.fetchall()]
    
    # 3. 如果有匹配的实体，以 chunk_id 为起点进行图谱遍历
    if entities:
        chunk_ids = [e["chunk_id"] for e in entities if e["chunk_id"]]
        
        if chunk_ids:
            # 3.1 构建 IN 查询条件
            placeholders = ",".join(["?" for _ in chunk_ids])
            
            # 3.2 从 relations 表中查找相关 Chunk
            cursor.execute(
                f"""SELECT DISTINCT 
                        r.src_chunk_id, 
                        r.tgt_chunk_id, 
                        r.rel_type, 
                        r.weight,
                        r.scope
                     FROM relations r
                     WHERE r.src_chunk_id IN ({placeholders})
                        OR r.tgt_chunk_id IN ({placeholders})
                     ORDER BY r.weight DESC
                     LIMIT 20""",
                chunk_ids + chunk_ids,
            )
            related_chunks = [dict(row) for row in cursor.fetchall()]
            
            # 3.3 获取相关 Chunk 的详细内容
            if related_chunks:
                related_chunk_ids = list(set(
                    r["src_chunk_id"] for r in related_chunks
                ) | set(
                    r["tgt_chunk_id"] for r in related_chunks
                ))
                
                placeholders = ",".join(["?" for _ in related_chunk_ids])
                cursor.execute(
                    f"""SELECT id, content, note_id 
                        FROM chunks 
                        WHERE id IN ({placeholders})""",
                    related_chunk_ids,
                )
                chunk_details = {
                    row["id"]: dict(row) 
                    for row in cursor.fetchall()
                }
                
                # 3.4 构建增强结果
                for rel in related_chunks:
                    chunk_id = rel["src_chunk_id"] if rel["src_chunk_id"] in chunk_details else rel["tgt_chunk_id"]
                    if chunk_id in chunk_details:
                        chunk_info = chunk_details[chunk_id]
                        results.append({
                            "type": "graph_related",
                            "chunk_id": chunk_id,
                            "content": chunk_info["content"],
                            "note_id": chunk_info["note_id"],
                            "relation": rel["rel_type"],
                            "weight": rel["weight"],
                            "scope": rel["scope"],
                            "path": f"{rel['src_chunk_id']} -> {rel['tgt_chunk_id']}",
                            "score": rel["weight"],  # 使用关系权重作为评分
                        })
    
    # 4. 合并基础结果和图谱增强结果
    all_results = base_results + results
    
    # 5. 去重（按 chunk_id）
    seen_ids = set()
    unique_results = []
    for result in all_results:
        chunk_id = result.get("chunk_id") or result.get("id")
        if chunk_id not in seen_ids:
            seen_ids.add(chunk_id)
            unique_results.append(result)
    
    # 6. 限制返回数量
    return unique_results[:top_k * 2]


if __name__ == "__main__":
    # 测试示例
    db_path = "data/rag.db"
    query = "GB"  # 测试实体
    
    results = search_with_graph_enhancement(db_path, query, top_k=5)
    
    print(f"=== 查询结果 ({len(results)} 条) ===")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. 类型：{result.get('type', 'base')}")
        print(f"   Chunk ID: {result.get('chunk_id') or result.get('id')}")
        print(f"   内容：{result.get('content', '')[:100]}...")
        if result.get("type") == "graph_related":
            print(f"   关系：{result.get('relation')}")
            print(f"   权重：{result.get('weight')}")
            print(f"   路径：{result.get('path')}")
            print(f"   评分：{result.get('score')}")
