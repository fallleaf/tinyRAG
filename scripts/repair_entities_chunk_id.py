#!/usr/bin/env python3
"""
数据回填脚本 - 修复 entities 表缺失的 chunk_id 字段

用于修复旧数据，将缺失 chunk_id 的实体回填到对应的 chunks 中。
"""

import sqlite3
import re
import sys


def clean_entity_name(name: str) -> bool:
    """检查实体名称是否有效（过滤脏数据）"""
    # 过滤特殊字符
    if re.match(r'^[\W_]+$', name):
        return False
    
    # 过滤过长的随机字符串
    if len(name) > 50 and re.match(r'^[A-Za-z0-9]+$', name):
        return False
    
    # 过滤长度异常的实体
    if len(name) < 2:
        return False
    
    return True


def repair_entities(db_path: str):
    """修复 entities 表中的 chunk_id 字段"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("=== 开始修复 entities 表 ===")
    
    # 1. 统计缺失 chunk_id 的实体数量
    cursor.execute("SELECT COUNT(*) FROM entities WHERE chunk_id IS NULL")
    total_missing = cursor.fetchone()[0]
    print(f"需要修复的实体数量：{total_missing}")
    
    if total_missing == 0:
        print("✅ 所有实体已有 chunk_id，无需修复")
        conn.close()
        return
    
    # 2. 获取所有缺少 chunk_id 的实体
    cursor.execute("""
        SELECT id, canonical_name, type 
        FROM entities 
        WHERE chunk_id IS NULL
        ORDER BY id
    """)
    entities = cursor.fetchall()
    
    repaired_count = 0
    skipped_count = 0
    failed_count = 0
    
    for ent_id, name, ent_type in entities:
        # 3. 过滤脏数据
        if not clean_entity_name(name):
            skipped_count += 1
            continue
        
        # 4. 在 chunks 表中搜索包含该实体的内容
        cursor.execute("""
            SELECT id FROM chunks 
            WHERE content LIKE ? 
            LIMIT 1
        """, (f'%{name}%',))
        
        result = cursor.fetchone()
        if result:
            chunk_id = result[0]
            cursor.execute(
                "UPDATE entities SET chunk_id = ? WHERE id = ?",
                (chunk_id, ent_id)
            )
            repaired_count += 1
        else:
            failed_count += 1
    
    conn.commit()
    conn.close()
    
    # 5. 统计结果
    print("\n=== 修复结果 ===")
    print(f"成功修复：{repaired_count}")
    print(f"跳过脏数据：{skipped_count}")
    print(f"匹配失败：{failed_count}")
    print(f"总计：{repaired_count + skipped_count + failed_count}")
    
    if repaired_count > 0:
        print("\n✅ 修复完成！")
    else:
        print("\n⚠️ 未修复任何实体，建议重建索引")


if __name__ == "__main__":
    db_path = "data/rag.db"
    
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    
    repair_entities(db_path)
