#!/usr/bin/env python3
"""
方案 C 实施验证脚本

验证实体提取逻辑优化效果
"""

import sys
sys.path.insert(0, '/home/fallleaf/tinyRAG')

import sqlite3
from plugins.tinyrag_memory_graph.extractor import clean_entity_name


def check_current_entities():
    """检查当前实体数据"""
    conn = sqlite3.connect('/home/fallleaf/tinyRAG/data/rag.db')
    cursor = conn.cursor()
    
    # 统计实体总数
    cursor.execute("SELECT COUNT(*) FROM entities")
    total = cursor.fetchone()[0]
    
    # 统计有 chunk_id 的实体
    cursor.execute("SELECT COUNT(*) FROM entities WHERE chunk_id IS NOT NULL")
    with_chunk = cursor.fetchone()[0]
    
    # 统计无 chunk_id 的实体
    cursor.execute("SELECT COUNT(*) FROM entities WHERE chunk_id IS NULL")
    without_chunk = cursor.fetchone()[0]
    
    # 检查脏数据
    cursor.execute("""
        SELECT canonical_name, type, source 
        FROM entities 
        WHERE canonical_name LIKE '%`%' 
           OR canonical_name LIKE '%"' 
           OR canonical_name LIKE '%)%' 
           OR canonical_name LIKE '%{%' 
           OR canonical_name LIKE '%|%'
           OR LENGTH(canonical_name) < 2
    """)
    dirty_entities = cursor.fetchall()
    
    conn.close()
    
    print("=== 当前实体数据统计 ===")
    print(f"实体总数：{total}")
    print(f"有 chunk_id 的实体：{with_chunk}")
    print(f"无 chunk_id 的实体：{without_chunk}")
    print(f"脏数据实体：{len(dirty_entities)}")
    
    if dirty_entities:
        print("\n脏数据示例:")
        for name, entity_type, source in dirty_entities[:10]:
            print(f"  - {repr(name)} ({entity_type}, {source})")
    
    return total, with_chunk, without_chunk, len(dirty_entities)


def test_cleaning_effect():
    """测试清洗函数的过滤效果"""
    
    # 从数据库中获取所有实体名称
    conn = sqlite3.connect('/home/fallleaf/tinyRAG/data/rag.db')
    cursor = conn.cursor()
    cursor.execute("SELECT canonical_name FROM entities")
    entity_names = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    cleaned_count = 0
    filtered_count = 0
    filtered_names = []
    
    for name in entity_names:
        cleaned = clean_entity_name(name)
        if cleaned:
            cleaned_count += 1
        else:
            filtered_count += 1
            filtered_names.append(name)
    
    print("\n=== 清洗函数过滤效果 ===")
    print(f"总实体数：{len(entity_names)}")
    print(f"通过清洗：{cleaned_count}")
    print(f"被过滤：{filtered_count}")
    
    if filtered_names:
        print("\n被过滤的实体示例:")
        for name in filtered_names[:20]:
            print(f"  - {repr(name)}")
    
    return cleaned_count, filtered_count


def main():
    print("=== 方案 C 实施验证 ===\n")
    
    print("1. 检查当前实体数据")
    total, with_chunk, without_chunk, dirty_count = check_current_entities()
    
    print("\n2. 测试清洗函数过滤效果")
    cleaned_count, filtered_count = test_cleaning_effect()
    
    print("\n=== 总结 ===")
    print(f"当前实体总数：{total}")
    print(f"预计过滤后实体数：{cleaned_count}")
    print(f"预计减少实体数：{filtered_count}")
    print(f"过滤比例：{filtered_count/total*100:.2f}%")
    
    if filtered_count > 0:
        print("\n✅ 方案 C 实施后，预计可减少脏数据")
    else:
        print("\n⚠️ 当前实体数据质量较好，无需大量过滤")


if __name__ == "__main__":
    main()
