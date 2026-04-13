import sqlite3

conn = sqlite3.connect("data/rag.db")
conn.row_factory = sqlite3.Row

# 检查向量数量
vec_count = conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
print(f"向量数量: {vec_count}")

# 检查 chunks 数量
chunk_count = conn.execute("SELECT COUNT(*) FROM chunks WHERE is_deleted=0").fetchone()[0]
print(f"活跃 chunks 数量: {chunk_count}")

# 检查 files 数量
file_count = conn.execute("SELECT COUNT(*) FROM files WHERE is_deleted=0").fetchone()[0]
print(f"活跃 files 数量: {file_count}")

# 检查向量是否有数据
if vec_count > 0:
    cursor = conn.execute("SELECT chunk_id, length(embedding) as emb_len FROM vectors LIMIT 1")
    row = cursor.fetchone()
    print(f"示例向量: chunk_id={row[0]}, embedding_bytes={row[1]}")
else:
    print("⚠️ 向量表为空!")

# 测试向量检索
print("\\n测试向量检索...")
import array

# 创建一个随机向量
test_vec = array.array("f", [0.1] * 512).tobytes()
try:
    cursor = conn.execute(
        """
        SELECT chunk_id, distance FROM vectors 
        WHERE embedding MATCH ? 
        ORDER BY distance LIMIT 5
    """,
        (test_vec,),
    )
    results = cursor.fetchall()
    print(f"向量检索结果数: {len(results)}")
    if results:
        for r in results:
            print(f"  chunk_id={r[0]}, distance={r[1]}")
except Exception as e:
    print(f"向量检索错误: {e}")

conn.close()
