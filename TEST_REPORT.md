# tinyRAG 增量索引功能测试报告

## 测试时间
2026-04-09 01:00 - 01:10

## 测试环境
- 系统：Linux x86_64
- Python：3.12.3
- 数据库：SQLite + sqlite-vec
- 向量模型：BAAI/bge-small-zh-v1.5 (512维)

## 测试项目

### 1. 新增文件测试 ✅

**测试步骤：**
1. 创建测试文件 `test_incremental_index.md`
2. 运行 `build_index.py` 进行增量索引

**测试结果：**
- ✅ 检测到新文件：`00.任务清单/test_incremental_index.md`
- ✅ 创建文件记录：ID 1960
- ✅ 创建 16 个 chunks
- ✅ 创建 16 个 vectors
- ✅ 数据一致性：总文件数 982，总 chunks 14538，总 vectors 14538

**日志输出：**
```
➕ 检测到新文件：00.任务清单/test_incremental_index.md
📊 扫描结果：新增: 1, 修改: 0, 移动: 0, 删除: 0, 仅时间戳更新: 0
✅ 扫描报告处理完成（共 1 项变更）
🚀 开始处理 2 个文件...
🧩 待向量化块总数：16，采用 Batch Size: 128
✅ 向量化完成：16 条 (0.64s)
🎉 索引构建完成！耗时：1.71s
```

### 2. 修改文件测试 ✅

**测试步骤：**
1. 修改测试文件内容（新增章节和内容）
2. 运行 `build_index.py` 进行增量索引

**测试结果：**
- ✅ 检测到文件修改：`00.任务清单/test_incremental_index.md`
- ✅ 旧 chunks 被删除
- ✅ 创建新的 13 个 chunks
- ✅ 创建新的 13 个 vectors
- ✅ 数据一致性：总文件数 982，总 chunks 14535，总 vectors 14535

**日志输出：**
```
📝 检测到内容修改：00.任务清单/test_incremental_index.md
📊 扫描结果：新增: 0, 修改: 1, 移动: 0, 删除: 0, 仅时间戳更新: 0
✅ 扫描报告处理完成（共 1 项变更）
🚀 开始处理 1 个文件...
🧩 待向量化块总数：13，采用 Batch Size: 128
✅ 向量化完成：13 条 (1.04s)
🎉 索引构建完成！耗时：1.96s
```

### 3. 移动文件测试 ✅

**测试步骤：**
1. 将文件从 `test_incremental_index.md` 移动到 `test_moved_file.md`
2. 运行 `build_index.py` 进行增量索引

**测试结果：**
- ✅ 检测到文件移动：`personal/00.任务清单/test_incremental_index.md → personal/00.任务清单/test_moved_file.md`
- ✅ 文件路径更新为 `00.任务清单/test_moved_file.md`
- ✅ Chunks 正确关联到新文件 ID (1960)
- ✅ Vectors 正确关联到新 chunks
- ✅ 旧路径记录已清理
- ✅ 数据一致性：总文件数 982，总 chunks 14535，总 vectors 14535

**日志输出：**
```
🔄 检测到文件移动：personal/00.任务清单/test_incremental_index.md → personal/00.任务清单/test_moved_file.md
📊 扫描结果：新增: 0, 修改: 0, 移动: 1, 删除: 0, 仅时间戳更新: 0
✅ 扫描报告处理完成（共 1 项变更）
```

### 4. 删除文件测试 ✅

**测试步骤：**
1. 删除测试文件 `test_moved_file.md`
2. 运行 `build_index.py` 进行增量索引

**测试结果：**
- ✅ 检测到文件删除：`personal/00.任务清单/test_moved_file.md`
- ✅ 文件标记为已删除：`is_deleted = 1`
- ✅ Chunks 级联删除：活跃 chunks 数量为 0
- ✅ Vectors 级联删除：vectors 数量为 0
- ✅ 数据一致性：总文件数 981，总 chunks 14522，总 vectors 14522

**日志输出：**
```
🗑️ 检测到文件删除：personal/00.任务清单/test_moved_file.md
📊 扫描结果：新增: 0, 修改: 0, 移动: 0, 删除: 1, 仅时间戳更新: 0
✅ 扫描报告处理完成（共 1 项变更）
```

### 5. 检索功能测试 ✅

**测试步骤：**
1. 使用 CLI 工具进行检索：`python rag_cli.py search "增量索引" --top-k 3`
2. 使用 MCP 工具进行检索：`mcp_tinyRAG_search`

**测试结果：**
- ✅ CLI 检索正常工作
- ✅ MCP 检索正常工作
- ✅ 混合检索正常工作（语义 + 关键词）
- ✅ 置信度评分正常
- ✅ 返回相关结果

**CLI 输出：**
```
📊 检索结果 (3 条,耗时 1.47s):

1. [最终=0.962 | 语义=0.399 | 关键词=0.938 | 置信度=0.72]
   来源:/home/fallleaf/NanobotMemory/02.收集/增量索引测试.md
   类型:personal / header | 章节:增量索引测试
   内容:# 增量索引测试

这是一个测试文件，用于验证增量索引功能。
```

**MCP 输出：**
```json
{
  "query": "增量索引",
  "total": 3,
  "results": [
    {
      "rank": 1,
      "file": "02.收集/增量索引测试.md",
      "abs_path": "/home/fallleaf/NanobotMemory/02.收集/增量索引测试.md",
      "content": "# 增量索引测试\n\n这是一个测试文件，用于验证增量索引功能。",
      "score": 0.9622,
      "confidence": 0.72
    }
  ]
}
```

### 6. 扫描功能测试 ✅

**测试步骤：**
1. 使用 MCP 工具进行扫描：`mcp_tinyRAG_scan_index`

**测试结果：**
- ✅ 扫描功能正常工作
- ✅ 正确检测到无变更状态

**MCP 输出：**
```json
{
  "status": "success",
  "summary": "新增: 0, 修改: 0, 移动: 0, 删除: 0, 仅时间戳更新: 0",
  "new": 0,
  "modified": 0,
  "moved": 0,
  "deleted": 0,
  "touched": 0
}
```

## Bug 修复

### 修复 1：MoveEvent 属性访问错误

**问题描述：**
在 `mcp_server/server.py` 中，尝试访问 `MoveEvent.absolute_path` 属性，但该属性不存在。

**错误信息：**
```
AttributeError: 'MoveEvent' object has no attribute 'absolute_path'. Did you mean: 'new_absolute_path'?
```

**修复方案：**
修改 `mcp_server/server.py` 第 434 行，正确访问 `MoveEvent.new_absolute_path` 属性。

**修复代码：**
```python
# 修复前
changed_paths = [f.absolute_path for f in report.new_files + report.modified_files + report.moved_files]

# 修复后
changed_paths = []
for f in report.new_files + report.modified_files:
    changed_paths.append(f.absolute_path)
for f in report.moved_files:
    changed_paths.append(f.new_absolute_path)
```

## 测试总结

### 测试通过项目
- ✅ 新增文件检测和索引
- ✅ 修改文件检测和索引
- ✅ 移动文件检测和索引
- ✅ 删除文件检测和清理
- ✅ 数据一致性维护
- ✅ 混合检索功能
- ✅ MCP 工具集成
- ✅ CLI 工具功能

### 性能指标
- 新增文件索引：1.71s (16 chunks)
- 修改文件索引：1.96s (13 chunks)
- 检索响应时间：1.47s (3 条结果)
- 向量化速度：~9-10 块/秒

### 数据库状态
- 总文件数：981
- 总 Chunks 数：14522
- 总 Vectors 数：14522
- 向量维度：512
- 向量模型：BAAI/bge-small-zh-v1.5

## 结论

tinyRAG 的增量索引功能经过全面测试，所有测试项目均通过。系统能够正确检测和处理文件的增删改操作，保持数据一致性，并提供高效的检索功能。

修复了 MoveEvent 属性访问错误后，MCP 工具的 scan_index 功能能够正常工作，为用户提供了完整的增量索引解决方案。

---
*测试人员：nanobot*
*测试日期：2026-04-09*
