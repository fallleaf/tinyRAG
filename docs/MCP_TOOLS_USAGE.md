# tinyRAG MCP 工具使用文档

## 概述

tinyRAG 提供了 MCP (Model Context Protocol) 工具，允许 LLM 直接调用知识库检索功能。

## 工具列表

### 1. `mcp_tinyRAG_search` - 知识库检索

#### 功能描述

混合检索知识库，支持语义检索和关键词检索的 RRF (Reciprocal Rank Fusion) 融合。

#### 参数说明

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `query` | string | ✅ | - | 搜索关键词，支持自然语言查询 |
| `mode` | string | ❌ | `hybrid` | 检索模式：`semantic`(纯语义), `keyword`(纯关键词), `hybrid`(混合) |
| `top_k` | integer | ❌ | `10` | 返回结果数量，范围 1-100 |
| `alpha` | number | ❌ | `None` | 语义检索权重 (0.0-1.0)，默认使用 config.yaml 中的值 |
| `beta` | number | ❌ | `None` | 关键词检索权重 (0.0-1.0)，默认使用 config.yaml 中的值 |
| `vaults` | array | ❌ | `None` | 指定检索的仓库名称列表，不指定则检索所有启用的 vault |

#### 返回格式

```json
{
  "query": "搜索关键词",
  "total": 5,
  "results": [
    {
      "rank": 1,
      "file": "文件路径",
      "abs_path": "绝对路径",
      "content": "内容摘要（前300字符）",
      "score": 0.95,
      "confidence": 0.85,
      "confidence_reason": "置信度原因"
    }
  ]
}
```

#### 使用示例

##### 示例 1: 基础检索

```python
mcp_tinyRAG_search(query="酒精炉")
```

##### 示例 2: 指定返回数量

```python
mcp_tinyRAG_search(query="国际网络一体化", top_k=20)
```

##### 示例 3: 纯语义检索

```python
mcp_tinyRAG_search(query="水培生菜", mode="semantic")
```

##### 示例 4: 纯关键词检索

```python
mcp_tinyRAG_search(query="OTN 设备", mode="keyword")
```

##### 示例 5: 自定义权重（技术术语）

```python
mcp_tinyRAG_search(
    query="OTN 设备",
    alpha=0.5,
    beta=0.5
)
```

##### 示例 6: 自定义权重（生活兴趣）

```python
mcp_tinyRAG_search(
    query="酒精炉",
    alpha=0.7,
    beta=0.3
)
```

##### 示例 7: 自定义权重（项目政策）

```python
mcp_tinyRAG_search(
    query="国际网络一体化",
    alpha=0.6,
    beta=0.4
)
```

##### 示例 8: 只指定 alpha（beta 自动计算）

```python
mcp_tinyRAG_search(
    query="测试",
    alpha=0.7
)
# beta 自动计算为 0.3
```

##### 示例 9: 只指定 beta（alpha 自动计算）

```python
mcp_tinyRAG_search(
    query="测试",
    beta=0.3
)
# alpha 自动计算为 0.7
```

##### 示例 10: 指定 vault

```python
mcp_tinyRAG_search(
    query="酒精炉",
    vaults=["personal"]
)
```

##### 示例 11: 指定多个 vault

```python
mcp_tinyRAG_search(
    query="OTN 设备",
    vaults=["personal", "work"]
)
```

##### 示例 12: 完整参数

```python
mcp_tinyRAG_search(
    query="酒精炉安全",
    mode="hybrid",
    top_k=10,
    alpha=0.7,
    beta=0.3,
    vaults=["personal"]
)
```

---

### 2. `mcp_tinyRAG_scan_index` - 增量扫描索引

#### 功能描述

增量扫描文件并更新索引。

#### 参数说明

无参数。

#### 使用示例

```python
mcp_tinyRAG_scan_index()
```

---

### 3. `mcp_tinyRAG_rebuild_index` - 重建索引

#### 功能描述

强制清空并重建完整索引。

#### 参数说明

无参数。

#### 使用示例

```python
mcp_tinyRAG_rebuild_index()
```

---

### 4. `mcp_tinyRAG_maintenance` - 数据库维护

#### 功能描述

执行数据库清理和 VACUUM 操作。

#### 参数说明

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `dry_run` | boolean | ❌ | `false` | 是否只检查不执行 |

#### 使用示例

```python
# 检查是否需要维护
mcp_tinyRAG_maintenance(dry_run=true)

# 执行维护
mcp_tinyRAG_maintenance()
```

---

### 5. `mcp_tinyRAG_stats` - 获取统计信息

#### 功能描述

获取知识库统计信息。

#### 参数说明

无参数。

#### 返回格式

```json
{
  "total_files": 100,
  "total_chunks": 1000,
  "vaults": [
    {
      "name": "personal",
      "files": 50,
      "chunks": 500
    }
  ]
}
```

#### 使用示例

```python
mcp_tinyRAG_stats()
```

---

### 6. `mcp_tinyRAG_config` - 获取配置

#### 功能描述

获取完整配置信息。

#### 参数说明

无参数。

#### 使用示例

```python
mcp_tinyRAG_config()
```

---

### 7. `mcp_tinyRAG_reload_config` - 重新加载配置

#### 功能描述

热重载 config.yaml 并重新初始化检索器和分词器。

#### 参数说明

无参数。

#### 使用示例

```python
mcp_tinyRAG_reload_config()
```

---

### 8. `mcp_tinyRAG_prompt_search_with_context` - 检索回答 Prompt

#### 功能描述

返回一个填充好的 prompt 模板，用于 LLM 基于检索结果回答问题。

#### 参数说明

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `query` | string | ✅ | - | 搜索关键词，支持自然语言查询 |
| `top_k` | string | ❌ | `5` | 返回结果数量（字符串类型） |
| `alpha` | string | ❌ | `None` | 语义检索权重 (0.0-1.0)，默认使用 config.yaml 中的值（字符串类型） |
| `beta` | string | ❌ | `None` | 关键词检索权重 (0.0-1.0)，默认使用 config.yaml 中的值（字符串类型） |
| `vaults` | array | ❌ | `None` | 指定检索的仓库名称列表，不指定则检索所有启用的 vault |

#### 返回格式

```json
{
  "description": "检索回答: 查询关键词",
  "messages": [
    {
      "role": "user",
      "content": {
        "type": "text",
        "text": "填充好的 prompt 模板"
      }
    }
  ]
}
```

#### 使用示例

##### 示例 1: 基础检索

```python
mcp_tinyRAG_prompt_search_with_context(query="酒精炉")
```

##### 示例 2: 指定返回数量

```python
mcp_tinyRAG_prompt_search_with_context(query="国际网络一体化", top_k="10")
```

##### 示例 3: 自定义权重

```python
mcp_tinyRAG_prompt_search_with_context(
    query="OTN 设备",
    alpha="0.5",
    beta="0.5"
)
```

##### 示例 4: 指定 vault

```python
mcp_tinyRAG_prompt_search_with_context(
    query="酒精炉",
    vaults=["personal"]
)
```

##### 示例 5: 完整参数

```python
mcp_tinyRAG_prompt_search_with_context(
    query="酒精炉安全",
    top_k="5",
    alpha="0.7",
    beta="0.3",
    vaults=["personal"]
)
```

---

### 9. `mcp_tinyRAG_prompt_summarize_document` - 文档摘要 Prompt

#### 功能描述

返回一个填充好的 prompt 模板，用于 LLM 摘要单个文档。

#### 参数说明

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `file_path` | string | ✅ | - | 文件路径 |

#### 返回格式

```json
{
  "description": "摘要: 文件路径",
  "messages": [
    {
      "role": "user",
      "content": {
        "type": "text",
        "text": "填充好的 prompt 模板"
      }
    }
  ]
}
```

#### 使用示例

```python
mcp_tinyRAG_prompt_summarize_document(file_path="04.生活/酒精炉.md")
```

---

## 智能检索策略

### 推荐权重配置

| 查询类型 | Alpha | Beta | 说明 |
|---|---|---|---|
| **技术术语** | 0.5 | 0.5 | OTN、PTN、SDH、MPLS、OLT 等精确匹配 |
| **项目政策** | 0.6 | 0.4 | 国际网络一体化、极简网络、南方宽带等平衡检索 |
| **网络协议** | 0.5 | 0.5 | IPLC、IEPL、SD-WAN、城域网等均衡检索 |
| **生活兴趣** | 0.7 | 0.3 | 酒精炉、水培生菜、吉他等语义理解优先 |
| **通用查询** | 0.6 | 0.4 | 默认配置 |

### LLM 调用建议

LLM 应根据查询内容自动选择合适的权重：

```
用户：检索酒精炉
LLM：调用 mcp_tinyRAG_search(query="酒精炉", alpha=0.7, beta=0.3)

用户：查找国际网络一体化项目
LLM：调用 mcp_tinyRAG_search(query="国际网络一体化", alpha=0.6, beta=0.4)

用户：搜索 OTN 设备信息
LLM：调用 mcp_tinyRAG_search(query="OTN 设备", alpha=0.5, beta=0.5)
```

---

## 错误处理

### 常见错误

#### 1. alpha + beta != 1

```json
{
  "error": "alpha + beta 必须等于 1，当前：alpha=0.6, beta=0.6",
  "query": "测试",
  "total": 0,
  "results": []
}
```

**解决方案**：确保 `alpha + beta = 1`，或只指定其中一个参数。

#### 2. vaults 不存在或未启用

```json
{
  "error": "指定的 vaults 不存在或未启用：['不存在的vault']",
  "query": "测试",
  "total": 0,
  "results": []
}
```

**解决方案**：检查 vault 名称是否正确，或检查 vault 是否在 config.yaml 中启用。

---

## 与 rag_cli.py 的一致性

| 参数 | rag_cli.py | MCP server.py | 状态 |
|---|---|---|---|
| `query` | ✅ 必填 | ✅ 必填 | ✅ 一致 |
| `top_k` | ✅ 1-100，默认 10 | ✅ 1-100，默认 10 | ✅ 一致 |
| `mode` | ✅ 枚举 | ✅ 枚举 | ✅ 一致 |
| `alpha` | ✅ 可选，覆盖默认 | ✅ 可选，覆盖默认 | ✅ 一致 |
| `beta` | ✅ 可选，覆盖默认 | ✅ 可选，覆盖默认 | ✅ 一致 |
| `vaults` | ✅ 可选 | ✅ 可选 | ✅ 一致 |

---

## 性能优化建议

1. **使用 vaults 参数**：指定 vault 可以减少检索范围，提高速度
2. **合理设置 top_k**：根据需求设置返回数量，避免过多结果
3. **选择合适的 mode**：
   - 精确匹配：使用 `keyword` 模式
   - 语义理解：使用 `semantic` 模式
   - 平衡检索：使用 `hybrid` 模式

---

## 版本历史

### v1.0 (2026-04-23)

- ✅ 添加 `alpha/beta` 参数支持
- ✅ 添加 `vaults` 参数支持
- ✅ 参数验证逻辑
- ✅ 自动计算缺失的权重
- ✅ 与 rag_cli.py 完全一致
