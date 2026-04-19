# tinyRAG - 知识库检索增强系统

> 基于 RAG (Retrieval-Augmented Generation) 的本地知识库系统，支持混合检索与图谱增强。

## 核心功能

| 功能 | 说明 |
|------|------|
| **智能分块** | Markdown 深度解析，支持 Frontmatter 元数据提取 |
| **混合检索** | 向量检索 + FTS5 关键词检索，RRF 融合排序 |
| **图谱增强** | WikiLink 关联发现，实体抽取，关系遍历 |
| **MCP 协议** | 标准 Model Context Protocol 接口，AI 助手集成 |

## 快速开始

```bash
# 1. 创建虚拟环境
python3 -m venv .venv && source .venv/bin/activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 安装中文 NLP 模型（图谱功能需要）
python -m spacy download zh_core_web_sm

# 4. 构建索引
python build_index.py --force

# 5. 搜索测试
python rag_cli.py search "人工智能" --top-k 5
```

## 配置说明

### 核心配置 (config.yaml)

```yaml
# 数据库
db_path: "./data/rag.db"

# 嵌入模型 (中文优化)
embedding_model:
  name: "BAAI/bge-small-zh-v1.5"
  dimensions: 512

# 分块配置
chunking:
  max_tokens: 512
  overlap: 50

# 检索权重
retrieval:
  alpha: 0.7    # 语义权重
  beta: 0.3     # 关键词权重

# 知识库
vaults:
  - path: "/path/to/documents"
    name: knowledge_base
    enabled: true

# 图谱插件
plugins:
  enabled: true
  plugins:
    - name: tinyrag_memory_graph
      enabled: true
```

## MCP 工具

| 工具 | 功能 |
|------|------|
| `search` | 混合检索（语义+关键词+图谱增强） |
| `scan_index` | 增量扫描更新索引 |
| `rebuild_index` | 强制重建索引 |
| `stats` | 获取知识库统计 |
| `config` | 获取完整配置 |
| `reload_config` | 热重载配置 |
| `maintenance` | 数据库清理与 VACUUM |

## 图谱增强

图谱插件通过 WikiLink（`[[文档名]]`）建立文档间关联：

```
查询 → 基础检索 → 种子 chunks → 图谱遍历 → 发现关联文档 → 融合排序
```

**工作原理：**
1. 搜索结果中包含 WikiLink 的 chunk 作为种子
2. 通过关系表遍历发现关联 chunk
3. 图谱分数 `graph_score` 作为加成权重
4. 最终分数 = 基础分 + β×图谱分 + γ×偏好分

## 目录结构

```
tinyRAG/
├── build_index.py        # 索引构建
├── rag_cli.py            # CLI 工具
├── config.yaml           # 配置文件
├── data/
│   ├── rag.db            # SQLite 数据库
│   └── cache.db          # 查询缓存
├── documents/            # 知识库目录
├── mcp_server/           # MCP 服务
├── plugins/              # 插件目录
│   └── tinyrag_memory_graph/
└── docs/                 # 文档
```

## 常用命令

```bash
# 搜索
python rag_cli.py search "关键词" --mode hybrid --top-k 10

# 纯语义检索
python rag_cli.py search "关键词" --mode semantic

# 纯关键词检索
python rag_cli.py search "关键词" --mode keyword

# 查看状态
python rag_cli.py status

# 增量扫描
python rag_cli.py scan

# 重建索引
python build_index.py --force
```

## 依赖说明

| 依赖 | 用途 |
|------|------|
| `fastembed` | 向量嵌入模型引擎 |
| `sqlite-vec` | SQLite 向量扩展 |
| `jieba` | 中文分词 |
| `tiktoken` | Token 计数 |
| `spacy` + `zh_core_web_sm` | NLP 实体抽取（图谱） |
| `mcp` | Model Context Protocol SDK |

## 故障排查

| 问题 | 解决方案 |
|------|---------|
| 模型下载失败 | 设置代理: `export HTTP_PROXY=http://proxy:port` |
| 数据库损坏 | `python build_index.py --force` |
| 图谱不生效 | 确保文档包含 WikiLink，重建索引 |
| 插件初始化失败 | 检查 `spacy` 和 `zh_core_web_sm` 是否安装 |

---

*更新日期：2026-04-19*
