# tinyRAG - 轻量级中文 RAG 系统

> **v1.2.0** | 检索增强生成系统，支持图谱增强检索

## 项目简介

tinyRAG 是一个轻量级的检索增强生成（RAG）系统，专为中文知识库设计。系统采用向量检索与全文检索（FTS5）双引擎混合架构，并通过图谱插件实现 WikiLink 关联发现，显著提升检索召回率。

### 核心特性

| 特性 | 说明 |
|------|------|
| **混合检索引擎** | 向量语义检索 + FTS5 关键词检索双引擎融合，RRF 排序 |
| **图谱增强检索** | WikiLink 关联发现、实体抽取、关系遍历，提升召回率 |
| **智能分块** | 按标题层级、代码块、表格等内容类型智能分割 |
| **动态置信度** | 基于文档类型、状态、日期衰减动态计算置信度 |
| **模型自动卸载** | 嵌入模型空闲自动卸载，优化内存占用 |
| **增量索引** | 两阶段扫描，支持文件增删改移动的增量更新 |
| **MCP 服务** | 完整支持 Tools + Resources + Prompts 三大接口 |

## 系统架构

```
tinyRAG/
├── main.py              # MCP 服务入口
├── config.py            # Pydantic v2 配置管理
├── build_index.py       # 流式+并行索引构建
├── rag_cli.py           # CLI 工具
├── config.yaml          # 配置文件
├── data/
│   ├── rag.db           # SQLite 数据库
│   └── cache.db         # 查询缓存
├── chunker/             # Markdown 智能分块
├── embedder/            # 懒加载嵌入引擎
├── retriever/           # 混合检索引擎
├── scanner/             # 两阶段文件扫描
├── storage/             # 数据库 + 缓存
├── mcp_server/          # MCP 协议实现
├── plugins/             # 插件目录
│   └── tinyrag_memory_graph/  # 图谱增强插件
└── docs/                # 文档
```

## 快速开始

### 环境要求
- Python 3.10+
- SQLite 3.35+（支持 FTS5）

### 安装依赖

```bash
# 克隆项目
git clone https://github.com/fallleaf/tinyRAG.git
cd tinyRAG

# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 安装中文 NLP 模型（图谱功能需要）
python -m spacy download zh_core_web_sm
```

### 构建索引

```bash
# 首次构建（强制重建）
python build_index.py --force

# 增量更新
python build_index.py
```

### 命令行检索

```bash
# 混合检索
python rag_cli.py search "极简网络" --top-k 5

# 纯关键词检索
python rag_cli.py search "网络优化" --mode keyword

# 查看状态
python rag_cli.py status
```

### MCP 服务

```bash
# 启动 MCP 服务
python main.py
```

## MCP 接口

### Tools（工具）

| 工具 | 功能 | 参数 |
|------|------|------|
| `search` | 混合检索（语义+关键词+图谱增强） | query, mode, top_k |
| `stats` | 知识库统计 | 无 |
| `config` | 获取完整配置 | 无 |
| `scan_index` | 增量扫描 | 无 |
| `rebuild_index` | 重建索引 | 无 |
| `reload_config` | 热重载配置 | 无 |
| `maintenance` | 数据库清理与 VACUUM | 无 |

### Resources（资源）

| URI | 说明 |
|-----|------|
| `tinyrag://stats` | 知识库统计信息 |
| `tinyrag://config` | 配置信息 |
| `tinyrag://vault/{name}` | 指定 Vault 统计 |
| `tinyrag://file/{id}` | 文件内容 |
| `tinyrag://chunks/{id}` | 文件分块 |

### Prompts（提示词）

| 提示词 | 功能 |
|--------|------|
| `search_with_context` | 检索增强回答 |
| `summarize_document` | 文档摘要 |

## 图谱增强检索

图谱插件通过 WikiLink（`[[文档名]]`）建立文档间关联：

```
查询 → 基础检索 → 种子 chunks → 图谱遍历 → 发现关联文档 → 融合排序
```

**工作原理：**
1. 搜索结果中包含 WikiLink 的 chunk 作为种子
2. 通过关系表遍历发现关联 chunk
3. 图谱分数 `graph_score` 作为加成权重
4. 最终分数 = 基础分 + β×图谱分 + γ×偏好分

## 配置示例

```yaml
# config.yaml
db_path: "./data/rag.db"

embedding_model:
  name: "BAAI/bge-small-zh-v1.5"
  dimensions: 512
  batch_size: 64

chunking:
  max_tokens: 512
  overlap: 50

retrieval:
  alpha: 0.7  # 语义权重
  beta: 0.3   # 关键词权重

vaults:
  - name: "knowledge_base"
    path: "/path/to/documents"
    enabled: true

# 图谱插件配置
plugins:
  enabled: true
  plugins:
    - name: tinyrag_memory_graph
      enabled: true
      config:
        graph:
          enabled: true
        retrieval:
          vector_weight: 0.6
          graph_weight: 0.4
          max_hops: 2
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
| 图谱不生效 | 确保文档包含 WikiLink，检查 spacy 模型安装 |
| 插件初始化失败 | 确认 `plugins.enabled: true` 且 spacy 模型已安装 |

## 文档

- [项目总览](docs/README.md) - 详细功能介绍
- [部署指南](docs/DEPLOYMENT.md) - 生产环境部署
- [快速参考](docs/QUICK_REFERENCE.md) - 命令速查

## 许可证

MIT License

---

*版本: v1.2.0 | 最后更新: 2026-04-19*
