# tinyRAG - 轻量级中文 RAG 系统

> **v1.1.5** | 检索增强生成系统，专为中文知识库设计

## 项目简介

tinyRAG 是一个轻量级的检索增强生成（RAG）系统，专为中文知识库设计。系统采用向量检索与全文检索（FTS5）双引擎混合架构，支持 Markdown 文档的智能分块、向量化索引和语义检索。

### 核心特性

| 特性 | 说明 |
|------|------|
| **混合检索引擎** | 向量语义检索 + FTS5 关键词检索双引擎融合，RRF 排序 |
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
├── chunker/             # Markdown 智能分块
├── embedder/            # 懒加载嵌入引擎
├── retriever/           # 混合检索引擎
├── scanner/             # 两阶段文件扫描
├── storage/             # 数据库 + 缓存
├── mcp_server/          # MCP 协议实现
└── prompts/             # 提示词模板
```

## 快速开始

### 环境要求
- Python 3.10+
- SQLite 3.35+（支持 FTS5）
- 可选：sqlite-vec 扩展

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
| `search` | 混合检索 | query, mode, top_k, alpha, beta, vaults |
| `stats` | 知识库统计 | 无 |
| `scan_index` | 增量扫描 | 无 |
| `rebuild_index` | 重建索引 | 无 |
| `maintenance` | 数据库维护 | dry_run |
| `config` | 获取配置 | 无 |
| `reload_config` | 重新加载配置 | 无 |

### Resources（资源）

| URI | 说明 |
|-----|------|
| `tinyrag://stats` | 知识库统计信息 |
| `tinyrag://config` | 配置信息 |
| `tinyrag://vault/{name}` | 指定 Vault 统计 |
| `tinyrag://file/{id}` | 文件内容 |
| `tinyrag://chunks/{id}` | 文件分块 |

### Prompts（提示词）

| 提示词 | 功能 | 参数 |
|--------|------|------|
| `search_with_context` | 检索增强回答 | query, top_k, alpha, beta, vaults |
| `summarize_document` | 文档摘要 | file_path |

## 配置示例

```yaml
# config.yaml
db_path: "./data/rag.db"

embedding_model:
  name: "BAAI/bge-small-zh-v1.5"
  dimensions: 512
  batch_size: 128

chunking:
  max_tokens: 512
  overlap: 50

retrieval:
  alpha: 0.7  # 语义权重
  beta: 0.3   # 关键词权重

vaults:
  - name: "personal"
    path: "~/NanobotMemory"
    enabled: true
```

## 文档

- [项目总览](docs/README.md) - 详细功能介绍
- [部署指南](docs/DEPLOYMENT.md) - 生产环境部署
- [快速参考](docs/QUICK_REFERENCE.md) - 命令速查
- [MCP 工具使用](docs/MCP_TOOLS_USAGE.md) - MCP 工具完整使用文档

## 许可证

MIT License

---

*版本: v1.1.5 | 最后更新: 2026-04-23*
