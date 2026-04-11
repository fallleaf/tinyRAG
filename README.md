# tinyRAG - 轻量级中文 RAG 系统

## 项目简介

tinyRAG 是一个轻量级的检索增强生成（RAG）系统，专为中文知识库设计。系统采用向量检索与全文检索（FTS5）双引擎混合架构，支持 Markdown 文档的智能分块、向量化索引和语义检索。

### 核心特性

- **混合检索引擎**：向量语义检索 + FTS5 关键词检索双引擎融合，支持 RRF（Reciprocal Rank Fusion）排序
- **智能 Markdown 分块**：按标题层级、代码块、表格等内容类型智能分割，保留语义完整性
- **动态置信度权重**：基于文档类型、状态、日期衰减等因素动态计算检索置信度
- **模型自动卸载**：嵌入模型空闲自动卸载，优化内存占用
- **增量索引**：两阶段扫描机制，支持文件新增/修改/移动/删除的增量更新
- **MCP 服务支持**：通过 Model Context Protocol 提供 AI 助手集成能力

## 系统架构

tinyRAG/
├── main.py              # 主程序入口（MCP 服务器）
├── config.py            # 配置管理与校验
├── build_index.py       # 全量索引构建
├── rag_cli.py           # 命令行工具
├── chunker/             # 文档分块模块
├── embedder/            # 向量化模块
├── retriever/           # 检索模块
├── scanner/             # 文件扫描模块
├── storage/             # 存储模块
├── utils/               # 工具模块
└── mcp_server/          # MCP 服务模块

## 快速开始

### 环境要求
- Python 3.10+
- SQLite 3.35+（支持 FTS5）
- 可选：sqlite-vec 扩展

### 安装依赖
pip install pydantic pyyaml loguru jieba fastembed mcp tqdm

### 构建索引
python build_index.py --force  # 全量构建
python build_index.py          # 增量更新

### 命令行检索
python rag_cli.py search "极简网络" --top-k 5
python rag_cli.py status

