# RAG System New

> 本地知识库检索增强系统，支持混合检索（向量 + 关键词），专为电信行业知识管理设计。

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🚀 快速开始

```bash
# 克隆项目
git clone <repo-url> rag_system_new
cd rag_system_new

# 安装依赖
pip install -r requirements.txt

# 初始化环境
./init_env.py

# 构建索引
./build_index.py --force

# 检索测试
./rag_cli.py search "关键词"
```

## ✨ 核心功能

- **智能文档解析**: 支持 Markdown 深度解析，自动提取元数据
- **混合检索引擎**: 向量检索 + 关键词检索，RRF 融合算法
- **多知识库管理**: 支持多 Vault 配置，类似 Obsidian
- **MCP 协议支持**: 可与 AI 助手无缝集成
- **本地化部署**: 数据完全本地存储，隐私安全

## 📚 详细文档

- [📖 项目总览](docs/README.md) - 功能介绍、架构、使用方法
- [🛠️ 部署指南](docs/DEPLOYMENT.md) - 安装、配置、运维、故障排查
- [⚡ 快速参考](docs/QUICK_REFERENCE.md) - 命令速查表
- [📑 文档索引](docs/SUMMARY.md) - 导航所有文档

## 🏗️ 系统架构

```
rag_system_new/
├── build_index.py        # 索引构建
├── rag_cli.py            # CLI 工具
├── init_env.py           # 环境初始化
├── config.yaml           # 配置文件
├── docs/                 # 文档
├── chunker/              # 文档分块
├── embedder/             # 向量化
├── retriever/            # 检索引擎
├── scanner/              # 文件扫描
├── storage/              # 数据存储
├── mcp_server/           # MCP 服务
└── utils/                # 工具函数
```

## 🔧 技术栈

- **Python 3.10+**: 核心语言
- **fastembed**: 轻量级嵌入模型
- **SQLite + sqlite-vec**: 本地向量数据库
- **Pydantic v2**: 配置校验
- **loguru**: 日志系统
- **MCP SDK**: 模型上下文协议

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE)

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---
*最后更新：2026-04-05*
