# tinyRAG - 知识库检索增强系统

> **RAG (Retrieval-Augmented Generation)** 系统，基于本地知识库构建，支持混合检索（向量 + 关键词），专为电信行业知识管理设计。

---

## 📋 目录
- [核心功能](#核心功能)
- [系统架构](#系统架构)
- [快速开始](#快速开始)
- [使用方法](#使用方法)
- [部署指南](#部署指南)
- [配置说明](#配置说明)
- [常见问题](#常见问题)

---

## 🚀 核心功能

### 1. **智能文档解析**
- 支持 Markdown 格式的深度解析
- 自动提取 Frontmatter (YAML) 元数据
- 多级标题追踪 (H1-H6) 构建完整路径
- 识别代码块/表格/列表，防止误切分
- 滑动窗口 + Overlap 算法，保护语义完整性

### 2. **混合检索引擎**
- **向量检索**：基于 `fastembed` 本地嵌入模型
- **关键词检索**：基于 SQLite FTS5 全文索引
- **RRF 融合**：Reciprocal Rank Fusion 算法融合两种检索结果
- **置信度评分**：结合路径权重和类型权重进行排序

### 3. **知识库管理**
- 多 Vault 支持（类似 Obsidian 的多仓库管理）
- 增量扫描与全量重建索引
- 数据库自动清理与优化
- 支持动态添加/删除知识库文件

### 4. **MCP 协议支持**
- 提供标准 MCP (Model Context Protocol) 接口
- 支持 `search`, `scan`, `rebuild` 等工具调用
- 可与 AI 助手（如 nanobot）无缝集成

---

## 🏗️ 系统架构

```
tinyRAG/
├── build_index.py        # 索引构建主程序
├── rag_cli.py            # 命令行工具
├── init_env.py           # 环境初始化与自检
├── config.py             # 配置管理
├── vacuum.py             # 数据库清理工具
├── main.py               # 主入口（可选）
│
├── chunker/              # 文档分块模块
│   └── markdown_splitter.py
│
├── embedder/             # 向量化模块
│   ├── embed_engine.py
│   └── model_factory.py
│
├── retriever/            # 检索引擎
│   └── hybrid_engine.py
│
├── scanner/              # 文件扫描模块
│   └── scan_engine.py
│
├── storage/              # 数据存储
│   ├── database.py
│   └── cache.py
│
├── mcp_server/           # MCP 服务器
│   └── server.py
│
├── utils/                # 工具函数
│   └── logger.py
│
└── data/                 # 数据存储目录（运行时生成）
    ├── rag.db            # SQLite 数据库
    └── logs/             # 日志文件
```

---

## 🏁 快速开始

### 1. 环境准备
```bash
# 确保 Python 3.10+
python3 --version

# 创建虚拟环境（推荐）
python3 -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 初始化环境
```bash
# 自动创建必要目录并检查依赖
./init_env.py
# 或
python3 init_env.py
```

### 3. 配置知识库
编辑 `config.yaml`：
```yaml
db_path: "./data/rag.db"
embedding_model:
  name: "sentence-transformers/all-MiniLM-L6-v2"
  cache_dir: "./models"
vaults:
  - path: "~/NanobotMemory"
    enabled: true
  - path: "~/project/docs"
    enabled: true
```

### 4. 构建索引
```bash
# 首次构建（强制重建）
./build_index.py --force

# 增量更新（默认）
./build_index.py
```

### 5. 检索测试
```bash
# 使用 CLI 检索
./rag_cli.py search "极简网络 提级转段" --mode hybrid

# 或使用 MCP 接口（需启动服务器）
./mcp_server/server.py
```

---

## 📖 使用方法

### 命令行工具 (rag_cli.py)

```bash
# 查看系统状态
./rag_cli.py status

# 搜索知识
./rag_cli.py search "关键词" --mode hybrid --top-k 10

# 指定 Vault 搜索
./rag_cli.py search "网络优化" --vaults NanobotMemory

# 增量扫描（更新索引）
./rag_cli.py scan

# 重建索引
./rag_cli.py rebuild

# 清理数据库
./rag_cli.py vacuum
```

### 参数说明
| 参数 | 说明 | 默认值 |
| :--- | :--- | :--- |
| `--mode` | 检索模式：`semantic`/`keyword`/`hybrid` | `hybrid` |
| `--top-k` | 返回结果数量 | `10` |
| `--vaults` | 指定搜索的 Vault 名称 | 全部 |
| `--alpha` | 向量检索权重 (0-1) | `0.6` |
| `--beta` | 关键词检索权重 (0-1) | `0.2` |

### MCP 接口调用
系统启动 MCP 服务器后，AI 助手可通过以下工具调用：
- `search`: 混合检索
- `scan_index`: 增量扫描
- `rebuild_index`: 重建索引

---

## 🛠️ 部署指南

### 1. 生产环境部署
```bash
# 1. 克隆代码
git clone <repo-url> tinyRAG
cd tinyRAG

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置环境变量（可选）
export RAG_CONFIG_PATH=/path/to/config.yaml

# 4. 初始化
./init_env.py

# 5. 构建索引
./build_index.py --force

# 6. 启动 MCP 服务（后台运行）
nohup ./mcp_server/server.py > logs/mcp.log 2>&1 &
```

### 2. Docker 部署（可选）
```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN chmod +x *.py

CMD ["./mcp_server/server.py"]
```

### 3. 系统服务配置 (systemd)
创建 `/etc/systemd/system/rag-system.service`:
```ini
[Unit]
Description=tinyRAG Service
After=network.target

[Service]
Type=simple
User=fallleaf
WorkingDirectory=/home/fallleaf/tinyRAG
ExecStart=/home/fallleaf/tinyRAG/.venv/bin/python ./mcp_server/server.py
Restart=always

[Install]
WantedBy=multi-user.target
```

启动服务：
```bash
sudo systemctl daemon-reload
sudo systemctl enable rag-system
sudo systemctl start rag-system
```

---

## ⚙️ 配置说明

### config.yaml 详解
```yaml
# 数据库路径
db_path: "./data/rag.db"

# 嵌入模型配置
embedding_model:
  name: "sentence-transformers/all-MiniLM-L6-v2"  # 模型名称
  cache_dir: "./models"                            # 模型缓存目录

# 分块配置
chunking:
  max_tokens: 512      # 最大 token 数
  overlap: 50          # 重叠 token 数

# 检索配置
retrieval:
  alpha: 0.6           # 向量权重
  beta: 0.2            # 关键词权重
  top_k: 10            # 默认返回数量

# 知识库列表
vaults:
  - path: "~/NanobotMemory"
    enabled: true
    name: "personal"   # 可选：自定义名称
  - path: "~/project/docs"
    enabled: true
    name: "work"
```

### 环境变量
| 变量名 | 说明 | 默认值 |
| :--- | :--- | :--- |
| `RAG_CONFIG_PATH` | 配置文件路径 | `./config.yaml` |
| `RAG_LOG_LEVEL` | 日志级别 | `INFO` |
| `HTTP_PROXY` | 代理设置（模型下载） | 无 |

---

## ❓ 常见问题

### Q1: 模型下载失败？
**A**: 检查网络代理设置，或手动下载模型：
```bash
export HTTP_PROXY=http://your-proxy:port
# 或手动下载模型到 cache_dir
```

### Q2: 索引构建速度慢？
**A**: 
- 检查磁盘 I/O 性能
- 减少 `max_tokens` 或 `top_k` 参数
- 使用 `--force` 参数重建索引（首次较慢）

### Q3: 检索结果不准确？
**A**: 
- 调整 `alpha` 和 `beta` 权重
- 检查文档分块是否合理
- 尝试 `--mode keyword` 纯关键词检索

### Q4: 数据库损坏？
**A**: 
```bash
# 备份后重建
cp data/rag.db data/rag.db.bak
./rag_cli.py rebuild --force
```

---

## 📄 许可证
MIT License

## 🤝 贡献
欢迎提交 Issue 和 Pull Request！

---
*最后更新：2026-04-05*
