# tinyRAG - 快速参考卡

## 🚀 常用命令速查

### 环境管理
```bash
# 激活虚拟环境
source .venv/bin/activate

# 退出虚拟环境
deactivate

# 重新安装依赖
pip install -r requirements.txt --force-reinstall
```

### 初始化与构建
```bash
# 环境初始化
./init_env.py

# 首次构建索引（强制）
./build_index.py --force

# 增量更新索引
./build_index.py

# 扫描新文件
./rag_cli.py scan
```

### 检索操作
```bash
# 混合检索（默认）
./rag_cli.py search "关键词"

# 纯向量检索
./rag_cli.py search "关键词" --mode semantic

# 纯关键词检索
./rag_cli.py search "关键词" --mode keyword

# 指定 Vault 搜索
./rag_cli.py search "关键词" --vaults NanobotMemory

# 调整检索参数
./rag_cli.py search "关键词" --alpha 0.7 --beta 0.2 --top-k 20
```

### 系统维护
```bash
# 查看系统状态
./rag_cli.py status

# 重建索引
./rag_cli.py rebuild

# 清理数据库
./rag_cli.py vacuum

# 查看日志
tail -f logs/rag.log
```

### MCP 服务
```bash
# 启动服务（前台）
./mcp_server/server.py

# 启动服务（后台）
nohup ./mcp_server/server.py > logs/mcp.log 2>&1 &

# 停止服务
pkill -f "server.py"

# 查看服务状态
ps aux | grep server.py
```

---

## ⚙️ 配置关键参数

### config.yaml 核心配置
```yaml
# 数据库
db_path: "./data/rag.db"

# 模型
embedding_model:
  name: "sentence-transformers/all-MiniLM-L6-v2"

# 分块
chunking:
  max_tokens: 512
  overlap: 50

# 检索
retrieval:
  alpha: 0.6    # 向量权重
  beta: 0.2     # 关键词权重
  top_k: 10

# 知识库
vaults:
  - path: "~/NanobotMemory"
    enabled: true
```

---

## 🔧 故障排查速查

| 问题 | 命令 |
| :--- | :--- |
| 模型下载失败 | `export HTTP_PROXY=...` 或手动下载 |
| 数据库损坏 | `./rag_cli.py rebuild --force` |
| 内存不足 | 减小 `max_tokens` 和 `batch_size` |
| 检索不准 | 调整 `alpha`/`beta` 或改用 `--mode keyword` |
| 服务未启动 | `sudo systemctl status rag-system` |
| 查看日志 | `tail -f logs/rag.log` 或 `journalctl -u rag-system -f` |

---

## 📁 目录结构速览

```
tinyRAG/
├── data/           # 数据库和缓存
│   └── rag.db
├── models/         # 嵌入模型
├── logs/           # 日志文件
├── docs/           # 文档
├── build_index.py  # 索引构建
├── rag_cli.py      # CLI 工具
└── config.yaml     # 配置文件
```

---

## 🌐 环境变量

```bash
export RAG_CONFIG_PATH=/path/to/config.yaml
export RAG_LOG_LEVEL=DEBUG
export RAG_DB_PATH=/custom/path/rag.db
export HTTP_PROXY=http://proxy:port  # 模型下载代理
```

---

## 📞 快速帮助

```bash
# 查看帮助
./rag_cli.py --help
./build_index.py --help

# 查看版本
python3 -c "import sys; print(sys.version)"

# 检查依赖
pip list | grep -E "fastembed|pydantic|sqlite-vec"
```

---

*生成时间：2026-04-05*
