# RAG System New - 部署与运维指南

## 📋 目录
- [环境要求](#环境要求)
- [安装步骤](#安装步骤)
- [配置详解](#配置详解)
- [服务部署](#服务部署)
- [运维监控](#运维监控)
- [故障排查](#故障排查)
- [备份与恢复](#备份与恢复)

---

## 🖥️ 环境要求

### 硬件要求
| 组件 | 最低配置 | 推荐配置 |
| :--- | :--- | :--- |
| CPU | 2 核心 | 4 核心 + |
| 内存 | 4GB | 8GB+ |
| 磁盘 | 10GB SSD | 50GB NVMe SSD |
| 网络 | 宽带 | 千兆宽带 |

### 软件要求
- **操作系统**: Linux (Ubuntu 20.04+, CentOS 7+), macOS 12+, Windows 10+ (WSL2)
- **Python**: 3.10 - 3.12
- **依赖库**: `fastembed`, `pydantic`, `sqlite-vec`, `loguru`

---

## 📦 安装步骤

### 1. 基础环境安装

#### Ubuntu/Debian
```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装 Python 3.12
sudo apt install -y python3.12 python3.12-venv python3.12-dev

# 安装构建工具
sudo apt install -y build-essential git curl
```

#### CentOS/RHEL
```bash
sudo yum install -y python3.12 python3.12-devel git curl
```

#### macOS
```bash
brew install python@3.12 git
```

### 2. 项目安装

```bash
# 克隆项目
git clone <repository-url> rag_system_new
cd rag_system_new

# 创建虚拟环境
python3.12 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# 升级 pip
pip install --upgrade pip

# 安装依赖
pip install -r requirements.txt

# 验证安装
python3 -c "import fastembed; print('✅ fastembed installed')"
```

### 3. 环境初始化

```bash
# 运行初始化脚本
./init_env.py

# 输出示例：
# 🔍 RAG System 环境初始化...
# ✅ 目录就绪: ./data
# ✅ 目录就绪: ./models
# ✅ 目录就绪: ./logs
# ✅ 核心依赖已安装
# ✅ 配置加载成功 | DB: ./data/rag.db | Model: ./models
# ✅ 有效 Vault 数量: 2
# 📦 预热嵌入模型 (首次运行将下载约 100MB)...
# ✅ 模型缓存就绪
# 🎉 初始化完成！请执行: python build_index.py --force
```

---

## ⚙️ 配置详解

### config.yaml 完整配置

```yaml
# ==================== 数据库配置 ====================
db_path: "./data/rag.db"  # SQLite 数据库路径

# ==================== 嵌入模型配置 ====================
embedding_model:
  name: "sentence-transformers/all-MiniLM-L6-v2"  # 模型名称
  cache_dir: "./models"                            # 模型缓存目录
  # 可选模型列表:
  # - "sentence-transformers/all-MiniLM-L6-v2" (轻量，384维)
  # - "sentence-transformers/all-mpnet-base-v2" (中等，768维)
  # - "BAAI/bge-small-en-v1.5" (中文优化)

# ==================== 分块配置 ====================
chunking:
  max_tokens: 512      # 最大 token 数 (建议 256-1024)
  overlap: 50          # 重叠 token 数 (建议 30-100)
  max_chars: 2000      # 最大字符数 (保底切分)

# ==================== 检索配置 ====================
retrieval:
  alpha: 0.6           # 向量检索权重 (0-1)
  beta: 0.2            # 关键词检索权重 (0-1)
  top_k: 10            # 默认返回结果数量
  min_score: 0.3       # 最低置信度阈值

# ==================== 知识库配置 ====================
vaults:
  - path: "~/NanobotMemory"
    enabled: true
    name: "personal"   # 自定义名称（可选）
    priority: 1        # 优先级（数字越小优先级越高）
  
  - path: "~/project/docs"
    enabled: true
    name: "work"
    priority: 2

  - path: "~/downloads/tech_papers"
    enabled: false     # 禁用该知识库
    name: "papers"

# ==================== 日志配置 ====================
logging:
  level: "INFO"        # DEBUG, INFO, WARNING, ERROR, CRITICAL
  file: "./logs/rag.log"
  max_size: "10MB"     # 日志文件最大大小
  backup_count: 5      # 保留的备份文件数量

# ==================== 性能配置 ====================
performance:
  batch_size: 32       # 批量处理大小
  max_workers: 4       # 最大工作线程数
  cache_enabled: true  # 启用缓存
```

### 环境变量覆盖

```bash
# 覆盖配置文件中的设置
export RAG_DB_PATH="/custom/path/rag.db"
export RAG_EMBEDDING_MODEL="BAAI/bge-small-zh-v1.5"
export RAG_LOG_LEVEL="DEBUG"
export RAG_VAULTS="~/custom/vault1,~/custom/vault2"
```

---

## 🚀 服务部署

### 方式一：手动运行（开发/测试）

```bash
# 启动 MCP 服务器
./mcp_server/server.py

# 后台运行
nohup ./mcp_server/server.py > logs/mcp.log 2>&1 &

# 查看日志
tail -f logs/mcp.log
```

### 方式二：Systemd 服务（生产环境）

#### 1. 创建服务文件

```bash
sudo nano /etc/systemd/system/rag-system.service
```

内容：
```ini
[Unit]
Description=RAG System Knowledge Base Service
Documentation=https://github.com/your-org/rag_system_new
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=fallleaf
Group=fallleaf
WorkingDirectory=/home/fallleaf/rag_system_new
Environment="PATH=/home/fallleaf/rag_system_new/.venv/bin"
Environment="RAG_CONFIG_PATH=/home/fallleaf/rag_system_new/config.yaml"
ExecStart=/home/fallleaf/rag_system_new/.venv/bin/python ./mcp_server/server.py
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=rag-system

# 资源限制
MemoryMax=2G
CPUQuota=80%

# 安全增强
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/home/fallleaf/rag_system_new/data
ReadWritePaths=/home/fallleaf/rag_system_new/logs

[Install]
WantedBy=multi-user.target
```

#### 2. 启动服务

```bash
# 重载配置
sudo systemctl daemon-reload

# 启用服务（开机自启）
sudo systemctl enable rag-system

# 启动服务
sudo systemctl start rag-system

# 查看状态
sudo systemctl status rag-system

# 查看日志
sudo journalctl -u rag-system -f
```

#### 3. 常用命令

```bash
# 停止服务
sudo systemctl stop rag-system

# 重启服务
sudo systemctl restart rag-system

# 查看服务日志（最近 100 行）
sudo journalctl -u rag-system -n 100

# 查看服务日志（今天）
sudo journalctl -u rag-system --since today
```

### 方式三：Docker 部署

#### 1. 构建镜像

```dockerfile
# Dockerfile
FROM python:3.12-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . .

# 设置权限
RUN chmod +x *.py && \
    mkdir -p data models logs

# 暴露端口（如果需要）
EXPOSE 8080

# 启动命令
CMD ["./mcp_server/server.py"]
```

#### 2. 构建并运行

```bash
# 构建镜像
docker build -t rag-system:latest .

# 运行容器
docker run -d \
  --name rag-system \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  -v ~/NanobotMemory:/app/vaults/nanobot:ro \
  -e RAG_CONFIG_PATH=/app/config.yaml \
  -p 8080:8080 \
  rag-system:latest

# 查看日志
docker logs -f rag-system
```

---

## 📊 运维监控

### 1. 健康检查

```bash
# 检查服务状态
curl -s http://localhost:8080/health

# 使用 CLI 检查
./rag_cli.py status
```

### 2. 日志分析

```bash
# 实时查看日志
tail -f logs/rag.log

# 搜索错误日志
grep "ERROR" logs/rag.log

# 统计错误数量
grep -c "ERROR" logs/rag.log

# 查看最近 100 条日志
tail -n 100 logs/rag.log
```

### 3. 性能监控

```bash
# 监控数据库大小
du -sh data/

# 监控内存使用
ps aux | grep rag_system

# 监控索引构建进度
tail -f logs/build_index.log
```

### 4. 定期维护

创建定时任务（crontab）：

```bash
# 编辑 crontab
crontab -e

# 每天凌晨 2 点清理数据库
0 2 * * * cd /home/fallleaf/rag_system_new && ./.venv/bin/python vacuum.py

# 每周日凌晨 3 点重建索引
0 3 * * 0 cd /home/fallleaf/rag_system_new && ./.venv/bin/python build_index.py --force

# 每天清理旧日志
0 3 * * * find /home/fallleaf/rag_system_new/logs -name "*.log.*" -mtime +30 -delete
```

---

## 🔧 故障排查

### 问题 1: 模型下载失败

**症状**: `❌ 模型下载失败: HTTPSConnectionPool...`

**解决方案**:
```bash
# 设置代理
export HTTP_PROXY=http://your-proxy:port
export HTTPS_PROXY=http://your-proxy:port

# 或手动下载模型
mkdir -p models
# 从 HuggingFace 手动下载模型文件到 models/
```

### 问题 2: 数据库损坏

**症状**: `database disk image is malformed`

**解决方案**:
```bash
# 备份数据库
cp data/rag.db data/rag.db.backup

# 重建索引
./rag_cli.py rebuild --force

# 如果仍然失败，删除数据库重建
rm data/rag.db*
./build_index.py --force
```

### 问题 3: 内存不足

**症状**: `MemoryError` 或 `Killed`

**解决方案**:
```yaml
# 调整 config.yaml
chunking:
  max_tokens: 256  # 减小分块大小

performance:
  batch_size: 16   # 减小批量大小
  max_workers: 2   # 减少线程数
```

### 问题 4: 检索结果不准确

**解决方案**:
```bash
# 调整检索参数
./rag_cli.py search "关键词" --alpha 0.8 --beta 0.1 --top-k 20

# 或修改 config.yaml
retrieval:
  alpha: 0.8
  beta: 0.1
  top_k: 20
```

---

## 💾 备份与恢复

### 备份策略

```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/backup/rag_system"
DATE=$(date +%Y%m%d_%H%M%S)
PROJECT_DIR="/home/fallleaf/rag_system_new"

# 创建备份目录
mkdir -p $BACKUP_DIR

# 备份数据库
cp $PROJECT_DIR/data/rag.db $BACKUP_DIR/rag.db.$DATE

# 备份配置文件
cp $PROJECT_DIR/config.yaml $BACKUP_DIR/config.$DATE

# 备份日志（可选）
tar -czf $BACKUP_DIR/logs.$DATE.tar.gz -C $PROJECT_DIR logs/

# 清理旧备份（保留 7 天）
find $BACKUP_DIR -name "rag.db.*" -mtime +7 -delete
find $BACKUP_DIR -name "config.*" -mtime +7 -delete

echo "✅ 备份完成：$BACKUP_DIR"
```

### 恢复步骤

```bash
# 1. 停止服务
sudo systemctl stop rag-system

# 2. 恢复数据库
cp /backup/rag_system/rag.db.20260405_120000 ./data/rag.db

# 3. 恢复配置
cp /backup/rag_system/config.20260405_120000 ./config.yaml

# 4. 启动服务
sudo systemctl start rag-system

# 5. 验证
./rag_cli.py status
```

---

## 📞 技术支持

- **Issue 反馈**: [GitHub Issues](https://github.com/your-org/rag_system_new/issues)
- **文档**: [Wiki](https://github.com/your-org/rag_system_new/wiki)
- **社区**: [Discord](https://discord.gg/your-invite)

---

*最后更新：2026-04-05*
