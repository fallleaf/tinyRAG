# tinyRAG 部署指南

## 环境要求

- Python 3.10+
- 内存 4GB+（推荐 8GB）
- 磁盘 10GB+

## 安装步骤

```bash
# 1. 克隆项目
git clone <repo-url> tinyRAG && cd tinyRAG

# 2. 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 3. 安装依赖
pip install --upgrade pip
pip install -r requirements.txt

# 4. 安装中文 NLP 模型（图谱功能）
python -m spacy download zh_core_web_sm

# 5. 配置知识库路径
# 编辑 config.yaml 中的 vaults.path

# 6. 构建索引
python build_index.py --force
```

## 生产部署

### Systemd 服务

```bash
# 创建服务文件
sudo nano /etc/systemd/system/tinyrag.service
```

```ini
[Unit]
Description=tinyRAG Service
After=network.target

[Service]
Type=simple
User=youruser
WorkingDirectory=/home/youruser/tinyRAG
ExecStart=/home/youruser/tinyRAG/.venv/bin/python mcp_server/server.py
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
# 启动服务
sudo systemctl daemon-reload
sudo systemctl enable tinyrag
sudo systemctl start tinyrag

# 查看日志
sudo journalctl -u tinyrag -f
```

### Docker 部署

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download zh_core_web_sm
COPY . .
CMD ["python", "mcp_server/server.py"]
```

```bash
docker build -t tinyrag .
docker run -d -v $(pwd)/data:/app/data -v $(pwd)/documents:/app/documents tinyrag
```

## 定期维护

```bash
# 添加到 crontab
crontab -e

# 每天凌晨增量扫描
0 2 * * * cd /home/user/tinyRAG && .venv/bin/python rag_cli.py scan

# 每周重建索引
0 3 * * 0 cd /home/user/tinyRAG && .venv/bin/python build_index.py --force
```

## 备份

```bash
# 备份数据库
cp data/rag.db data/rag.db.backup

# 恢复
cp data/rag.db.backup data/rag.db
```

---

*更新：2026-04-19*
