# tinyRAG 快速参考卡

## 常用命令

```bash
# 激活环境
source .venv/bin/activate

# 构建索引
python build_index.py --force    # 强制重建
python build_index.py            # 增量更新

# 搜索
python rag_cli.py search "关键词"                    # 混合检索
python rag_cli.py search "关键词" --mode semantic    # 语义检索
python rag_cli.py search "关键词" --mode keyword     # 关键词检索
python rag_cli.py search "关键词" --top-k 20         # 指定数量

# 维护
python rag_cli.py status         # 系统状态
python rag_cli.py scan           # 增量扫描
python rag_cli.py config         # 查看配置

# 图谱功能需要
python -m spacy download zh_core_web_sm
```

## 配置要点 (config.yaml)

```yaml
embedding_model:
  name: "BAAI/bge-small-zh-v1.5"  # 中文优化模型
  dimensions: 512

retrieval:
  alpha: 0.7    # 语义权重
  beta: 0.3     # 关键词权重

vaults:
  - path: "/path/to/documents"
    enabled: true

plugins:
  enabled: true  # 启用图谱插件
```

## MCP 工具列表

| 工具 | 用途 |
|------|------|
| search | 混合检索 |
| stats | 系统统计 |
| config | 查看配置 |
| scan_index | 增量扫描 |
| rebuild_index | 重建索引 |
| maintenance | 数据库维护 |

## 故障排查

| 问题 | 解决 |
|------|------|
| 模型下载慢 | `export HTTP_PROXY=...` |
| 图谱不工作 | 检查 spacy 模型安装 |
| 检索不准 | 调整 alpha/beta 参数 |

---

*更新：2026-04-19*
