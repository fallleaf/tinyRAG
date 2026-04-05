# tinyRAG - 文档索引

## 📚 文档列表

| 文档 | 说明 | 适用场景 |
| :--- | :--- | :--- |
| [README.md](README.md) | **项目总览** | 首次接触项目，了解核心功能 |
| [DEPLOYMENT.md](DEPLOYMENT.md) | **部署与运维** | 生产环境部署、服务配置、故障排查 |
| [QUICK_REFERENCE.md](QUICK_REFERENCE.md) | **快速参考** | 日常使用命令速查 |

---

## 🎯 按场景查找文档

### 我是新用户
1. 阅读 [README.md](README.md) 了解项目功能
2. 执行 `快速开始` 章节的安装步骤
3. 保存 [QUICK_REFERENCE.md](QUICK_REFERENCE.md) 到桌面随时查阅

### 我要部署到生产环境
1. 阅读 [DEPLOYMENT.md](DEPLOYMENT.md) 的 `环境要求` 和 `安装步骤`
2. 配置 `config.yaml`（参考 `配置详解` 章节）
3. 选择部署方式：
   - **手动运行**: 参考 `方式一：手动运行`
   - **Systemd 服务**: 参考 `方式二：Systemd 服务`（推荐）
   - **Docker**: 参考 `方式三：Docker 部署`
4. 配置定时任务（参考 `定期维护` 章节）

### 我遇到了问题
1. 查看 [DEPLOYMENT.md](DEPLOYMENT.md) 的 `故障排查` 章节
2. 检查日志：`tail -f logs/rag.log`
3. 使用 `./rag_cli.py status` 检查系统状态
4. 如果仍未解决，提交 Issue 并附上日志

### 我要调整检索效果
1. 修改 `config.yaml` 中的 `retrieval` 配置
   - `alpha`: 向量检索权重（默认 0.6）
   - `beta`: 关键词检索权重（默认 0.2）
   - `top_k`: 返回结果数量（默认 10）
2. 或使用命令行参数临时调整：
   ```bash
   ./rag_cli.py search "关键词" --alpha 0.8 --beta 0.1
   ```

### 我要维护系统
1. **每日**: 查看日志 `tail -f logs/rag.log`
2. **每周**: 清理数据库 `./rag_cli.py vacuum`
3. **每月**: 重建索引 `./rag_cli.py rebuild --force`
4. **定期**: 备份数据（参考 `备份与恢复` 章节）

---

## 🔗 外部资源

- **RAG 技术原理**: [Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401)
- **SQLite FTS5**: [Full-Text Search in SQLite](https://www.sqlite.org/fts5.html)
- **fastembed**: [Qdrant/fastembed](https://github.com/qdrant/fastembed)
- **RRF 算法**: [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)

---

## 📝 文档更新日志

| 日期 | 版本 | 更新内容 | 作者 |
| :--- | :--- | :--- | :--- |
| 2026-04-05 | v1.0 | 初始版本，包含完整文档 | nanobot |

---

*文档最后更新：2026-04-05*
