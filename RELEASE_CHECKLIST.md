# tinyRAG - 发布准备清单

## ✅ 已完成项

- [x] **代码质量检查**: Ruff 检查通过（已修复所有 RUF015, F401, RUF013, W291, SIM105, E722 等问题）
- [x] **Shebang 统一**: 所有可执行脚本已添加 `#!/usr/bin/env python3`
- [x] **可执行权限**: 所有脚本已设置 `chmod +x`
- [x] **文档完善**: 已创建完整的 `docs/` 目录（README, DEPLOYMENT, QUICK_REFERENCE, SUMMARY）
- [x] **依赖管理**: `requirements.txt` 已更新
- [x] **配置示例**: `config.yaml` 已存在
- [x] **环境初始化**: `init_env.py` 可自动检查环境和依赖

## ⚠️ 待办项（发布前必须完成）

### 1. 版本控制 (Git)
- [ ] **初始化 Git 仓库**: `git init`
- [ ] **创建 .gitignore**: 排除虚拟环境、数据库、日志等
- [ ] **首次提交**: `git add . && git commit -m "Initial release v0.1.0"`
- [ ] **创建 Git 标签**: `git tag v0.1.0`
- [ ] **推送到远程**: `git push origin main --tags`

### 2. 标准文件
- [ ] **LICENSE**: 添加开源许可证（推荐 MIT）
- [ ] **CHANGELOG.md**: 记录版本变更历史
- [ ] **根目录 README.md**: 项目入口文档（指向 docs/README.md）
- [ ] **CONTRIBUTING.md**: 贡献指南（可选）

### 3. 测试
- [ ] **单元测试**: 添加 `tests/` 目录和基础测试
- [ ] **集成测试**: 验证完整流程（初始化 -> 构建 -> 检索）
- [ ] **CI/CD**: 配置 GitHub Actions 自动测试

### 4. 打包与分发
- [ ] **setup.py / pyproject.toml**: 配置 Python 包元数据
- [ ] **Dockerfile**: 容器化部署支持
- [ ] **release 包**: 创建 `.tar.gz` 或 `.zip` 发布包

### 5. 安全与隐私
- [ ] **敏感信息**: 确保 `config.yaml` 中无硬编码密码/密钥
- [ ] **安全扫描**: 运行 `pip-audit` 或 `safety check`
- [ ] **数据隐私**: 明确说明数据处理方式（本地/云端）

---

## 📋 快速执行脚本

```bash
#!/bin/bash
# release_prep.sh - 发布准备自动化脚本

echo "🚀 开始发布准备..."

# 1. 创建 .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
.venv/
venv/
ENV/

# 数据与日志
data/
logs/
*.db
*.db-wal
*.db-shm

# 模型缓存
models/

# IDE
.idea/
.vscode/
*.swp
*.swo

# 系统文件
.DS_Store
Thumbs.db

# Ruff
.ruff_cache/

# 配置文件（可选：如果包含敏感信息）
# config.yaml
EOF
echo "✅ 已创建 .gitignore"

# 2. 创建 LICENSE (MIT)
cat > LICENSE << 'EOF'
MIT License

Copyright (c) 2026 fallleaf

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
EOF
echo "✅ 已创建 LICENSE (MIT)"

# 3. 创建根目录 README.md
cat > README.md << 'EOF'
# tinyRAG

> 本地知识库检索增强系统，支持混合检索（向量 + 关键词），专为电信行业知识管理设计。

## 🚀 快速开始

```bash
# 克隆项目
git clone <repo-url> tinyRAG
cd tinyRAG

# 安装依赖
pip install -r requirements.txt

# 初始化环境
./init_env.py

# 构建索引
./build_index.py --force

# 检索测试
./rag_cli.py search "关键词"
```

## 📚 详细文档

- [项目总览](docs/README.md)
- [部署指南](docs/DEPLOYMENT.md)
- [快速参考](docs/QUICK_REFERENCE.md)

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE)
EOF
echo "✅ 已创建根目录 README.md"

# 4. 创建 CHANGELOG.md
cat > CHANGELOG.md << 'EOF'
# 变更日志

## [0.1.0] - 2026-04-05

### 新增
- 核心功能：文档解析、向量化、混合检索
- CLI 工具：`rag_cli.py` 支持搜索、扫描、重建
- MCP 服务：支持 AI 助手集成
- 文档：完整的部署和使用文档

### 修复
- 修复 Ruff 检查问题（全角标点、可变默认值等）
- 统一 Shebang 和可执行权限
- 优化数据库初始化逻辑

### 改进
- 改进日志系统（loguru）
- 优化检索算法（RRF 融合）
- 增强配置管理（Pydantic v2）
EOF
echo "✅ 已创建 CHANGELOG.md"

# 5. 初始化 Git
if [ ! -d ".git" ]; then
    git init
    git add .
    git commit -m "Initial release v0.1.0"
    git tag v0.1.0
    echo "✅ Git 仓库已初始化，首次提交完成"
else
    echo "⏭️  Git 仓库已存在，跳过初始化"
fi

echo ""
echo "🎉 发布准备完成！"
echo ""
echo "下一步："
echo "1. 推送到远程仓库: git push origin main --tags"
echo "2. 创建 GitHub Release"
echo "3. 配置 CI/CD (可选)"
```

---

## 📊 当前项目状态

| 类别 | 状态 | 说明 |
| :--- | :--- | :--- |
| **代码质量** | ✅ 优秀 | Ruff 检查全通过 |
| **文档** | ✅ 完善 | 4 个核心文档已就绪 |
| **依赖** | ✅ 完整 | `requirements.txt` 已更新 |
| **Git** | ⚠️ 缺失 | 需初始化仓库和 `.gitignore` |
| **许可证** | ⚠️ 缺失 | 需添加 `LICENSE` |
| **测试** | ⚠️ 缺失 | 需添加单元测试 |
| **打包** | ⚠️ 缺失 | 需配置 `pyproject.toml` |

---

## 🚀 推荐发布流程

1. **执行 `release_prep.sh`**: 自动创建缺失的标准文件
2. **手动检查**: 确认 `config.yaml` 无敏感信息
3. **运行测试**: `./rag_cli.py status` 验证系统正常
4. **提交代码**: `git add . && git commit -m "Release v0.1.0"`
5. **打标签**: `git tag v0.1.0 && git push origin main --tags`
6. **创建 Release**: 在 GitHub/Gitee 创建 Release 并上传源码包
7. **通知用户**: 发布更新公告

---

*生成时间：2026-04-05 22:21*
