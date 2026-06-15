# 📚 文档目录

所有项目文档都在这里。根据你的需求选择阅读：

## 📖 文档列表

| 文档 | 内容 |
|------|------|
| **[QUICKSTART.md](QUICKSTART.md)** | 快速入门：CLI / API 模式、邮箱配置、故障排除 |
| **[FONT_SCANNING.md](FONT_SCANNING.md)** | 🔤 字体扫描模式（`--font`）：按站点获取字体授权与版权 |
| **[API_USAGE.md](API_USAGE.md)** | API 详细使用指南和示例 |
| **[DOCKER.md](DOCKER.md)** | 🐳 Docker 容器化使用与部署（CLI / API / 生产） |
| **[CRATE_IO_INTEGRATION.md](CRATE_IO_INTEGRATION.md)** | Rust crates.io 包分析集成说明 |
| **[CHANGES.md](CHANGES.md)** | 项目改动说明 / 变更记录 |

## 🧭 按场景选择

| 我想做什么 | 查看 |
|----------|------|
| 快速上手项目 | [QUICKSTART.md](QUICKSTART.md) |
| 扫描字体授权（input.xlsx 全为字体） | [FONT_SCANNING.md](FONT_SCANNING.md) |
| 使用 / 集成 HTTP API | [API_USAGE.md](API_USAGE.md) |
| 用 Docker 运行或部署 | [DOCKER.md](DOCKER.md) |
| 分析 Rust crate | [CRATE_IO_INTEGRATION.md](CRATE_IO_INTEGRATION.md) |
| 了解项目改动 | [CHANGES.md](CHANGES.md) |

## ⚡ 关键命令速查

```bash
# CLI 模式（软件包，处理 input.xlsx）
python main.py

# 字体扫描模式（input.xlsx 全为字体）
python main.py --font

# API / Web 模式
python main.py --api            # 访问 http://localhost:8000/docs

# Docker
docker-compose --profile cli run analyzer-cli    # CLI
docker-compose --profile api up -d               # API
```

---

**提示**：更顶层的项目介绍见仓库根目录的 [`README.md`](../README.md)。
