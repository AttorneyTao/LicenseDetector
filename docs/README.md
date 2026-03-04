# 📚 文档目录

所有项目文档都在这里。根据你的需求选择阅读：

## 🚀 快速开始

- **[QUICKSTART.md](QUICKSTART.md)** - 5分钟快速入门指南
- **[DOCKER_QUICK_REF.md](DOCKER_QUICK_REF.md)** - Docker快速参考

## 📖 完整文档

### API相关
- **[API_USAGE.md](API_USAGE.md)** - API详细使用指南和示例

### Docker/容器化
- **[DOCKER_GUIDE.md](DOCKER_GUIDE.md)** - Docker完整使用指南
- **[DOCKER.md](DOCKER.md)** - Docker超简洁快速开始（推荐首先阅读）
- **[CONTAINERIZATION_REPORT.md](CONTAINERIZATION_REPORT.md)** - 容器化实现细节

### 项目信息
- **[CHANGES.md](CHANGES.md)** - 项目改动说明
- **[IMPLEMENTATION_REPORT.md](IMPLEMENTATION_REPORT.md)** - API实现报告
- **[DELIVERY_CHECKLIST.md](DELIVERY_CHECKLIST.md)** - 项目交付清单
- **[DOCKER_COMPLETE.md](DOCKER_COMPLETE.md)** - 容器化完成总结

## 📋 按使用场景选择

### 场景1：我是新用户，想快速上手
```
1. 阅读: QUICKSTART.md
2. 阅读: DOCKER.md (如果用容器)
3. 开始使用!
```

### 场景2：我想了解API功能
```
1. 阅读: API_USAGE.md
2. 查看: examples/ 目录下的示例
```

### 场景3：我想部署到Docker/容器
```
1. 阅读: DOCKER.md (3分钟快速了解)
2. 参考: DOCKER_GUIDE.md (详细说明)
```

### 场景4：我是项目维护者
```
1. 阅读: CHANGES.md (了解改动)
2. 阅读: IMPLEMENTATION_REPORT.md (实现细节)
3. 参考: DELIVERY_CHECKLIST.md (交付清单)
```

## 🔍 文档速查表

| 问题 | 查看文件 |
|------|---------|
| 如何快速开始? | QUICKSTART.md 或 DOCKER.md |
| 如何使用API? | API_USAGE.md |
| Docker怎么用? | DOCKER_GUIDE.md 或 DOCKER_QUICK_REF.md |
| 项目有什么改变? | CHANGES.md |
| 如何部署? | DOCKER_GUIDE.md 的部署章节 |
| 常见问题? | 各文档的"常见问题"或"故障排除"章节 |

## 📌 关键要点

### CLI模式 (本地处理)
```bash
uv run main.py
```

### API模式 (HTTP服务)
```bash
python main.py --api
```

### Docker模式
```bash
docker-compose --profile cli run analyzer-cli    # CLI
docker-compose --profile api up -d                # API
```

## 🎯 超快速参考

> **3行快速开始**
> ```bash
> docker build -t analyzer .
> docker-compose --profile cli run analyzer-cli
> # 结果在 outputs/ 目录
> ```

## 💬 需要帮助?

1. 查看对应文档的"常见问题"部分
2. 查看对应文档的"故障排除"部分  
3. 检查项目日志: `logs/` 目录

---

**提示**: 所有文档都在 `docs/` 目录下。建议按上面的场景选择合适的文档阅读。
