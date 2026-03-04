# 🚀 Docker 快速开始

> **最简单的方式** - 只需3步启动项目

## 三种运行模式

### CLI 模式 - 处理本地文件
```bash
docker-compose --profile cli run analyzer-cli
```
✅ 处理 `input.xlsx` → 输出到 `outputs/` → 自动退出

### API 模式 - HTTP 服务
```bash
docker-compose --profile api up -d
# 访问: http://localhost:8000/docs
```
✅ 长期运行 → 支持并发 → 支持邮件发送

### 生产模式 - 完整部署  
```bash
docker-compose -f docker-compose.prod.yml up -d
# 访问: http://localhost
```
✅ Nginx代理 + 健康检查 + 持久化

---

## 首次使用

### 1️⃣ 构建镜像（第一次需要5-10分钟）
```bash
docker build -t github-analyzer .
```

### 2️⃣ 运行（选择上面三种方式之一）

### 3️⃣ 查看结果
- CLI: `outputs/` 目录
- API: http://localhost:8000/docs  
- 生产: http://localhost

---

## 常见操作

```bash
# 查看日志
docker-compose logs -f

# 停止容器
docker-compose down

# 进入容器
docker-compose exec analyzer-api bash

# 清理资源
docker system prune -a -f
```

---

## 快捷工具

### Linux/Mac (Makefile)
```bash
make docker-cli
make docker-api-bg
make docker-logs
make docker-stop
```

### Windows (PowerShell)
```bash
.\docker.ps1 cli
.\docker.ps1 api
.\docker.ps1 stop
```

---

## 部署到其他服务器

### 方式A：复制整个项目
```bash
scp -r . user@server:/app/analyzer
docker-compose -f docker-compose.prod.yml up -d
```

### 方式B：推送到Docker Hub
```bash
docker build -t yourusername/analyzer:latest .
docker push yourusername/analyzer:latest
# 任何地方拉取: docker pull yourusername/analyzer:latest
```

---

## 需要详细说明?

查看完整文档: [`docs/DOCKER_GUIDE.md`](DOCKER_GUIDE.md)

---

**🎉 就这么简单！** 
