# 🐳 Docker 容器化使用指南

## 概述

本项目已完全容器化，支持以下部署方式：

1. **CLI模式** - 处理本地文件
2. **API模式** - HTTP服务
3. **生产模式** - 包含反向代理

---

## 快速开始

### 前置条件

- Docker 安装完成
- Docker Compose（推荐）
- `.env` 文件已配置（已有）

### 方式1️⃣：使用Docker Compose（推荐）

#### CLI 模式 - 处理本地文件

```bash
# 方式A：使用docker-compose CLI profile
docker-compose --profile cli run analyzer-cli

# 方式B：直接运行（不使用profile）
docker-compose run --rm \
  -v ./input.xlsx:/app/input.xlsx:ro \
  -v ./outputs:/app/outputs \
  -v ./logs:/app/logs \
  -v ./.env:/app/.env:ro \
  analyzer-cli
```

**工作流：**
1. `input.xlsx` 在当前目录
2. 运行上述命令
3. 结果保存到 `outputs/` 目录

#### API 模式 - HTTP服务

```bash
# 启动API服务（默认8000端口）
docker-compose --profile api up

# 后台运行
docker-compose --profile api up -d

# 查看日志
docker-compose logs -f analyzer-api

# 停止服务
docker-compose --profile api down
```

**使用API：**
```bash
# 访问API文档
http://localhost:8000/docs

# 健康检查
curl http://localhost:8000/health

# 上传文件并分析
curl -X POST "http://localhost:8000/api/v1/analyze-and-download" \
  -F "file=@input.xlsx" \
  --output result.xlsx

# 上传文件并邮件
curl -X POST "http://localhost:8000/api/v1/analyze" \
  -F "file=@input.xlsx" \
  -F "email=user@example.com"
```

### 方式2️⃣：使用原生Docker命令

#### 构建镜像

```bash
# 构建镜像
docker build -t github-license-analyzer:latest .

# 或指定标签版本
docker build -t github-license-analyzer:v0.1.0 .
```

#### 运行CLI模式

```bash
docker run --rm \
  -v $(pwd)/input.xlsx:/app/input.xlsx:ro \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/.env:/app/.env:ro \
  github-license-analyzer:latest
```

#### 运行API模式

```bash
# 前台运行
docker run -it --rm \
  -p 8000:8000 \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/.env:/app/.env:ro \
  github-license-analyzer:latest \
  --api --host 0.0.0.0

# 后台运行
docker run -d \
  --name analyzer-api \
  -p 8000:8000 \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/.env:/app/.env:ro \
  github-license-analyzer:latest \
  --api --host 0.0.0.0
```

---

## 生产部署

### 方式：Docker Compose + Nginx

生产环境配置包含反向代理和健康检查。

```bash
# 启动生产环境
docker-compose -f docker-compose.prod.yml up -d

# 查看日志
docker-compose -f docker-compose.prod.yml logs -f

# 停止服务
docker-compose -f docker-compose.prod.yml down
```

**配置说明：**
- API 服务在 `http://localhost:80` （通过Nginx代理）
- 自动健康检查
- 日志、输出、临时文件持久化
- 支持HTTPS（需要配置证书）

### 配置HTTPS（可选）

1. 获取SSL证书（例如使用Let's Encrypt）
   ```bash
   mkdir -p ssl
   # 将cert.pem和key.pem放入ssl目录
   ```

2. 编辑 `nginx.conf`，取消注释HTTPS部分

3. 修改服务器名称
   ```bash
   sed -i 's/your-domain.com/your.domain.com/g' nginx.conf
   ```

4. 重启服务
   ```bash
   docker-compose -f docker-compose.prod.yml restart nginx
   ```

---

## 文件说明

| 文件 | 用途 |
|------|------|
| `Dockerfile` | 容器镜像定义 |
| `docker-compose.yml` | 开发/测试配置 |
| `docker-compose.prod.yml` | 生产环境配置 |
| `nginx.conf` | Nginx反向代理配置 |
| `.dockerignore` | Docker构建排除文件 |

---

## 常见操作

### 查看容器日志

```bash
# 查看最近日志
docker-compose logs analyzer-api

# 实时跟踪日志
docker-compose logs -f analyzer-api

# 查看指定行数
docker-compose logs --tail=100 analyzer-api
```

### 进入容器调试

```bash
# 进入运行中的容器
docker-compose exec analyzer-api /bin/bash

# 或直接运行命令
docker-compose exec analyzer-api python verify_installation.py
```

### 清理资源

```bash
# 停止所有容器
docker-compose down

# 删除镜像
docker rmi github-license-analyzer:latest

# 清理所有未使用的资源
docker system prune -a
```

### 查看容器信息

```bash
# 查看运行中的容器
docker-compose ps

# 查看镜像
docker images | grep analyzer

# 查看volume
docker volume list | grep analyzer
```

---

## 环境变量配置

### 从.env文件读取

容器会自动读取宿主机的 `.env` 文件中的变量：

```env
GITHUB_TOKEN=...
GEMINI_API_KEY=...
DASHSCOPE_API_KEY=...
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SENDER_EMAIL=...
SENDER_PASSWORD=...
```

### 在命令行传入

```bash
docker-compose run --rm \
  -e GITHUB_TOKEN=xxx \
  -e SENDER_EMAIL=user@gmail.com \
  -e SENDER_PASSWORD=password \
  analyzer-cli
```

---

## 性能和限制

### 资源限制

可以在 `docker-compose.yml` 中添加资源限制：

```yaml
services:
  analyzer-api:
    # ... 其他配置
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
        reservations:
          cpus: '1'
          memory: 1G
```

### 存储管理

- `outputs/` - 输出文件
- `logs/` - 日志文件
- `temp/` - 临时文件

建议定期清理 `temp/` 目录：

```bash
docker-compose exec analyzer-api rm -rf /app/temp/*
```

---

## 故障排除

### 问题1：容器启动失败

```bash
# 查看错误日志
docker-compose logs analyzer-api

# 检查.env文件是否存在
ls -la .env

# 检查input.xlsx（CLI模式）
ls -la input.xlsx
```

### 问题2：API无法连接

```bash
# 检查容器是否运行
docker-compose ps

# 检查端口映射
docker-compose port analyzer-api 8000

# 查看网络
docker network list
```

### 问题3：邮件发送失败

```bash
# 检查环境变量
docker-compose exec analyzer-api env | grep SMTP

# 查看日志中的邮件错误
docker-compose logs analyzer-api | grep -i email
```

---

## Docker Hub 发布（可选）

### 构建和推送镜像

```bash
# 登录Docker Hub
docker login

# 构建镜像
docker build -t yourusername/github-license-analyzer:latest .

# 推送镜像
docker push yourusername/github-license-analyzer:latest

# 其他机器拉取使用
docker pull yourusername/github-license-analyzer:latest
docker run ... yourusername/github-license-analyzer:latest
```

---

## 最佳实践

### 1. 使用最小化镜像
- 使用 `python:3.13-slim` 而不是完整镜像
- 清理不必要的文件
- 使用多阶段构建（可选）

### 2. 安全性
- 不在Dockerfile中硬编码敏感信息
- 使用.env文件存储配置
- 容器内以非root用户运行（可选）

### 3. 日志管理
- 定期清理日志文件
- 考虑使用日志聚合工具
- 监控容器日志

### 4. 数据持久化
- 使用volume挂载重要目录
- 定期备份数据
- 避免在容器内存储重要数据

---

## 参考命令速查

| 任务 | 命令 |
|------|------|
| 构建镜像 | `docker build -t analyzer .` |
| 查看镜像 | `docker images` |
| 运行CLI | `docker-compose --profile cli run analyzer-cli` |
| 运行API | `docker-compose --profile api up` |
| 查看日志 | `docker-compose logs -f` |
| 进入容器 | `docker-compose exec analyzer-api bash` |
| 停止容器 | `docker-compose down` |
| 删除镜像 | `docker rmi analyzer` |
| 清理资源 | `docker system prune -a` |

---

## 更多帮助

```bash
# 查看Docker Compose文档
docker-compose --help

# 查看Dockerfile参考
# https://docs.docker.com/engine/reference/builder/

# 查看Docker最佳实践
# https://docs.docker.com/develop/dev-best-practices/
```

---

**项目已容器化！** 🎉

可以将 `Dockerfile` 和相关文件打包到任何支持Docker的环境运行。
