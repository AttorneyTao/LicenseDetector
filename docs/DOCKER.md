# 🐳 Docker 容器化使用指南

## 概述

本项目已完全容器化，支持以下部署方式：

1. **CLI模式** - 处理本地文件（软件包模式，或字体扫描模式 `--font`）
2. **API模式** - HTTP服务
3. **生产模式** - 包含反向代理

### 前置条件

- Docker 安装完成 + Docker Compose（推荐）
- `.env` 文件已配置（GitHub Token、LLM Key、邮件配置等）
- 首次构建镜像：`docker build -t github-license-analyzer:latest .`（约 5–10 分钟）

---

## 快捷工具（推荐）

### Linux/Mac (Makefile)
```bash
make docker-cli      # CLI 模式处理 input.xlsx
make docker-api-bg   # 后台启动 API 服务
make docker-logs     # 查看日志
make docker-stop     # 停止容器
```

### Windows (PowerShell)
```bash
.\docker.ps1 cli
.\docker.ps1 api
.\docker.ps1 stop
```

---

## 方式1️⃣：Docker Compose

### CLI 模式 - 处理本地文件

```bash
# 方式A：使用 docker-compose CLI profile
docker-compose --profile cli run analyzer-cli

# 方式B：直接运行（不使用profile）
docker-compose run --rm \
  -v ./input.xlsx:/app/input.xlsx:ro \
  -v ./outputs:/app/outputs \
  -v ./logs:/app/logs \
  -v ./.env:/app/.env:ro \
  analyzer-cli
```

**工作流：** `input.xlsx` 放在当前目录 → 运行命令 → 结果保存到 `outputs/`。

> **字体扫描模式**：当 `input.xlsx` 全部为字体时，在 CLI 启动命令后追加 `--font`
> 参数即可（详见 [FONT_SCANNING.md](FONT_SCANNING.md)）。例如：
> `docker-compose run --rm ... analyzer-cli --font`

### API 模式 - HTTP服务

```bash
docker-compose --profile api up -d          # 后台启动（默认 8000 端口）
docker-compose logs -f analyzer-api         # 查看日志
docker-compose --profile api down           # 停止服务
```

**使用 API：**
```bash
# API 文档：http://localhost:8000/docs    健康检查：curl http://localhost:8000/health

# 上传文件并直接下载结果
curl -X POST "http://localhost:8000/api/v1/analyze-and-download" \
  -F "file=@input.xlsx" --output result.xlsx

# 上传文件并邮件发送
curl -X POST "http://localhost:8000/api/v1/analyze" \
  -F "file=@input.xlsx" -F "email=user@example.com"
```

---

## 方式2️⃣：原生 Docker 命令

```bash
# 构建镜像
docker build -t github-license-analyzer:latest .

# CLI 模式
docker run --rm \
  -v $(pwd)/input.xlsx:/app/input.xlsx:ro \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/.env:/app/.env:ro \
  github-license-analyzer:latest

# API 模式（后台）
docker run -d --name analyzer-api \
  -p 8000:8000 \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/.env:/app/.env:ro \
  github-license-analyzer:latest --api --host 0.0.0.0
```

---

## 生产部署：Docker Compose + Nginx

```bash
docker-compose -f docker-compose.prod.yml up -d      # 启动
docker-compose -f docker-compose.prod.yml logs -f    # 日志
docker-compose -f docker-compose.prod.yml down       # 停止
```

**配置说明：** API 通过 Nginx 代理在 `http://localhost:80`；自动健康检查；日志/输出/临时文件持久化；支持 HTTPS。

### 配置 HTTPS（可选）
1. 获取 SSL 证书放入 `ssl/`（`cert.pem`、`key.pem`）
2. 编辑 `nginx.conf`，取消注释 HTTPS 部分
3. 修改服务器名：`sed -i 's/your-domain.com/your.domain.com/g' nginx.conf`
4. 重启：`docker-compose -f docker-compose.prod.yml restart nginx`

### 部署到其他服务器
```bash
# 方式A：复制整个项目
scp -r . user@server:/app/analyzer && docker-compose -f docker-compose.prod.yml up -d

# 方式B：推送到 Docker Hub
docker build -t yourusername/analyzer:latest . && docker push yourusername/analyzer:latest
```

---

## 文件说明

| 文件 | 用途 |
|------|------|
| `Dockerfile` | 容器镜像定义 |
| `docker-compose.yml` | 开发/测试配置 |
| `docker-compose.prod.yml` | 生产环境配置 |
| `nginx.conf` | Nginx 反向代理配置 |
| `.dockerignore` | Docker 构建排除文件 |

---

## 环境变量配置

容器会自动读取宿主机 `.env` 文件中的变量；也可在命令行用 `-e KEY=VALUE` 传入：

```env
GITHUB_TOKEN=...
GEMINI_API_KEY=...
DASHSCOPE_API_KEY=...
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SENDER_EMAIL=...
SENDER_PASSWORD=...
```

---

## 常见操作与故障排除

```bash
docker-compose logs -f analyzer-api          # 实时日志
docker-compose exec analyzer-api /bin/bash   # 进入容器
docker-compose ps                            # 查看状态
docker-compose down                          # 停止
docker system prune -a                       # 清理资源
```

| 问题 | 排查 |
|------|------|
| 容器启动失败 | `docker-compose logs analyzer-api`；检查 `.env` / `input.xlsx` 是否存在 |
| API 无法连接 | `docker-compose ps`；`docker-compose port analyzer-api 8000` |
| 邮件发送失败 | `docker-compose exec analyzer-api env \| grep SMTP`；查日志中的 email 错误 |
