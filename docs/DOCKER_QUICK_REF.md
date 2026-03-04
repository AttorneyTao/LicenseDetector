# 🐳 Docker 快速参考

## 最常用的命令

### 构建镜像
```bash
docker build -t github-analyzer .
```

### 运行CLI模式
```bash
docker-compose --profile cli run analyzer-cli
```

### 运行API模式
```bash
docker-compose --profile api up -d
```

### 查看日志
```bash
docker-compose logs -f analyzer-api
```

### 停止所有容器
```bash
docker-compose down
```

---

## 三种常见场景

### 场景1：本地文件一次性处理
```bash
# 确保input.xlsx在当前目录
docker-compose --profile cli run analyzer-cli

# 结果在outputs/目录
ls outputs/
```

### 场景2：启动API服务供其他程序调用
```bash
# 启动（后台）
docker-compose --profile api up -d

# 查看日志
docker-compose logs -f analyzer-api

# 上传文件
curl -X POST "http://localhost:8000/api/v1/analyze-and-download" \
  -F "file=@input.xlsx" \
  --output result.xlsx

# 停止
docker-compose --profile api down
```

### 场景3：生产部署（含Nginx）
```bash
# 启动
docker-compose -f docker-compose.prod.yml up -d

# 访问: http://localhost （Nginx代理）

# 停止
docker-compose -f docker-compose.prod.yml down
```

---

## 排查问题

### 容器无法启动
```bash
# 查看错误
docker-compose logs analyzer-api

# 检查配置
docker-compose config

# 手动运行看详细错误
docker-compose run --rm analyzer-cli
```

### 需要进入容器调试
```bash
# 进入运行中的容器
docker-compose exec analyzer-api bash

# 检查文件
ls -la /app/

# 查看环境变量
env | grep SMTP
```

### 清理空间
```bash
# 停止所有容器
docker-compose down

# 删除镜像
docker rmi github-analyzer

# 删除所有未使用资源
docker system prune -a
```

---

## 文件说明

| 文件 | 说明 |
|------|------|
| Dockerfile | 镜像构建文件 |
| docker-compose.yml | 开发配置（CLI+API） |
| docker-compose.prod.yml | 生产配置（含Nginx） |
| nginx.conf | Nginx配置 |
| .dockerignore | Docker构建忽略文件 |

---

## API地址

启动API后访问：

- **Swagger文档**：http://localhost:8000/docs
- **健康检查**：http://localhost:8000/health
- **分析上传**：http://localhost:8000/api/v1/analyze
- **分析下载**：http://localhost:8000/api/v1/analyze-and-download

---

## 更多详细信息

查看完整文档：
```bash
cat DOCKER_GUIDE.md
```

🚀 **现在你可以在任何地方运行这个容器化项目了！**
