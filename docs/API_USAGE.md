# GitHub License Analyzer - API 使用指南

## 概述

本项目支持两种运行模式：
1. **CLI模式**（默认）：处理本地input.xlsx文件，输出到outputs目录
2. **API模式**（新增）：通过HTTP API接收文件，处理后可发送到邮箱

## 快速开始

### 1. 安装依赖

```bash
# 更新依赖
uv sync

# 或使用pip
pip install -r requirements.txt
```

### 2. 配置邮件（可选）

编辑`.env`文件，配置邮件发送参数：

```env
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SENDER_EMAIL=your_email@gmail.com
SENDER_PASSWORD=your_app_password_here
```

**常见邮箱配置：**
- **Gmail**: SMTP服务器 smtp.gmail.com，端口 587（需要使用应用专用密码）
- **网易邮箱**: SMTP服务器 smtp.163.com，端口 587
- **QQ邮箱**: SMTP服务器 smtp.qq.com，端口 587
- **Outlook**: SMTP服务器 smtp.office365.com，端口 587

### 3. 运行

#### CLI模式（原有功能）
```bash
# 处理本地input.xlsx，结果保存到outputs目录
python main.py
# 或使用uv
uv run main.py
```

#### API模式（新增功能）
```bash
# 启动API服务
python main.py --api

# 或指定Host和Port
python main.py --api --host 0.0.0.0 --port 8000
```

## API 端点

API服务启动后，可通过以下地址访问：
- **API文档**: http://localhost:8000/docs
- **备用文档**: http://localhost:8000/redoc
- **健康检查**: http://localhost:8000/health

### 端点 1: 分析并发送邮件

**URL**: `POST /api/v1/analyze`

**请求参数**:
- `file` (required): 上传的Excel文件
- `email` (required): 接收结果的邮箱地址
- `smtp_server` (optional): SMTP服务器地址（不提供则使用环境变量）
- `smtp_port` (optional): SMTP端口（不提供则使用环境变量）

**请求示例**:

```bash
curl -X POST "http://localhost:8000/api/v1/analyze" \
  -H "accept: application/json" \
  -F "file=@input.xlsx" \
  -F "email=recipient@example.com"
```

**响应示例**:
```json
{
  "status": "success",
  "message": "分析完成，结果已发送至邮箱",
  "processed_rows": 50,
  "email_sent": true,
  "email": "recipient@example.com",
  "timestamp": "2024-01-15T10:30:00.123456"
}
```

### 端点 2: 分析并下载文件

**URL**: `POST /api/v1/analyze-and-download`

如果不想发送邮件，可以直接下载Excel文件。

**请求参数**:
- `file` (required): 上传的Excel文件

**请求示例**:

```bash
curl -X POST "http://localhost:8000/api/v1/analyze-and-download" \
  -F "file=@input.xlsx" \
  --output result.xlsx
```

**响应**: 返回处理后的Excel文件

### 端点 3: 健康检查

**URL**: `GET /health`

检查API服务状态。

**请求示例**:
```bash
curl http://localhost:8000/health
```

**响应示例**:
```json
{
  "status": "ok",
  "timestamp": "2024-01-15T10:30:00.123456",
  "service": "GitHub License Analyzer API"
}
```

## Python 客户端示例

### 使用requests库

```python
import requests

# 分析并发送邮件
response = requests.post(
    "http://localhost:8000/api/v1/analyze",
    files={"file": open("input.xlsx", "rb")},
    data={"email": "recipient@example.com"}
)
print(response.json())

# 分析并下载
response = requests.post(
    "http://localhost:8000/api/v1/analyze-and-download",
    files={"file": open("input.xlsx", "rb")}
)
with open("result.xlsx", "wb") as f:
    f.write(response.content)
```

## 错误处理

API返回标准HTTP状态码：
- `200 OK`: 请求成功
- `400 Bad Request`: 请求参数错误（如无效的邮箱）
- `500 Internal Server Error`: 服务器错误

错误响应示例：
```json
{
  "detail": "无效的邮箱地址"
}
```

## 日志

API运行中的日志保存在`logs/`目录下：
- `main.log`: 主日志文件
- `substep.log`: 子步骤日志
- `url_construction.log`: URL构造日志

## 性能和并发

API默认使用以下配置：
- **最大并发数**: 5（可在core/config.py中修改MAX_CONCURRENCY）
- **临时文件**: 存放在`temp/`目录

长时间运行的分析任务会定期保存临时结果。

## 常见问题

### 1. 邮件发送失败
- 确认SMTP服务器配置正确
- 检查邮箱密码（某些邮箱需要应用专用密码）
- 确认网络连接正常

### 2. 文件上传超时
- 使用`/api/v1/analyze-and-download`先获取结果，再手动发送
- 增加API请求超时时间

### 3. 内存占用过高
- 减少并发数（修改MAX_CONCURRENCY）
- 处理较小的文件

## 部署建议

### Docker 部署

可创建Dockerfile：
```dockerfile
FROM python:3.13-slim

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt

CMD ["python", "main.py", "--api", "--host", "0.0.0.0", "--port", "8000"]
```

### 生产环境

建议使用Gunicorn或其他ASGI服务器：
```bash
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker api:app --bind 0.0.0.0:8000
```

## 文件说明

- `main.py`: 主入口，支持CLI和API模式
- `api.py`: FastAPI应用实现
- `core/email_utils.py`: 邮件发送工具
- `pyproject.toml`: 已更新，包含fastapi、uvicorn等依赖

## 反馈和支持

如有问题，请在项目日志中查看详细错误信息。
