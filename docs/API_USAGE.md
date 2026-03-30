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

### 端点 2: 流式日志输出（仅日志）

**URL**: `POST /api/v1/analyze-stream`

实时流式返回分析过程中的日志（错误、警告和进度信息），无需等待处理完成。仅输出日志，不产生文件或邮件。

**请求参数**:
- `file` (required): 上传的Excel文件

**请求示例**:

```bash
# 使用curl（--no-buffer保证实时输出）
curl -X POST "http://localhost:8000/api/v1/analyze-stream" \
  -F "file=@input.xlsx" \
  --no-buffer

# 将流式日志保存到文件
curl -X POST "http://localhost:8000/api/v1/analyze-stream" \
  -F "file=@input.xlsx" \
  --no-buffer > analysis_logs.txt
```

**响应示例**（流式文本）:
```
[START] 开始处理文件: input.xlsx
[INFO] 读取了 50 行数据
[INFO] GitHub API 客户端已初始化
[INFO] 开始处理仓库...
[INFO] 检测到 Go 包 URL: go.mod
[WARNING] 无法获取许可证信息: 404 Not Found
[ERROR] 处理失败 https://github.com/example/repo: Timeout
[SUCCESS] 处理完成，共处理 50 行数据
```

**日志级别**:
- `[START]` - 开始处理
- `[INFO]` - 信息提示
- `[WARNING]` - 警告信息
- `[ERROR]` - 错误信息
- `[SUCCESS]` - 完成

### 端点 3: 流式日志 + 邮件发送

**URL**: `POST /api/v1/analyze-stream-email`

流式输出分析日志，处理完成后自动发送邮件。结合了流式监控和邮件发送的优点。

**请求参数**:
- `file` (required): 上传的Excel文件
- `email` (required): 接收结果的邮箱地址
- `smtp_server` (optional): SMTP服务器地址（不提供则使用环境变量）
- `smtp_port` (optional): SMTP端口（不提供则使用环境变量）

**请求示例**:

```bash
curl -X POST "http://localhost:8000/api/v1/analyze-stream-email" \
  -F "file=@input.xlsx" \
  -F "email=recipient@example.com" \
  --no-buffer
```

**响应示例**（流式）:
```
[START] 开始处理文件: input.xlsx
[INFO] 读取了 50 行数据
[INFO] GitHub API 客户端已初始化
[INFO] 开始处理仓库...
[INFO] 处理完成，共处理 50 行数据
[INFO] 生成输出文件...
[INFO] 输出文件已生成
[INFO] 正在发送结果到邮箱: recipient@example.com
[SUCCESS] 邮件已发送到: recipient@example.com
```

### 端点 4: 流式日志 + 文件下载

**URL**: `POST /api/v1/analyze-stream-download`

流式输出分析日志，处理完成后返回Excel文件。适合需要同时监控进度和获取返回文件的场景。

**请求参数**:
- `file` (required): 上传的Excel文件

**请求示例**:

```bash
# 流式查看日志，最后下载文件
curl -X POST "http://localhost:8000/api/v1/analyze-stream-download" \
  -F "file=@input.xlsx" \
  --no-buffer \
  --output result.xlsx
```

**响应**: 流式输出日志，最后返回Excel文件

### 端点 5: 分析并下载文件

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

### 端点 6: 健康检查

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

# 分析并发送邮件（等待完成后返回结果）
response = requests.post(
    "http://localhost:8000/api/v1/analyze",
    files={"file": open("input.xlsx", "rb")},
    data={"email": "recipient@example.com"}
)
print(response.json())

# 流式日志输出（仅监控进度）
response = requests.post(
    "http://localhost:8000/api/v1/analyze-stream",
    files={"file": open("input.xlsx", "rb")},
    stream=True
)
for line in response.iter_lines():
    if line:
        print(line.decode('utf-8'))

# 流式日志 + 邮件发送
response = requests.post(
    "http://localhost:8000/api/v1/analyze-stream-email",
    files={"file": open("input.xlsx", "rb")},
    data={"email": "recipient@example.com"},
    stream=True
)
for line in response.iter_lines():
    if line:
        print(line.decode('utf-8'))

# 流式日志 + 文件下载（日志+文件都会返回）
response = requests.post(
    "http://localhost:8000/api/v1/analyze-stream-download",
    files={"file": open("input.xlsx", "rb")},
    stream=True
)
# 这种情况较复杂，建议使用curl或其他工具

# 分析并下载（等待完成后返回文件）
response = requests.post(
    "http://localhost:8000/api/v1/analyze-and-download",
    files={"file": open("input.xlsx", "rb")}
)
with open("result.xlsx", "wb") as f:
    f.write(response.content)
```

### 高级用法：带进度条的流式请求

```python
import requests
from tqdm import tqdm

# 流式监控分析进度
response = requests.post(
    "http://localhost:8000/api/v1/analyze-stream",
    files={"file": open("input.xlsx", "rb")},
    stream=True
)

print("分析进程日志:")
for line in response.iter_lines():
    if line:
        print(line.decode('utf-8'))

# 流式监控 + 邮件发送
response = requests.post(
    "http://localhost:8000/api/v1/analyze-stream-email",
    files={"file": open("input.xlsx", "rb")},
    data={"email": "recipient@example.com"},
    stream=True
)

print("分析进程日志（含邮件发送状态）:")
for line in response.iter_lines():
    if line:
        print(line.decode('utf-8'))
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

### 1. 如何选择合适的端点？

| 需求 | 推荐端点 | 说明 |
|------|--------|------|
| 仅监控日志 | `/api/v1/analyze-stream` | 纯日志流，不产生文件或邮件 |
| 日志 + 邮件 | `/api/v1/analyze-stream-email` | 流式日志，完成后发送邮件 |
| 日志 + 下载文件 | `/api/v1/analyze-stream-download` | 流式日志，完成后返回文件 |
| 仅下载文件 | `/api/v1/analyze-and-download` | 等待完成后一次性返回文件 |
| 仅发送邮件 | `/api/v1/analyze` | 等待完成后发送邮件 |

### 2. 流式端点和非流式端点的区别？

- **流式端点** (`analyze-stream*`): 实时输出日志，可立即看到处理进度，适合长时间处理任务
- **非流式端点** (`analyze*`): 等待完成后一次性返回结果，适合短时间任务或需要最终结果的场景

### 3. 如何在流式过程中监控分析进度？

使用流式端点（`/api/v1/analyze-stream` 或 `/api/v1/analyze-stream-email`）：

```bash
curl -X POST "http://localhost:8000/api/v1/analyze-stream" \
  -F "file=@input.xlsx" \
  --no-buffer
```

或在Python中：
```python
response = requests.post(
    "http://localhost:8000/api/v1/analyze-stream",
    files={"file": open("input.xlsx", "rb")},
    stream=True
)
for line in response.iter_lines():
    print(line.decode('utf-8'))
```

### 4. 邮件发送失败
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
- `api.py`: FastAPI应用实现，包含6个API端点和健康检查
  - `POST /api/v1/analyze`: 分析并发送邮件
  - `POST /api/v1/analyze-stream`: 流式日志输出（仅日志）
  - `POST /api/v1/analyze-stream-email`: 流式日志 + 邮件发送
  - `POST /api/v1/analyze-stream-download`: 流式日志 + 文件下载
  - `POST /api/v1/analyze-and-download`: 分析并下载文件
  - `GET /health`: 健康检查
- `core/email_utils.py`: 邮件发送工具
- `pyproject.toml`: 已更新，包含fastapi、uvicorn等依赖

## 反馈和支持

如有问题，请在项目日志中查看详细错误信息。
