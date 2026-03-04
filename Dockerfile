# 使用官方Python runtime作为基础镜像
FROM python:3.13-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
# 注意：pyproject.toml中readme: README.md，但Docker中可能在另一个位置
# 为了避免构建失败，我们先尝试复制，如果不存在则继续
COPY pyproject.toml ./
# 如果存在README.md则复制，不存在也继续
COPY README.md* ./
# 复制.env文件（环境变量配置）
COPY .env* ./
COPY core/ ./core/
COPY main.py api.py prompts.yaml ./

# 安装Python依赖（直接从requirements）
RUN pip install --no-cache-dir \
    pandas>=2.0.0 \
    openpyxl>=3.1.0 \
    requests>=2.31.0 \
    tqdm>=4.66.0 \
    rapidfuzz>=3.0.0 \
    python-dotenv>=1.0.0 \
    google-generativeai>=0.8.5 \
    google-genai>=1.18.0 \
    pyyaml>=6.0.2 \
    packaging>=25.0 \
    beautifulsoup4>=4.13.4 \
    aiofiles>=24.1.0 \
    httpx>=0.28.1 \
    tenacity>=9.1.2 \
    pytest>=8.4.1 \
    pytest-asyncio>=1.1.0 \
    openai>=1.106.1 \
    aiohttp>=3.13.0 \
    fastapi>=0.104.0 \
    uvicorn>=0.24.0 \
    python-multipart>=0.0.6

# 创建必要的目录
RUN mkdir -p /app/outputs /app/logs /app/temp

# 默认入口 - 支持通过环境变量或命令参数切换模式
# CLI模式（默认）: docker run myapp
# API模式: docker run myapp --api
ENTRYPOINT ["python", "main.py"]
CMD []
