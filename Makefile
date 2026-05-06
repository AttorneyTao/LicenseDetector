.PHONY: help build build-prod run-cli run-api run-api-bg logs stop clean install test docker-cli docker-api docker-prod docker-logs docker-stop docker-clean verify

help:
	@echo "================================"
	@echo "GitHub License Analyzer"
	@echo "================================"
	@echo ""
	@echo "=== 本地开发 ==="
	@echo "make install        - 安装依赖"
	@echo "make verify         - 验证安装"
	@echo "make run-cli        - 运行CLI模式"
	@echo "make run-api        - 运行API模式（前台）"
	@echo "make run-api-bg     - 运行API模式（后台）"
	@echo ""
	@echo "=== Docker 开发 ==="
	@echo "make docker-build   - 构建Docker镜像"
	@echo "make docker-cli     - 运行Docker CLI模式"
	@echo "make docker-api     - 运行Docker API模式"
	@echo "make docker-api-bg  - 运行Docker API模式（后台）"
	@echo "make docker-prod    - 运行生产环境"
	@echo "make docker-logs    - 查看Docker日志"
	@echo "make docker-stop    - 停止Docker容器"
	@echo ""
	@echo "=== 工具 ==="
	@echo "make clean          - 清理本地缓存"
	@echo "make docker-clean   - 清理Docker资源"
	@echo ""

# ============ 本地开发命令 ============

install:
	@echo "安装依赖..."
	uv sync

verify:
	@echo "验证安装..."
	python verify_installation.py

run-cli:
	@echo "运行CLI模式..."
	uv run main.py

run-api:
	@echo "运行API模式（前台）..."
	uv run main.py --api

run-api-bg:
	@echo "运行API模式（后台）..."
	uv run main.py --api > logs/api.log 2>&1 &
	@echo "API已在后台启动，访问: http://localhost:8000/docs"

test:
	@echo "测试API..."
	python test_api.py

# ============ Docker 命令 ============

docker-build:
	@echo "构建Docker镜像..."
	docker build -t github-license-analyzer:latest .
	@echo "✓ 镜像构建完成"

docker-cli:
	@echo "运行Docker CLI模式..."
	docker-compose --profile cli run --rm analyzer-cli

docker-api:
	@echo "运行Docker API模式（前台）..."
	docker-compose --profile api up analyzer-api

docker-api-bg:
	@echo "运行Docker API模式（后台）..."
	docker-compose --profile api up -d analyzer-api
	@echo "✓ API已启动，访问: http://localhost:8000/docs"

docker-prod:
	@echo "运行生产环境..."
	docker-compose -f docker-compose.prod.yml up -d
	@echo "✓ 生产环境已启动，访问: http://localhost"

docker-logs:
	@echo "查看Docker日志..."
	docker-compose logs -f

docker-stop:
	@echo "停止Docker容器..."
	docker-compose down
	@echo "✓ 容器已停止"

docker-clean:
	@echo "清理Docker资源..."
	docker system prune -a -f
	@echo "✓ Docker资源已清理"

docker-shell:
	@echo "进入容器..."
	docker-compose exec analyzer-api /bin/bash

# ============ 清理命令 ============

clean:
	@echo "清理本地缓存..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache build dist *.egg-info
	@echo "✓ 缓存已清理"

# ============ 快捷命令 ============

.DEFAULT_GOAL := help

# 打印信息的辅助函数
.SILENT: help
