#!/bin/bash
#
# GitHub License Analyzer - Docker 助手脚本
# 用于简化Docker命令

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印帮助信息
show_help() {
    echo -e "${BLUE}GitHub License Analyzer - Docker 助手${NC}\n"
    echo -e "${YELLOW}用法: ./docker.sh [command]${NC}\n"
    echo -e "${GREEN}命令:${NC}"
    echo "  build          构建Docker镜像"
    echo "  cli            运行CLI模式"
    echo "  api            运行API模式（前台）"
    echo "  api-bg         运行API模式（后台）"
    echo "  prod           运行生产环境"
    echo "  logs           查看日志"
    echo "  stop           停止容器"
    echo "  clean          清理资源"
    echo "  shell          进入容器"
    echo "  help           显示此帮助信息\n"
    echo -e "${GREEN}示例:${NC}"
    echo "  ./docker.sh cli                # 处理本地input.xlsx"
    echo "  ./docker.sh api                # 启动API服务"
    echo "  ./docker.sh api-bg             # 后台启动API"
    echo "  ./docker.sh prod               # 启动生产环境\n"
}

# 构建镜像
build() {
    echo -e "${YELLOW}构建Docker镜像...${NC}"
    docker build -t github-license-analyzer:latest .
    echo -e "${GREEN}✓ 镜像构建完成${NC}"
}

# 运行CLI模式
run_cli() {
    echo -e "${YELLOW}运行CLI模式...${NC}"
    docker-compose --profile cli run --rm analyzer-cli
}

# 运行API模式（前台）
run_api() {
    echo -e "${YELLOW}运行API模式（前台）...${NC}"
    docker-compose --profile api up analyzer-api
}

# 运行API模式（后台）
run_api_bg() {
    echo -e "${YELLOW}运行API模式（后台）...${NC}"
    docker-compose --profile api up -d analyzer-api
    echo -e "${GREEN}✓ API已启动${NC}"
    echo -e "${BLUE}访问: http://localhost:8000/docs${NC}\n"
}

# 运行生产环境
run_prod() {
    echo -e "${YELLOW}运行生产环境...${NC}"
    docker-compose -f docker-compose.prod.yml up -d
    echo -e "${GREEN}✓ 生产环境已启动${NC}"
    echo -e "${BLUE}访问: http://localhost${NC}\n"
}

# 查看日志
show_logs() {
    echo -e "${YELLOW}查看日志...${NC}"
    docker-compose logs -f
}

# 停止容器
stop_containers() {
    echo -e "${YELLOW}停止容器...${NC}"
    docker-compose down
    echo -e "${GREEN}✓ 容器已停止${NC}"
}

# 清理资源
clean_resources() {
    echo -e "${YELLOW}清理Docker资源...${NC}"
    docker system prune -a -f
    echo -e "${GREEN}✓ Docker资源已清理${NC}"
}

# 进入容器
enter_shell() {
    echo -e "${YELLOW}进入容器...${NC}"
    docker-compose exec analyzer-api /bin/bash
}

# 主函数
main() {
    local cmd="${1:-help}"
    
    case "$cmd" in
        build)
            build
            ;;
        cli)
            run_cli
            ;;
        api)
            run_api
            ;;
        api-bg)
            run_api_bg
            ;;
        prod)
            run_prod
            ;;
        logs)
            show_logs
            ;;
        stop)
            stop_containers
            ;;
        clean)
            clean_resources
            ;;
        shell)
            enter_shell
            ;;
        help)
            show_help
            ;;
        *)
            echo -e "${RED}未知命令: $cmd${NC}"
            show_help
            exit 1
            ;;
    esac
}

# 运行主函数
main "$@"
