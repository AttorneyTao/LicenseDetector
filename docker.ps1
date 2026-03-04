#!/usr/bin/env pwsh
<#
.SYNOPSIS
    GitHub License Analyzer Docker 脚本

.DESCRIPTION
    提供简化的Docker命令行工具

.EXAMPLE
    .\docker.ps1 cli       # 运行CLI模式
    .\docker.ps1 api       # 运行API模式
    .\docker.ps1 prod      # 运行生产环境
#>

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet('build', 'cli', 'api', 'api-bg', 'prod', 'logs', 'stop', 'clean', 'help', 'shell')]
    [string]$Command = 'help'
)

function Show-Help {
    Write-Host "GitHub License Analyzer - Docker 助手`n" -ForegroundColor Cyan
    Write-Host "用法: .\docker.ps1 [command]`n" -ForegroundColor Yellow
    Write-Host "命令:" -ForegroundColor Green
    Write-Host "  build          构建Docker镜像"
    Write-Host "  cli            运行CLI模式"
    Write-Host "  api            运行API模式（前台）"
    Write-Host "  api-bg         运行API模式（后台）"
    Write-Host "  prod           运行生产环境"
    Write-Host "  logs           查看日志"
    Write-Host "  stop           停止容器"
    Write-Host "  clean          清理资源"
    Write-Host "  shell          进入容器"
    Write-Host "  help           显示此帮助信息`n"
    Write-Host "示例:" -ForegroundColor Green
    Write-Host "  .\docker.ps1 cli                # 处理本地input.xlsx"
    Write-Host "  .\docker.ps1 api                # 启动API服务"
    Write-Host "  .\docker.ps1 api-bg             # 后台启动API"
    Write-Host "  .\docker.ps1 prod               # 启动生产环境`n"
}

function Invoke-Build {
    Write-Host "构建Docker镜像..." -ForegroundColor Yellow
    docker build -t github-license-analyzer:latest .
    Write-Host "✓ 镜像构建完成" -ForegroundColor Green
}

function Invoke-CLI {
    Write-Host "运行CLI模式..." -ForegroundColor Yellow
    docker-compose --profile cli run --rm analyzer-cli
}

function Invoke-API {
    Write-Host "运行API模式（前台）..." -ForegroundColor Yellow
    docker-compose --profile api up analyzer-api
}

function Invoke-APIBackground {
    Write-Host "运行API模式（后台）..." -ForegroundColor Yellow
    docker-compose --profile api up -d analyzer-api
    Write-Host "✓ API已启动" -ForegroundColor Green
    Write-Host "访问: http://localhost:8000/docs`n" -ForegroundColor Cyan
}

function Invoke-Prod {
    Write-Host "运行生产环境..." -ForegroundColor Yellow
    docker-compose -f docker-compose.prod.yml up -d
    Write-Host "✓ 生产环境已启动" -ForegroundColor Green
    Write-Host "访问: http://localhost`n" -ForegroundColor Cyan
}

function Invoke-Logs {
    Write-Host "查看日志..." -ForegroundColor Yellow
    docker-compose logs -f
}

function Invoke-Stop {
    Write-Host "停止容器..." -ForegroundColor Yellow
    docker-compose down
    Write-Host "✓ 容器已停止" -ForegroundColor Green
}

function Invoke-Clean {
    Write-Host "清理Docker资源..." -ForegroundColor Yellow
    docker system prune -a -f
    Write-Host "✓ Docker资源已清理" -ForegroundColor Green
}

function Invoke-Shell {
    Write-Host "进入容器..." -ForegroundColor Yellow
    docker-compose exec analyzer-api /bin/bash
}

# 执行命令
switch ($Command) {
    'build' { Invoke-Build }
    'cli' { Invoke-CLI }
    'api' { Invoke-API }
    'api-bg' { Invoke-APIBackground }
    'prod' { Invoke-Prod }
    'logs' { Invoke-Logs }
    'stop' { Invoke-Stop }
    'clean' { Invoke-Clean }
    'shell' { Invoke-Shell }
    'help' { Show-Help }
    default { Show-Help }
}
