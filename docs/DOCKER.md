# Docker 配置与当前限制

仓库提供 [`Dockerfile`](../Dockerfile)、开发用 [`docker-compose.yml`](../docker-compose.yml) 和含 Nginx 的 [`docker-compose.prod.yml`](../docker-compose.prod.yml)。当前 master 自动部署采用 self-hosted runner + systemd，未使用此 Compose 路径，详见[部署说明](DEPLOYMENT.md)。

## 使用前检查

现有文件有以下可从代码确认的差异，不能视为已经验证的完整生产镜像：

| 项目 | 当前状态及影响 |
| --- | --- |
| 构建上下文 | `.dockerignore` 排除了 `.env`、`README.md` 和 `*.md`，Dockerfile 却包含相应 `COPY`；干净构建可能失败 |
| 依赖 | Dockerfile 手工列举依赖，未与 `pyproject.toml`/`uv.lock` 同步，未列出近期使用的 `zstandard` |
| 风险表 | Dockerfile 未复制 `license_risk.yaml`；缺失时风险加载器退回未知风险 |
| 健康检查 | 生产 Compose 使用 `curl`，当前镜像只显式安装 gcc，未保证 curl 存在 |
| provider | 生产 Compose 未显式传递 `LLM_PROVIDER` 等全部可配置变量，不能假设宿主 `.env` 的所有键都会进入容器 |
| 服务健康 | 容器 `/health` 仅探测 API 存活，不验证模型、邮箱或审核结果 |

本轮仅更新文档，没有修改以上构建/配置文件。使用容器部署前应先核对并另行修复所需项目，然后实际构建及验证，不能只根据下列命令存在就认定可用。

## 配置入口

从根目录运行，准备 `.env`、CLI 使用的 `input.xlsx` 和输出目录。开发 Compose 使用 `env_file: .env`，挂载 outputs/logs/temp，并将模型缓存置于命名卷 `/var/lib/licensedetector`。

以下为构建前核对及构建入口：

```bash
docker compose --profile api config --quiet
docker compose -f docker-compose.prod.yml config --quiet
docker compose --profile api build analyzer-api
```

`config --quiet` 只检查 Compose 配置解析；它不构建镜像，也不验证应用依赖或运行效果。

## 开发 Compose 命令

完成构建问题处理并验证镜像后：

```bash
# CLI 处理 input.xlsx，完成即退出
docker compose --profile cli run --rm analyzer-cli

# 字体 CLI
docker compose --profile cli run --rm analyzer-cli --font

# API 后台运行，映射宿主机 8000
docker compose --profile api up -d analyzer-api
docker compose --profile api logs -f analyzer-api
curl --fail-with-body http://127.0.0.1:8000/health

# 停止此 Compose 项目的服务
docker compose --profile api down
```

仓库 Makefile、`docker.sh`、`docker.ps1` 使用旧式 `docker-compose` 命令；若环境只有 Compose v2，优先使用上述 `docker compose` 入口。这些脚本的测试/验证目标还可能引用已不存在的根目录测试脚本，详见[测试说明](TESTING.md)。

## 生产 Compose 与 Nginx

生产配置使用命名卷持久化 outputs、logs、temp 和模型缓存；API 只 expose 8000，由 Nginx 映射 80/443。HTTPS server 在 `nginx.conf` 中仍为注释，映射 443 本身不会启用 HTTPS。

在镜像、配置、证书和网络都已核对后，入口为：

```bash
docker compose -f docker-compose.prod.yml up -d --build
docker compose -f docker-compose.prod.yml ps
docker compose -f docker-compose.prod.yml logs -f analyzer-api
```

当前 Nginx 配置的 `client_max_body_size` 为 100M、`proxy_read_timeout` 为 60 秒，关闭了代理缓冲。长时间无响应的批量下载可能超过读取超时，需要按实际入口调整；API 流式端点语义见 [API_USAGE.md](API_USAGE.md)。

命名卷持久化不代表可以随意清理。不要将 `docker system prune -a -f` 或删除卷作为常规启动/排障步骤；它们可能影响其他项目或历史数据。
