# 部署与运维

## 当前自动部署

仓库的 [`.github/workflows/deploy.yml`](../.github/workflows/deploy.yml) 定义 `Deploy to Production`：

1. 推送到 `master` 时触发，在 self-hosted runner 上运行。
2. 进入 `/root/LicenseDetector`，执行 `git fetch origin master` 和 `git reset --hard origin/master`。
3. 执行 `uv sync`。
4. 通过 `timeout 120 systemctl restart licensedetector.service` 重启服务。
5. 最多检查 10 次 `systemctl is-active`，间隔 2 秒；失败时输出服务状态和最近 journal。
6. 成功/失败均尝试调用 `/root/ops/notify_deploy.py` 通知；通知失败不改变部署步骤的结果。

工作流没有 `paths` 过滤，**仅修改文档的 master 推送也会触发部署**。没有自动测试前置步骤，也没有 HTTP、模型、SMTP 或真实扫描检查。推送成功只证明代码进入远端，服务是否运行需核对 Actions 和服务器。

服务单元和 `/root/ops/notify_deploy.py` 未保存在本仓库；其监听地址、环境及通知目标不能仅从 workflow 推断。self-hosted runner 需具备 Git、uv、systemd 权限以及目标目录。`reset --hard` 会覆盖服务器受版本控制文件的本地修改，因此配置/业务结果不要只保存在会被覆盖的跟踪文件中。

## 部署后只读核对

以下命令在服务器上执行；路径和服务名与当前 workflow 一致。检查前不需要重复重启。

```bash
cd /root/LicenseDetector
git rev-parse HEAD
git log -1 --oneline
systemctl is-active licensedetector.service
systemctl status licensedetector.service --no-pager
systemctl cat licensedetector.service
curl --fail-with-body http://127.0.0.1:8000/health
journalctl -u licensedetector.service -n 100 --no-pager
```

将服务器 SHA 与本次推送提交对照。若服务单元使用其他端口，按实际配置检查。`/health` 返回成功不代表外部 GitHub、注册表、模型或邮箱可用，也不证明某批扫描已完成。

需要回滚时应明确目标提交并按既有发布流程处理；当前工作流每次重新取 `origin/master`，单独在服务器检出旧提交可能在下次推送时被覆盖。不要在排障过程中未经必要核对就重跑生产批次。

## 本地 API 和长期运行

```bash
uv sync
uv run main.py --api --host 127.0.0.1 --port 8000
```

长期运行由 systemd 或现有进程管理器管理，工作目录应为仓库根目录，并配置可访问的 `.env`、日志和缓存目录。项目 `start.sh` 固定了服务器目录，包含历史前端路径提示；本地运行优先采用上述入口，前端实际位于 `static/index.html`。

应用没有内置认证、持久化作业队列或跨 worker 日志广播。不要直接套用多 worker 示例并假定日志和批次状态能够共享；反向代理和进程模型须与这些限制匹配。

## 日志、结果与缓存

| 位置 | 用途 |
| --- | --- |
| `logs/github_license_analyzer.log` | 主审核日志 |
| `logs/substep.log` | 仓库处理步骤 |
| `logs/url_construction.log` | URL 解析及构造 |
| `logs/version_resolve.log` | 版本匹配 |
| `logs/llm_interaction.log` | 模型请求和响应详情 |
| `logs/llm_cache.log` | 缓存状态及调用统计 |
| `logs/npm.log`、`logs/maven.log` 等 | 相应生态日志，具体是否产生由路径决定 |
| `outputs/` | CLI 最终 Excel；API 附件通常使用系统临时文件 |
| `temp/temp_results_latest.csv` | CLI 已完成行的中间快照，另有带时间戳备份 |
| 代码目录外的 SQLite | 模型缓存；默认路径见[配置说明](CONFIGURATION.md) |

共享轮转处理器在跨日或达到 50 MiB 时轮转，归档形如 `*.log.YYYY-MM-DD.001`，按归档修改时间清理超过 7 天的文件。应同时查看当前文件和日期归档；临时 CSV 不是自动恢复检查点。

日志可能包含输入文件名、组件名、URL、模型提示词和响应；当前没有统一匿名化处理。分享诊断材料时按实际内容控制范围。全局 SSE 同样可能展示多个作业活动。

## 扫描问题定位

按以下证据对应同一条输入：

1. 原表流水号、`name`、`version`、`github_url` 和批次时间。
2. 输出行的 `input_url`、`resolved_version`、许可、链接和判定理由。
3. 日志实际分发路径：注册表、GitHub、归档或 purl。
4. 下载成功后是否找到证据文件，模型是否返回有效结果。
5. 最终链接优化是否仅替换链接，或前序已经选错版本/许可。

下载成功不能单独证明内部审核成功；许可证网页存在也不能单独证明它适用于输入文件。不要用另一个版本的页面补写历史许可。

## 常见部署问题

| 问题 | 核对内容 |
| --- | --- |
| 推送后未部署 | master 触发条件、Actions 是否排队、self-hosted runner 在线状态 |
| Actions 成功但 API 不通 | workflow 只检查 systemd active；另查实际监听端口、HTTP 与反向代理 |
| 服务重启卡住 | workflow 的 120 秒上限、systemd 状态、journal |
| 扫描有结果但没收到邮件 | `email_sent` 或流中邮件结果、SMTP 配置及服务器发送日志 |
| 额度快速消耗 | 多批次并行数、版本匹配/仓库发现请求、缓存 hit/miss/audit、失败重试；OpenClaw 与扫描模型调用分开核对 |
| 长请求被代理断开 | 当前 nginx `proxy_read_timeout` 为 60 秒；批量规模、客户端读取超时和日志流行为需共同考虑 |

Docker 作为另一套配置入口，当前限制见 [DOCKER.md](DOCKER.md)。本文的核对命令不启动扫描，也不会发送业务邮件。
