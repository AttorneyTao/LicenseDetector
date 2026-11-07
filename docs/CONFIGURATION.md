# 配置说明

从项目根目录启动，使用 `.env` 或进程环境变量。代码通过 python-dotenv 加载配置；重启服务后再验证修改是否生效。当前没有提交 `.env.example`，也没有 CLI 参数覆盖所有配置项。

## 最小配置

```dotenv
GITHUB_TOKEN=your_github_token
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_key
USE_LLM=true
LLM_CACHE_MODE=read_write
```

所选模型提供方需要对应 key；Qwen 使用 `LLM_PROVIDER=qwen` 和 `DASHSCOPE_API_KEY`。这两种 provider 是可选配置，当前没有自动从 Gemini 切换到 Qwen 的统一灾备流程。

| 环境变量 | 默认/用途 |
| --- | --- |
| `GITHUB_TOKEN` | GitHub API 访问令牌，批量审核应配置 |
| `LLM_PROVIDER` | `gemini`；可选 `qwen` |
| `GEMINI_API_KEY` | Gemini 提供方凭据 |
| `DASHSCOPE_API_KEY` | Qwen 提供方凭据 |
| `USE_LLM` | `true`；部分调用点据此跳过模型分析/版本匹配 |
| `HTTP_PROXY`、`HTTPS_PROXY` | 网络客户端使用的代理；各客户端行为以实现为准 |
| `MAVEN_REPOSITORY_BASE_URLS` | 逗号分隔的额外 Maven 仓库根地址 |
| `ARCHIVE_MAX_SIZE_BYTES` | `1073741824`，归档下载最大字节数 |
| `ARCHIVE_DOWNLOAD_TIMEOUT` | `900`，归档总下载超时，单位秒 |

`USE_LLM=false` 不是所有 provider 调用的统一硬开关；某些直接使用 `complete_sync/async` 的路径仍需检查调用方。不能以此保证绝不产生模型请求，也不能将其结果等同于完整审核。

[`core/config.py`](../core/config.py) 中当前模型为 Gemini `gemini-3.5-flash`、Qwen `qwen3-max`，Qwen base URL 为 `https://dashscope.aliyuncs.com/compatible-mode/v1`。这些值来自代码配置，当前未提供通用 `MODEL` 环境变量覆盖；不要填写代码不读取的变量并假定已切换模型。

## 代码配置

| 配置位置 | 内容 |
| --- | --- |
| [`core/config.py`](../core/config.py) | `MAX_CONCURRENCY=5`、`SCORE_THRESHOLD=65`、第三方目录关键词、输出优先顺序、provider 模型配置 |
| [`license_risk.yaml`](../license_risk.yaml) | 许可证风险分类、显示标签和未知风险默认值 |
| [`prompts.yaml`](../prompts.yaml) | 版本匹配、许可分析、版权提取、仓库发现等提示词 |
| [`core/llm_policies.py`](../core/llm_policies.py) | 可缓存任务、响应校验、策略版本及过期规则 |

并发数是每个批次内的组件并发；多个 API 批次同时执行时总请求量可能超过该值。更改输出顺序、提示词或风险分类前先运行对应[回归测试](TESTING.md)。

## 模型缓存

| 环境变量 | 默认/用途 |
| --- | --- |
| `LLM_CACHE_MODE` | `read_write`；可选 `off`、`read_only`。无效值按 off 处理 |
| `LLM_CACHE_PATH` | 覆盖 SQLite 路径，建议放在代码目录外 |
| `LLM_CACHE_AUDIT_RATE` | `0.01`，已确认缓存命中的实时抽样复核比例 |
| `LLM_CACHE_EPOCH` | `1`，改变后旧键整体失效 |
| `LLM_CACHE_MODEL_REVISION` | 空；用于区分模型服务端修订 |

默认路径：Linux 等非 macOS 环境为 `/var/lib/licensedetector/llm-cache.sqlite`，macOS 为 `~/Library/Application Support/LicenseDetector/llm-cache.sqlite`。进程须有目录读写权限；Docker Compose 配置了持久化缓存卷。

仅缓存有任务校验策略、输入和输出可验证的响应。键包含任务、完整提示词的哈希、模型/端点、调用参数、策略版本及 epoch。首个有效回答记为候选，第二次独立回答语义一致后方可命中；不一致会隔离，过期或不合法结果继续实时请求。缓存故障不应阻断审核。

缓存不保存完整提示词，但模型响应可能包含敏感内容；日志另可能包含提示词和响应。两次一致及抽样只能降低重复错误风险，不能证明许可结论正确。

```bash
# 以下命令会清除缓存记录；按排障需要选择执行
uv run python -m core.llm_cache invalidate --task license_analysis
uv run python -m core.llm_cache invalidate --all
```

`logs/llm_cache.log` 的 `LLM_CACHE_STATS` 按任务及进程记录 hit/miss/audit/bypass/provider_error 等结果。命中率分母是可缓存调用，实际调用避免率分母是全部入口请求；audit 仍访问模型，不计为避免调用。跨进程/日期统计按逐条事件计算，避免累加累计值。

## 邮件

```dotenv
SMTP_SERVER=smtp.example.com
SMTP_PORT=587
SENDER_EMAIL=sender@example.com
SENDER_PASSWORD=your_smtp_credential
```

SMTP 默认服务器为 `smtp.gmail.com`，默认端口 587，账号和密码没有默认值。实现使用 `smtplib.SMTP` 后执行 STARTTLS；当前 API 表单没有切换隐式 TLS 的参数，不能仅改成端口 465 就假定兼容。使用邮箱服务商要求的 SMTP 授权码或应用凭据。

收件人和可选邮件主题在请求中传递；没有统一的默认收件邮箱。发送是否成功检查 `email_sent` 或流中的邮件结果。

## 运行与部署区别

本地 `.env`、systemd 环境和 Docker environment 的来源不同。生产 Compose 只显式传入部分变量，尤其未列出 `LLM_PROVIDER`；改变宿主机 `.env` 后，应核对变量是否确实进入容器。Docker 的当前限制见 [DOCKER.md](DOCKER.md)。

本项目不读取 OpenClaw 账户额度配置。OpenClaw 调用 API 与扫描服务调用模型是不同链路，应分别核查运行环境及请求日志。
