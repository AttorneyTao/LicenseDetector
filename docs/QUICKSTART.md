# 快速开始

## 安装和配置

需要 Python 3.13+。从仓库根目录执行：

```bash
uv sync
uv run main.py --help
```

不用 uv 时，可先创建虚拟环境，再安装依赖：

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Windows PowerShell 的激活命令为 `.venv\Scripts\Activate.ps1`。后续示例使用 `uv run`；已激活环境时可改用 `python`。

手动创建 `.env`，填写实际凭据；当前没有随仓库提交的 `.env.example`。

```dotenv
GITHUB_TOKEN=your_github_token
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_key
USE_LLM=true
```

Qwen 配置、缓存、代理和邮件参数见[配置说明](CONFIGURATION.md)。程序使用根目录的 `prompts.yaml`、`input.xlsx` 和相对日志路径，请在项目根目录启动。

## 准备输入

创建 `input.xlsx`，使用默认首张工作表，第一行为字段名：

| name | version | github_url | package_manager | serial_number |
| --- | --- | --- | --- | --- |
| requests | 2.31.0 | https://pypi.org/project/requests/ | pypi | 00001 |
| serde | 1.0.197 | pkg:cargo/serde@1.0.197 | cargo | 00002 |

`github_url` 为必要定位字段。原三列格式仍可使用，后两个可选字段会带入输出。流水号在录入前设为文本；单纯改变显示格式无法恢复 Excel 已截断的数字。[输入与输出](INPUT_OUTPUT.md)列出完整规则。

## 选择入口

### 本地软件包扫描

```bash
uv run main.py
```

结果位于 `outputs/output_latest.xlsx` 和带时间戳的副本，工作表为“分析结果”。中间结果位于 `temp/`，详细日志位于 `logs/`；中间结果不会在下次启动时自动恢复。

### Web 页面与 API

```bash
uv run main.py --api --host 127.0.0.1 --port 8000
```

访问 `http://127.0.0.1:8000/` 上传 Excel；接口文档位于 `/docs` 或 `/redoc`。需要其他机器访问时再设置 `--host 0.0.0.0` 并配置网络访问控制。

```bash
curl --fail-with-body http://127.0.0.1:8000/health
curl --fail-with-body -X POST http://127.0.0.1:8000/api/v1/analyze-and-download \
  -F 'file=@input.xlsx' --output result.xlsx
```

不要将 `/api/v1/analyze-stream` 的文本日志保存为 `.xlsx`；直接下载使用上述接口。[API 使用](API_USAGE.md)说明各端点的真实响应类型。

### 邮件结果

先配置 SMTP，再提交明确的收件邮箱：

```bash
curl --fail-with-body -X POST http://127.0.0.1:8000/api/v1/analyze \
  -F 'file=@input.xlsx' -F 'email=recipient@example.com' \
  -F 'email_subject=开源组件许可审核结果'
```

该请求等待审核及邮件处理后返回 JSON。检查 `email_sent`；HTTP 200 或 `status=success` 不能单独证明邮件发送成功。过程中查看日志可用 `analyze-stream-email`。

### 字体

```bash
uv run main.py --font
```

用于整张输入表都是字体的场景。当前 API 上传端点没有字体模式参数，`--api --font` 也不会将 API 切换为字体模式。详见[字体模式](FONT_SCANNING.md)。

## 检查结果和排障

先核对行数和流水号，再看请求与实际版本、许可结论、证据链接、风险等级及判定理由。版权字段可能是最终兜底文本；`AND Others` 表示检测到第三方内容提示。

| 现象 | 排查方法 |
| --- | --- |
| 无法读取 `input.xlsx` | 确认工作目录、首行表头及 `github_url` 列；业务表需先整理为标准输入 |
| 模型初始化失败 | 核对 `LLM_PROVIDER` 与所选提供方的 API key |
| API 无法访问 | 核对监听地址、端口、服务日志与反向代理；`/health` 只反映 API 存活 |
| 使用了其他版本 | 对照 `input_version`、`resolved_version`、`used_default_branch` 与判定理由 |
| 邮件未收到 | 检查 `email_sent` 和 SMTP 日志；接口默认 STARTTLS，避免直接套用隐式 TLS 端口 |
| 下载文件无法打开 | 检查状态、Content-Type 和内容，确认未将 JSON 错误或文本日志保存为 Excel |
| Docker 构建或健康检查失败 | 先看 [Docker 当前限制](DOCKER.md)，避免按旧文档假定配置已完整验证 |

部署见[部署与运维](DEPLOYMENT.md)，回归测试见[测试与开发](TESTING.md)。
