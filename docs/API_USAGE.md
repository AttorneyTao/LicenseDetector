# API 使用

## 启动与输入

从根目录执行 `uv run main.py --api --host 127.0.0.1 --port 8000`。未指定 host 时默认监听 `0.0.0.0`；`/` 跳转到 `/static/index.html`，自动生成文档在 `/docs` 和 `/redoc`。

批量接口接受 `multipart/form-data` 的 Excel 附件，不是 JSON 组件数组。首行字段为 `name, version, github_url, package_manager, serial_number`，具体必填与透传规则见[输入与输出](INPUT_OUTPUT.md)。当前上传接口没有字体模式开关。

应用没有内置用户认证和持久化作业队列。请求中执行审核，进程重启、连接中断和超时不能按后台持久任务的语义处理。公网部署需由现有网关控制访问；详见[部署与运维](DEPLOYMENT.md)。

## 端点总览

| 方法 | 路径 | 请求 | 实际响应 |
| --- | --- | --- | --- |
| GET | `/` | 无 | 重定向至前端 |
| GET | `/health` | 无 | API 存活 JSON |
| GET | `/api/v1/logs/live` | 无 | 全局日志 SSE，持续连接 |
| POST | `/api/v1/detect-license` | JSON：`text`、可选 `source_url` | 文本许可识别 JSON |
| POST | `/api/v1/analyze` | `file`、`email` 及可选邮件参数 | 等待处理并发送邮件后返回 JSON |
| POST | `/api/v1/analyze-and-download` | `file` | 等待处理后返回 Excel 附件 |
| POST | `/api/v1/analyze-stream` | `file` | 文本日志流，不导出最终 Excel |
| POST | `/api/v1/analyze-stream-email` | `file`、`email` 及可选邮件参数 | 文本日志流，完成后发送邮件 |
| POST | `/api/v1/analyze-stream-download` | `file` | 当前实现等待完成后返回 Excel 附件，不在响应中混合日志 |

前端由 FastAPI 静态目录提供；应用端点及实现见 [`api.py`](../api.py)。

## 直接下载

```bash
curl --fail-with-body -X POST http://127.0.0.1:8000/api/v1/analyze-and-download \
  -F 'file=@input.xlsx' --output result.xlsx
```

成功返回 `application/vnd.openxmlformats-officedocument.spreadsheetml.sheet`，附件名为 `output_YYYYMMDD_HHMMSS.xlsx`，工作表为“分析结果”。输入临时文件在请求结束时清理，输出附件在发送完成后由后台清理任务删除。

`analyze-stream-download` 也返回同类附件。不要按名字推断它提供日志与二进制文件混合协议；需要同时看日志时可另开 `/api/v1/logs/live`。

```python
import requests

with open('input.xlsx', 'rb') as source:
    response = requests.post(
        'http://127.0.0.1:8000/api/v1/analyze-and-download',
        files={'file': ('input.xlsx', source)},
        timeout=(10, 3600),
    )
response.raise_for_status()
content_type = response.headers.get('content-type', '')
if 'spreadsheetml.sheet' not in content_type:
    raise RuntimeError(f'Unexpected response type: {content_type}')
with open('result.xlsx', 'wb') as target:
    target.write(response.content)
```

示例超时只是客户端配置，需按批量规模和反向代理超时共同调整。

## 邮件

`analyze` 和 `analyze-stream-email` 都接受：

| 参数 | 必填 | 说明 |
| --- | --- | --- |
| `file` | 是 | Excel 附件 |
| `email` | 是 | 收件邮箱，代码仅做基本 `@` 校验 |
| `smtp_server` | 否 | 覆盖环境中的 SMTP 地址 |
| `smtp_port` | 否 | 覆盖环境中的 SMTP 端口 |
| `email_subject` | 否 | 自定义邮件主题 |

发送账号与密码由服务端配置，见[邮件配置](CONFIGURATION.md)。请求参数没有发送账号、密码或 TLS 模式开关。

```bash
curl --fail-with-body -X POST http://127.0.0.1:8000/api/v1/analyze \
  -F 'file=@input.xlsx' -F 'email=recipient@example.com' \
  -F 'email_subject=开源组件审核结果'
```

非流式响应字段为 `status`、`message`、`processed_rows`、`email_sent`、`email`、`timestamp`。**须检查 `email_sent`**：邮件发送函数返回 False 时，接口仍可返回 HTTP 200 和 `status=success`，message 会提示邮件失败；配置初始化等异常也可能直接返回 HTTP 500。

## 两类日志流

### 请求级文本流

```bash
curl --no-buffer -X POST http://127.0.0.1:8000/api/v1/analyze-stream \
  -F 'file=@input.xlsx'

curl --no-buffer -X POST http://127.0.0.1:8000/api/v1/analyze-stream-email \
  -F 'file=@input.xlsx' -F 'email=recipient@example.com'
```

响应为 `text/plain`，逐行输出 `[START]`、`[INFO]`、`[PROGRESS]`、`[ERROR]`、`[SUCCESS]` 等文本。流开始后出现的错误通过文本报告，HTTP 状态可能仍为 200；应读取到流结束并检查错误及邮件结果，不能只看响应头。

`analyze-stream` 会执行审核，但不调用最终 Excel 导出和邮件发送；要获取结果，提交到对应下载或邮件端点。不要为“先看进度再下载”重复提交同一批；可使用一次下载请求加独立全局日志监视。

### 全局 SSE

```bash
curl --no-buffer http://127.0.0.1:8000/api/v1/logs/live
```

消息形如 `data: {"job_id":"...","type":"info","message":"...","timestamp":"..."}`，以空行分隔，并有心跳注释。每条关联作业的消息带 `job_id`；系统消息使用 `system`。最多 10 个订阅连接，超出返回 503。

该日志流包含当前进程的全局 API 活动，没有按用户隔离；日志处理器也并非严格的作业事件总线。不要把全局完成提示作为 Excel 已正确导出或邮件发送成功的唯一依据。订阅状态与广播器在进程内，多 worker 部署不会自动共享。

## 单段文本识别

```bash
curl --fail-with-body -X POST http://127.0.0.1:8000/api/v1/detect-license \
  -H 'Content-Type: application/json' \
  -d '{"text":"Licensed under the MIT License.","source_url":"https://example.org/project/LICENSE"}'
```

`source_url` 用于追溯，本接口将 `text` 交给许可分析函数，不据此下载该网址或证明文本属于某个组件。返回结构取决于分析结果，可含 `spdx_expression`、`licenses`、`bundled_licenses`、双许可关系及置信度；不保证识别一定成功。

## 错误与集成注意事项

| 情况 | 处理 |
| --- | --- |
| 缺少 Form/JSON 必填字段、字段类型错误 | FastAPI 通常返回 422 |
| 无法读取 Excel、非流式邮件邮箱无效、识别文本为空 | 400 |
| 运行时异常 | 500，JSON 通常含 `detail` |
| SSE 超出连接数 | 503 |
| 行级审核失败 | 可能仍正常返回 Excel，在 `status`、`error` 查看 |
| 流开始后的业务异常 | 检查流中的 `[ERROR]`，不能仅依赖 HTTP 200 |

`/health` 只返回存活信息，不探测 GitHub、模型额度、注册表或 SMTP。API 默认每个批次的组件并发为 5，并非全服务统一的 5 个请求上限。并行提交多个批次会放大模型和网络请求量。
