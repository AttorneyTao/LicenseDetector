# LicenseDetector

LicenseDetector 批量分析开源组件的许可证、版权声明和许可证据，支持本地 Excel、HTTP API 和 Web 页面。输入可以是源码仓库、包管理器页面、发布包下载地址或 Package URL（purl）；输出为 Excel。

审核会使用注册表元数据、发布包内文件、仓库中的 LICENSE/README，以及 LLM 内容分析。各生态有独立的版本和证据规则，详见[审核流程](docs/AUDIT_FLOW.md)。`status=success` 表示处理成功；判断是否取得许可及是否匹配请求版本，还需查看 `concluded_license`、`resolved_version`、`license_files` 和判定理由。

## 快速开始

需要 Python **3.13+**，从项目根目录执行：

```bash
uv sync
```

手动创建 `.env`；仓库目前没有 `.env.example`。替换以下占位符：

```dotenv
GITHUB_TOKEN=your_github_token
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_key
USE_LLM=true
```

也可设置 `LLM_PROVIDER=qwen` 和 `DASHSCOPE_API_KEY`。模型名称、并发数及配置边界见[配置说明](docs/CONFIGURATION.md)。

在 `input.xlsx` 第一行填写下列表头，每行一个组件：

| name | version | github_url | package_manager | serial_number |
| --- | --- | --- | --- | --- |
| requests | 2.31.0 | https://pypi.org/project/requests/ | pypi | 00001 |
| serde | 1.0.197 | https://crates.io/crates/serde | cargo | 00002 |
| commons-lang3 | 3.12.0 | pkg:maven/org.apache.commons/commons-lang3@3.12.0 | maven | 00003 |

`github_url` 为必要定位字段，其余字段可选。流水号须在 Excel 中按**文本**录入，以保留前导零和长编号；完整字段规则见[输入与输出](docs/INPUT_OUTPUT.md)。

```bash
# 分析当前目录的 input.xlsx
uv run main.py

# API 与 Web 页面
uv run main.py --api --host 127.0.0.1 --port 8000

# 全表为字体时使用独立 CLI 模式
uv run main.py --font
```

API 启动后访问 `http://127.0.0.1:8000/`，交互文档位于 `/docs`，健康检查为 `/health`。

```bash
curl --fail-with-body -X POST http://127.0.0.1:8000/api/v1/analyze-and-download \
  -F 'file=@input.xlsx' --output result.xlsx
```

CLI 将结果保存为 `outputs/output_YYYY-MM-DD_HH-MM-SS.xlsx` 和 `outputs/output_latest.xlsx`，工作表名为“分析结果”。API 下载接口直接返回附件，邮件接口向请求中指定的邮箱发送结果。临时 CSV 仅用于保留中间结果，当前没有自动恢复批次的入口。

## 支持的来源

| 来源 | 入口示例或定位方式 |
| --- | --- |
| GitHub | 仓库根、`tree/<ref>`、`blob/<ref>/...` |
| npm | `https://www.npmjs.com/package/@scope/name`、npm purl |
| PyPI | `https://pypi.org/project/name/`；经发布记录核对的 `files.pythonhosted.org` 源码包/wheel |
| Rust crates.io | 包页面、带版本的 API `/download` 链接、cargo purl |
| Maven | mvnrepository、Central/Sonatype、Maven 源站和可解析的私服地址 |
| Go | `pkg.go.dev`、受支持的 Go 模块定位方式、golang purl |
| Dart/Flutter | `pub.dev/packages/...`、pub purl |
| NuGet | `nuget.org/packages/...`、nuget purl |
| Debian/Ubuntu | sources.debian.org、Launchpad、deb purl |
| R | CRAN、Bioconductor 包页面及对应源码包 |
| Conda | anaconda.org、api.anaconda.org、conda.anaconda.org 的包/版本/构建链接 |
| Hugging Face | 模型及 datasets 仓库、tree/blob/resolve 链接；只读取元数据与许可文本，不下载权重 |
| 直接发布包 | 支持后缀的归档下载链接，当前置流程未取得许可时读取包内文件 |
| 字体 | GitHub、Google Fonts、Fontshare、猫啃网及其他可抓取来源，使用 `--font` |

具体识别形式和失败处理以[审核流程](docs/AUDIT_FLOW.md)为准。通用下载域名、同名项目或 monorepo 根许可证不能单独证明某个文件的许可适用性。

## 结果阅读

- `input_url` 保留原输入；`input_version` 和 `resolved_version` 分别记录请求与实际版本。
- `concluded_license` 保留许可证表达式；紧随其后的 `concluded_license_semicolon` 提供无分组括号、去重的分号展示，`AND Others` 提示保持原样。
- `risk_level` 根据 `license_risk.yaml` 与去重后的项目逐项匹配，项目内 `OR` 取较低的已知风险，用分号对应列示。
- `license_files` 优先采用可确认的许可证网页或含许可的页面；包下载地址作为没有合适网页时的兜底。最终链接优化只调整链接，不改许可结论。
- 最终输出中仍为空的 `copyright_notice` 填为 `Copyright <input_name> Original author and authors`。该文本是系统兜底内容，应与从原文提取的声明区分。

## 部署与维护

当前 [GitHub Actions](.github/workflows/deploy.yml) 在推送到 `master` 时使用 self-hosted runner 自动拉取代码、同步依赖并重启 `licensedetector.service`。**文档提交也会触发该工作流**，没有按路径过滤或自动化测试前置步骤。推送成功与服务部署成功须分别核对；详见[部署与运维](docs/DEPLOYMENT.md)。

仓库另附 Docker 配置，其构建文件与近期依赖存在已知差异，使用前先阅读 [Docker 说明](docs/DOCKER.md)。

## 文档索引

| 文档 | 内容 |
| --- | --- |
| [快速开始](docs/QUICKSTART.md) | 安装、CLI、Web、下载、邮件与常见问题 |
| [输入与输出](docs/INPUT_OUTPUT.md) | Excel 表头、purl、版本、输出字段、分号与风险 |
| [审核流程](docs/AUDIT_FLOW.md) | 生态分发、证据来源、归档及 purl 回退、链接选择 |
| [API 使用](docs/API_USAGE.md) | 所有应用端点、请求格式、流式响应与错误处理 |
| [配置说明](docs/CONFIGURATION.md) | 模型、缓存、网络、邮件和代码配置 |
| [部署与运维](docs/DEPLOYMENT.md) | Actions 自动部署、systemd、日志及排障 |
| [Docker](docs/DOCKER.md) | Compose 用法、构建前检查及当前限制 |
| [测试与开发](docs/TESTING.md) | 代码结构、离线回归、真实联网验证边界 |
| [crates.io](docs/CRATE_IO_INTEGRATION.md) | 下载链接归一与标准 crate 流程 |
| [字体模式](docs/FONT_SCANNING.md) | 字体站点适配、CLI 范围和联网抽样 |
| [Hugging Face](docs/HUGGINGFACE.md) | 模型/数据集、版本固定及禁止权重下载 |
| [变更记录](docs/CHANGES.md) | 有代码/提交依据的近期变更 |

本项目采用 [Apache License 2.0](LICENSE)。
