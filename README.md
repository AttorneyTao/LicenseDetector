# GitHub Repository License Analyzer

一个强大的开源许可证分析工具，专为GitHub仓库设计，支持复杂场景下的许可证识别和合规审查。

## 功能特性

### 🔍 核心功能
- **智能许可证识别**: 自动识别GitHub仓库的主许可证类型（SPDX标准）
- **双重许可证检测**: 识别并分析双重许可证关系（AND/OR）
- **第三方组件分析**: 发现和定位第三方依赖许可证信息
- **版权声明提取**: 自动提取或构造版权声明
- **多URL支持**: 处理GitHub URL和包管理器URL（npm、PyPI、NuGet、Go模块）
- **字体扫描模式**: 通过 `--font` 启动，按字体来源站点（GitHub / Google Fonts / Fontshare / 猫啃网 / 微信公众号等）获取授权与版权，详见 [docs/FONT_SCANNING.md](docs/FONT_SCANNING.md)
- **版本解析**: 支持特定版本分析和默认分支回退
- **冲突检测**: 识别README与License文件中的许可证不一致

### 🤖 AI驱动分析
- **LLM智能分析**: 集成Gemini API进行自然语言理解
- **模糊许可证文本处理**: 识别非标准许可证声明
- **上下文理解**: 理解复杂的许可证关系表达

### 🔧 技术特性
- **异步并发处理**: 支持高效的批量分析
- **完善的日志系统**: 详细的分析过程记录
- **错误恢复机制**: 自动重试和故障处理
- **代理支持**: 适应企业网络环境

## 系统要求

- Python 3.13+
- GitHub API Token
- Gemini API Key（用于LLM分析）
- 网络连接（支持代理配置）

## 前端 Web 界面

项目内置了一个开箱即用的 Web 前端，启动 API 服务后即可通过浏览器使用全部功能。

### 功能一览

| 功能 | 说明 |
|------|------|
| 服务状态监控 | 实时显示后端健康状态（绿色/红色指示灯，30秒自动刷新） |
| 拖拽上传 | 支持拖拽或点击上传 `.xlsx` 文件，自动预览前5行数据 |
| 实时日志 | 分析过程中流式显示服务端日志，按级别着色 |
| 进度条 | 从日志中自动提取进度百分比，动态更新 |
| 下载结果 | 分析完成后一键下载结果 Excel 文件 |
| 邮件发送 | 填写邮箱后，分析完成自动发送结果 |
| 日志工具 | 支持按级别过滤、关键词搜索、复制、导出为 .txt |
| 分析历史 | 本地记录最近20条分析任务（文件名、行数、耗时、状态） |
| 暗色模式 | 点击右上角切换，偏好自动保存 |

### 使用步骤

1. **启动 API 服务**（见下方"快速开始"）
2. **浏览器访问** `http://localhost:8000/`，自动跳转到前端页面
3. **上传文件**：将准备好的 `input.xlsx` 拖入上传区，或点击选择
4. **选择模式**：
   - **下载结果**（默认）：分析完成后点击"下载结果文件"按钮
   - **发送到邮箱**：填写邮箱地址，分析完成自动发送
5. **点击"开始分析"**，实时查看日志和进度

> **注意**：前端通过浏览器直接请求本机 API，确保 API 服务已启动且可访问。

---

## 快速开始

### 1. 环境配置

创建 `.env` 文件并配置必要的环境变量：

```bash
# 必需配置
GITHUB_TOKEN=your_github_token_here
GEMINI_API_KEY=your_gemini_api_key_here

# 可选配置
USE_LLM=true  # 启用/禁用LLM分析
HTTP_PROXY=http://127.0.0.1:7897  # HTTP代理配置
HTTPS_PROXY=http://127.0.0.1:7897  # HTTPS代理配置
DASHSCOPE_API_KEY=your_qwen_api_key  # 备用LLM配置
LLM_CACHE_MODE=read_write  # off / read_only / read_write；默认 read_write
# LLM_CACHE_PATH=/var/lib/licensedetector/llm-cache.sqlite  # 可选，缓存须放在代码目录外
# LLM_CACHE_AUDIT_RATE=0.01  # 已确认缓存命中的抽样实时复核比例
# LLM_CACHE_EPOCH=1  # 修改此值可以让全部旧缓存立即失效
# LLM_CACHE_MODEL_REVISION=  # 模型服务端版本变化时可设置新的修订标识
MAVEN_REPOSITORY_BASE_URLS=https://repo.example.com/repository/releases  # 可选；多个私服根地址用逗号分隔
```

### 2. 安装依赖

```bash
# 使用 uv 安装（推荐）
uv sync

# 或使用 pip 安装
pip install -e .
```

### 3. 准备输入文件

创建 `input.xlsx` 文件，包含以下列：
- `github_url`: GitHub仓库URL、包管理器URL，或 [purl](https://github.com/package-url/purl-spec)（Package URL）
- `version`: （可选）指定分析的版本
- `name`: （可选）组件名称

#### 使用 purl 作为输入

`github_url` 列直接填 purl 即可，无需新增列，系统会自动识别并翻译成对应生态的注册表地址后按原有流程分析：

| purl 示例 | 实际分析的地址 |
| --- | --- |
| `pkg:npm/%40babel/core@7.24.0` | `https://www.npmjs.com/package/@babel/core` |
| `pkg:pypi/requests@2.31.0` | `https://pypi.org/project/requests` |
| `pkg:maven/org.apache.commons/commons-lang3@3.12.0` | `https://mvnrepository.com/artifact/org.apache.commons/commons-lang3` |
| `pkg:golang/github.com/gin-gonic/gin@v1.9.1` | `https://pkg.go.dev/github.com/gin-gonic/gin` |
| `pkg:cargo/serde@1.0.197` | `https://crates.io/crates/serde` |
| `pkg:pub/http@1.2.0` | `https://pub.dev/packages/http` |
| `pkg:nuget/Newtonsoft.Json@13.0.3` | `https://www.nuget.org/packages/Newtonsoft.Json` |
| `pkg:github/torvalds/linux@v6.1` | `https://github.com/torvalds/linux` |
| `pkg:deb/adduser@3.137ubuntu1` | `https://launchpad.net/ubuntu/+source/adduser/3.137ubuntu1` |
| `pkg:deb/debian/curl@8.21.0-2` | `https://sources.debian.org/src/curl/8.21.0-2/` |
| `pkg:generic/openssl@3.0.0?download_url=...` | qualifier 中的 `download_url`（走源码包下载分析） |

说明：

- purl 自带的 `@版本` 与包名会自动回填到 `version` / `name`；与列中已填的值冲突时**以 purl 为准**，并在日志中记录 WARNING。
- 未覆盖的 purl type（如 `pkg:rpm`、`pkg:conan`）会原样透传，走 LLM 查找 GitHub 仓库的兜底逻辑；若 purl 带有 `download_url` 或 `vcs_url` qualifier，则优先使用该地址。
- 普通 URL 输入的行为完全不变。

#### Debian / Ubuntu（deb）包的处理方式

deb 是少数拥有**强制性机器可读许可证元数据**的生态，因此这条路径以确定性解析为主，LLM 只作兜底：

1. **判定发行版**：purl 的 namespace（`pkg:deb/debian/…`）优先，其次看 `?distro=` qualifier，再看版本号特征（含 `ubuntu` 即 Ubuntu）。判定不出或查不到时会自动尝试另一个发行版。
2. **解析源码包与版本**：Ubuntu 走 Launchpad API，Debian 走 sources.debian.org API，同时取到 `component` / `area`。SBOM 里常见的**二进制包名**会映射回源码包（如 `libssl3` → `openssl`）——Ubuntu 经 Launchpad build 资源，Debian 经 ftp-master madison（snapshot.debian.org 备用）。
3. **下载 `debian/copyright` 原文**，每个发行版都配了主源与备用源。
4. **确定性解析 DEP-5**：`Files: *` 段的 `License:` 是主许可证，其余段是内嵌第三方许可证，用 `AND` 合成 SPDX 表达式；`Copyright:` 直接作为版权声明；头部 `Source:` 填入 `repo_url`。Debian 的短名（`GPL-2+`、`Expat`、`BSD-3-clause`…）经映射表转 SPDX，未收录的短名原样保留、风险等级落到"未知"，便于后续补表。只有 copyright 文件不是 DEP-5 格式（老包为自由文本）或解析不出许可证时，才把原文交给 LLM。

`license_files` 输出的是带版本的 copyright 文件地址，可直接作为核查证据。注意 epoch 处理在两边是相反的：Ubuntu 的 pool 路径与 Debian 的 metadata 备用源不含 epoch，而 sources.debian.org 的 data 路径必须保留 epoch。


### 4. 运行分析

**方式一：Web 界面（推荐）**

```bash
# 启动 API 服务 + Web 前端
python main.py --api

# 指定端口（默认 8000）
python main.py --api --port 8080
```

启动后访问 `http://localhost:8000/` 使用图形界面。

**方式二：命令行**

```bash
# 直接处理 input.xlsx（软件包模式），结果保存到 outputs/ 目录
python main.py

# 字体扫描模式：当 input.xlsx 全部为字体时使用
python main.py --font
```

> 字体扫描模式按字体来源站点（GitHub / Google Fonts / Fontshare / 猫啃网 / 微信公众号 等）
> 自动获取授权与版权信息，详见 [docs/FONT_SCANNING.md](docs/FONT_SCANNING.md)。

**方式三：Docker**

```bash
# Web 模式
make docker-api-bg
# 访问 http://localhost:8000/

# CLI 模式
make docker-cli
```

### 5. 查看结果

分析结果将保存在 `outputs/` 目录下：
- `output_latest.xlsx`: 最新分析结果
- `output_YYYY-MM-DD_HH-MM-SS.xlsx`: 带时间戳的结果文件
- `temp/`: 中间结果和备份文件

## 支持的URL类型

本工具支持多种类型的URL输入：

### GitHub URLs
```
https://github.com/owner/repo
https://github.com/owner/repo/tree/branch
https://github.com/owner/repo/tree/tag
https://github.com/owner/repo/blob/branch/path/to/file
```

### 包管理器URLs
```
# NPM
https://www.npmjs.com/package/package-name
npm://package-name

# PyPI
https://pypi.org/project/package-name/
pypi://package-name

# NuGet
https://www.nuget.org/packages/PackageName/
nuget://PackageName

# Go模块
https://pkg.go.dev/module-path
go://module-path

# Maven（mvnrepository、Maven Central 官方前端、repo1 源站、Nexus/Artifactory）
https://mvnrepository.com/artifact/groupId/artifactId[/version]
https://central.sonatype.com/artifact/groupId/artifactId[/version]
https://repo1.maven.org/maven2/group/path/artifactId/version/
```

## 分析流程

系统采用15步渐进式分析流程：

### 🔍 第一阶段：URL处理与验证
1. **URL验证**: 检查URL有效性和GitHub仓库可访问性
2. **URL解析**: 提取仓库所有者、名称和路径信息
3. **仓库信息获取**: 获取仓库基本信息和默认分支
4. **版本解析**: 将指定版本解析为具体的commit/tag/branch

### 📝 第二阶段：许可证信息搜集
5. **GitHub API许可证检查**: 尝试通过GitHub API直接获取许可证信息
6. **仓库树结构分析**: 获取完整的仓库文件结构
7. **README分析**: 搜索并分析README文件中的许可证信息
8. **License文件搜索**: 在指定路径中搜索许可证文件

### 🤖 第三阶段：AI分析与处理
9. **许可证内容分析**: 使用LLM分析许可证文件内容
10. **版权声明提取**: 从许可证文件和README中提取版权信息
11. **许可证冲突检测**: 比较README和许可证文件中的许可证信息

### 🔍 第四阶段：扩展搜索与验证
12. **仓库级搜索**: 在整个仓库中搜索许可证文件
13. **仓库级许可证检查**: 检查仓库级别的许可证信息
14. **第三方许可证检测**: 识别和定位第三方组件许可证
15. **最终分析汇总**: 整合所有信息，生成综合分析结果

## 输出结果说明

分析结果包含以下关键字段：

### 基本信息
- `input_url`: 原始输入URL
- `repo_url`: GitHub仓库URL
- `component_name`: 组件/仓库名称
- `input_version`: 请求的版本
- `resolved_version`: 实际分析的版本
- `used_default_branch`: 是否使用了默认分支

### 许可证信息
- `concluded_license`: 综合判定的最终许可证
- `license_type`: 主许可证类型（SPDX标识符）
- `license_files`: 找到的许可证文件URL列表
- `readme_license`: README中发现的许可证
- `license_file_license`: 许可证文件中的许可证
- `has_license_conflict`: 是否存在许可证冲突

### 双重许可证信息
- `is_dual_licensed`: 是否为双重许可证
- `dual_license_relationship`: 双重许可证关系（AND/OR/none）

### 第三方许可证信息
- `has_third_party_licenses`: 是否包含第三方许可证
- `third_party_license_location`: 第三方许可证位置
- `thirdparty_dirs`: 第三方目录列表

### 其他信息
- `copyright_notice`: 提取的版权声明
- `license_analysis`: 详细的许可证分析结果
- `license_determination_reason`: 许可证判定理由
- `status`: 分析状态（success/error/skipped）
- `error`: 错误信息（如果有）

## 特殊功能详解

### 🔄 双重许可证检测
系统能够理解和分析复杂的双重许可证声明：
- `"Licensed under MIT OR Apache-2.0"` → OR 关系
- `"Dual licensed under MIT and Apache-2.0"` → AND 关系
- `"Available under either MIT or BSD-3-Clause"` → OR 关系


### 📁 第三方许可证检测
智能识别和定位第三方组件的许可证信息：
- 自动发现 `LICENSE-THIRD-PARTY` 文件
- 识别 `third-party/`, `vendor/`, `dependencies/` 目录
- 分析README中的依赖部分
- 识别常见的第三方关键词

### 📋 版权声明处理
自动提取和构造版权声明：
- 从许可证文件中提取现有版权声明
- 从 README 文件中提取版权信息
- 自动构造版权声明（如果未找到）：
  - 使用仓库创建/更新年份
  - 包含组件名称
  - 添加通用版权语句

## 性能与可靠性

### 🚀 高性能并发处理
- **异步并发**: 默认支持20个并发任务
- **智能限流**: 自动处理GitHub API速率限制
- **断点续传**: 定期保存中间结果，支持故障恢复
- **进度追踪**: 实时显示处理进度

### 🔁 错误恢复机制
- **自动重试**: 对于网络错误和API限制自动重试
- **代理回退**: 代理失败时自动尝试直连
- **友好错误处理**: 详细的错误信息和解决建议
- **部分失败容忍**: 单个仓库失败不影响整体进程

## 日志系统

系统生成多个分类日志文件：

- `logs/github_license_analyzer.log`: 主程序日志
- `logs/url_construction.log`: URL处理和解析详情
- `logs/llm_interaction.log`: LLM交互详情和响应
- `logs/llm_cache.log`: 缓存命中、隔离及按任务统计的命中率
- `logs/substep.log`: 分步骤执行详情
- `logs/repository_trees.log`: 仓库结构信息

## 高级功能

### LLM 响应缓存

仅对有明确校验规则的任务缓存完全相同的提示词；任务类型、模型、调用参数和策略版本也参与缓存键。第一次有效回答只记为候选；第二次独立调用得到相同的有效结论后才供后续调用复用。低置信度、空值、无效格式或不在候选列表内的回答不缓存；两次结论冲突时隔离该键，继续实时调用。命中时再次按当前输入校验；默认 1% 命中会实时复核。缓存定期过期，过期记录在后续写入时清理。缓存故障直接回退实时调用，不影响主流程。

默认缓存文件位于 Linux 的 `/var/lib/licensedetector/llm-cache.sqlite` 或 macOS 的 `~/Library/Application Support/LicenseDetector/llm-cache.sqlite`，不在 Git 工作目录。Docker Compose 使用持久化卷。可用 `LLM_CACHE_MODE=off` 立即关闭，或设为 `read_only` 暂停写入。按任务清除示例：`uv run python -m core.llm_cache invalidate --task license_analysis`；清空全部：`uv run python -m core.llm_cache invalidate --all`。调整 `LLM_CACHE_EPOCH` 可使旧键整体失效。缓存文件可能含模型输出，应限制访问和备份范围。

缓存不能从根本上证明模型结论正确；两次一致、字段校验和抽样复核只能降低错误被长期复用的风险。涉及高风险结论时可关闭缓存并人工复核。

每次模型入口调用结束后，`logs/llm_cache.log` 都写入一条 `LLM_CACHE_STATS`：包含任务名、结果（`hit` / `miss` / `audit` / `bypass` / `provider_error`）、该任务与全局的累计请求数、可缓存请求数、实际命中数及命中率。`hit_rate` 的分母是可缓存请求；`global_avoidance_rate` 的分母是所有模型入口请求，表示实际避免模型调用的比例。抽样复核仍调用模型，因此不计为命中。计数按服务器本地日期和进程统计，`run` 字段用于区分服务重启；跨进程或跨日期汇总时应按日志中的逐次结果统计，不要直接累加累计值。

### 配置自定义

在 `core/config.py` 中可以调整：
- `MAX_CONCURRENCY`: 最大并发数（默认20）
- `SCORE_THRESHOLD`: 模糊匹配阈值（默认65）
- `THIRD_PARTY_KEYWORDS`: 第三方目录关键词

### 提示词模板

在 `prompts.yaml` 中自定义LLM提示词：
- `license_analysis`: 许可证分析提示词
- `version_resolve`: 版本解析提示词
- `copyright_extract`: 版权提取提示词
- `github_url_finder`: GitHub URL查找提示词

## 常见问题与解决

### Q: 如何处理API速率限制？
A: 系统自动检测并处理GitHub API速率限制，会等待限制重置后自动继续。

### Q: 如何在企业网络环境中使用？
A: 在 `.env` 文件中配置 `HTTP_PROXY` 和 `HTTPS_PROXY` 即可。

### Q: 如何禁用LLM分析？
A: 在 `.env` 文件中设置 `USE_LLM=false`。

### Q: 如何处理大型仓库？
A: 系统支持并发处理和进度追踪，大型仓库会需要更长时间。

## 📚 完整文档

更多文档见 [`docs/`](docs/README.md) 目录：

| 文档 | 内容 |
|------|------|
| [docs/QUICKSTART.md](docs/QUICKSTART.md) | 快速入门（CLI / API、邮箱配置、故障排除） |
| [docs/FONT_SCANNING.md](docs/FONT_SCANNING.md) | 字体扫描模式（`--font`） |
| [docs/API_USAGE.md](docs/API_USAGE.md) | API 使用指南 |
| [docs/DOCKER.md](docs/DOCKER.md) | Docker 容器化与部署 |
| [docs/CRATE_IO_INTEGRATION.md](docs/CRATE_IO_INTEGRATION.md) | Rust crates.io 集成 |
| [docs/CHANGES.md](docs/CHANGES.md) | 变更记录 |

## 贡献指南

1. Fork 项目仓库
2. 创建功能分支: `git checkout -b feature/new-feature`
3. 提交更改: `git commit -am 'Add new feature'`
4. 推送分支: `git push origin feature/new-feature`
5. 创建 Pull Request

## 许可证

本项目采用 [Apache License 2.0](LICENSE) 许可证。

## 支持和反馈

如果遇到问题或有功能建议，请：
1. 查看日志文件了解详细错误信息
2. 在 GitHub Issues 中提交问题报告
3. 提供输入数据和错误日志以便复现问题

---

*此工具专为开源许可证合规审查设计，适用于法务人员、合规工程师和软件供应链安全分析人员。*
