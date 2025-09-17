# GitHub Repository License Analyzer

一个强大的开源许可证分析工具，专为GitHub仓库设计，支持复杂场景下的许可证识别和合规审查。

## 功能特性

### 🔍 核心功能
- **智能许可证识别**: 自动识别GitHub仓库的主许可证类型（SPDX标准）
- **双重许可证检测**: 识别并分析双重许可证关系（AND/OR）
- **第三方组件分析**: 发现和定位第三方依赖许可证信息
- **版权声明提取**: 自动提取或构造版权声明
- **多URL支持**: 处理GitHub URL和包管理器URL（npm、PyPI、NuGet、Go模块）
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
- `github_url`: GitHub仓库URL或包管理器URL
- `version`: （可选）指定分析的版本
- `name`: （可选）组件名称


### 4. 运行分析

```bash
python main.py
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
- `logs/substep.log`: 分步骤执行详情
- `logs/repository_trees.log`: 仓库结构信息

## 高级功能

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
