# Crates.io 集成指南

## 概述

本项目现已支持 Rust crates.io 包的分析功能，可以自动从 crates.io 获取包的许可证、版权、仓库等信息。

## 功能特性

### 支持的输入格式

`crate_utils.py` 支持以下格式的输入：

1. **纯 crate 名称**
   ```
   serde
   tokio
   reqwest
   ```

2. **crates.io URL**
   ```
   https://crates.io/crates/serde
   https://crates.io/crates/tokio/1.0.0
   ```

3. **带版本号的 crate**
   ```
   serde@1.0.0
   tokio@1.0
   ```

### 主要功能

- ✅ **自动版本解析**：支持精确匹配、范围匹配（如 `1.x`）、部分匹配和 LLM 智能匹配
- ✅ **许可证分析**：从 crates.io 元数据、GitHub 仓库、README 中提取许可证信息
- ✅ **版权声明提取**：自动识别和构造版权声明
- ✅ **GitHub 仓库关联**：如果 crate 有关联的 GitHub 仓库，会自动获取更详细的信息
- ✅ **第三方目录检测**：分析包中的第三方依赖目录

## 使用方法

### 方法一：通过 main.py 批量处理

在 `input.xlsx` 中添加 crates.io 的 URL 或 crate 名称，然后运行：

```bash
python main.py
```

示例 `input.xlsx` 内容：

| name | github_url | version |
|------|-----------|---------|
| serde | https://crates.io/crates/serde | |
| tokio | https://crates.io/crates/tokio | 1.0 |
| reqwest | crates.io/crates/reqwest | 0.11.0 |

### 方法二：单独调用 API

```python
from core.crate_utils import process_crate_repository
import asyncio

async def analyze_crate():
    # 方式 1: 使用 crate 名称
    result = await process_crate_repository("serde", "1.0.0")
    
    # 方式 2: 使用完整 URL
    result = await process_crate_repository(
        "https://crates.io/crates/tokio", 
        "1.0"
    )
    
    print(result)

asyncio.run(analyze_crate())
```

### 方法三：使用测试脚本

运行提供的测试脚本：

```bash
python test_crate.py
```

## 输出字段说明

`process_crate_repository` 返回的字典包含以下字段：

| 字段名 | 说明 |
|--------|------|
| `input_url` | 输入的 URL 或 crate 名称 |
| `repo_url` | GitHub 仓库地址（如果有） |
| `component_name` | crate 名称 |
| `resolved_version` | 解析后的版本号 |
| `input_version` | 用户输入的版本号 |
| `used_default_branch` | 是否使用了默认版本 |
| `license_type` | 许可证类型 |
| `license_files` | 许可证文件 URL |
| `license_analysis` | 详细的许可证分析结果 |
| `has_license_conflict` | 是否存在许可证冲突 |
| `readme_license` | 从 README 中提取的许可证 |
| `license_file_license` | 从 LICENSE 文件中提取的许可证 |
| `copyright_notice` | 版权声明 |
| `status` | 处理状态（success/error） |
| `homepage` | 项目主页 |
| `documentation` | 文档地址 |
| `readme` | README 内容（前 5000 字符） |

## 版本解析逻辑

系统会按照以下顺序尝试解析版本：

1. **精确匹配**：忽略大小写和 'v' 前缀
   - `1.0.0` 匹配 `1.0.0`
   - `v1.0.0` 匹配 `1.0.0`

2. **范围匹配**：支持 `x` 通配符
   - `1.x` 匹配 `1.x.x` 系列
   - `1.2.x` 匹配 `1.2.x` 系列

3. **部分匹配**：子字符串匹配
   - `1.0` 可能匹配 `1.0.0`

4. **LLM 智能匹配**：如果以上都失败，使用大语言模型推断最可能的版本

## API 端点说明

crates.io 提供以下公开 API：

- `GET https://crates.io/api/v1/crates/{crate_name}` - 获取 crate 基本信息
- `GET https://crates.io/api/v1/crates/{crate_name}/{version}` - 获取特定版本信息
- `GET https://crates.io/api/v1/crates/{crate_name}/versions` - 获取所有版本
- `GET https://crates.io/api/v1/crates/{crate_name}/owners` - 获取维护者信息

## 与 npm_utils 的对比

`crate_utils.py` 参照 `npm_utils.py` 设计，保持了相似的接口和行为：

| 特性 | npm_utils | crate_utils |
|------|-----------|-------------|
| 包管理器 | npm | cargo (crates.io) |
| 注册表 | registry.npmjs.org | crates.io |
| 包格式 | @scope/package | crate_name |
| 版本格式 | semver | semver |
| 许可证字段 | license | license |
| 仓库字段 | repository | repository |

## 注意事项

1. **网络连接**：需要能够访问 crates.io API 和 GitHub
2. **LLM 配置**：版本解析失败时需要 LLM API key（可选）
3. **错误处理**：所有异常都会被捕获并返回错误状态
4. **异步支持**：完全异步实现，支持高并发处理

## 故障排除

### 常见问题

**Q: 提示 "CrateAPIError: HTTP Error 404"**
A: crate 名称不存在或拼写错误

**Q: 版本解析不正确**
A: 检查输入版本格式，或尝试提供完整版本号

**Q: 无法获取 GitHub 仓库信息**
A: 某些 crate 可能没有关联 GitHub 仓库，这是正常的

### 日志查看

查看相关日志：

```bash
# 查看 crate 相关日志
tail -f logs/crate.log

# 查看版本解析日志
tail -f logs/version_resolve.log

# 查看 LLM 交互日志
tail -f logs/llm_interaction.log
```

## 贡献

如需添加更多功能或修复问题，请参考现有的 `npm_utils.py` 实现模式。

## 参考资料

- [crates.io 官方文档](https://crates.io/)
- [crates.io API 文档](https://crates.io/data-access)
- [Rust Cargo 包管理](https://doc.rust-lang.org/cargo/)
