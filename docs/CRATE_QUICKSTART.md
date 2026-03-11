# Crates.io 集成 - 快速开始

## 📦 新增功能

本项目现已支持 **Rust crates.io** 包的分析功能！

## 🚀 快速使用

### 1. 准备输入文件

在 `input.xlsx` 中添加 Rust crate 信息：

| name | github_url | version |
|------|-----------|---------|
| serde | https://crates.io/crates/serde | |
| tokio | https://crates.io/crates/tokio | 1.0 |
| reqwest | crates.io/crates/reqwest | 0.11.0 |
| serde_json | https://crates.io/crates/serde_json | 1.0.0 |

### 2. 运行分析

```bash
python main.py
```

### 3. 查看结果

输出文件位于 `outputs/output_*.xlsx`，包含以下信息：

- ✅ 许可证类型（MIT、Apache-2.0、BSD 等）
- ✅ 版权声明
- ✅ GitHub 仓库地址
- ✅ 版本信息
- ✅ 第三方依赖目录

## 📋 支持的输入格式

```
# 仅 crate 名称
serde

# crates.io URL
https://crates.io/crates/tokio

# 带版本的 URL
https://crates.io/crates/reqwest/0.11.0

# 简写 URL
crates.io/crates/serde
```

## 🔍 测试功能

运行测试脚本验证安装：

```bash
python test_crate.py
```

## 📖 详细文档

完整的使用说明和 API 文档请查看：
- [CRATE_IO_INTEGRATION.md](./CRATE_IO_INTEGRATION.md)

## 💡 示例代码

```python
from core.crate_utils import process_crate_repository
import asyncio

async def analyze():
    # 分析单个 crate
    result = await process_crate_repository("serde", "1.0.0")
    
    print(f"Crate: {result['component_name']}")
    print(f"Version: {result['resolved_version']}")
    print(f"License: {result['license_type']}")
    print(f"Repository: {result['repo_url']}")

asyncio.run(analyze())
```

## 🎯 主要特性

- **自动版本解析** - 支持 semver、范围匹配、智能推断
- **许可证分析** - 多源验证（元数据、README、LICENSE 文件）
- **GitHub 集成** - 自动关联并获取仓库详细信息
- **版权提取** - 智能识别版权声明
- **异步高并发** - 支持批量处理

## ⚙️ 配置

无需额外配置，直接使用现有的环境变量和 LLM 配置。

如需禁用 LLM 版本解析：

```bash
export USE_LLM=false
python main.py
```

## 📝 注意事项

1. 确保网络可以访问 `crates.io` 和 `github.com`
2. 某些 crate 可能没有关联的 GitHub 仓库
3. 建议使用最新的 Python 3.x 版本

---

**Happy analyzing! 🎉**
