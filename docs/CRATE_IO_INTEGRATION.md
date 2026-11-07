# crates.io 集成

## 批量输入

CLI 和 API 使用同一类 crate 识别规则：

```text
https://crates.io/crates/zerocopy/0.6.1
https://crates.io/api/v1/crates/zerocopy/0.6.1/download
pkg:cargo/zerocopy@0.6.1
```

标准输入仍为 `name, version, github_url, package_manager, serial_number`，`package_manager` 可填 `cargo`。纯名称支持属于内部 `process_crate_repository` 的解析能力，批量输入建议使用明确 URL 或 purl，避免触发通用仓库发现。

## 下载链接归一

[`process_crate_repository`](../core/crate_utils.py) 将带版本的 API `/download` 链接转换为：

```text
https://crates.io/crates/<crate_name>/<version>
```

随后走现有 crate 注册表处理：读取元数据、解析版本、取得该版本许可，必要时分析关联仓库，输出对应 crate 版本页。当前实现没有独立的 `.crate` 下载并解包审核模块；不能将历史设计或临时测试代码描述成现行流程。

## 版本与结果

crate 处理器支持精确匹配、范围/部分匹配及 LLM 匹配，匹配失败可使用注册表默认稳定版本。因此应对照 `input_version`、`resolved_version` 和 `used_default_branch`，不能假定输入版本永远匹配。

`license_files` 使用带实际版本的 crates.io 页面；`license_type` 主要来自版本元数据，其他许可证/版权信息可能来自关联仓库或文本分析。最终仍走共享输出组装和版权空值填充，字段说明见[输入与输出](INPUT_OUTPUT.md)。

内部调用示例（真实联网，可能调用模型）：

```python
import asyncio
from core.crate_utils import process_crate_repository

async def main():
    result = await process_crate_repository(
        'https://crates.io/api/v1/crates/zerocopy/0.6.1/download', '0.6.1'
    )
    print(result['resolved_version'], result['license_type'], result['license_files'])

asyncio.run(main())
```

## 回归与排障

```bash
uv run python -m pytest -q \
  test/test_log_rotation_and_crate_routing.py \
  test/test_crate_license_link.py \
  test/test_crate_copyright_fallback.py
```

查看主日志、`substep.log` 和 `version_resolve.log`。URL 末尾没有归档后缀的 `/download` 依靠 crate 专用识别处理，不能要求通用后缀归档判断自动识别它。
