# 输入与输出

## 标准 Excel 输入

CLI 读取当前目录 `input.xlsx`；所有上传 Excel 的 API 端点使用同一个 `read_input_excel` 函数。读取默认首张工作表，第一行为字段名，未实现业务表中文表头、合并标题或多行表头的自动转换。

| 字段 | 是否必要 | 含义 |
| --- | --- | --- |
| `github_url` | 是 | 原始定位值：仓库 URL、受支持的注册表 URL、发布包地址或 purl |
| `name` | 否 | 输入组件名称；建议填写，供最终 purl 回退和版权兜底使用 |
| `version` | 否 | 请求版本；建议将版本单元格设为文本，避免 `1.10` 被 Excel 改写 |
| `package_manager` | 否 | 包管理器原值透传；受支持的值还可供最终 purl 回退构造坐标 |
| `serial_number` | 否 | 原流水号，按文本读取并透传，不参与许可判定 |

建议列顺序为 `name, version, github_url, package_manager, serial_number`。不要在数据区放备注行或第二套表头。原三列格式继续适用；可选标识字段缺失时输出为空。读取函数会对已存在的 `package_manager`、`serial_number` 列再按文本读取，但无法恢复文件写入前已经丢失的数字。

`package_manager` 通常不会覆盖原 URL 的生态识别；R 处理器可使用明确的 `cran`、`bioconductor`、`bioc` 和合法 `name` 提供定位。Conda 仅按支持的 URL 识别，单填 `conda` 不会把任意 URL 改送 Conda 流程。

## purl 输入

直接在 `github_url` 填写 purl，不需要另建列。映射代码位于 [`resolve_input_ref`](../core/github_utils.py) 和 [`purl_to_url`](../core/utils.py)。

| purl | 注册表或仓库入口 |
| --- | --- |
| `pkg:npm/%40babel/core@7.24.0` | `https://www.npmjs.com/package/@babel/core` |
| `pkg:pypi/requests@2.31.0` | `https://pypi.org/project/requests` |
| `pkg:maven/org.apache.commons/commons-lang3@3.12.0` | `https://mvnrepository.com/artifact/org.apache.commons/commons-lang3` |
| `pkg:golang/github.com/gin-gonic/gin@v1.9.1` | `https://pkg.go.dev/github.com/gin-gonic/gin` |
| `pkg:cargo/serde@1.0.197` | `https://crates.io/crates/serde` |
| `pkg:pub/http@1.2.0` | `https://pub.dev/packages/http` |
| `pkg:nuget/Newtonsoft.Json@13.0.3` | `https://www.nuget.org/packages/Newtonsoft.Json` |
| `pkg:github/torvalds/linux@v6.1` | `https://github.com/torvalds/linux` |
| `pkg:deb/debian/curl@8.21.0-2` | `https://sources.debian.org/src/curl/8.21.0-2/` |

purl 内明确的名称和版本优先于输入列，并记录冲突日志。未覆盖的类型优先使用其 `download_url` 或 `vcs_url` qualifier；没有可映射地址时透传至既有处理逻辑，不能据此宣称已支持该生态。当前没有 Conda/CRAN 专用 purl 映射；使用其网页或发布包地址。

## 版本含义

`input_version` 记录请求，`resolved_version` 记录实际采用的版本或仓库引用；`used_default_branch` 在不同生态中也用于表示使用默认版本，不仅表示 Git 分支。各生态可能保留 latest/default 回退，详见[审核流程](AUDIT_FLOW.md)。请求版本不可用时不会统一终止为 Unknown。

PyPI 下载文件另有入口核对：只将域名准确为 `files.pythonhosted.org` 的 HTTPS 地址作为候选，解析文件名后核对对应版本 JSON 中的包名、版本、文件名和下载路径。输入版本冲突或核对失败时保留原流程。

## 输出位置与列顺序

CLI 保存 `outputs/output_YYYY-MM-DD_HH-MM-SS.xlsx` 及 `outputs/output_latest.xlsx`；工作表为“分析结果”。API 下载返回临时附件，发送完成后清理；邮件接口临时附件在请求结束时清理。API 不保证在 `outputs/` 留下同一份文件。

[`RESULT_COLUMNS_ORDER`](../core/config.py) 定义优先顺序。只排列实际存在的字段，额外生态字段追加到末尾，不能假定每个输出都有所有可选列：

```text
serial_number, package_manager, input_url, repo_url, component_name,
input_version, resolved_version, used_default_branch,
concluded_license, concluded_license_semicolon, license_files, risk_level,
copyright_notice, license_type, license_analysis, has_license_conflict,
readme_license, license_file_license, status, license_determination_reason,
license_text, error
```

| 字段 | 阅读方式 |
| --- | --- |
| `input_url` | 保留原输入，包括原始 purl 或下载地址 |
| `repo_url` | 处理器采用的仓库或注册表来源，不保证为 GitHub |
| `component_name` | 处理器识别的名称，可能与输入 `name` 不同 |
| `concluded_license` | 综合主许可、README、许可文件及处理状态生成的表达式 |
| `license_type` / `readme_license` / `license_file_license` | 分别保留处理器主结论和不同证据的许可值 |
| `license_files` | 最终许可证据 URL，也可能是含许可的注册表页面或保留的下载地址 |
| `license_text` | 已取得的证据文本；网页/元数据路径未必包含完整许可证正文，空值也不表示已经读取全文 |
| `copyright_notice` | 提取或构造的版权文本，最终空值会填充通用声明 |
| `license_analysis` | 结构化分析和生态补充信息；Excel 中通常显示为文本形式 |
| `has_license_conflict` | 不同证据中检测到的许可差异，结合版本和理由复核 |
| `license_determination_reason` | 选证据、版本及回退原因 |
| `status` / `error` | 处理状态和失败信息；成功处理与取得明确许可须分开判断 |

`input_name`、`thirdparty_dirs`、第三方许可位置等可能作为额外字段保留。`status` 常见 `success`、`error`、`skipped`；字体另有 `not_found` 等状态。

## 分号展示与风险对应

`concluded_license` 保留原表达式。`concluded_license_semicolon` 将 AND 项展开、去除分组括号，按完整项目忽略大小写和多余空白去重，保留首次出现顺序和写法。OR 组合及 WITH 例外仍属于项目内部内容。`AND Others` 是第三方内容提示，保持原样并且不单独占用风险项。

| 原表达式 | 分号展示 | 风险 |
| --- | --- | --- |
| `MIT AND (Apache-2.0 AND MIT)` | `MIT; Apache-2.0` | `低风险;低风险` |
| `(MIT OR GPL-3.0-only) AND LGPL-2.1-only` | `MIT OR GPL-3.0-only; LGPL-2.1-only` | `低风险;中风险` |

风险映射来自 [`license_risk.yaml`](../license_risk.yaml)。同一 OR 项取能确定的较低风险，均无法匹配时为未知风险；不同去重项目即使风险相同也分别列示。分号列用于展示，不应作为可重新解析、语义无损的 SPDX 原表达式使用。

## 版权和第三方内容

最终输出组装完成后，仅对空值/空白 `copyright_notice` 填充：

```text
Copyright <input_name> Original author and authors
```

已有非空声明保持原值。该兜底文案不证明实际权利人身份；更早的各生态处理器也可能构造版权声明，应结合证据判断。

`thirdparty_dirs` 或 `AND Others` 反映目录/文件中的第三方内容线索，不代表已经完整扫描包管理器依赖图，也不等同于披露了所有传递依赖。

实现与验证：[`core/utils.py`](../core/utils.py)、[`main.py`](../main.py)、[`api.py`](../api.py)、[输入字段测试](../test/test_input_metadata.py)、[表达式测试](../test/test_output_license_expression.py)、[版权兜底测试](../test/test_output_copyright_fallback.py)。
