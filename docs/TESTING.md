# 测试与开发

## 代码位置

| 文件/目录 | 职责 |
| --- | --- |
| `main.py` | CLI、字体入口、本地批次分发与输出保存 |
| `api.py` | Web、HTTP、上传/下载、邮件及流式日志；API 批次分发 |
| `core/github_utils.py` | 仓库访问、版本/路径解析、purl 入口、共享仓库处理 |
| `core/*_utils.py` | 各包管理器、字体、邮件等专用逻辑 |
| `core/archive_utils.py` | 下载、解包、包内目录分析与生命周期 |
| `core/r_utils.py`、`core/conda_utils.py` | R 网页证据、Conda 版本/构建范围 |
| `core/purl_fallback.py` | 最后一次按已知坐标重试，控制是否采用候选结果 |
| `core/license_links.py` | 最终证据链接优化，仅修改链接 |
| `core/utils.py` | Excel 读取、许可表达式、风险、版权兜底等共享函数 |
| `core/llm_provider.py`、`core/llm_service.py` | provider 抽象及统一模型调用 |
| `core/llm_cache.py`、`core/llm_policies.py`、`core/llm_metrics.py` | 缓存、任务响应校验与统计 |
| `core/logging_utils.py` | 分类日志、轮转及保留 |
| `static/` | Web 前端 |
| `test/` | 回归测试、固定样本和部分真实联网验证脚本 |

CLI 与 API 的分发及输出组装存在平行实现。修改其中一处后应检查另一处；仅测试处理器返回值不能证明 Excel 上传和最终导出保持一致。

## 安装与离线回归

从根目录执行 `uv sync`。以下为使用 mock/固定证据的重点回归集合，不启动真实批次或发送邮件：

```bash
uv run python -m pytest -q \
  test/test_pythonhosted_reference.py \
  test/test_pypi_entrypoints.py \
  test/test_pypi_evidence_boundaries.py \
  test/test_pypi_github_url.py \
  test/test_archive_utils.py \
  test/test_purl_fallback.py \
  test/test_purl_fallback_regression.py \
  test/test_license_links.py \
  test/test_versioned_registry_fallback.py \
  test/test_r_utils.py \
  test/test_conda_utils.py \
  test/test_conda_spdx.py \
  test/test_input_metadata.py \
  test/test_output_copyright_fallback.py \
  test/test_output_license_expression.py \
  test/test_risk_level.py \
  test/test_log_rotation_and_crate_routing.py
```

其他主题可单独选择：

| 主题 | 测试文件 |
| --- | --- |
| Maven POM、父属性、来源边界 | `test_maven_analysis.py`、`test_maven_output.py`、`test_maven_parent_properties.py`、`test_maven_source_boundaries.py` |
| npm 版本、仓库与版权年份 | `test_npm_utils.py`、`test_npm_repo_license_conflict.py`、`test_npm_copyright_year.py` |
| crate 证据链接与版权 | `test_crate_license_link.py`、`test_crate_copyright_fallback.py` |
| 模型缓存和调用统计 | `test_llm_cache.py`、`test_llm_metrics.py`、`test_empty_llm_inputs.py` |
| 许可正文及选择 | `test_license_text.py`、`test_github_license_reuse.py`、`test_component_license_locator.py` |

路径均位于 `test/`。报告应分别记录 passed、failed、xfailed、deselected；预期失败和跳过不计为通过，固定通过数量也不应写成后续版本的验收条件。

仓库没有统一的 live 测试标记覆盖所有历史测试。运行 `pytest test/` 前应检查目标是否包含真实网络和模型调用。Makefile 的 `make test`、`make verify` 引用根目录旧脚本，当前有效的安装检查脚本位于 `test/verify_installation.py`；常规回归优先显式选择测试文件。

## 真实联网验证

`test/verify_crate_integration.py`、`test/verify_versioned_registry_fallback.py` 和 `test/test_font_adapters.py` 等脚本涉及真实来源或输入样本。运行前阅读脚本确认目标、凭据、输出路径和调用成本，不将它们混入默认离线检查。

验证单个输入时，先关联原输入、实际版本、结果行和日志。使用指定小样本即可完成的验证，不自行扩展成服务器批次重跑。下载校验或 PyPI 元数据核对也不等于已重新完成整个包的许可审核。

## 回退测试重点

- 正确包、正确版本、明确许可：保留现有证据，不启动不必要的回退。
- 请求版本与文件/元数据冲突：不把其他包或版本当作已确认结果。
- GitHub 默认分支与历史发布许可不同：按各生态规则选择，保留实际版本与原因。
- 网络错误、空许可、未知许可、找不到仓库：验证回退次数有限且失败不覆盖原结果。
- purl 输入和最后 purl 重试：避免递归、死锁及较差证据替换。
- 最终链接优化：保持许可证、版权和版本字段，只调整验证过的链接。
- 输出：原流水号和包管理器透传，原输入保留，分号去重和风险逐项对应，版权仅对最终空值填充。
- R/Conda：网页/构建范围不明时不挪用其他发布证据，归档复用共享生命周期。

已有失败先在修改前代码或对应基线复现。修复范围之外的问题单独记录，避免借回归测试改动无关规则。

## 文档和发布检查

文档更新应核对本地链接、API 实际签名、CLI `--help`、配置读取点及测试文件。不要把计划、临时补丁或旧测试结果写成当前实现，也不要在示例放入真实凭据、收件邮箱或业务输入。

```bash
git diff --check
git diff --name-only
uv run main.py --help
```

文档专属提交只应包含 Markdown；推送 `master` 仍会触发现有自动部署。当前 Actions 未配置测试步骤，发布前验证须另行完成，部署核对见 [DEPLOYMENT.md](DEPLOYMENT.md)。
