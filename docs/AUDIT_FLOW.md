# 审核流程与回退规则

本文描述已提交实现中的实际调用路径。入口为 [`main.process_all_repos`](../main.py) 与 [`api._process_repositories`](../api.py)。两者各自维护分发代码，通过回归测试验证一致性；修改生态路由时需同时检查两处。

## 总体顺序

1. 读取原输入，解析 purl 并回填其名称和版本，保留原 `input_url`。
2. 按来源分发到 Conda、R、pub、Go、npm、crate、deb、Maven 或通用仓库处理。PyPI 及 pythonhosted 在共享仓库入口识别并转交 PyPI 处理器。
3. 当前置结果未成功或没有主许可，且输入/处理器提供了受支持归档地址时，尝试共享归档审核。Conda 的归档选择由 Conda 处理器控制，外层不再通用下载兜底。
4. 满足触发条件时，利用输入的包管理器、名称和版本，最后进行一次 purl 重试。
5. 所有许可分析结束后，执行许可证据链接优化；只更新 `license_files`。
6. 生成最终表达式、分号展示、风险和版权兜底列，输出 Excel。

各路径有不同失败语义，不能将上述流程理解为每一行都会依次运行所有生态。字体使用单独的 CLI 分发器，见[字体模式](FONT_SCANNING.md)。

## 各生态的证据与版本

| 路径 | 主要行为与边界 |
| --- | --- |
| GitHub | 解析仓库/子路径，匹配 tag、branch 或 commit；失败时可能使用默认分支。搜集许可文件、README 和第三方目录线索，调用 LLM 分析/选择证据。非 GitHub 通用地址可通过 LLM 查找仓库 |
| npm | 查询 npm 注册表、选择版本并尝试关联仓库，必要时审核发布包。结果须结合包名、版本及证据文件范围判断，不能笼统保证 monorepo 根 LICENSE 适用于所有输入子包 |
| PyPI | 根据选中发布读取该版本的元数据，筛选真实源码仓库；GitHub 未匹配版本时优先保留有许可的该发布元数据。请求版本不存在或历史元数据不可用时保留现有 latest 回退并记录实际版本 |
| pythonhosted | 标准解析 `.tar.gz`/`.zip` 源码包或 `.whl` 文件名，核对确切发布 JSON 的名称、版本和文件路径，再拼成带版本的 PyPI 页面并复用 PyPI 处理器。核对失败时不猜测包身份，继续原流程 |
| crates.io | API `/download` 地址先归一成带版本 crate 页面，复用标准 crate 处理；注册表版本许可为主要证据，必要时合并关联仓库信息。不会建立独立的 `.crate` 下载审核流程 |
| Maven | 从原仓库 POM 及必要的父 POM/属性取许可；POM 无可用许可时再发现源码仓库。最终许可证据统一使用对应版本的 mvnrepository 页面，POM 仍可作为内部核实来源 |
| Go | 从 pkg.go.dev 等定位源码仓库；GitHub 使用默认分支时，可经 Go proxy 校验版本后采用带版本 pkg.go.dev 链接 |
| pub | 优先使用 pubspec 关联 GitHub；仓库未成功匹配版本时读取 pub 发布包 |
| NuGet | 对受支持的 nuget.org 入口查询 NuGet 包许可；没有可用结果时继续既有仓库发现路径 |
| Debian/Ubuntu | 解析源码包及版本，必要时由二进制名映射源码包；读取版本化 `debian/copyright`，优先解析 DEP-5，自由文本/无法解析时才使用 LLM |

版本选择由各处理器维护；`used_default_branch` 在注册表处理器中也可能表示默认版本。必须结合 `resolved_version`、证据链接及判定理由核对，不能只看一个布尔值。

### CRAN / Bioconductor

[`core/r_utils.py`](../core/r_utils.py) 优先读取可验证的包网页、DESCRIPTION 和页面明确链接的许可文件。识别 CRAN 官方/常用镜像页面及 `src/contrib` 下载链接，也识别 Bioconductor `release`、`devel`、数值发行版下的 bioc、annotation、experiment、workflows 路径。

- 校验包身份和版本；历史请求不直接借用当前版本页面的许可。
- Bioconductor 的历史查找只跟随包页面已公布的发行版链接；数值发行版输入不任意跨发行版。
- `release`/`devel` 别名会尝试固定到实际发行版页面再核对。
- 网页证据不足时，由外层调用共享归档流程；R 处理器本身不另建递归下载/仓库猜测流程。
- 共享归档分析支持 `DESCRIPTION` 的 License 字段及其文件引用。

### Conda

[`core/conda_utils.py`](../core/conda_utils.py) 只按明确 URL 坐标定位 channel、包名、版本及可选构建/label。元数据来自 `api.anaconda.org`，重定向及数据体大小受到校验。

- 包级输入按请求版本选 release；无明确 URL 版本且请求不可用时可保留 latest 回退。明确版本/构建 URL 不静默切换其他版本。
- 汇总所选版本下符合条件的构建许可，保留不同声明；不会用一项宽泛声明覆盖其余构建。
- SPDX 收敛仅允许相同许可及相同版本的 `-only`/`-or-later` 等有界情况，不跨许可家族或版本做交集推断。
- 元数据无有效许可时，仅在构建选择足够明确时使用共享归档；校验包内 `info/index.json` 名称、版本和 build，结合许可文件和 `info/about.json`。
- 支持 `.tar.bz2` 与 `.conda`（包含 zstd 内容）；多构建未限定时不会任取一个压缩包代表全部构建。

## 共享归档审核

[`core/archive_utils.py`](../core/archive_utils.py) 的通用 URL 后缀为 `.tar.gz`、`.tar.xz`、`.tar.bz2`、`.tgz`、`.txz`、`.tbz2`、`.zip`、`.tar`。`.whl` 的 PyPI 识别和 `.conda` 的专用解包由相应生态路径处理；不能据此认为任意 wheel/Conda URL 都属于通用归档兜底。

下载后安全解包，在解压目录中查找许可文件、README，以及 R DESCRIPTION 等证据，再复用既有内容分析。压缩包有大小和超时限制，临时下载及解压目录最终清理。下载地址缺少网页证据时可以保留为 `license_files`。

配置默认值：归档最大 1 GiB、总下载超时 900 秒，详见[配置说明](CONFIGURATION.md)。

## 最终一次 purl 回退

实现位于 [`core/purl_fallback.py`](../core/purl_fallback.py)。触发条件是前序未获得有效许可，或指定版本的 GitHub 结果使用默认分支。

- 原输入已经是 purl 时不重复重试；每行最多一次，重试不重新获取同一并发信号量。
- 需要 `package_manager` 与 `name`，版本可选；不猜测缺失的 namespace。
- 支持 cargo、npm、pypi、maven、golang、pub、nuget、deb；接受 pip、yarn、pnpm、gradle、go、dart、flutter、apt 等已配置别名。
- Maven 名称需为 `groupId:artifactId` 或 `groupId/artifactId`，Go 需完整模块路径，npm scoped 包需保留 `@scope/name`。
- 指定版本时，重试结果须取得有效许可并通过版本链接校验才采用；原结果已有同版本有效证据时优先保留。异常或不满足条件时保留原结果。
- Conda/R 的专用审核不等于已新增相应的通用 purl 回退映射。

## 最终链接选择

[`core/license_links.py`](../core/license_links.py) 在上述回退结束后运行。对已知有效证据网页直接保留；遇到下载地址或跳转至下载内容时，检查有来源依据的候选页面及其许可信息。

优先返回许可证文件网页或包含许可的注册表/项目页面；没有合适网页时保留原下载链接。单次优化限制为 30 秒，网页正文限制为 1,000,000 字节。候选不可达、许可不匹配或过程异常时保留已有链接。

该步骤只替换 `license_files`，不会重新调用整套生态审核，也不会改写许可证、版权或版本结论。版本和文件身份需要在前面的处理器中确认；链接更美观不能补足许可适用证据。

## 测试入口

对应测试包括 `test_pythonhosted_reference.py`、`test_pypi_entrypoints.py`、`test_r_utils.py`、`test_conda_utils.py`、`test_conda_spdx.py`、`test_purl_fallback_regression.py`、`test_license_links.py` 和 `test_versioned_registry_fallback.py`。运行方法见[测试与开发](TESTING.md)。
