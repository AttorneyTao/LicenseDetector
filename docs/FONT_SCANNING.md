# 🔤 字体扫描模式（Font Scanning）

> 新增功能：当 `input.xlsx` 全部为「字体」而非软件包时，使用字体扫描模式按字体来源站点
> 自动获取**授权(license)**与**版权/作者(copyright)**信息。

## 一、如何通过 CLI 启动

字体扫描通过 `--font` 启动参数触发，输入文件仍为项目根目录的 `input.xlsx`：

```bash
# 字体扫描模式（处理 input.xlsx 中的字体，结果保存到 outputs/）
python main.py --font

# 对照：默认（软件包）模式
python main.py

# 对照：API / Web 模式
python main.py --api
```

> Docker 下可在对应 CLI 启动命令后追加 `--font` 参数。

输入文件列与默认模式一致：

| 列 | 说明 |
|---|---|
| `name` | 字体名称（可选，用于辅助识别与输出） |
| `version` | 版本（可选） |
| `github_url` | 字体来源 URL（github / Google Fonts / Fontshare / 猫啃网 / 微信公众号 等） |

输出文件与默认模式一致（`outputs/output_latest.xlsx` 等），其中
`license_files` 列为**授权证据 URL**（指向实际据以判定的页面/文件，而非泛化的许可证模板页）。

## 二、设计原则

字体来源站点高度分散（实测一个 723 条样本里有 82 个域名），多数没有标准化 API。核心约束：

- **绝不把整页 HTML 丢给 LLM**：先把页面裁剪到「授权/版权/作者」相关的小片段
  （默认上限 1800 字符）再交给 LLM，控制 token 成本。
- **能复用尽量复用、能用规则就不调 LLM**：github / Google Fonts / Fontshare 全程不调用 LLM。

## 三、站点路由与处理方式

按 URL 域名路由到对应适配器（见 [`core/font_utils.py`](../core/font_utils.py) 的 `classify_font_source`）：

| 类别 | 站点 | 处理方式 | 是否用 LLM |
|---|---|---|---|
| `github` | github.com | 复用既有 `process_github_repository` 流程 | 否 |
| `google_fonts` | fonts.google.com | 复用 GitHub API client 读 `google/fonts` 仓库 | 否 |
| `fontshare` | fontshare.com | 调官方 API 按每个字体的 `license_type` 判定 | 否 |
| `maoken` | maoken.com（猫啃网） | 定向截取授权区块片段 → 小片段 LLM 归一化 | 小片段 |
| `crawl_llm` | 微信公众号 / 站酷 / B站 / 字体家 等长尾站点 | 通用爬虫 → 关键词窗口截取 → 小片段 LLM | 小片段 |

### Google Fonts（不假定 OFL）

`google/fonts` 仓库按授权分了 **4 个顶层目录**：`ofl/`、`apache/`、`ufl/`、`cc-by-sa/`。
适配器在这 4 个目录中**定位** family 实际所在目录，再读取该目录下**真实的 LICENSE 文件 +
`METADATA.pb`** 判定授权与版权——不写死 OFL（例如 Roboto 已从 Apache 迁到 OFL，会被正确识别）。

### Fontshare（区分两种授权）

Fontshare 同时提供两类免费字体，必须按每个字体区分：

- `itf_ffl` → **ITF Free Font License**（闭源，ITF 自有）
- `sil_ofl` → **OFL-1.1**（开源，归各设计师/发布者所有）

适配器加载并缓存官方目录（`https://api.fontshare.com/v2/fonts`），按每个字体的
`license_type` 字段判定，并从 `designers` / `publisher` 提取版权。目录中找不到的
slug（如部分商业付费字体）会如实标为 `not_found`，**不会被错误标成免费**。

## 四、测试与报告

测试脚本会从真实 `input.xlsx` 按站点类别**分层抽样**（长尾 `crawl_llm` 按域名频次降序
优先覆盖微信公众号、站酷等高频站点），跑通所有适配器并生成报告。

```bash
# 默认每类约 6 个，合计约 30 个用例
python test/test_font_adapters.py

# 每类最多 8 个
python test/test_font_adapters.py 8

# 每类最多 8 个，总量上限 10
python test/test_font_adapters.py 8 10
```

产物（均已加入 `.gitignore`）：

- `test/fixtures/font_samples.xlsx` — 本次抽样的样本
- `test/font_report/font_test_report_latest.md` / `.xlsx` — 报告（含分类别成功率、明细、原始 JSON）

## 五、已知限制（属数据/站点问题，非适配器缺陷）

- **JS 渲染 / 反爬站点**（如 booth.pm）：纯 HTTP 抓不到正文，标为 `error`。如需可加浏览器渲染回退。
- **通用列表页 URL**（如 `58pic.com/...?m=qtwFonts&a=index`）：本身不指向具体字体，无法解析。
- **非免费目录字体**（Fontshare 上的部分商业字体）：标为 `not_found`，需人工核实来源 URL。

## 六、状态值说明

| status | 含义 |
|---|---|
| `success` | 成功获取授权信息 |
| `not_found` | 来源目录中未找到该字体（如非免费目录字体） |
| `error` | 抓取失败 / 解析失败（站点反爬、JS 渲染、空正文等） |
| `pending_font_adapter` | 该来源类别的适配器尚未实现（保留状态） |
