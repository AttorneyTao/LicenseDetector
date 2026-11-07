# 字体扫描

## 入口与范围

整张输入表都是字体时，使用单独的 CLI 模式：

```bash
uv run main.py --font
```

输入仍为根目录 `input.xlsx`，使用 `name`、`version`、`github_url`，可增加 `package_manager`、`serial_number` 透传。结果与软件包 CLI 一样写入 `outputs/`，并在最后执行输出列组装和版权空值填充。

当前上传 API 没有字体模式字段；同时传 `--api --font` 时仍启动普通 API。不要把字体和软件包混在一张表中，并假定每行可自动切换扫描模式。

## 站点适配器

分发实现为 [`core/font_utils.py`](../core/font_utils.py)：

| 来源 | 处理方式 | 模型使用 |
| --- | --- | --- |
| GitHub | 复用 `process_github_repository(..., font_mode=True)`，必要时读取自定义字体授权 README | 继承仓库内容分析，可能调用 LLM |
| Google Fonts | 在 google/fonts 的 ofl、apache、ufl、cc-by-sa 目录定位 family，读取实际许可和 METADATA.pb | 专用适配器主要按文件解析，不统一假定 OFL |
| Fontshare | 按官方字体目录中的 license_type、设计者和发布者信息判断 | 专用适配器按目录字段解析 |
| 猫啃网 | 抓取并截取授权片段，再作归一化分析 | 小片段 LLM |
| 其他可抓取站点 | 抓取正文、按关键词窗口截取授权/版权相关部分 | 小片段 LLM |

通用授权片段默认上限为 1800 字符，避免将整页正文直接送模型。GitHub 路径不能笼统标为“全程不调用 LLM”。

Fontshare 的 `itf_ffl` 与 `sil_ofl` 分别对应 ITF Free Font License 和 OFL-1.1，目录中未找到的字体不会自动判为免费。Google Fonts 使用实际目录和文件，不能用当前流行字体许可覆盖历史版本。

## 输出与限制

`license_files` 应指向实际判断所用证据；字体站点可能只有授权页面或自定义条款。`success`、`not_found`、`error`、`pending_font_adapter` 等状态须结合处理器结果和判定理由阅读。

JavaScript 渲染、反爬、列表页地址、目录未收录或来源缺失都会影响结果；当前没有统一浏览器渲染回退。`not_found` 反映来源中未找到记录，不能推定禁止使用或免费。

## 联网抽样脚本

[`test/test_font_adapters.py`](../test/test_font_adapters.py) 会从实际 `input.xlsx` 分层抽样并生成报告，访问网络且可能调用模型。只有确认输入与执行范围后才运行：

```bash
# 每类最多 8 项，总量上限 10
uv run python test/test_font_adapters.py 8 10
```

产物包括 `test/fixtures/font_samples.xlsx` 和 `test/font_report/` 下的 Markdown、Excel 报告。它属于真实样本验证，不能当成无网络、无费用的常规单元测试。
