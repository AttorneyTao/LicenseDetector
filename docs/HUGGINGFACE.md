# Hugging Face 模型和数据集审核

## 输入和版本

支持 `https://huggingface.co/<owner>/<repo>` 模型仓库、`/datasets/<owner>/<repo>` 数据集仓库，以及其中的 tree、blob、resolve、raw 链接。Spaces、集合、用户主页不作为模型或数据集处理。输入仍使用现有 Excel 字段，流水号和原名称透传；无需安装 Hugging Face SDK。

URL 明确的 revision 优先于 version 列，并记录冲突；否则采用 version 或 main。请求 revision 不存在时，仅尝试同一仓库的 main，记录实际 commit 和回退原因。401、403、网络失败等不触发其他生态处理。

所有证据固定到 API 返回的 40 位 commit SHA。文件输入必须在该 commit 的文件列表中存在；目录输入也需确认对应目录。官方 API 重定向可用于确认仓库改名，不能只按名称近似替换仓库。

## 严格不下载权重

此流程不调用 Git clone、snapshot download、模型加载或完整仓库下载。输入 `.safetensors` 等地址仅用于识别仓库、revision 和文件路径；文件存在性通过 API 元数据确认，不请求该文件正文。

- 请求前拦截 `.safetensors`、`.bin`、`.gguf`、`.pt`、`.pth`、`.ckpt`、`.onnx` 等权重，以及数据/归档路径。
- 每次重定向后重新检查地址，跳转到权重也不继续请求。
- 读取正文前检查 Content-Type，仅允许 JSON 元数据及受支持文本类型，拒绝二进制响应。
- 元数据上限 12,000,000 字节，单份文本上限 2,000,000 字节；Content-Length 超限时不读正文，无 Content-Length 的分块传输同样累计限制。
- 最多读取 8 个本地证据文件，外加卡片明确引用的许可条款；元数据/文本网络阶段总超时 120 秒，单请求 30 秒。

读取路径及字节数记入 `[HF] Read metadata/text evidence` 日志。模型分析和版权提取复用现有调用及其超时配置。

## 许可证据与输出

按输入文件所在目录及祖先目录搜集 LICENSE、README、NOTICE 等小型文本，读取同一 commit 的卡片声明。自定义 `license_link` 仅按发布方明确引用读取，不搜索其他仓库；外部许可页面不接收 HF token。

复用现有 LLM 许可及版权提取；卡片中的基础模型、训练数据、示例代码等链接不能自动替代当前对象的许可。多个许可标签不直接推断为 AND/OR。最终版权空值和风险等级继续使用共享输出规则。

`license_files` 返回固定 commit 的 blob 网页，优先实际许可文件，只有卡片证据时返回 README。`license_analysis` 保留仓库类型、commit、输入路径、原许可声明、证据链接及不可读文件说明。

命中 HF 域名后使用独立分发，即使 URL 格式不支持、文件不存在、证据缺失或请求异常，也不进入 GitHub、通用归档、purl 或最终通用链接查找。其他生态入口保持原有行为。

公开仓库无需 token。需要访问受限证据时可配置可选 `HF_TOKEN`；不自动申请访问、不自动接受条款。访问失败与无许可分开记录。

## 测试

```bash
uv run python -m pytest -q test/test_huggingface_utils.py
```

包含模型/数据集路由、revision 固定与回退、文件身份、官方重定向、自定义许可和列表、CLI/API 失败隔离，以及权重路径、二进制响应、超大响应、分块超限和凭据不向外部传播等测试。
