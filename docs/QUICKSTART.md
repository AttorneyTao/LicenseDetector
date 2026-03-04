# 🚀 快速开始指南

## 原有功能（保持不变）

### 1. CLI 模式 - 处理本地文件

```bash
# 方式一: 使用uv运行（推荐）
uv run main.py

# 方式二: 激活虚拟环境后运行
.venv\Scripts\activate
python main.py
```

**工作流程：**
1. 将需要分析的文件保存为 `input.xlsx`
2. 运行上述命令
3. 输出文件自动保存到 `outputs/` 目录
4. 检查 `logs/` 目录了解执行详情

---

## 新增功能 - API 服务

### 2. API 模式 - 通过HTTP接口发送文件

#### 步骤1：启动 API 服务

```bash
# 默认监听 0.0.0.0:8000
python main.py --api

# 或指定端口
python main.py --api --port 8080

# 或指定地址和端口
python main.py --api --host 127.0.0.1 --port 8000
```

#### 步骤2：配置邮件（可选）

编辑 `.env` 文件：

```env
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SENDER_EMAIL=your_email@gmail.com
SENDER_PASSWORD=your_app_password
```

> 💡 **为何需要应用专用密码？**
> 出于安全考虑，Gmail等邮箱不允许直接使用账户密码。
> 需要生成专用的应用密码用于此程序。

#### 步骤3：上传文件并分析

**方案A：发送邮件（需配置邮箱）**

```bash
curl -X POST "http://localhost:8000/api/v1/analyze" \
  -F "file=@input.xlsx" \
  -F "email=recipient@example.com"
```

**方案B：直接下载（无需邮箱配置）**

```bash
curl -X POST "http://localhost:8000/api/v1/analyze-and-download" \
  -F "file=@input.xlsx" \
  --output result.xlsx
```

---

## 📊 查看 API 文档

启动 API 服务后，访问：

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **健康检查**: http://localhost:8000/health

---

## 🐍 Python 脚本使用

### 方式1：使用provided的测试脚本

```bash
# 启动 API 服务（新终端）
python main.py --api &

# 运行测试脚本
python test_api.py
```

### 方式2：自己的脚本

```python
import requests

# 上传文件并获取结果
with open("input.xlsx", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/v1/analyze-and-download",
        files={"file": f}
    )
    
    # 保存结果
    with open("output.xlsx", "wb") as out:
        out.write(response.content)
```

---

## ⚙️ 常见邮箱配置

### Gmail
```env
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SENDER_EMAIL=your_email@gmail.com
SENDER_PASSWORD=xxxx xxxx xxxx xxxx  # 16位应用专用密码
```
> 获取应用密码: https://myaccount.google.com/apppasswords

### 网易邮箱
```env
SMTP_SERVER=smtp.163.com
SMTP_PORT=587
SENDER_EMAIL=your_email@163.com
SENDER_PASSWORD=your_password
```

### QQ邮箱
```env
SMTP_SERVER=smtp.qq.com
SMTP_PORT=587
SENDER_EMAIL=your_qq@qq.com
SENDER_PASSWORD=your_password
```

### Outlook
```env
SMTP_SERVER=smtp.office365.com
SMTP_PORT=587
SENDER_EMAIL=your_email@outlook.com
SENDER_PASSWORD=your_password
```

---

## 🔍 故障排除

### API 无法连接
```
错误: Failed to connect to localhost:8000
解决: 確保 API 已启动 (python main.py --api)
```

### 邮件发送失败
```
错误: Authentication failed
解决: 检查 SENDER_EMAIL 和 SENDER_PASSWORD 配置
```

### 找不到 input.xlsx
```
错误: ModuleNotFoundError: input.xlsx not found
解决: 确保 input.xlsx 在当前工作目录
```

---

## 📁 文件结构

```
.
├── main.py                 # 主程序（支持CLI和API）
├── api.py                  # FastAPI 应用
├── test_api.py             # API 测试脚本
├── prompts.yaml            # LLM 提示词
├── pyproject.toml          # 项目配置和依赖
├── .env                    # 本地配置（含邮件设置）
├── .env.example            # 配置模板
├── API_USAGE.md            # API 详细文档
├── CHANGES.md              # 改动说明
├── core/
│   ├── email_utils.py      # 邮件工具（新）
│   ├── github_utils.py     # GitHub API
│   ├── config.py           # 配置
│   └── ...                 # 其他工具
├── input.xlsx              # 输入文件（由用户提供）
├── outputs/                # 输出目录
├── logs/                   # 日志目录
└── temp/                   # 临时文件目录
```

---

## 💡 使用建议

### 场景1：一次性分析（推荐CLI模式）
```bash
# 放入 input.xlsx
uv run main.py
# 输出保存到 outputs/
```

### 场景2：服务器部署（推荐API模式）
```bash
# 启动 API 服务
python main.py --api --host 0.0.0.0 --port 8000

# 其他系统通过 API 调用
# curl -F "file=@input.xlsx" http://server:8000/api/v1/...
```

### 场景3：批量处理（混合模式）
```bash
# 启动 API（后台）
python main.py --api &

# 写脚本批量上传
for file in *.xlsx; do
    curl -X POST "http://localhost:8000/api/v1/analyze-and-download" \
      -F "file=@$file" \
      --output "output_$file"
done
```

---

## 📚 更多帮助

查看详细文档：
- **API_USAGE.md** - API 完整使用指南
- **CHANGES.md** - 项目改动说明
- **main.py --help** - 命令行帮助

---

## ✅ 验证安装

```bash
# 检查依赖是否安装
uv sync

# 检查 Python 版本
python --version  # 需要 >= 3.13

# 运行语法检查
python -m py_compile main.py api.py

# 测试导入
python -c "import api; import core.email_utils; print('OK')"
```

---

## 🎯 下一步

1. **配置邮箱**（可选）- 编辑 `.env`
2. **启动 API**（可选）- `python main.py --api`
3. **测试功能** - 使用 `test_api.py` 或 `curl`
4. **部署上线**（可选）- 参考 API_USAGE.md 的部署建议

祝您使用愉快！🎉
