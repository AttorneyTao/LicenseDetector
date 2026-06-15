基于用户要求，项目成功添加API功能，同时保持原有程序入口（uv run main.py）不变。

---

## 🆕 字体扫描模式（Font Scanning）

新增 `--font` 启动参数，用于处理「全部为字体」的 `input.xlsx`，按字体来源站点自动获取
授权(license)与版权(copyright)信息。详见 [FONT_SCANNING.md](FONT_SCANNING.md)。

- **启动方式**：`python main.py --font`（默认软件包模式与 API 模式不受影响）
- **站点路由**：github（复用既有流程）/ google_fonts（读 google/fonts 真实 LICENSE，不假定 OFL）
  / fontshare（按官方 API 的 `license_type` 区分 ITF-FFL 与 OFL）/ maoken（猫啃网）/ 长尾站点通用爬虫
- **Token 控制**：绝不把整页丢给 LLM，仅把授权/版权相关小片段（≤1800 字符）交给 LLM；
  github/google_fonts/fontshare 全程不调用 LLM
- **新增文件**：[`core/font_utils.py`](../core/font_utils.py)（站点适配器与分发器）、
  [`test/test_font_adapters.py`](../test/test_font_adapters.py)（分层抽样集成测试 + 报告）
- **改动文件**：`main.py`（`--font` 参数 + `process_all_fonts` 并发流程）

---

## 📋 项目改动总结

### ✅ 完成的功能

1. **API服务** - FastAPI实现
   - POST /api/v1/analyze - 分析并发送邮件到指定邮箱
   - POST /api/v1/analyze-and-download - 分析并直接下载Excel
   - GET /health - 健康检查

2. **邮件发送功能**
   - 支持多种SMTP服务器（Gmail、网易、QQ等）
   - 自动附加Excel结果文件
   - 处理成功/失败状态

3. **程序入口保持不变**
   - `uv run main.py` - 保持原有CLI模式
   - `python main.py --api` - 启动API服务（新增）

### 📁 新增文件

1. **core/email_utils.py** - 邮件发送工具
   - EmailConfig: 邮件配置类
   - EmailSender: 邮件发送器
   - send_analysis_result: 发送分析结果

2. **api.py** - FastAPI应用
   - FastAPI应用配置
   - 3个API端点
   - 异步处理逻辑

3. **API_USAGE.md** - API使用指南
   - 快速开始
   - API端点说明
   - Python客户端示例
   - 常见问题

4. **test_api.py** - API测试脚本
   - 健康检查测试
   - 分析下载测试
   - 邮件发送测试

5. **.env.example** - 邮件配置模板

### 📝 修改的文件

1. **main.py**
   - 添加argparse支持命令行参数
   - main()函数支持API/CLI模式选择
   - 保留原有main_async()作为CLI模式入口

2. **pyproject.toml**
   - 添加fastapi >= 0.104.0
   - 添加uvicorn >= 0.24.0
   - 添加python-multipart >= 0.0.6

### 🚀 使用方式

#### CLI模式（原有功能，保持不变）
```bash
uv run main.py
```

#### API模式（新增）
```bash
# 启动API服务（默认8000端口）
python main.py --api

# 指定端口和地址
python main.py --api --host 0.0.0.0 --port 8080
```

#### 测试API
```bash
# 先启动API服务
python main.py --api &

# 在另一个终端运行测试
python test_api.py
```

### 📊 API端点说明

#### 1. 分析并发送邮件
```
POST /api/v1/analyze
参数:
  - file: Excel文件 (必需)
  - email: 收件人邮箱 (必需)
  - smtp_server: SMTP服务器 (可选，使用环境变量)
  - smtp_port: SMTP端口 (可选，使用环境变量)

示例:
curl -X POST "http://localhost:8000/api/v1/analyze" \
  -F "file=@input.xlsx" \
  -F "email=user@example.com"
```

#### 2. 分析并下载
```
POST /api/v1/analyze-and-download
参数:
  - file: Excel文件 (必需)

示例:
curl -X POST "http://localhost:8000/api/v1/analyze-and-download" \
  -F "file=@input.xlsx" \
  --output result.xlsx
```

#### 3. 健康检查
```
GET /health

示例:
curl http://localhost:8000/health
```

### ⚙️ 配置说明

#### 邮件配置（可选）
编辑 `.env` 文件:
```
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SENDER_EMAIL=your_email@gmail.com
SENDER_PASSWORD=your_app_password
```

常见邮箱配置:
- Gmail: smtp.gmail.com:587 (使用应用专用密码)
- 网易: smtp.163.com:587
- QQ: smtp.qq.com:587
- Outlook: smtp.office365.com:587

### 🔍 API文档

启动API服务后，访问:
- Swagger Doc: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### 📦 依赖安装

已通过 `uv sync` 安装:
- fastapi==0.135.1
- uvicorn==0.41.0
- python-multipart==0.0.22

### ✨ 特点

1. **保持向后兼容** - 原有CLI模式完全不变
2. **异步处理** - 支持并发请求
3. **自动附件** - 邮件自动附加Excel结果
4. **灵活配置** - 支持环境变量和运行时参数
5. **完整日志** - 记录所有操作到logs目录
6. **API文档** - 自动生成Swagger和ReDoc文档

### 🧪 验证

所有文件已通过:
- ✓ Python语法验证
- ✓ 模块导入验证
- ✓ 命令行参数验证
- ✓ 依赖安装验证

### 📚 文档

- API_USAGE.md - 详细API使用指南
- .env.example - 邮件配置示例
- test_api.py - API测试脚本

### 🔄 工作流

1. 用户上传input.xlsx到API
2. API并发处理所有仓库
3. 本地处理结果
4. 可选：自动发送到指定邮箱
5. 返回结果或下载文件

### ⚡ 性能

- 默认最大并发数: 5 (可配置)
- 临时文件存储: temp/ 目录
- 定期保存中间结果
- 支持大文件处理
