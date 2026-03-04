# 🚀 快速参考卡

## 三种快速开始方式

### 方式1️⃣：原有CLI功能（保持不变）
```bash
uv run main.py
# 输出文件在 outputs/ 目录
```

### 方式2️⃣：启动API服务
```bash
python main.py --api
# 访问 http://localhost:8000/docs 查看API文档
```

### 方式3️⃣：API服务自定义端口
```bash
python main.py --api --port 8080
```

---

## 5分钟快速上手

### Step 1: 查看文档
```bash
cat QUICKSTART.md  # 快速开始指南
```

### Step 2: 验证安装
```bash
python verify_installation.py  # 验证所有文件和依赖
```

### Step 3: 选择运行方式
```bash
# 方式A: 处理本地文件（原有功能）
uv run main.py

# 方式B: 启动API服务（新增功能）
python main.py --api
```

### Step 4: 配置邮件（可选）
编辑 `.env` 文件，参考 `.env.example` 模板

### Step 5: 开始使用
```bash
# CLI: 放入input.xlsx，运行main.py
# API: 使用curl或test_api.py上传文件
python test_api.py
```

---

## 常用命令速查

| 任务 | 命令 |
|------|------|
| 查看帮助 | `python main.py --help` |
| CLI分析 | `uv run main.py` |
| 启动API | `python main.py --api` |
| 自定义端口 | `python main.py --api --port 8080` |
| 验证安装 | `python verify_installation.py` |
| 测试API | `python test_api.py` |

---

## API快速用例

### 使用curl上传文件并邮件
```bash
curl -X POST "http://localhost:8000/api/v1/analyze" \
  -F "file=@input.xlsx" \
  -F "email=user@example.com"
```

### 使用curl上传文件并下载
```bash
curl -X POST "http://localhost:8000/api/v1/analyze-and-download" \
  -F "file=@input.xlsx" \
  --output result.xlsx
```

### 查看API文档
启动API后访问：http://localhost:8000/docs

---

## 文件导航

| 需要... | 查看文件 |
|--------|---------|
| 快速开始 | `QUICKSTART.md` |
| API详情 | `API_USAGE.md` |
| 改动说明 | `CHANGES.md` |
| 邮箱配置 | `.env.example` |
| 测试API | `test_api.py` |
| 验证安装 | `verify_installation.py` |

---

## 邮箱配置速查

### Gmail
```env
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SENDER_EMAIL=your@gmail.com
SENDER_PASSWORD=<16位应用密码>
```

### 网易
```env
SMTP_SERVER=smtp.163.com
SMTP_PORT=587
SENDER_EMAIL=your@163.com
SENDER_PASSWORD=<密码>
```

### QQ
```env
SMTP_SERVER=smtp.qq.com
SMTP_PORT=587
SENDER_EMAIL=your@qq.com
SENDER_PASSWORD=<密码>
```

---

## 功能对比

| 功能 | CLI | API |
|------|-----|-----|
| 本地文件处理 | ✓ | ✓ |
| 远程文件上传 | ✗ | ✓ |
| 自动邮件发送 | ✗ | ✓ |
| 直接下载 | ✓ | ✓ |
| 异步并发 | ✓ | ✓ |
| HTTP接口 | ✗ | ✓ |

---

## 故障排除

### Q: API无法连接
**A:** 确保API已启动 `python main.py --api`

### Q: 邮件发送失败
**A:** 检查 `.env` 邮件配置，参考 `.env.example`

### Q: 找不到input.xlsx
**A:** 确保 `input.xlsx` 在当前工作目录

### Q: 依赖缺失
**A:** 运行 `uv sync` 重新安装依赖

---

## 更多帮助

```bash
python main.py --help          # 查看命令帮助
python verify_installation.py  # 验证安装
python test_api.py             # 测试API
cat QUICKSTART.md              # 快速开始
cat API_USAGE.md               # API详细文档
```

---

## 项目状态

✅ **所有功能已实现并验证通过**
- ✓ CLI模式保持不变
- ✓ API服务完整实现
- ✓ 邮件发送功能就绪
- ✓ 文档完善
- ✓ 已通过所有验证

**准备好开始了吗？** 🚀

```bash
# 最简单的开始方式
cat QUICKSTART.md  # 阅读快速开始
python verify_installation.py  # 验证
uv run main.py     # 或 python main.py --api
```
