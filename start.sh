#!/bin/bash
# LicenseDetector - 启动脚本

echo "🦐 开源虾 - License Analyzer"
echo "================================"
echo ""

# 检查并启动 API 服务
cd /root/LicenseDetector

# 检查 API 是否已在运行
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ API 服务已在运行"
else
    echo "🚀 启动 API 服务..."
    uv run main.py --api > /dev/null 2>&1 &
    API_PID=$!

    # 等待 API 启动
    echo "⏳ 等待 API 服务启动..."
    for i in {1..10}; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo "✅ API 服务启动成功 (PID: $API_PID)"
            break
        fi
        sleep 1
        echo "   等待中... ($i/10)"
    done
fi

# 打开前端页面
echo ""
echo "🌐 打开前端页面..."
echo "   文件位置: /root/LicenseDetector/frontend/index.html"
echo "   直接在浏览器中打开即可使用"
echo ""
echo "📋 使用说明："
echo "   1. 确保 API 服务运行在 http://localhost:8000"
echo "   2. 在浏览器中打开 index.html"
echo "   3. 上传 .xlsx 文件并输入邮箱"
echo "   4. 点击'开始分析'查看实时日志"
echo ""
echo "💡 提示：如需修改默认邮箱或 API 地址，请编辑 index.html 中的配置"