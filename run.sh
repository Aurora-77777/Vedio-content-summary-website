#!/bin/bash

# 视频字幕提取与关键词生成工具启动脚本

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "视频字幕提取与关键词生成工具"
echo "=========================================="
echo "工作目录: $SCRIPT_DIR"
echo ""

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到 Python3，请先安装 Python"
    exit 1
fi

# 检查是否已安装依赖
echo "检查依赖..."
if ! python3 -c "import flask" 2>/dev/null; then
    echo "正在安装依赖..."
    pip3 install -r requirements.txt
fi

# 检查 ffmpeg（Whisper 需要）
if ! command -v ffmpeg &> /dev/null; then
    echo "警告: 未找到 ffmpeg，Whisper 可能无法正常工作"
    echo "请安装 ffmpeg:"
    echo "  macOS: brew install ffmpeg"
    echo "  Ubuntu: sudo apt-get install ffmpeg"
fi

# 创建必要的目录
mkdir -p uploads transcripts transcripts_clean

# 启动应用
echo ""
echo "启动 Web 应用..."
PORT=${1:-5001}  # 默认使用 5001 端口，避免与 macOS AirPlay Receiver 冲突
echo "访问地址: http://localhost:$PORT"
echo ""
echo "提示: 如果端口被占用，可以指定其他端口: ./run.sh 8080"
echo ""
python3 app.py $PORT

