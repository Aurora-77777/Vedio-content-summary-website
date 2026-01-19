# 1. 使用支持 Python 3.9 的基础镜像
FROM python:3.9-slim

# 2. 设置工作目录
WORKDIR /app

# 3. 安装系统依赖 (ffmpeg是必须的，用于Whisper转录)
# git 用于安装某些依赖，curl/wget 用于调试
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 4. 复制依赖文件并安装
COPY requirements.txt .
# 使用清华源加速安装 Python 库
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 5. 复制项目所有代码到容器
COPY . .

# 6. 创建必要的临时文件夹并给予读写权限
# uploads: 上传视频, transcripts: 转录结果, pgdata: 临时数据库(如果本地跑)
RUN mkdir -p uploads transcripts transcripts_clean && \
    chmod -R 777 uploads transcripts transcripts_clean

# 7. 暴露端口 (ModelScope/HuggingFace 默认使用 7860)
EXPOSE 7860

# 8. 启动命令 (指定端口 7860)
CMD ["python", "app.py", "7860"]# force rebuild
