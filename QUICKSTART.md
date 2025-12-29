# 快速开始指南

## 5 分钟快速上手

### 1. 安装依赖

```bash
# 安装 Python 依赖
pip install -r requirements.txt

# 安装 ffmpeg (Whisper 需要)
# macOS:
brew install ffmpeg

# Ubuntu:
sudo apt-get install ffmpeg

# Windows: 下载 ffmpeg 并添加到 PATH
```

### 2. 启动应用

**方式一：使用启动脚本（推荐）**
```bash
./run.sh
```

**方式二：直接运行**
```bash
python app.py
```

### 3. 打开浏览器

访问 `http://localhost:5001` or `http://localhost:5000`

### 4. 使用流程

1. **上传视频**
   - 拖拽视频文件到上传区域
   - 等待自动转录完成（可能需要几分钟）

2. **输入信息**（可选）
   - 填写标题、报告人、原摘要等信息

3. **提取关键词**
   - 输入 OpenAI API Key（如果使用 GPT 模式）
   - 选择使用 OpenAI 或本地 jieba
   - 点击"提取关键词"

4. **生成摘要**（可选）
   - 点击"生成摘要"
   - 等待生成完成

## 常见问题

### Q: Whisper 转录很慢？
A: 首次运行会下载模型（约 150MB），之后会快很多。可以使用更小的模型（如 `tiny` 或 `small`）加快速度。

### Q: 如何获取 OpenAI API Key？
A: 访问 https://platform.openai.com/api-keys 注册并创建 API Key

### Q: 可以不使用 OpenAI 吗？
A: 可以，取消勾选"使用 OpenAI GPT"选项，将使用免费的 jieba 分词，但准确度较低。

### Q: 支持哪些视频格式？
A: 支持 MP4, AVI, MOV, MKV, FLV, WEBM, MP3, WAV, M4A 等常见格式

### Q: 文件保存在哪里？
A: 
- 上传的文件：`uploads/`
- 转录文本：`transcripts/`
- 清洗后的文本：`transcripts_clean/`

## 下一步

查看 [README_WEB.md](README_WEB.md) 了解详细功能和 API 文档。

