# 视频字幕提取与关键词生成 Web 工具

这是一个基于 Flask 的 Web 应用，用于批量处理视频文件，提取字幕，生成关键词和摘要。

## 功能特性

1. **视频上传与转录**
   - 支持拖拽上传多个视频文件
   - 自动使用 Whisper 转录音频为文本
   - 自动清洗转录文本（去除填充词、重复等）

2. **关键词提取**
   - 支持两种模式：
     - OpenAI GPT（更准确，需要 API Key）
     - 本地 jieba 分词（免费，准确度较低）
   - 自动过滤通用词，提取学术术语

3. **摘要生成**
   - 基于原摘要、关键词和转录文本生成新的学术摘要
   - 使用 OpenAI GPT 生成高质量摘要

## 安装步骤

### 1. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

### 2. 安装 Whisper（用于音频转录）

```bash
pip install openai-whisper
```

或者使用 conda：

```bash
conda install -c conda-forge ffmpeg
pip install openai-whisper
```

**注意**: Whisper 需要 ffmpeg，请确保系统已安装：
- macOS: `brew install ffmpeg`
- Ubuntu: `sudo apt-get install ffmpeg`
- Windows: 下载 ffmpeg 并添加到 PATH

### 3. 运行应用

```bash
python app.py
```

应用将在 `http://localhost:5000` 启动。

### 4. 访问界面

在浏览器中打开 `http://localhost:5000` 即可使用。

## 使用说明

### 步骤 1: 上传视频文件
- 点击上传区域或拖拽视频文件
- 支持格式: MP4, AVI, MOV, MKV, MP3, WAV 等
- 系统会自动转录音频并显示转录文本

### 步骤 2: 输入视频信息（可选）
- 填写标题、报告人、原摘要等信息
- 这些信息将用于后续的关键词提取和摘要生成

### 步骤 3: 提取关键词
- 输入 OpenAI API Key（如果使用 GPT 模式）
- 选择使用 OpenAI GPT 或本地 jieba
- 点击"提取关键词"按钮
- 关键词会自动显示并可以复制

### 步骤 4: 生成摘要（可选）
- 点击"生成摘要"按钮
- 系统会基于原摘要、关键词和转录文本生成新的学术摘要

## API 接口说明

### POST /api/transcribe
上传视频文件并转录音频

**请求**: multipart/form-data
- `file`: 视频文件

**响应**:
```json
{
  "success": true,
  "transcript": "转录文本...",
  "filename": "video.mp4",
  "output_path": "transcripts/video.txt"
}
```

### POST /api/clean-transcript
清洗转录文本

**请求**: application/json
```json
{
  "transcript": "原始转录文本...",
  "filename": "video.mp4"
}
```

**响应**:
```json
{
  "success": true,
  "cleaned_transcript": "清洗后的文本...",
  "output_path": "transcripts_clean/video.txt"
}
```

### POST /api/extract-keywords
提取关键词

**请求**: application/json
```json
{
  "transcript": "转录文本...",
  "description": "原摘要...",
  "api_key": "sk-...",
  "use_openai": true
}
```

**响应**:
```json
{
  "success": true,
  "keywords": ["关键词1", "关键词2", ...],
  "keywords_str": "关键词1，关键词2，...",
  "method": "openai"
}
```

### POST /api/generate-abstract
生成摘要

**请求**: application/json
```json
{
  "original_abstract": "原摘要...",
  "keywords": "关键词1，关键词2",
  "title": "视频标题",
  "speaker_name": "报告人",
  "transcript": "转录文本...",
  "api_key": "sk-..."
}
```

**响应**:
```json
{
  "success": true,
  "abstract": "生成的摘要..."
}
```

## 文件结构

```
.
├── app.py                 # Flask 后端主文件
├── templates/
│   └── index.html        # 前端 HTML 界面
├── uploads/              # 上传的文件（自动创建）
├── transcripts/          # 原始转录文本（自动创建）
├── transcripts_clean/    # 清洗后的转录文本（自动创建）
├── requirements.txt      # Python 依赖
└── README_WEB.md         # 本文档
```

## 注意事项

1. **OpenAI API Key**: 
   - API Key 仅在浏览器和服务器之间传输，不会保存
   - 建议使用环境变量或配置文件管理 API Key（生产环境）

2. **文件存储**:
   - 上传的文件会保存在 `uploads/` 目录
   - 转录文本保存在 `transcripts/` 和 `transcripts_clean/` 目录
   - 建议定期清理这些目录

3. **性能优化**:
   - 大文件转录可能需要较长时间
   - 可以考虑使用队列系统（如 Celery）处理长时间任务

4. **安全性**:
   - 当前版本为开发版本，不建议直接暴露到公网
   - 生产环境建议添加认证、限流等安全措施

## 部署建议

### 使用 Gunicorn（生产环境）

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### 使用 Docker

创建 `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

构建和运行:

```bash
docker build -t video-tool .
docker run -p 5000:5000 video-tool
```

## 故障排除

### Whisper 安装失败
- 确保已安装 ffmpeg
- 尝试使用 conda 安装

### 转录速度慢
- Whisper 模型较大，首次运行会下载模型
- 可以考虑使用更小的模型（如 `base` 或 `small`）

### API 调用失败
- 检查 API Key 是否正确
- 检查网络连接
- 查看浏览器控制台的错误信息

## 许可证

MIT License

