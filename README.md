---
title: Cas Report Agent
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

# 📹 视频内容分析与专家观点生成工具

## 🎯 功能概述

这是一个完整的Web应用，用于：
- 📹 **视频内容分析**：支持视频链接解析和文件上传
- 🎤 **自动转录**：使用Whisper将视频音频转为文本
- 🔑 **关键词提取**：支持多种AI模型（Gemini/ChatGPT/DeepSeek/Qwen）
- 📝 **专家观点生成**：基于Few-Shot学习生成结构化文章
- 🔍 **RAG智能搜索**：基于向量数据库的学术报告检索系统

---

## 🚀 快速开始（5分钟）

### 1. 安装依赖

```bash
# 进入项目目录
cd Vedio-content-summary-website

# 安装Python依赖
pip install -r requirements.txt

# 安装ffmpeg（必需，用于音频处理）
# macOS:
brew install ffmpeg
# Ubuntu:
sudo apt-get install ffmpeg
# Windows: 下载ffmpeg并添加到PATH
```

### 2. 启动应用

```bash
# 方式1：使用启动脚本（推荐）
./run.sh

# 方式2：直接运行
python app.py
```

### 3. 打开浏览器

访问：`http://localhost:5001`

---

## 🔑 获取API Key（必需）

### 至少需要配置一个AI模型的API Key：

1. **Gemini API Key**（推荐，支持视频直接分析）
   - 🔗 https://makersuite.google.com/app/apikey
   - 免费额度充足，支持视频分析

2. **OpenAI API Key**（ChatGPT）
   - 🔗 https://platform.openai.com/api-keys
   - 注意：代码中使用 `gpt-5.1`，如不可用请修改 `app.py` 第28行为 `gpt-4o-mini`

3. **DeepSeek API Key**（可选，性价比高）
   - 🔗 https://platform.deepseek.com/api_keys

4. **Qwen API Key**（可选，阿里云百炼）
   - 🔗 https://dashscope.console.aliyun.com/apiKey

---

## ✨ 核心功能

### ✅ 视频内容分析
- ✅ 支持视频链接（YouTube, Bilibili等）
- ✅ 支持视频文件上传
- ✅ 自动下载和压缩（超过2GB自动处理）
- ✅ Whisper自动转录（支持中文）

### ✅ AI模型支持
- ✅ Gemini 2.5 Pro（视频直接分析）
- ✅ ChatGPT（关键词、专家观点、摘要）
- ✅ DeepSeek（关键词、专家观点）
- ✅ Qwen（关键词、专家观点）

### ✅ 关键词提取
- ✅ AI模型提取（高质量）
- ✅ Jieba本地分词（备用）

### ✅ 专家观点生成
- ✅ Few-Shot学习支持
- ✅ 多模型支持
- ✅ 结构化JSON输出

### ✅ RAG智能搜索（可选）
- ✅ 向量数据库检索
- ✅ 语义搜索
- ✅ 分类筛选
- ⚠️ 需要PostgreSQL数据库（如不使用可忽略）

---

## 📋 使用流程

### 场景1：分析视频链接（最简单）

1. 打开 `http://localhost:5001`
2. 在"全局配置"输入 **Gemini API Key**
3. 在"视频输入"区域：
   - 粘贴视频链接
   - 选择"Gemini 2.5 Pro"
   - 输入Few-Shot示例（可选）
   - 点击"分析视频链接"
4. 等待完成，查看结果：
   - ✅ 转录文本
   - ✅ 关键词
   - ✅ 专家观点文章

### 场景2：上传视频文件

1. 点击"上传视频文件"
2. 拖拽或选择视频文件
3. 等待Whisper转录
4. 输入Few-Shot示例（可选）
5. 选择AI模型并提取关键词
6. 生成专家观点

---

## ⚙️ 配置说明

### RAG功能（可选，如不使用可忽略）

如果需要使用RAG智能搜索，需要：

1. **安装PostgreSQL和pgvector**
```bash
# macOS
brew install postgresql pgvector

# Ubuntu
sudo apt-get install postgresql postgresql-contrib
# 然后安装pgvector扩展
```

2. **修改数据库配置**
编辑 `app.py` 第128-135行，修改数据库连接信息

3. **构建向量索引**
```bash
python build_index.py
```

**注意**：如果不使用RAG功能，可以完全忽略此配置，其他功能不受影响。

---

## 🐛 常见问题

### Q: 启动后无法访问？
- 检查端口是否被占用（默认5001）
- 尝试其他端口：`python app.py 8080`

### Q: Whisper转录很慢？
- 首次运行会下载模型（约150MB），需要等待
- 大文件需要较长时间，请耐心等待

### Q: 视频下载失败？
- 检查网络连接
- 检查磁盘空间
- 某些网站可能需要VPN

### Q: API调用失败？
- 检查API Key是否正确
- 检查网络连接
- 检查API余额
- 查看浏览器控制台错误信息

### Q: ChatGPT模型报错？
- 代码中使用 `gpt-5.1`，如果不可用请修改 `app.py` 第28行：
  ```python
  OPENAI_MODEL = "gpt-4o-mini"  # 改为可用的模型
  ```

---

## 📁 项目结构

```
Vedio-content-summary-website/
├── app.py                 # Flask后端主文件
├── requirements.txt       # Python依赖
├── run.sh                # 启动脚本
├── DEPLOYMENT.md         # 详细部署文档
├── README_FOR_MENTOR.md  # 本文件
├── templates/
│   ├── home.html         # 首页
│   ├── index.html        # 视频分析页面
│   └── index2.html       # RAG搜索页面
├── uploads/              # 上传文件目录（自动创建）
├── transcripts/          # 转录文本目录（自动创建）
└── transcripts_clean/    # 清洗文本目录（自动创建）
```

---

## ✅ 功能完整性检查

- ✅ 视频链接解析和下载
- ✅ 视频文件上传
- ✅ 自动转录（Whisper）
- ✅ 文本清洗
- ✅ 关键词提取（多模型）
- ✅ 专家观点生成（Few-Shot）
- ✅ 摘要生成
- ✅ RAG搜索（可选）
- ✅ 前端界面完整
- ✅ 错误处理完善
- ✅ API Key管理
- ✅ 跨页面数据共享

---

## 🔒 安全提示

1. **API Key安全**：API Key仅在浏览器localStorage中保存，刷新页面后需要重新输入
2. **文件存储**：上传的文件保存在本地 `uploads/` 目录，建议定期清理
3. **生产环境**：当前版本为开发版本，不建议直接暴露到公网

---

## 📞 技术支持

如遇问题，请检查：
1. ✅ Python版本（需要3.8+）
2. ✅ 依赖是否完整安装
3. ✅ ffmpeg是否安装
4. ✅ 网络连接是否正常
5. ✅ API Key是否有效

---

## 📝 总结

**✅ 可以直接使用**：只要安装了依赖和ffmpeg，配置了API Key，就可以直接使用所有核心功能。

**✅ 功能完善**：所有主要功能都已实现并测试。

**⚠️ 注意事项**：
- RAG功能需要PostgreSQL数据库（可选）
- ChatGPT模型名称可能需要根据实际情况调整
- 首次运行Whisper会下载模型，需要等待

---

**祝使用愉快！** 🎉
