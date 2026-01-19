# 部署和使用指南

## 📦 快速部署（给导师使用）

### 系统要求
- Python 3.8+ 
- 网络连接（用于API调用）
- 至少 2GB 可用磁盘空间（用于视频处理和Whisper模型）

### 一键启动步骤

#### macOS / Linux:
```bash
# 1. 进入项目目录
cd Vedio-content-summary-website

# 2. 安装依赖（首次运行）
pip3 install -r requirements.txt

# 3. 安装 ffmpeg（如果未安装）
# macOS:
brew install ffmpeg
# Ubuntu:
sudo apt-get install ffmpeg

# 4. 启动应用
./run.sh
# 或者
python3 app.py
```

#### Windows:
```bash
# 1. 进入项目目录
cd Vedio-content-summary-website

# 2. 安装依赖（首次运行）
pip install -r requirements.txt

# 3. 安装 ffmpeg（下载并添加到PATH）
# 下载地址: https://ffmpeg.org/download.html

# 4. 启动应用
python app.py
```

### 访问应用
打开浏览器访问：`http://localhost:5001`

---

## 🔑 API Key 配置

### 必需配置（至少选择一个AI模型）

1. **Gemini API Key**（推荐，支持视频直接分析）
   - 获取地址：https://makersuite.google.com/app/apikey
   - 用途：视频内容分析、关键词提取、专家观点生成

2. **OpenAI API Key**（ChatGPT）
   - 获取地址：https://platform.openai.com/api-keys
   - 用途：关键词提取、专家观点生成、摘要生成
   - 注意：代码中使用的是 `gpt-5.1`，如果不可用请修改 `app.py` 第28行

3. **DeepSeek API Key**（可选）
   - 获取地址：https://platform.deepseek.com/api_keys
   - 用途：关键词提取、专家观点生成

4. **Qwen API Key**（可选，阿里云百炼）
   - 获取地址：https://dashscope.console.aliyun.com/apiKey
   - 用途：关键词提取、专家观点生成

### 使用方式
- 在 Web 界面首页的"全局配置"区域输入对应的 API Key
- API Key 会保存在浏览器 localStorage 中，刷新页面后需要重新输入
- API Key 不会发送到服务器保存，仅在当前会话使用

---

## ✨ 功能清单

### ✅ 已实现功能

1. **视频内容分析**
   - ✅ 支持视频文件上传（MP4, AVI, MOV等）
   - ✅ 支持视频链接解析（YouTube, Bilibili等）
   - ✅ 自动视频下载和压缩（超过2GB自动压缩）
   - ✅ Whisper 音频转录（支持中文）
   - ✅ 文本清洗（去除填充词、重复等）

2. **AI模型支持**
   - ✅ Gemini 2.5 Pro（视频直接分析）
   - ✅ ChatGPT（关键词提取、专家观点、摘要）
   - ✅ DeepSeek（关键词提取、专家观点）
   - ✅ Qwen（关键词提取、专家观点）

3. **关键词提取**
   - ✅ AI模型提取（Gemini/ChatGPT/DeepSeek/Qwen）
   - ✅ Jieba本地分词（备用方案）
   - ✅ 自动过滤通用词和学术术语提取

4. **专家观点生成**
   - ✅ Few-Shot学习支持
   - ✅ 多模型支持（Gemini/ChatGPT/DeepSeek/Qwen）
   - ✅ 结构化输出（JSON格式）

5. **摘要生成**
   - ✅ 基于原摘要、关键词和转录文本生成
   - ✅ 可调节长度和来源比例

6. **RAG智能搜索**（需要PostgreSQL数据库）
   - ✅ 向量数据库检索
   - ✅ 语义搜索
   - ✅ 分类筛选
   - ✅ 查询意图分析

---

## ⚙️ 配置说明

### RAG功能配置（可选）

如果需要使用RAG智能搜索功能，需要配置PostgreSQL数据库：

1. **安装PostgreSQL和pgvector扩展**
```bash
# macOS
brew install postgresql
brew install pgvector

# Ubuntu
sudo apt-get install postgresql postgresql-contrib
# 然后安装pgvector扩展
```

2. **修改数据库配置**
编辑 `app.py` 第128-135行：
```python
RAG_DB_CONFIG = {
    'DB_NAME': 'postgres',           # 数据库名
    'DB_USER': 'postgres',           # 用户名
    'DB_PASSWORD': 'your_password',  # 密码
    'DB_HOST': 'localhost',          # 主机
    'DB_PORT': '5433',               # 端口
    'TABLE_NAME': 'cas_reports'      # 表名
}
```

3. **构建向量索引**
```bash
# 运行构建索引脚本（需要先准备好markdown文件）
python build_index.py
```

**注意**：如果不使用RAG功能，可以忽略此配置，其他功能不受影响。

---

## 🐛 常见问题

### Q: 启动后无法访问？
A: 
- 检查端口是否被占用（默认5001）
- 尝试使用其他端口：`python app.py 8080`
- 检查防火墙设置

### Q: Whisper转录很慢？
A: 
- 首次运行会下载模型（约150MB），需要等待
- 大文件需要较长时间，请耐心等待
- 可以在 `app.py` 中修改Whisper模型大小（第147行）

### Q: 视频下载失败？
A: 
- 检查网络连接
- 检查磁盘空间（需要至少视频文件大小的2倍空间）
- 某些网站可能需要VPN

### Q: API调用失败？
A: 
- 检查API Key是否正确
- 检查网络连接（需要能访问对应API服务）
- 检查API余额是否充足
- 查看浏览器控制台错误信息

### Q: RAG搜索报错？
A: 
- 如果不使用RAG功能，可以忽略
- 如果使用，确保PostgreSQL数据库已启动
- 确保已构建向量索引
- 检查数据库连接配置

---

## 📝 使用流程示例

### 场景1：分析视频链接（推荐）

1. 打开 `http://localhost:5001`
2. 在"全局配置"输入 Gemini API Key
3. 在"视频输入"区域：
   - 粘贴视频链接（如YouTube链接）
   - 选择AI模型（推荐Gemini）
   - 输入Few-Shot示例（可选）
   - 点击"分析视频链接"
4. 等待分析完成，查看结果：
   - 转录文本
   - 关键词
   - 专家观点文章

### 场景2：上传视频文件

1. 在"视频输入"区域点击"上传视频文件"
2. 拖拽或选择视频文件
3. 等待Whisper转录完成
4. 输入Few-Shot示例（可选）
5. 选择AI模型并提取关键词
6. 生成专家观点和摘要

### 场景3：使用RAG搜索

1. 确保已配置PostgreSQL数据库和向量索引
2. 访问 `http://localhost:5001/index2.html`
3. 在Video Content Analysis页面输入Gemini API Key
4. 在RAG搜索页面输入搜索关键词
5. 选择分类筛选（可选）
6. 查看搜索结果

---

## 🔒 安全建议

1. **不要将API Key提交到代码仓库**
2. **生产环境建议添加用户认证**
3. **限制文件上传大小和类型**
4. **定期清理临时文件**（`uploads/`, `transcripts/`目录）

---

## 📞 技术支持

如遇问题，请检查：
1. Python版本（需要3.8+）
2. 依赖是否完整安装
3. ffmpeg是否安装
4. 网络连接是否正常
5. API Key是否有效

---

## 📄 许可证

MIT License
