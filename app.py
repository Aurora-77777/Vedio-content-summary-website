from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
import os
import json
import time
import pandas as pd
from pathlib import Path
from werkzeug.utils import secure_filename
from openai import OpenAI
import re
import jieba
import jieba.analyse
from typing import Dict, List
import subprocess
import tempfile
import shutil
import whisper

app = Flask(__name__)
CORS(app)

# 配置
UPLOAD_FOLDER = Path("uploads")
TRANSCRIPTS_FOLDER = Path("transcripts")
TRANSCRIPTS_CLEAN_FOLDER = Path("transcripts_clean")
UPLOAD_FOLDER.mkdir(exist_ok=True)
TRANSCRIPTS_FOLDER.mkdir(exist_ok=True)
TRANSCRIPTS_CLEAN_FOLDER.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'flv', 'webm', 'mp3', 'wav', 'm4a'}

# 关键词提取配置
MAX_KEYWORDS_TOTAL = 30
MIN_WORD_LENGTH = 2
MAX_TRANSCRIPT_LENGTH = 3000
MAX_DESCRIPTION_LENGTH = 500

COMMON_WORDS = [
    '研究', '方法', '技术', '系统', '问题', '应用', '发展', '实现',
    '分析', '设计', '开发', '构建', '优化', '提出', '创新', '改进',
    '验证', '测试', '评估', '策略', '方案', '工艺', '过程', '结果',
    '数据', '信息', '内容', '方面', '领域', '方向', '工作', '项目'
]

METHOD_KEYWORDS = [
    '设计', '开发', '构建', '优化', '实现', '提出', '创新',
    '改进', '研究', '分析', '验证', '测试', '评估', '应用',
    '方法', '策略', '方案', '技术', '工艺'
]

FILLER_WORDS_CHINESE = [
    '这个', '那个', '然后', '就是', '就是说',
    '嗯', '呃', '啊', '那个那个', '这个这个', '然后然后',
    '呃呃', '嗯嗯', '啊啊'
]

FILLER_WORDS_ENGLISH = [
    r'\bum\b', r'\buh\b', r'\byou know\b', r'\blike\b'
]

# 初始化 jieba
try:
    jieba.initialize()
except:
    pass


def create_openai_client(api_key: str):
    old_key = os.environ.get('OPENAI_API_KEY', None)
    
    try:
        # 临时设置环境变量
        os.environ['OPENAI_API_KEY'] = api_key
        
        # 使用环境变量初始化（不传任何参数，避免参数冲突）
        client = OpenAI()
        print("成功: 使用环境变量方式初始化 OpenAI 客户端")
        return client
    except Exception as e:
        error_msg = str(e)
        print(f"环境变量方式失败: {error_msg}")
        
        # 如果环境变量方式失败，尝试直接传参
        try:
            # 清除环境变量，使用直接传参
            if 'OPENAI_API_KEY' in os.environ:
                del os.environ['OPENAI_API_KEY']
            
            client = OpenAI(api_key=api_key)
            print("成功: 使用直接传参方式初始化 OpenAI 客户端")
            return client
        except Exception as e2:
            # 恢复原来的环境变量
            if old_key:
                os.environ['OPENAI_API_KEY'] = old_key
            elif 'OPENAI_API_KEY' in os.environ:
                del os.environ['OPENAI_API_KEY']
            
            raise Exception(
                f"OpenAI 客户端初始化失败。\n"
                f"环境变量方式错误: {error_msg}\n"
                f"直接传参方式错误: {e2}\n"
                f"请检查 OpenAI SDK 版本: pip show openai\n"
                f"或尝试更新: pip install --upgrade openai"
            )
    finally:
        # 确保恢复环境变量（如果之前有的话）
        if old_key and 'OPENAI_API_KEY' not in os.environ:
            os.environ['OPENAI_API_KEY'] = old_key


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    return text.strip()


def remove_filler_words(text: str) -> str:
    for word in FILLER_WORDS_CHINESE:
        text = text.replace(word, '')
    for pattern in FILLER_WORDS_ENGLISH:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    return text


def remove_repeats(text: str) -> str:
    text = re.sub(r'(\S+)(\s+|\W+)\1{2,}', r'\1', text)
    return text


def fix_punctuation(text: str) -> str:
    text = re.sub(r'[，,]{2,}', '，', text)
    text = re.sub(r'[。.]{2,}', '。', text)
    text = re.sub(r'[！!]{2,}', '！', text)
    text = re.sub(r'[？?]{2,}', '？', text)
    return text


def merge_broken_sentences(text: str) -> str:
    lines = text.split('\n')
    merged = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if len(line) < 10 and i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            if next_line:
                merged.append(line + next_line)
                i += 2
                continue
        merged.append(line)
        i += 1
    return '\n'.join(merged)


def clean_text(text: str) -> str:
    text = normalize_text(text)
    text = remove_filler_words(text)
    text = remove_repeats(text)
    text = fix_punctuation(text)
    text = merge_broken_sentences(text)
    text = normalize_text(text)
    return text


def extract_keywords_by_category(text: str) -> Dict[str, List[str]]:
    if not text or len(text.strip()) < 50:
        return {'all': [], 'methodology': []}
    
    keywords_noun = jieba.analyse.extract_tags(
        text,
        topK=MAX_KEYWORDS_TOTAL * 3,
        withWeight=True,
        allowPOS=('n', 'nz', 'nt', 'ns', 'nr')
    )
    
    keywords_verb_adj = jieba.analyse.extract_tags(
        text,
        topK=MAX_KEYWORDS_TOTAL * 2,
        withWeight=True,
        allowPOS=('v', 'vn', 'a', 'an')
    )
    
    all_candidates = {}
    for word, weight in keywords_noun + keywords_verb_adj:
        if len(word) >= MIN_WORD_LENGTH:
            if word not in all_candidates or weight > all_candidates[word]:
                all_candidates[word] = weight
    
    long_words = []
    short_words = []
    
    for word, weight in sorted(all_candidates.items(), key=lambda x: x[1], reverse=True):
        if len(word) >= 3:
            long_words.append((word, weight))
        elif len(word) == 2:
            short_words.append((word, weight))
    
    filtered_keywords = []
    
    for word, weight in long_words:
        filtered_keywords.append(word)
        if len(filtered_keywords) >= MAX_KEYWORDS_TOTAL:
            break
    
    if len(filtered_keywords) < MAX_KEYWORDS_TOTAL:
        for word, weight in short_words:
            if word in filtered_keywords:
                continue
            is_common = any(cw == word for cw in COMMON_WORDS)
            if not is_common:
                filtered_keywords.append(word)
                if len(filtered_keywords) >= MAX_KEYWORDS_TOTAL:
                    break
    
    if len(filtered_keywords) < MAX_KEYWORDS_TOTAL // 2:
        for word, weight in sorted(all_candidates.items(), key=lambda x: x[1], reverse=True):
            if word not in filtered_keywords and len(word) >= MIN_WORD_LENGTH:
                filtered_keywords.append(word)
                if len(filtered_keywords) >= MAX_KEYWORDS_TOTAL:
                    break
    
    methodology_keywords = []
    for word in filtered_keywords:
        if any(mk in word for mk in METHOD_KEYWORDS):
            methodology_keywords.append(word)
    
    return {
        'all': filtered_keywords[:MAX_KEYWORDS_TOTAL],
        'methodology': methodology_keywords
    }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/transcribe', methods=['POST'])
def transcribe():
    """使用 Whisper 转录音频/视频文件"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': '没有上传文件'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': '文件名为空'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': '不支持的文件格式'}), 400
    
    try:
        filename = secure_filename(file.filename)
        filepath = UPLOAD_FOLDER / filename
        file.save(str(filepath))
        
        # 使用 whisper 转录音频
        output_path = TRANSCRIPTS_FOLDER / f"{Path(filename).stem}.txt"
        
        # 检查是否已存在转录文件
        if output_path.exists():
            transcript = output_path.read_text(encoding='utf-8', errors='ignore')
            return jsonify({
                'success': True,
                'transcript': transcript,
                'filename': filename,
                'output_path': str(output_path)
            })
        
        transcript = None
        try:
            print(f"使用 Whisper Python API 转录: {filename}")
            model_name = "small"
            try:
                model = whisper.load_model(model_name)
                print(f"已加载 {model_name} 模型")
            except Exception as e:
                print(f"无法加载 {model_name} 模型: {e}，尝试 base 模型")
                try:
                    model_name = "base"
                    model = whisper.load_model(model_name)
                    print(f"已加载 {model_name} 模型")
                except Exception as e2:
                    print(f"无法加载 base 模型: {e2}，使用 tiny 模型")
                    model_name = "tiny"
                    model = whisper.load_model(model_name)
            result = model.transcribe(
                str(filepath),
                language="zh",
                task="transcribe",
                fp16=False,
                verbose=False,
                initial_prompt="""这是一段学术演讲，包含专业术语和技术内容，请尽量准确地转写。
1. 能识别语音转写（ASR）或 OCR 文本中出现的专业术语错误；
2. 能将错误或不规范的词语，校正为该领域通用、规范的学术术语；

输出格式：
- 只输出转写后的正文
- 不要添加标题、注释、解释
"""
            )
            if "segments" in result and result["segments"]:
                transcript_parts = []
                for segment in result["segments"]:
                    text = segment.get("text", "").strip()
                    if text:
                        transcript_parts.append(text)
                transcript = "\n".join(transcript_parts)
            else:
                transcript = result.get("text", "")
            output_path.write_text(transcript, encoding='utf-8')
            print(f"转录完成: {len(transcript)} 字符 (使用 {model_name} 模型)")
        except ImportError:
            print(f"Whisper Python API 不可用，尝试命令行: {filename}")
            cmd = [
                'whisper',
                str(filepath),
                '--language', 'zh',
                '--output_dir', str(TRANSCRIPTS_FOLDER),
                '--output_format', 'txt',
                '--model', 'small',  # 使用 small 模型（质量与速度的平衡）
                '--verbose', 'False',
                '--fp16', 'False'  # 在某些系统上可以提高准确性
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)  # 20分钟超时（small模型）
            
            if result.returncode != 0:
                print(f"small 模型失败，尝试 base 模型")
                cmd[6] = 'base'  # 修改模型参数
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode != 0:
                error_msg = result.stderr[:500] if result.stderr else "未知错误"
                print(f"Whisper 命令行失败: {error_msg}")
                return jsonify({
                    'success': False,
                    'error': f'Whisper 转录失败: {error_msg}',
                    'hint': '请确保已安装 whisper: pip install openai-whisper 和 ffmpeg'
                }), 500
            
            if output_path.exists():
                transcript = output_path.read_text(encoding='utf-8', errors='ignore')
            else:
                possible_path = TRANSCRIPTS_FOLDER / f"{Path(filename).stem}.txt"
                if possible_path.exists():
                    transcript = possible_path.read_text(encoding='utf-8', errors='ignore')
                else:
                    return jsonify({
                        'success': False,
                        'error': '转录文件未生成，请检查 Whisper 是否正常工作'
                    }), 500
        
        if not transcript:
            return jsonify({
                'success': False,
                'error': '转录结果为空，请检查音频文件是否有效'
            }), 500
        
        return jsonify({
            'success': True,
            'transcript': transcript,
            'filename': filename,
            'output_path': str(output_path)
        })
    
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"转录异常: {error_trace}")
        return jsonify({
            'success': False,
            'error': f'处理失败: {str(e)}',
            'detail': error_trace[-500:] if len(error_trace) > 500 else error_trace
        }), 500


@app.route('/api/clean-transcript', methods=['POST'])
def clean_transcript():
    data = request.json
    transcript = data.get('transcript', '')
    filename = data.get('filename', 'unknown')
    
    if not transcript:
        return jsonify({'success': False, 'error': '转录文本为空'}), 400
    
    try:
        cleaned = clean_text(transcript)
        output_path = TRANSCRIPTS_CLEAN_FOLDER / f"{Path(filename).stem}.txt"
        output_path.write_text(cleaned, encoding='utf-8')
        
        return jsonify({
            'success': True,
            'cleaned_transcript': cleaned,
            'output_path': str(output_path)
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': f'清洗失败: {str(e)}'}), 500


@app.route('/api/extract-keywords', methods=['POST'])
def extract_keywords():
    data = request.json
    transcript = data.get('transcript', '')
    description = data.get('description', '')
    use_openai = data.get('use_openai', False)
    api_key = data.get('api_key', '')
    
    if not transcript and not description:
        return jsonify({'success': False, 'error': '转录文本和描述都为空'}), 400
    
    try:
        combined_text = ""
        if description and description.strip():
            combined_text = description.strip() + "\n" + description.strip() + "\n"
        if transcript and transcript.strip():
            combined_text += transcript.strip()
        
        if use_openai and api_key:
            client = create_openai_client(api_key)
            
            desc_short = description[:MAX_DESCRIPTION_LENGTH] if description else ""
            transcript_short = transcript[:MAX_TRANSCRIPT_LENGTH] if transcript else ""
            
            prompt = f"""从以下内容提取10-15个学术术语关键词：

A) description（摘要，核心信息）:
{desc_short}

B) transcript（转写文本，详细内容）:
{transcript_short}

提取要求（按优先级）：
1. 【学术术语优先】优先提取专业术语、技术名词、学科概念词
   - 长词（≥3字）优先：如"神经网络"、"电化学"、"钙钛矿"、"离子传输"
   - 专业名词优先：如"晶体管"、"器件"、"电池"、"算法"
   - 避免通用词：如"研究"、"方法"、"系统"、"功能"、"性能"、"实现"、"应用"、"发展"、"问题"、"技术"、"界面"、"策略"、"方案"

2. 【来源约束】关键词必须在A或B中明确出现（允许同义词/变体）

3. 【候选词使用】优先从candidates中选择，如果candidates质量不高或不足，可以从A/B中补充
   - 如果candidates中学术术语≥10个，优先使用candidates
   - 如果candidates中学术术语<10个，从A/B补充至10-15个

4. 【过滤规则】必须排除：
   - 泛词：研究/方法/技术/系统/功能/性能/实现/应用/发展/问题/界面/策略/方案/过程/结果/数据/信息/内容/方面/领域/方向/工作/项目
   - 机构名：学校名、公司名
   - 人名/地名
   - 低信息量词：如"界面"、"知识"、"应用"、"三座大山"、"第一桶金"、"策略"、"方案"、"过程"、"结果"、"数据"、"信息"、"内容"、"方面"、"领域"、"方向"、"工作"、"项目"

5. 【词性偏好】优先名词性词汇（专业术语通常是名词），减少动词/形容词/副词

6. 【数量】最终输出10-15个关键词，确保都是该领域的核心学术术语

输出格式（JSON）：
{{
  "keywords": ["词1","词2",...]
}}"""
            
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "你是信息抽取助手，只输出JSON格式。"},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=400
            )
            
            result_text = resp.choices[0].message.content.strip()
            result = json.loads(result_text)
            keywords = result.get('keywords', [])
            
            return jsonify({
                'success': True,
                'keywords': keywords,
                'keywords_str': '，'.join(keywords),
                'method': 'openai'
            })
        else:
            keywords_dict = extract_keywords_by_category(combined_text)
            
            return jsonify({
                'success': True,
                'keywords': keywords_dict['all'],
                'keywords_methodology': keywords_dict['methodology'],
                'keywords_str': '，'.join(keywords_dict['all']),
                'method': 'jieba'
            })
    
    except Exception as e:
        return jsonify({'success': False, 'error': f'关键词提取失败: {str(e)}'}), 500


@app.route('/api/generate-abstract', methods=['POST'])
def generate_abstract():
    data = request.json
    original_abstract = data.get('original_abstract', '')
    keywords = data.get('keywords', '')
    title = data.get('title', '')
    speaker_name = data.get('speaker_name', '')
    transcript = data.get('transcript', '')
    api_key = data.get('api_key', '')
    
    abstract_length = data.get('abstract_length', 400)
    main_source = data.get('main_source', 'balanced')  # transcript, abstract, balanced
    use_original_abstract = data.get('use_original_abstract', True)
    use_keywords = data.get('use_keywords', True)
    use_transcript = data.get('use_transcript', True)
    
    if not api_key:
        return jsonify({'success': False, 'error': '需要 OpenAI API Key'}), 400
    
    if not use_original_abstract and not use_keywords and not use_transcript:
        return jsonify({'success': False, 'error': '请至少选择一个输入源'}), 400
    
    if use_original_abstract and not original_abstract.strip():
        return jsonify({'success': False, 'error': '已选择原摘要但内容为空'}), 400
    if use_keywords and not keywords.strip():
        return jsonify({'success': False, 'error': '已选择关键词但内容为空'}), 400
    if use_transcript and not transcript.strip():
        return jsonify({'success': False, 'error': '已选择转录文本但内容为空'}), 400
    
    try:
        client = create_openai_client(api_key)
        
        # 根据用户选择准备输入内容
        input_parts = []
        
        if use_original_abstract and original_abstract:
            desc_short = original_abstract[:8000]
            input_parts.append(f"- 原摘要 desc：{desc_short}")
        
        if use_keywords and keywords:
            input_parts.append(f"- 关键词（参考，不得编造）：{keywords}")
        
        if use_transcript and transcript:
            transcript_short = transcript[:3000]
            input_parts.append(f"- 转写 transcript：{transcript_short}")
        
        input_section = "\n".join(input_parts) if input_parts else "无输入内容"
        
        # 根据主要内容来源设置优先级说明
        if main_source == 'transcript':
            priority_note = "【重要】请以转写文本 transcript 为主要事实来源，原摘要和关键词仅作为补充参考。"
        elif main_source == 'abstract':
            priority_note = "【重要】请以原摘要 desc 为主要事实来源，转写文本和关键词仅作为补充参考。"
        else:  # balanced
            priority_note = "【重要】请平衡使用所有输入源，确保摘要全面准确。"
        
        # 计算字数范围（允许±20%的浮动）
        min_length = int(abstract_length * 0.9)
        max_length = int(abstract_length * 1.1)
        
        prompt = f"""你是学术编辑且是一名熟悉该领域的科研助理。请基于以下输入生成一段 {min_length}–{max_length} 字（目标 {abstract_length} 字）的中文学术摘要。

输入：
- 标题：{title if title else '未提供'}
- 报告人：{speaker_name if speaker_name else '未提供'}
{input_section}

{priority_note}
你具备以下能力：
1. 能识别语音转写（ASR）或 OCR 文本中出现的专业术语错误；
2. 能将错误或不规范的词语，校正为该领域通用、规范的学术术语；
3. 在不引入原文未出现的新事实的前提下，基于校正后的语义撰写学术摘要。

重要约束：
1） 若发现明显不符合学术语境的词（如“气贱”“养化物”等），必须进行术语校正；
2） 不允许保留明显错误的术语进入最终摘要；
3） 摘要内容必须可被该领域研究人员理解和接受。

硬约束：
1) 摘要中的事实/方法/结果必须能在提供的输入源中找到依据；不确定就不要写。
2) 允许重组表达与学术化改写，但不得新增实验、数据、结论。
3) 摘要长度应控制在 {min_length}–{max_length} 字之间（目标 {abstract_length} 字）。

输出格式（JSON）：
{{
  "abstract": "摘要正文（{min_length}-{max_length}字）",
  "self_check": {{
    "fidelity_score": 4,
    "fallback": false
  }}
}}"""
        
        # 根据字数调整 max_tokens（中文字符大约 1 token = 1-2 字符）
        max_tokens = max(400, int(abstract_length * 1.5))  # 预留一些空间
        
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "你是学术编辑助手，只输出JSON格式。"},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=max_tokens
        )
        
        result_text = resp.choices[0].message.content.strip()
        result = json.loads(result_text)
        abstract = result.get('abstract', original_abstract)
        
        return jsonify({
            'success': True,
            'abstract': abstract
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': f'摘要生成失败: {str(e)}'}), 500


@app.route('/api/download-transcript', methods=['GET'])
def download_transcript():
    filename = request.args.get('filename', '')
    if not filename:
        return jsonify({'success': False, 'error': '文件名不能为空'}), 400
    
    filepath = TRANSCRIPTS_FOLDER / filename
    if not filepath.exists():
        return jsonify({'success': False, 'error': '文件不存在'}), 404
    
    return send_file(str(filepath), as_attachment=True, download_name=filename)


if __name__ == '__main__':
    import sys
    port = 5001
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"警告: 无效的端口号 '{sys.argv[1]}'，使用默认端口 {port}")
    
    print(f"\n{'='*60}")
    print(f"启动 Flask 应用")
    print(f"访问地址: http://localhost:{port}")
    print(f"{'='*60}\n")
    
    app.run(debug=True, host='0.0.0.0', port=port)

