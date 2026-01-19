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
from typing import Dict, List, Optional, Union
import subprocess
import tempfile
import shutil
import whisper
import sys
import traceback

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

app = Flask(__name__)
CORS(app)

GEMINI_MODEL = "gemini-2.5-pro"
OPENAI_MODEL = "gpt-5.1"
DEEPSEEK_MODEL = "deepseek-chat"
QWEN_MODEL = "qwen-max"
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

UPLOAD_FOLDER = Path("uploads")
TRANSCRIPTS_FOLDER = Path("transcripts")
TRANSCRIPTS_CLEAN_FOLDER = Path("transcripts_clean")
UPLOAD_FOLDER.mkdir(exist_ok=True)
TRANSCRIPTS_FOLDER.mkdir(exist_ok=True)
TRANSCRIPTS_CLEAN_FOLDER.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'flv', 'webm', 'mp3', 'wav', 'm4a'}

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

GEMINI_AVAILABLE = False
FEW_SHOT_EXAMPLE_TEXT = ""

try:
    from google import genai
    from google.genai import types
    import trafilatura
    from yt_dlp import YoutubeDL
    GEMINI_AVAILABLE = True
    
    try:
        gemini_try_path = Path(__file__).parent.parent.parent / "gemini_try.py"
        if gemini_try_path.exists():
            with open(gemini_try_path, 'r', encoding='utf-8') as f:
                content = f.read()
                match = re.search(r'FEW_SHOT_EXAMPLE_TEXT\s*=\s*"""(.*?)"""', content, re.DOTALL)
                if match:
                    FEW_SHOT_EXAMPLE_TEXT = match.group(1).strip()
    except:
        pass
except ImportError as e:
    print(f"警告: 无法导入Gemini相关模块: {e}")

try:
    from openai import OpenAI as DeepSeekClient
    DEEPSEEK_AVAILABLE = True
except:
    DEEPSEEK_AVAILABLE = False

try:
    jieba.initialize()
except:
    pass

# RAG相关配置和初始化
RAG_AVAILABLE = False
RAG_ENGINE = None
RAG_ENGINE_API_KEY = None  # 跟踪当前RAG引擎使用的API Key

try:
    from llama_index.core import VectorStoreIndex
    from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterOperator
    from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
    from llama_index.llms.google_genai import GoogleGenAI
    from llama_index.core import Settings
    try:
        from llama_index.vector_stores.postgres import PGVectorStore
    except ImportError:
        try:
            from llama_index_postgres import PGVectorStore
        except ImportError:
            PGVectorStore = None
    
    RAG_AVAILABLE = True
except ImportError as e:
    print(f"警告: RAG模块未安装: {e}")

RAG_DB_CONFIG = {
    'DB_NAME': 'postgres' 
    'DB_USER': 'postgres.ivngzzdpdvcryhzevhsi' 
    'DB_PASSWORD': 'ZssHjq19880302_' 
    'DB_HOST': 'aws-0-ap-southeast-1.pooler.supabase.com' 
    'DB_PORT': '6543'            
}

BROAD_CATEGORIES = [
    "数学", "力学", "物理学", "化学", "天文学", "地球科学", "生物学",
    "农学", "林学", "畜牧/兽医科学", "水产学", "医药卫生",
    "测绘科学技术", "材料科学与工程", "冶金工程技术", "机械与运载工程",
    "核科学技术", "电子与通信技术", "计算机科学技术", "化学工程",
    "轻工纺织科学技术", "食品科学技术", "土木建筑工程", "水利工程",
    "交通运输", "航空航天科学技术", "环境及资源", "安全科学技术",
    "管理学", "社会人文", "其他"
]

def create_openai_client(api_key: str, base_url: str = None):
    old_key = os.environ.get('OPENAI_API_KEY', None)
    try:
        os.environ['OPENAI_API_KEY'] = api_key
        if base_url:
            client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            client = OpenAI()
        return client
    except Exception as e:
        try:
            if 'OPENAI_API_KEY' in os.environ:
                del os.environ['OPENAI_API_KEY']
            client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
            return client
        except Exception as e2:
            if old_key: os.environ['OPENAI_API_KEY'] = old_key
            raise Exception(f"OpenAI 客户端初始化失败: {str(e)} | {str(e2)}")
    finally:
        if old_key and 'OPENAI_API_KEY' not in os.environ:
            os.environ['OPENAI_API_KEY'] = old_key

def create_qwen_client(api_key: str):
    """创建Qwen（阿里云百炼）客户端，使用OpenAI兼容接口"""
    return create_openai_client(api_key, base_url=QWEN_BASE_URL)

def create_gemini_client(api_key: str):
    if not GEMINI_AVAILABLE:
        raise ImportError("Gemini 模块未安装或不可用")
    from google import genai
    return genai.Client(api_key=api_key)

def cleanup_files(*file_paths):
    for path in file_paths:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except Exception as e:
                print(f"清理文件失败 {path}: {e}")

def transcribe_with_whisper(filepath: str) -> str:
    """
    """
    print(f"开始 Whisper 转录: {filepath}")
    
    try:
        model_name = "small"
        model = None
        
        try:
            model = whisper.load_model("small")
            print("已加载 small 模型")
        except Exception:
            try:
                print("加载 small 失败，尝试 base")
                model_name = "base"
                model = whisper.load_model("base")
            except Exception:
                print("加载 base 失败，尝试 tiny")
                model_name = "tiny"
                model = whisper.load_model("tiny")
        
        result = model.transcribe(
            str(filepath),
            language="zh",
            task="transcribe",
            fp16=False,
            verbose=False,
            initial_prompt="""这是一段学术演讲，包含专业术语和技术内容，请尽量准确地转写。
1. 能识别语音转写（ASR）或 OCR 文本中出现的专业术语错误；
2. 能将错误或不规范的词语，校正为该领域通用、规范的学术术语；
输出格式：只输出转写后的正文，不要添加标题、注释、解释"""
        )
        
        if "segments" in result and result["segments"]:
            return "\n".join([s.get("text", "").strip() for s in result["segments"] if s.get("text")])
        return result.get("text", "")

    except ImportError:
        print(f"Whisper Python API 不可用，尝试命令行: {filepath}")
        cmd = [
            'whisper', str(filepath),
            '--language', 'zh',
            '--output_dir', str(TRANSCRIPTS_FOLDER),
            '--output_format', 'txt',
            '--model', 'small',
            '--verbose', 'False',
            '--fp16', 'False'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
        if result.returncode != 0:
            print("CLI small 模型失败，尝试 base")
            cmd[6] = 'base'
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
        if result.returncode != 0:
            raise Exception(f"Whisper 命令行失败: {result.stderr[:200]}")
            
        expected_output = TRANSCRIPTS_FOLDER / f"{Path(filepath).stem}.txt"
        if expected_output.exists():
            return expected_output.read_text(encoding='utf-8', errors='ignore')
        else:
            raise Exception("转录文件未生成")

def generate_keyword_extraction_prompt(description: str, transcript: str) -> str:
    desc_short = description[:MAX_DESCRIPTION_LENGTH] if description else ""
    transcript_short = transcript[:MAX_TRANSCRIPT_LENGTH] if transcript else ""
    
    return f"""从以下内容提取10-15个学术术语关键词：

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

3. 【过滤规则】必须排除：
   - 泛词：研究/方法/技术/系统/功能/性能/实现/应用/发展/问题/界面/策略/方案/过程/结果/数据/信息/内容/方面/领域/方向/工作/项目
   - 机构名：学校名、公司名
   - 人名/地名
   - 低信息量词：如"界面"、"知识"、"应用"、"策略"、"方案"、"过程"、"结果"、"数据"、"信息"、"内容"、"方面"、"领域"、"方向"、"工作"、"项目"
   - 口语化词汇：如"基本上"、"有没有"、"相当于"、"舒服"、"肯定"、"证出来"等
   - 动词性通用词：如"能够"、"收获"、"接水"等

4. 【词性偏好】优先名词性词汇（专业术语通常是名词），减少动词/形容词/副词

5. 【质量要求】
   - 确保每个关键词都是该领域的核心学术术语
   - 避免提取口语化表达、填充词、语气词
   - 关键词应该具有明确的学术含义和技术价值

6. 【数量】最终输出10-15个关键词，确保都是该领域的核心学术术语

输出格式（JSON）：
{{
  "keywords": ["词1","词2",...]
}}"""

def generate_expert_opinion_prompt(web_text, transcript, few_shot_text, has_video=False):
    if has_video:
        input_section = f"""
    1. **网页简介**：{web_text}
    2. **视频内容**：请重点分析视频中的 PPT 视觉信息（数据/架构图）和 语音论述。
"""
        visual_note = "如果视频 PPT 中展示了关键数据或模型，请在文章中用文字描述出来。"
        visual_evidence_desc = "列出视频中3个关键图表/PPT页面的详细内容"
    else:
        input_section = f"""
    1. **网页简介**：{web_text if web_text else '（未能提取到网页文字）'}
    2. **视频转录内容**：{transcript[:5000] if transcript else '（无转录内容）'}
"""
        visual_note = "如果转录文本中提到了关键数据或模型，请在文章中用文字描述出来。"
        visual_evidence_desc = "列出视频中3个关键图表/PPT页面的详细内容（如果转录文本中有提到）"
    
    return f"""
    你是由中国科学院学部分配的资深学术编辑。你的任务是根据提供的学术报告视频，撰写一篇高质量的《专家观点》文章。

    ### 核心参照范例 (Style Reference)：
    请模仿以下范例的**叙述口吻**和**文章结构**：
    {few_shot_text}
    
    ### 输入背景资料：
{input_section}
    ### 写作任务要求：
    1. **文体风格**：深度报道风格，使用"报告人指出"等第三人称表述。语言严谨、典雅。
    2. **结构要求**：约 1200 字，必须包含小标题，开头注明论坛名称。
    3. **视觉融合**：{visual_note}
    
    ### 输出格式 (JSON)：
    {{
        "visual_evidence": {{
            "description": "{visual_evidence_desc}",
            "details": [{{"timestamp": "MM:SS", "content": "..."}}]
        }},
        "speaker_emphasis": {{
            "description": "...",
            "points": ["..."]
        }},
        "comprehensive_summary": "..."
    }}
    """

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def remove_filler_words(text: str) -> str:
    for word in FILLER_WORDS_CHINESE:
        text = text.replace(word, '')
    for pattern in FILLER_WORDS_ENGLISH:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    return text

def remove_repeats(text: str) -> str:
    return re.sub(r'(\S+)(\s+|\W+)\1{2,}', r'\1', text)

def fix_punctuation(text: str) -> str:
    for char in ['，', '。', '！', '？']:
        text = re.sub(f'[{char}]{{2,}}', char, text)
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
    return normalize_text(text)

def extract_keywords_by_category(text: str) -> Dict[str, List[str]]:
    if not text or len(text.strip()) < 50:
        return {'all': [], 'methodology': []}
    
    keywords_noun = jieba.analyse.extract_tags(
        text, topK=MAX_KEYWORDS_TOTAL * 3, withWeight=True, allowPOS=('n', 'nz', 'nt', 'ns', 'nr')
    )
    keywords_verb_adj = jieba.analyse.extract_tags(
        text, topK=MAX_KEYWORDS_TOTAL * 2, withWeight=True, allowPOS=('v', 'vn', 'a', 'an')
    )
    
    all_candidates = {}
    for word, weight in keywords_noun + keywords_verb_adj:
        if len(word) >= MIN_WORD_LENGTH:
            if word not in all_candidates or weight > all_candidates[word]:
                all_candidates[word] = weight
    
    filtered_keywords = []
    sorted_words = sorted(all_candidates.items(), key=lambda x: x[1], reverse=True)
    long_words = [w for w, _ in sorted_words if len(w) >= 3]
    short_words = [w for w, _ in sorted_words if len(w) == 2]
    
    filtered_keywords.extend(long_words[:MAX_KEYWORDS_TOTAL])
    
    if len(filtered_keywords) < MAX_KEYWORDS_TOTAL:
        for word in short_words:
            if word not in filtered_keywords and word not in COMMON_WORDS:
                filtered_keywords.append(word)
                if len(filtered_keywords) >= MAX_KEYWORDS_TOTAL: break
                
    methodology_keywords = [w for w in filtered_keywords if any(mk in w for mk in METHOD_KEYWORDS)]
    
    return {
        'all': filtered_keywords[:MAX_KEYWORDS_TOTAL],
        'methodology': methodology_keywords
    }

def init_rag_engine(api_key=None):
    """初始化RAG搜索引擎（延迟加载）"""
    global RAG_ENGINE, RAG_ENGINE_API_KEY
    
    # 如果API Key未提供，尝试从环境变量获取
    if not api_key:
        import os
        api_key = os.environ.get('GOOGLE_API_KEY') or os.environ.get('GEMINI_API_KEY')
    
    if not api_key:
        raise ValueError("需要提供Gemini API Key来初始化RAG引擎")
    
    # 如果引擎已存在且API Key相同，直接返回
    if RAG_ENGINE is not None and RAG_ENGINE_API_KEY == api_key:
        return RAG_ENGINE
    
    if not RAG_AVAILABLE:
        raise ImportError("RAG模块未安装，请安装: pip install llama-index llama-index-vector-stores-postgres")
    
    if not GEMINI_AVAILABLE:
        raise ImportError("Gemini模块未安装，请安装: pip install google-genai")
    
    try:
        # 设置环境变量（LlamaIndex的GoogleGenAI会从环境变量读取）
        import os
        os.environ["GOOGLE_API_KEY"] = api_key
        
        # 配置LlamaIndex（会从环境变量读取API Key）
        Settings.embed_model = GoogleGenAIEmbedding(model_name="models/text-embedding-004")
        Settings.llm = GoogleGenAI(model="models/gemini-2.5-pro", temperature=0.1)
        
        # 连接PostgreSQL向量数据库
        vector_store = PGVectorStore.from_params(
            database=RAG_DB_CONFIG['DB_NAME'],
            host=RAG_DB_CONFIG['DB_HOST'],
            password=RAG_DB_CONFIG['DB_PASSWORD'],
            port=RAG_DB_CONFIG['DB_PORT'],
            user=RAG_DB_CONFIG['DB_USER'],
            table_name=RAG_DB_CONFIG['TABLE_NAME'],
            embed_dim=768
        )
        
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        RAG_ENGINE = SmartSearchEngine(index)
        RAG_ENGINE_API_KEY = api_key  # 保存API Key以便后续检查
        
        return RAG_ENGINE
    except Exception as e:
        raise RuntimeError(f"RAG引擎初始化失败: {str(e)}")

class SmartSearchEngine:
    """智能搜索引擎类（从RAG_LlamdaIndex3.py迁移）"""
    def __init__(self, index):
        self.index = index
        self.analyzer_model = GEMINI_MODEL
    
    def analyze_query(self, user_query, api_key):
        """分析用户查询意图"""
        if not GEMINI_AVAILABLE:
            return {"categories": [], "keywords": []}
        
        prompt = f"""
        你是一个学术搜索助手。用户正在搜索学术报告。
        请根据用户的问题，推断出最可能的【宽泛学科分类】（1-2个）和【扩展检索词】（3-5个）。
        
        【可选分类列表】：{", ".join(BROAD_CATEGORIES)}
        【用户问题】："{user_query}"
        
        要求：
        1. 分类必须完全来自给定的列表。如果无法判断，返回空列表。
        2. 扩展检索词应该是该领域具体的专业术语，用于帮助向量检索找到相关性高的内容。
        3. 请直接输出 JSON 格式，包含 "categories" 和 "keywords" 两个字段。
        
        示例 JSON 输出：
        {{
            "categories": ["化学", "生物学"],
            "keywords": ["分子动力学", "蛋白质结构", "药物配体", "有机合成"]
        }}
        """
        
        try:
            client = create_gemini_client(api_key)
            response = client.models.generate_content(
                model=self.analyzer_model,
                contents=[types.Content(parts=[types.Part(text=prompt)])],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.2
                )
            )
            text = response.text.replace("```json", "").replace("```", "").strip()
            analysis_result = json.loads(text)
            return analysis_result
        except Exception as e:
            print(f" 意图分析失败: {e}")
            traceback.print_exc()
            return {"categories": [], "keywords": []}
    
    def search(self, user_query, api_key, categories_filter=None):
        """执行搜索"""
        analysis = self.analyze_query(user_query, api_key)
        target_categories = categories_filter if categories_filter else analysis.get("categories", [])
        expanded_keywords = analysis.get("keywords", [])
        
        filters = None
        if target_categories:
            filter_list = [
                MetadataFilter(key="categories", value=cat, operator=FilterOperator.CONTAINS)
                for cat in target_categories
            ]
            filters = MetadataFilters(filters=filter_list, condition="or")
        
        enhanced_query = f"{user_query} {' '.join(expanded_keywords)}"
        
        try:
            query_engine = self.index.as_query_engine(
                filters=filters,
                similarity_top_k=10
            )
            
            response = query_engine.query(enhanced_query)
            
            results_dict = {}  # key: title或original_url, value: result with highest score
            if hasattr(response, 'source_nodes'):
                for node in response.source_nodes:
                    metadata = node.metadata if hasattr(node, 'metadata') else {}
                    text = node.text if hasattr(node, 'text') else ''
                    score = node.score if hasattr(node, 'score') else 0.0
                    
                    # 使用title或original_url作为唯一标识
                    unique_key = metadata.get('title') or metadata.get('original_url') or f"result_{len(results_dict)}"
                    
                    # 提取核心摘要（优先从metadata，否则从text中提取）
                    summary = metadata.get('summary', '')
                    if not summary and text:
                        # 尝试从text中提取【核心摘要】部分
                        summary_match = re.search(r'【核心摘要】\s*\n(.*?)(?=\n---|\n###|$)', text, re.DOTALL)
                        if summary_match:
                            summary = summary_match.group(1).strip()
                        else:
                            # 如果没有找到【核心摘要】，使用text的前500字符作为摘要
                            summary = text[:500] + ('...' if len(text) > 500 else '')
                    
                    # 如果该报告已存在，比较相似度，保留更高的
                    if unique_key in results_dict:
                        if score > (results_dict[unique_key].get('score') or 0):
                            results_dict[unique_key] = {
                                'text': summary,  # 只保存摘要
                                'full_text': text,  # 保存完整文本作为backup
                                'metadata': metadata,
                                'score': score
                            }
                    else:
                        results_dict[unique_key] = {
                            'text': summary,  # 只保存摘要
                            'full_text': text,  # 保存完整文本作为backup
                            'metadata': metadata,
                            'score': score
                        }
            
            # 转换为列表并按相似度排序
            results = list(results_dict.values())
            results.sort(key=lambda x: x.get('score', 0), reverse=True)
            
            return {
                'response': str(response),
                'results': results,
                'analysis': analysis
            }
        except Exception as e:
            raise RuntimeError(f"搜索失败: {str(e)}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/index2')
def index2():
    return render_template('index2.html')

@app.route('/.well-known/<path:path>')
def well_known(path):
    return '', 204

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/api/transcribe', methods=['POST'])
def transcribe():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': '没有上传文件'}), 400
    
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'success': False, 'error': '无效的文件'}), 400
    
    try:
        filename = secure_filename(file.filename)
        filepath = UPLOAD_FOLDER / filename
        file.save(str(filepath))
        
        output_path = TRANSCRIPTS_FOLDER / f"{Path(filename).stem}.txt"
        
        if output_path.exists():
            return jsonify({
                'success': True,
                'transcript': output_path.read_text(encoding='utf-8', errors='ignore'),
                'filename': filename,
                'output_path': str(output_path)
            })
        
        transcript = transcribe_with_whisper(str(filepath))
        
        output_path.write_text(transcript, encoding='utf-8')
        
        return jsonify({
            'success': True,
            'transcript': transcript,
            'filename': filename,
            'output_path': str(output_path)
        })
    
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"转录异常: {error_trace}")
        return jsonify({'success': False, 'error': str(e), 'detail': error_trace}), 500

@app.route('/api/clean-transcript', methods=['POST'])
def clean_transcript_route():
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
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/extract-keywords', methods=['POST'])
def extract_keywords():
    data = request.json
    transcript = data.get('transcript', '')
    description = data.get('description', '')
    use_openai = data.get('use_openai', False)
    use_gemini = data.get('use_gemini', False)
    api_key = data.get('api_key', '')
    
    if not transcript and not description:
        return jsonify({'success': False, 'error': '内容为空'}), 400
    
    try:
        if (use_gemini or use_openai) and api_key:
            prompt = generate_keyword_extraction_prompt(description, transcript)
            keywords = []
            method = ""
            
            if use_gemini and GEMINI_AVAILABLE:
                client = create_gemini_client(api_key)
                response = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=[types.Content(parts=[types.Part(text=prompt)])],
                    config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.3)
                )
                result = json.loads(response.text.strip())
                keywords = result.get('keywords', [])
                method = 'gemini'
                
            elif use_openai:
                client = create_openai_client(api_key)
                resp = client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": "你是信息抽取助手，只输出JSON格式。"},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.3, max_completion_tokens=400
                )
                result = json.loads(resp.choices[0].message.content.strip())
                keywords = result.get('keywords', [])
                method = 'openai'
                
            return jsonify({
                'success': True,
                'keywords': keywords,
                'keywords_str': '，'.join(keywords),
                'method': method
            })
            
        else:
            combined_text = (description + "\n" + transcript).strip()
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
    main_source = data.get('main_source', 'balanced')
    
    if not api_key:
        return jsonify({'success': False, 'error': '需要 OpenAI API Key'}), 400
    
    try:
        client = create_openai_client(api_key)
        
        input_parts = []
        if data.get('use_original_abstract') and original_abstract:
            input_parts.append(f"- 原摘要 desc：{original_abstract[:8000]}")
        if data.get('use_keywords') and keywords:
            input_parts.append(f"- 关键词：{keywords}")
        if data.get('use_transcript') and transcript:
            input_parts.append(f"- 转写 transcript：{transcript[:3000]}")
            
        if not input_parts:
            return jsonify({'success': False, 'error': '未选择任何输入源'}), 400
            
        priority_note = {
            'transcript': "【重要】以转写文本为主要来源。",
            'abstract': "【重要】以原摘要为主要来源。",
            'balanced': "【重要】平衡使用所有来源。"
        }.get(main_source, "")
        
        min_len, max_len = int(abstract_length * 0.9), int(abstract_length * 1.1)
        
        prompt = f"""你是学术编辑。请生成 {min_len}–{max_len} 字的中文学术摘要。
输入：标题：{title}, 报告人：{speaker_name}
{chr(10).join(input_parts)}
{priority_note}
要求：术语规范，不编造事实。
输出JSON: {{"abstract": "正文", "self_check": {{...}}}}"""

        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "只输出JSON。"},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_completion_tokens=max(400, int(abstract_length * 1.5))
        )
        
        result = json.loads(resp.choices[0].message.content.strip())
        return jsonify({'success': True, 'abstract': result.get('abstract', '')})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/download-transcript', methods=['GET'])
def download_transcript():
    filename = request.args.get('filename', '')
    filepath = TRANSCRIPTS_FOLDER / filename
    if not filename or not filepath.exists():
        return jsonify({'success': False, 'error': '文件不存在'}), 404
    return send_file(str(filepath), as_attachment=True, download_name=filename)

@app.route('/api/process-video-url', methods=['POST'])
def process_video_url_api():
    try:
        data = request.json
        video_url = data.get('video_url', '').strip()
        ai_model = data.get('ai_model', 'gemini')
        api_key = data.get('api_key', '')
        few_shot_text = data.get('few_shot_text', FEW_SHOT_EXAMPLE_TEXT)
        
        if not video_url or not api_key:
            return jsonify({'success': False, 'error': '缺少必要参数'}), 400
            
        if not GEMINI_AVAILABLE:
            return jsonify({'success': False, 'error': '缺少依赖库 yt-dlp/trafilatura'}), 500
            
        from gemini_try import download_video, compress_video
        
        local_video_path = None
        compressed_file_path = None
        
        try:
            local_video_path, web_text = download_video(video_url, "temp_lecture.mp4")
            if not local_video_path:
                return jsonify({'success': False, 'error': '视频下载失败'}), 500

            if ai_model == 'gemini':
                if not GEMINI_AVAILABLE:
                    return jsonify({'success': False, 'error': 'Gemini模块不可用，请安装: pip install google-genai'}), 500
                
                # 确保types模块已导入
                from google.genai import types
                
                client = create_gemini_client(api_key)
                
                file_size = os.path.getsize(local_video_path)
                upload_path = local_video_path
                if file_size > 2.0 * 1024 * 1024 * 1024: # > 2GB
                    compressed_file_path = compress_video(local_video_path)
                    if compressed_file_path: upload_path = compressed_file_path

                video_file = client.files.upload(file=upload_path)
                
                try:
                    start_time = time.time()
                    while video_file.state.name == "PROCESSING":
                        if time.time() - start_time > 600:
                            raise TimeoutError("Gemini 视频处理超时")
                        time.sleep(5)
                        video_file = client.files.get(name=video_file.name)
                    
                    if video_file.state.name == "FAILED":
                        raise Exception(f"Gemini 处理失败: {video_file.error.message}")
                    
                    prompt = f"""分析这个学术报告视频，提取以下信息：

1. **转录文本(transcript)**：请将视频中的语音内容完整转录为文本（包括演讲者的所有论述）
   - 转录文本要完整、准确，保留专业术语
   - 能识别并校正专业术语错误，使用规范的学术术语

2. **关键词(keywords)**：从转录文本中提取10-15个核心学术术语关键词
   
   提取要求（按优先级）：
   - 【学术术语优先】优先提取专业术语、技术名词、学科概念词
     * 长词（≥3字）优先：如"神经网络"、"电化学"、"钙钛矿"、"离子传输"
     * 专业名词优先：如"晶体管"、"器件"、"电池"、"算法"
     * 避免通用词：如"研究"、"方法"、"系统"、"功能"、"性能"、"实现"、"应用"、"发展"、"问题"、"技术"、"界面"、"策略"、"方案"
   
   - 【过滤规则】必须排除：
     * 泛词：研究/方法/技术/系统/功能/性能/实现/应用/发展/问题/界面/策略/方案/过程/结果/数据/信息/内容/方面/领域/方向/工作/项目
     * 口语化词汇：如"基本上"、"有没有"、"相当于"、"舒服"、"肯定"、"证出来"等
     * 动词性通用词：如"能够"、"收获"、"接水"等
     * 机构名、人名、地名
     * 低信息量词
   
   - 【词性偏好】优先名词性词汇（专业术语通常是名词），减少动词/形容词/副词
   
   - 【质量要求】确保每个关键词都是该领域的核心学术术语，具有明确的学术含义和技术价值

网页简介参考：{web_text if web_text else '（无网页简介）'}

输出格式（JSON）：
{{
    "transcript": "完整的转录文本内容...",
    "keywords": ["关键词1", "关键词2", "关键词3", ...]
}}"""

                    response = client.models.generate_content(
                        model=GEMINI_MODEL,
                        contents=[
                            types.Content(parts=[
                                types.Part(file_data=types.FileData(file_uri=video_file.uri, mime_type=video_file.mime_type)),
                                types.Part(text=prompt)
                            ])
                        ],
                        config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.2)
                    )
                    
                    # 安全解析JSON响应
                    try:
                        res_data = json.loads(response.text)
                    except json.JSONDecodeError as e:
                        print(f"JSON解析失败: {e}")
                        print(f"响应内容: {response.text[:500]}")
                        raise Exception(f"Gemini返回的JSON格式无效: {str(e)}")
                    
                    transcript = res_data.get('transcript', '')
                    keywords = res_data.get('keywords', [])
                    
                    expert_opinion = None
                    if few_shot_text and few_shot_text.strip():
                        try:
                            op_prompt = generate_expert_opinion_prompt(web_text, "", few_shot_text, has_video=True)
                            op_res = client.models.generate_content(
                                model=GEMINI_MODEL,
                                contents=[types.Content(parts=[
                                    types.Part(file_data=types.FileData(file_uri=video_file.uri, mime_type=video_file.mime_type)),
                                    types.Part(text=op_prompt)
                                ])],
                                config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.2)
                            )
                            try:
                                expert_opinion = json.loads(op_res.text)
                            except json.JSONDecodeError as e:
                                print(f"专家观点JSON解析失败: {e}")
                                print(f"响应内容: {op_res.text[:500]}")
                                expert_opinion = {'raw_text': op_res.text, 'error': 'JSON解析失败'}
                        except Exception as e:
                            print(f"生成专家观点失败（不影响转录和关键词提取）: {e}")
                            traceback.print_exc()

                    return jsonify({
                        'success': True,
                        'transcript': transcript,
                        'keywords': keywords,
                        'keywords_str': '，'.join(keywords),
                        'web_text': web_text,
                        'expert_opinion': expert_opinion,
                        'model': 'gemini'
                    })

                finally:
                    if 'video_file' in locals() and hasattr(video_file, 'name'):
                        try: client.files.delete(name=video_file.name)
                        except: pass

            elif ai_model in ['chatgpt', 'deepseek', 'qwen']:
                transcript = transcribe_with_whisper(local_video_path)
                
                # 使用AI模型提取关键词（而不是只用Jieba）
                keywords = []
                keywords_str = ''
                try:
                    keyword_prompt = generate_keyword_extraction_prompt(web_text, transcript)
                    
                    if ai_model == 'chatgpt':
                        client = create_openai_client(api_key)
                        resp = client.chat.completions.create(
                            model=OPENAI_MODEL,
                            messages=[
                                {"role": "system", "content": "你是信息抽取助手，只输出JSON格式。"},
                                {"role": "user", "content": keyword_prompt}
                            ],
                            response_format={"type": "json_object"},
                            temperature=0.3,
                            max_completion_tokens=400
                        )
                        result = json.loads(resp.choices[0].message.content.strip())
                        keywords = result.get('keywords', [])
                        keywords_str = '，'.join(keywords)
                    
                    elif ai_model == 'deepseek':
                        if not DEEPSEEK_AVAILABLE:
                            print("DeepSeek SDK 未安装，使用Jieba提取关键词")
                            keywords_dict = extract_keywords_by_category(transcript)
                            keywords = keywords_dict['all']
                            keywords_str = '，'.join(keywords)
                        else:
                            client = DeepSeekClient(api_key=api_key, base_url="https://api.deepseek.com")
                            resp = client.chat.completions.create(
                                model=DEEPSEEK_MODEL,
                                messages=[
                                    {"role": "system", "content": "你是信息抽取助手，只输出JSON格式。"},
                                    {"role": "user", "content": keyword_prompt}
                                ],
                                response_format={"type": "json_object"},
                                temperature=0.3
                            )
                            result = json.loads(resp.choices[0].message.content.strip())
                            keywords = result.get('keywords', [])
                            keywords_str = '，'.join(keywords)
                    
                    elif ai_model == 'qwen':
                        client = create_qwen_client(api_key)
                        resp = client.chat.completions.create(
                            model=QWEN_MODEL,
                            messages=[
                                {"role": "system", "content": "你是信息抽取助手，只输出JSON格式。"},
                                {"role": "user", "content": keyword_prompt}
                            ],
                            response_format={"type": "json_object"},
                            temperature=0.3
                        )
                        result = json.loads(resp.choices[0].message.content.strip())
                        keywords = result.get('keywords', [])
                        keywords_str = '，'.join(keywords)
                        
                except Exception as e:
                    print(f"AI关键词提取失败，回退到Jieba: {e}")
                    traceback.print_exc()
                    # 回退到Jieba
                    keywords_dict = extract_keywords_by_category(transcript)
                    keywords = keywords_dict['all']
                    keywords_str = '，'.join(keywords)
                
                # 如果提供了Few-Shot文本，在步骤1就生成专家观点
                expert_opinion = None
                if few_shot_text and few_shot_text.strip():
                    try:
                        prompt = generate_expert_opinion_prompt(web_text, transcript, few_shot_text, has_video=False)
                        
                        if ai_model == 'chatgpt':
                            client = create_openai_client(api_key)
                            resp = client.chat.completions.create(
                                model=OPENAI_MODEL,
                                messages=[{"role": "system", "content": "只输出JSON。"}, {"role": "user", "content": prompt}],
                                response_format={"type": "json_object"}, temperature=0.2
                            )
                            expert_opinion = json.loads(resp.choices[0].message.content.strip())
                        
                        elif ai_model == 'deepseek':
                            if not DEEPSEEK_AVAILABLE:
                                print("DeepSeek SDK 未安装，跳过专家观点生成")
                            else:
                                client = DeepSeekClient(api_key=api_key, base_url="https://api.deepseek.com")
                                resp = client.chat.completions.create(
                                    model=DEEPSEEK_MODEL,
                                    messages=[{"role": "system", "content": "只输出JSON。"}, {"role": "user", "content": prompt}],
                                    response_format={"type": "json_object"}, temperature=0.2
                                )
                                expert_opinion = json.loads(resp.choices[0].message.content.strip())
                                print(f"DeepSeek专家观点生成成功: {len(str(expert_opinion))} 字符")
                        
                        elif ai_model == 'qwen':
                            client = create_qwen_client(api_key)
                            resp = client.chat.completions.create(
                                model=QWEN_MODEL,
                                messages=[{"role": "system", "content": "只输出JSON。"}, {"role": "user", "content": prompt}],
                                response_format={"type": "json_object"}, temperature=0.2
                            )
                            expert_opinion = json.loads(resp.choices[0].message.content.strip())
                    except Exception as e:
                        print(f"步骤1生成专家观点失败（不影响转录和关键词提取）: {e}")
                        traceback.print_exc()
                
                return jsonify({
                    'success': True,
                    'transcript': transcript,
                    'keywords': keywords,
                    'keywords_str': keywords_str,
                    'web_text': web_text,
                    'expert_opinion': expert_opinion,  # 如果有Few-Shot文本，会生成专家观点
                    'model': ai_model
                })
            
            else:
                return jsonify({'success': False, 'error': f'不支持的模型: {ai_model}'}), 400

        except Exception as e:
            error_trace = traceback.format_exc()
            print(f"处理视频链接异常: {error_trace}")
            try:
                return jsonify({
                    'success': False,
                    'error': f'处理失败: {str(e)}',
                    'detail': error_trace[-1000:] if len(error_trace) > 1000 else error_trace
                }), 500
            except Exception as json_error:
                # 如果连JSON都无法返回，至少返回一个简单的响应
                return json.dumps({
                    'success': False,
                    'error': f'处理失败: {str(e)}'
                }), 500, {'Content-Type': 'application/json'}
        
        finally:
            cleanup_files(local_video_path, compressed_file_path)
            
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"请求解析异常: {error_trace}")
        try:
            return jsonify({
                'success': False,
                'error': f'请求解析失败: {str(e)}',
                'detail': error_trace[-500:] if len(error_trace) > 500 else error_trace
            }), 400
        except Exception as json_error:
            return json.dumps({
                'success': False,
                'error': f'请求解析失败: {str(e)}'
            }), 400, {'Content-Type': 'application/json'}

@app.route('/api/generate-expert-opinion', methods=['POST'])
def generate_expert_opinion_route():
    data = request.json
    transcript = data.get('transcript', '')
    web_text = data.get('web_text', '')
    few_shot_text = data.get('few_shot_text', FEW_SHOT_EXAMPLE_TEXT)
    ai_model = data.get('ai_model', 'chatgpt')
    api_key = data.get('api_key', '')
    
    if not (transcript or web_text) or not api_key:
        return jsonify({'success': False, 'error': '缺少必要参数'}), 400
        
    try:
        prompt = generate_expert_opinion_prompt(web_text, transcript, few_shot_text, has_video=False)
        result = {}
        
        if ai_model == 'chatgpt':
            client = create_openai_client(api_key)
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "system", "content": "只输出JSON。"}, {"role": "user", "content": prompt}],
                response_format={"type": "json_object"}, temperature=0.2
            )
            result = json.loads(resp.choices[0].message.content.strip())
            
        elif ai_model == 'deepseek':
            if not DEEPSEEK_AVAILABLE: raise Exception("DeepSeek SDK 未安装")
            client = DeepSeekClient(api_key=api_key, base_url="https://api.deepseek.com")
            resp = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[{"role": "system", "content": "只输出JSON。"}, {"role": "user", "content": prompt}],
                response_format={"type": "json_object"}, temperature=0.2
            )
            result = json.loads(resp.choices[0].message.content.strip())
            
        elif ai_model == 'qwen':
            client = create_qwen_client(api_key)
            resp = client.chat.completions.create(
                model=QWEN_MODEL,
                messages=[{"role": "system", "content": "只输出JSON。"}, {"role": "user", "content": prompt}],
                response_format={"type": "json_object"}, temperature=0.2
            )
            result = json.loads(resp.choices[0].message.content.strip())
            
        elif ai_model == 'gemini':
            client = create_gemini_client(api_key)
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[types.Content(parts=[types.Part(text=prompt)])],
                config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.2)
            )
            result = json.loads(resp.text.strip())
            
        return jsonify({'success': True, 'result': result, 'model': ai_model})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/rag/search', methods=['POST'])
def rag_search():
    try:
        data = request.json
        query = data.get('query', '').strip()
        categories_filter = data.get('categories', [])
        api_key = data.get('api_key', '')
        
        if not query:
            return jsonify({'success': False, 'error': '搜索关键词不能为空'}), 400
        
        if not api_key:
            return jsonify({'success': False, 'error': '需要Gemini API Key进行查询分析，请在Video Content Analysis页面配置'}), 400
        
        # 初始化RAG引擎（延迟加载，传递API Key）
        try:
            engine = init_rag_engine(api_key=api_key)
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'RAG引擎初始化失败: {str(e)}',
                'hint': '请检查数据库连接和配置'
            }), 500
        
        # 执行搜索
        try:
            search_result = engine.search(query, api_key, categories_filter)
            
            return jsonify({
                'success': True,
                'results': search_result['results'],
                'analysis': search_result['analysis'],
                'response': search_result['response']
            })
        except Exception as e:
            error_msg = str(e)
            if "password authentication failed" in error_msg.lower():
                return jsonify({
                    'success': False,
                    'error': '数据库认证失败，请检查数据库密码配置'
                }), 500
            elif "connection" in error_msg.lower():
                return jsonify({
                    'success': False,
                    'error': '数据库连接失败，请检查数据库是否运行'
                }), 500
            else:
                return jsonify({
                    'success': False,
                    'error': f'搜索失败: {error_msg}'
                }), 500
        
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"RAG搜索异常: {error_trace}")
        return jsonify({
            'success': False,
            'error': f'搜索失败: {str(e)}',
            'detail': error_trace[-500:] if len(error_trace) > 500 else error_trace
        }), 500

# ==========================================
# 新增：后台管理与知识库更新逻辑
# ==========================================
import threading
from datetime import datetime

# 设置一个简单的管理员密码
ADMIN_PASSWORD = "admin"  # 建议修改复杂一点

def mock_crawl_latest_urls():
    """
    [需自定义] 爬虫函数：获取最新的报告URL列表
    这里应该实现真实的爬虫逻辑，比如访问CAS官网解析列表
    """
    print("正在爬取最新报告列表...")
    # 示例数据
    return [
        "https://example.com/report1",
        "https://example.com/report2"
    ]

def background_update_task(api_key):
    """后台执行的更新任务"""
    print(f"[{datetime.now()}] 🚀 开始执行知识库更新任务...")
    
    try:
        # 1. 初始化 RAG 引擎 (连接数据库)
        # 注意：这里需要确保使用了正确的 Embedding 模型对应的 API Key
        # 如果是国内环境，建议换成 HuggingFaceEmbedding (本地模型) 就不需要这个 Key 了
        engine = init_rag_engine(api_key=api_key)
        
        # 2. 获取最新 URL
        latest_urls = mock_crawl_latest_urls()
        
        # 3. 检查数据库中已存在的 URL (防止重复)
        # 这里使用简单的 SQL 查询或者通过 LlamaIndex 的 docstore 检查
        # 为简化演示，这里假设所有爬到的都是新的
        new_urls = latest_urls 
        
        if not new_urls:
            print("没有发现新内容。")
            return

        print(f"发现 {len(new_urls)} 个新报告，开始处理...")
        
        # 4. 处理新内容 (下载 -> 转录 -> 向量化 -> 入库)
        new_documents = []
        
        # ⚠️ 这里需要你复用 build_index.py 里的逻辑或者 download_video 逻辑
        # from gemini_try import download_video
        # for url in new_urls:
        #     local_video, web_text = download_video(url, "temp.mp4")
        #     transcript = transcribe_with_whisper(local_video)
        #     doc = Document(text=transcript, metadata={"original_url": url, "summary": web_text})
        #     new_documents.append(doc)
        
        # 5. 插入数据库
        if new_documents:
            engine.index.insert_nodes(new_documents)
            print(f"✅ 成功更新 {len(new_documents)} 条数据到知识库！")
        else:
            print("⚠️ 未生成有效文档，更新跳过。")
            
    except Exception as e:
        print(f"❌ 更新任务失败: {str(e)}")
        traceback.print_exc()

@app.route('/admin')
def admin_page():
    """后台管理页面"""
    return render_template('admin.html')

@app.route('/api/admin/update-db', methods=['POST'])
def trigger_update_db():
    """触发更新的接口"""
    data = request.json
    password = data.get('password')
    api_key = data.get('api_key')
    
    if password != ADMIN_PASSWORD:
        return jsonify({'success': False, 'error': '管理员密码错误'}), 401
    
    if not api_key:
        return jsonify({'success': False, 'error': '需要提供 API Key 用于生成向量'}), 400

    # 启动后台线程，避免卡死网页
    thread = threading.Thread(target=background_update_task, args=(api_key,))
    thread.start()
    
    return jsonify({
        'success': True, 
        'message': '更新任务已在后台启动，请关注服务器日志或稍后刷新数据库。'
    })


if __name__ == '__main__':
    port = 5001
    if len(sys.argv) > 1:
        try: port = int(sys.argv[1])
        except: pass
    
    print(f"\n启动 Flask 应用: http://localhost:{port}\n")
    from werkzeug.serving import WSGIRequestHandler
    WSGIRequestHandler.timeout = 1800
    app.run(debug=True, host='0.0.0.0', port=port, threaded=True)