# =============================================================================
# AI 写作工作台 Pro - 完整版
# 包含：文案生成（项目一）+ 文档问答（项目二）
# 运行：python app.py
# 访问：http://127.0.0.1:8000
# =============================================================================

import os
import asyncio
import tempfile
import shutil
import json
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, StreamingResponse
import openai
import uvicorn
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pdfplumber
import tiktoken

# ========== 配置 ==========
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise ValueError("DEEPSEEK_API_KEY environment variable not set")
client = openai.AsyncOpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com/v1"
)
encoding = tiktoken.get_encoding("cl100k_base")
def count_tokens(text: str) -> int:
    return len(encoding.encode(text))
# =========================

app = FastAPI(title="AI 写作工作台 Pro")

# ---------- 项目一：文案生成器 ----------
@app.post("/generate")
async def generate(request: Request):
    data = await request.json()
    topic = data.get('topic')
    tone = data.get('tone')
    temperature = data.get('temperature', 0.7)
    num_points = data.get('num_points', 3)
    prompt = f"请用{tone}的语气，为产品「{topic}」写{num_points}个核心卖点，每条不超过20字，用数字序号分段。"
    input_tokens = count_tokens(prompt)
    async def stream():
        output_text = ""
        try:
            stream_resp = await client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                stream=True
            )
            async for chunk in stream_resp:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    output_text += content
                    yield f"data: {content}\n\n"
            output_tokens = count_tokens(output_text)
            usage = json.dumps({"type": "usage", "input": input_tokens, "output": output_tokens})
            yield f"data: {usage}\n\n"
        except Exception as e:
            yield f"data: 错误: {str(e)}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(stream(), media_type="text/event-stream")

# ---------- 项目二：RAG ----------
embedding_function = embedding_functions.OpenAIEmbeddingFunction(
    api_key=DEEPSEEK_API_KEY,
    api_base="https://api.deepseek.com/v1",
    model_name="bge-large-zh"
)
chroma_client = chromadb.PersistentClient(path="./chroma_db")
try:
    collection = chroma_client.get_collection("knowledge_base")
except:
    collection = chroma_client.create_collection(
        name="knowledge_base",
        embedding_function=embedding_function
    )

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        def parse_pdf():
            text = ""
            with pdfplumber.open(tmp_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""
            return text
        text = await asyncio.to_thread(parse_pdf)
        os.unlink(tmp_path)
        if not text.strip():
            return {"message": "⚠️ PDF 内容为空或无法解析"}
        def split_and_store():
            global collection, chroma_client
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_text(text)
            try:
                chroma_client.delete_collection("knowledge_base")
            except:
                pass
            new_collection = chroma_client.create_collection(
                name="knowledge_base",
                embedding_function=embedding_function
            )
            ids = [f"id_{i}" for i in range(len(chunks))]
            new_collection.add(documents=chunks, ids=ids)
            collection = new_collection
            return len(chunks)
        count = await asyncio.to_thread(split_and_store)
        return {"message": f"✅ 成功存储 {count} 个文档片段"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"message": f"❌ 错误: {str(e)}"}

@app.post("/ask")
async def ask(request: Request):
    data = await request.json()
    question = data.get('question')
    def retrieve():
        results = collection.query(query_texts=[question], n_results=6)
        return results['documents'][0] if results['documents'] else []
    contexts = await asyncio.to_thread(retrieve)
    context_text = "\n".join(contexts) if contexts else "（未找到相关文档片段）"
    prompt = f"""
    请根据以下提供的资料回答用户的问题。如果资料中没有相关信息，请直接回答"资料中未提及"。

    ===== 资料内容 =====
    {context_text}
    ===================

    用户问题：{question}
    回答：
    """
    async def stream():
        try:
            stream_resp = await client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                stream=True
            )
            async for chunk in stream_resp:
                if chunk.choices[0].delta.content:
                    yield f"data: {chunk.choices[0].delta.content}\n\n"
        except Exception as e:
            yield f"data: 错误: {str(e)}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(stream(), media_type="text/event-stream")

# ---------- 前端（完全复刻项目一风格，嵌入文档问答标签页） ----------
HTML = """
<!DOCTYPE html>
<html lang="zh" data-theme="light">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI 写作工作台 Pro</title>
    <style>
        /* ===== CSS 变量（暗黑/亮色主题） ===== */
        :root {
            --bg-body: #f8fafc;
            --bg-card: #ffffff;
            --bg-input: #ffffff;
            --bg-output: #f1f5f9;
            --bg-param: #f8fafc;
            --text-primary: #0f172a;
            --text-secondary: #475569;
            --text-muted: #94a3b8;
            --border-color: #e2e8f0;
            --shadow: 0 2px 10px rgba(0,0,0,0.05);
            --btn-primary: #1a202c;
            --btn-primary-text: white;
            --btn-secondary: #e2e8f0;
            --btn-secondary-text: #1a202c;
            --btn-danger: #fee2e2;
            --btn-danger-text: #991b1b;
            --btn-success: #22c55e;
            --btn-success-text: white;
            --output-border: #3b82f6;
            --error-bg: #fee2e2;
            --error-border: #dc2626;
        }
        [data-theme="dark"] {
            --bg-body: #0f172a;
            --bg-card: #1e293b;
            --bg-input: #334155;
            --bg-output: #1e293b;
            --bg-param: #1e293b;
            --text-primary: #f1f5f9;
            --text-secondary: #cbd5e1;
            --text-muted: #64748b;
            --border-color: #334155;
            --shadow: 0 2px 10px rgba(0,0,0,0.3);
            --btn-primary: #f1f5f9;
            --btn-primary-text: #0f172a;
            --btn-secondary: #334155;
            --btn-secondary-text: #e2e8f0;
            --btn-danger: #7f1d1d;
            --btn-danger-text: #fca5a5;
            --btn-success: #15803d;
            --btn-success-text: white;
            --output-border: #60a5fa;
            --error-bg: #7f1d1d;
            --error-border: #ef4444;
        }
        
        * { box-sizing: border-box; transition: background-color 0.3s, color 0.2s, border-color 0.3s; }
        body { font-family: system-ui, sans-serif; max-width: 920px; margin: 20px auto; padding: 0 20px; background: var(--bg-body); color: var(--text-primary); }
        .card { background: var(--bg-card); padding: 25px; border-radius: 16px; box-shadow: var(--shadow); margin-bottom: 20px; border: 1px solid var(--border-color); }
        h2, h3 { margin-top: 0; display: flex; align-items: center; gap: 10px; }
        .row { display: flex; gap: 15px; flex-wrap: wrap; margin-bottom: 10px; }
        .row > * { flex: 1; min-width: 140px; }
        input, select, button { padding: 12px; border: 1px solid var(--border-color); border-radius: 8px; font-size: 16px; background: var(--bg-input); color: var(--text-primary); }
        input:focus, select:focus { outline: 2px solid #3b82f6; outline-offset: 1px; }
        button { background: var(--btn-primary); color: var(--btn-primary-text); font-weight: 600; cursor: pointer; border: none; }
        button:disabled { opacity: 0.4; cursor: not-allowed; }
        button.secondary { background: var(--btn-secondary); color: var(--btn-secondary-text); }
        button.danger { background: var(--btn-danger); color: var(--btn-danger-text); }
        button.success { background: var(--btn-success); color: var(--btn-success-text); }
        #output, #ragOutput { margin-top: 20px; padding: 20px; background: var(--bg-output); border-radius: 8px; white-space: pre-wrap; min-height: 140px; border-left: 4px solid var(--output-border); font-size: 15px; line-height: 1.7; }
        .error { border-left-color: var(--error-border) !important; background: var(--error-bg) !important; }
        .toolbar { display: flex; gap: 10px; margin-top: 15px; flex-wrap: wrap; }
        .toolbar button { flex: 0 1 auto; padding: 8px 18px; width: auto; margin: 0; }
        .status-bar { display: flex; justify-content: space-between; flex-wrap: wrap; gap: 10px; margin-top: 12px; font-size: 14px; color: var(--text-secondary); border-top: 1px solid var(--border-color); padding-top: 12px; }
        .param-group { display: flex; align-items: center; gap: 15px; flex-wrap: wrap; background: var(--bg-param); padding: 12px 16px; border-radius: 10px; margin: 10px 0; border: 1px solid var(--border-color); }
        .param-group label { font-weight: 500; font-size: 14px; }
        .param-group input[type="range"] { flex: 1; min-width: 120px; padding: 0; margin: 0; background: transparent; }
        .param-group span { font-size: 14px; background: var(--bg-card); padding: 2px 12px; border-radius: 20px; border: 1px solid var(--border-color); }
        .param-group select { width: auto; padding: 6px 12px; margin: 0; }
        /* 历史记录 - 与项目一完全一致 */
        .history-list { margin-top: 15px; max-height: 300px; overflow-y: auto; }
        .history-item { display: flex; justify-content: space-between; align-items: center; padding: 10px 14px; border: 1px solid var(--border-color); border-radius: 8px; margin-bottom: 6px; cursor: pointer; }
        .history-item:hover { background: var(--bg-param); }
        .history-item .preview { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; margin-right: 15px; font-size: 14px; }
        .history-item .meta { font-size: 12px; color: var(--text-muted); min-width: 80px; text-align: right; }
        .history-item .del-btn { background: none; border: none; color: #ef4444; cursor: pointer; font-size: 18px; padding: 0 8px; }
        .empty-history { color: var(--text-muted); text-align: center; padding: 20px; }
        /* 亮暗模式按钮 - fixed 定位，与项目一完全一致 */
        .theme-toggle { position: fixed; top: 20px; right: 20px; background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 30px; padding: 8px 16px; cursor: pointer; font-size: 14px; z-index: 999; box-shadow: var(--shadow); }
        /* 标签页切换 */
        .tabs { display: flex; gap: 10px; margin-bottom: 0; }
        .tab { padding: 10px 20px; background: var(--bg-card); border: 1px solid var(--border-color); border-bottom: none; border-radius: 8px 8px 0 0; cursor: pointer; font-weight: 600; color: var(--text-secondary); transition: all 0.2s; }
        .tab.active { background: var(--btn-primary); color: var(--btn-primary-text); border-color: var(--btn-primary); }
        .tab:hover:not(.active) { background: var(--bg-param); }
        .panel { display: none; }
        .panel.active { display: block; }
        /* 上传区域 */
        .upload-area { border: 2px dashed var(--border-color); padding: 20px; text-align: center; border-radius: 12px; background: var(--bg-param); margin: 10px 0; }
        .upload-area input[type="file"] { display: none; }
        .upload-area label { background: var(--btn-primary); color: var(--btn-primary-text); padding: 8px 20px; border-radius: 8px; cursor: pointer; display: inline-block; }
        .status { font-size: 14px; color: var(--text-secondary); margin-top: 8px; }
        @media (max-width: 600px) {
            .theme-toggle { top: 10px; right: 10px; padding: 6px 12px; font-size: 12px; }
            .param-group { flex-direction: column; align-items: stretch; }
        }
    </style>
</head>
<body>
    <button class="theme-toggle" id="themeToggle">🌙 暗黑</button>

    <!-- ===== 标签页 ===== -->
    <div class="tabs">
        <div class="tab active" onclick="switchTab('writer')">✍️ 文案生成</div>
        <div class="tab" onclick="switchTab('rag')">📚 文档问答</div>
    </div>

    <!-- ===== 文案生成面板（项目一完整风格） ===== -->
    <div id="writer" class="panel active">
        <div class="card">
            <h2>✍️ AI 写作工作台 <span style="font-size:14px;font-weight:normal;color:var(--text-muted);">Pro</span></h2>
            
            <div class="row">
                <input id="topic" placeholder="输入主题，比如：防晒霜" value="智能无线耳机">
                <select id="tone">
                    <option value="幽默风趣">幽默风趣</option>
                    <option value="专业严谨" selected>专业严谨</option>
                    <option value="煽情走心">煽情走心</option>
                </select>
            </div>

            <div class="param-group">
                <label>🎨 创意度</label>
                <input type="range" id="temperature" min="0" max="1.0" step="0.1" value="0.7">
                <span id="tempDisplay">0.7</span>
                <label style="margin-left:10px;">📝 条数</label>
                <select id="numPoints">
                    <option value="3" selected>3 条</option>
                    <option value="5">5 条</option>
                    <option value="10">10 条</option>
                </select>
            </div>

            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                <button id="genBtn" style="flex:2;">🚀 生成</button>
                <button id="regenerateBtn" class="secondary" style="flex:1;">🔄 重新生成</button>
            </div>
            
            <div id="output" class="loading">✨ 输入主题，点击生成...</div>
            
            <div class="toolbar">
                <button id="copyBtn" class="secondary" disabled>📋 复制</button>
                <button id="exportBtn" class="secondary" disabled>📥 导出 MD</button>
                <button id="clearBtn" class="danger" disabled>🗑️ 清空</button>
                <button id="saveHistoryBtn" class="success" disabled>💾 保存</button>
            </div>

            <div class="status-bar">
                <span id="statusMsg">就绪</span>
                <span id="tokenInfo">📊 Token: 0 (输入: 0 / 输出: 0) | 💰 费用: $0.0000</span>
            </div>
        </div>

        <!-- 历史记录卡片（独立卡片，与项目一完全一致） -->
        <div class="card">
            <h3>📜 历史记录 <span style="font-size:14px;font-weight:normal;color:var(--text-muted);">（点击恢复，点击 ✕ 删除）</span></h3>
            <div style="display:flex; gap:10px; flex-wrap:wrap; margin-bottom:10px;">
                <button id="clearAllHistory" class="danger" style="width:auto;padding:6px 16px;font-size:13px;">清空所有</button>
                <button id="exportAllHistory" class="secondary" style="width:auto;padding:6px 16px;font-size:13px;">📤 导出全部为 MD</button>
            </div>
            <div id="historyContainer" class="history-list">
                <div class="empty-history">暂无历史记录</div>
            </div>
        </div>
    </div>

    <!-- ===== 文档问答面板（项目二） ===== -->
    <div id="rag" class="panel">
        <div class="card">
            <h2>📚 文档问答 <span style="font-size:14px;font-weight:normal;color:var(--text-muted);">RAG</span></h2>
            <div class="upload-area">
                <input type="file" id="fileInput" accept=".pdf">
                <label for="fileInput">📤 选择 PDF 文件</label>
                <div id="fileName" style="margin:10px 0;color:var(--text-muted);">未选择文件</div>
                <button id="uploadBtn" style="width:auto;padding:8px 30px;">🚀 上传并构建知识库</button>
                <div id="uploadStatus" class="status"></div>
            </div>
            <hr style="margin:20px 0;border-color:var(--border-color);">
            <h3>💬 提问</h3>
            <input id="question" placeholder="输入你的问题，比如：这份报告讲了什么？" value="请用中文总结这份文档的核心内容">
            <button id="askBtn">🤖 基于文档回答</button>
            <div id="ragOutput" class="loading">✨ 等待提问...</div>
        </div>
    </div>

    <script>
        // ===== 主题切换（与项目一完全一致） =====
        (function() {
            const themeToggle = document.getElementById('themeToggle');
            function setTheme(theme) {
                document.documentElement.setAttribute('data-theme', theme);
                localStorage.setItem('theme', theme);
                themeToggle.textContent = theme === 'dark' ? '☀️ 亮色' : '🌙 暗黑';
            }
            const savedTheme = localStorage.getItem('theme') || 'light';
            setTheme(savedTheme);
            themeToggle.addEventListener('click', () => {
                const current = document.documentElement.getAttribute('data-theme');
                setTheme(current === 'dark' ? 'light' : 'dark');
            });
        })();

        // ===== 标签页切换 =====
        function switchTab(tab) {
            document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.getElementById(tab).classList.add('active');
            if (tab === 'writer') {
                document.querySelector('.tab:first-child').classList.add('active');
            } else {
                document.querySelector('.tab:last-child').classList.add('active');
            }
        }

        // ============================================================
        // 文案生成模块（与项目一完全一致）
        // ============================================================
        (function() {
            const genBtn = document.getElementById('genBtn');
            const regenerateBtn = document.getElementById('regenerateBtn');
            const topicInput = document.getElementById('topic');
            const toneSelect = document.getElementById('tone');
            const tempSlider = document.getElementById('temperature');
            const tempDisplay = document.getElementById('tempDisplay');
            const numPointsSelect = document.getElementById('numPoints');
            const output = document.getElementById('output');
            const copyBtn = document.getElementById('copyBtn');
            const exportBtn = document.getElementById('exportBtn');
            const clearBtn = document.getElementById('clearBtn');
            const saveHistoryBtn = document.getElementById('saveHistoryBtn');
            const statusMsg = document.getElementById('statusMsg');
            const tokenInfo = document.getElementById('tokenInfo');
            const historyContainer = document.getElementById('historyContainer');
            const clearAllHistoryBtn = document.getElementById('clearAllHistory');
            const exportAllHistoryBtn = document.getElementById('exportAllHistory');

            let currentText = '';
            let currentTopic = '';
            let currentTone = '';
            let currentTemp = 0.7;
            let currentNum = 3;
            let isGenerating = false;
            let lastUsage = { input: 0, output: 0 };

            tempSlider.addEventListener('input', function() {
                tempDisplay.textContent = parseFloat(this.value).toFixed(1);
            });

            function updateButtons(enable) {
                const hasContent = currentText.trim().length > 0;
                copyBtn.disabled = !enable || !hasContent;
                exportBtn.disabled = !enable || !hasContent;
                clearBtn.disabled = !enable || !hasContent;
                saveHistoryBtn.disabled = !enable || !hasContent;
            }

            function setStatus(msg, isError = false) {
                statusMsg.textContent = msg;
                statusMsg.style.color = isError ? '#dc2626' : 'var(--text-secondary)';
            }

            function updateTokenDisplay(input, output) {
                const cost = (input * 0.14 + output * 0.28) / 1000000;
                tokenInfo.textContent = `📊 Token: ${input + output} (输入: ${input} / 输出: ${output}) | 💰 费用: $${cost.toFixed(6)}`;
                lastUsage = { input, output };
            }

            // ===== 历史记录（与项目一完全一致） =====
            function getHistory() {
                try { return JSON.parse(localStorage.getItem('ai_writer_history')) || []; } catch { return []; }
            }
            function setHistory(history) {
                localStorage.setItem('ai_writer_history', JSON.stringify(history));
            }

            function renderHistory() {
                const history = getHistory();
                if (history.length === 0) {
                    historyContainer.innerHTML = '<div class="empty-history">暂无历史记录</div>';
                    return;
                }
                let html = '';
                history.forEach(item => {
                    const rawContent = item.content || '（内容为空）';
                    const previewText = rawContent.replace(/\\n/g, ' ').substring(0, 60) + (rawContent.length > 60 ? '...' : '');
                    html += `
                        <div class="history-item">
                            <div class="preview" onclick="window._loadHistory(${item.id})">${previewText}</div>
                            <div class="meta">${item.timestamp || ''}</div>
                            <button class="del-btn" onclick="event.stopPropagation(); window._deleteHistory(${item.id})">✕</button>
                        </div>
                    `;
                });
                historyContainer.innerHTML = html;
                window._loadHistory = (id) => {
                    const item = getHistory().find(h => h.id === id);
                    if (item) loadHistoryItem(item);
                };
                window._deleteHistory = (id) => {
                    let history = getHistory();
                    history = history.filter(item => item.id !== id);
                    setHistory(history);
                    renderHistory();
                };
            }

            function loadHistoryItem(item) {
                currentText = item.content || '';
                currentTopic = item.topic || '未命名';
                output.innerHTML = currentText || '（内容为空）';
                output.className = '';
                updateButtons(true);
                setStatus(`📂 已恢复: ${item.topic} (${item.timestamp})`);
                topicInput.value = item.topic || '';
                if (item.tone) toneSelect.value = item.tone;
            }

            function saveHistory() {
                const content = currentText.trim();
                if (!content) {
                    setStatus('没有内容可保存，请先生成文案', true);
                    return;
                }
                const history = getHistory();
                history.unshift({
                    id: Date.now(),
                    topic: topicInput.value.trim() || '未命名',
                    tone: toneSelect.value || '通用',
                    content: content,
                    timestamp: new Date().toLocaleString(),
                    tokens: lastUsage.input + lastUsage.output
                });
                if (history.length > 50) history.pop();
                setHistory(history);
                renderHistory();
                setStatus('💾 已保存到历史记录！');
                setTimeout(() => setStatus(''), 1500);
            }

            function clearAllHistory() {
                if (confirm('确定清空所有历史记录吗？')) {
                    setHistory([]);
                    renderHistory();
                    setStatus('已清空所有历史');
                }
            }

            function exportAllHistory() {
                const history = getHistory();
                if (history.length === 0) { setStatus('没有历史记录可导出', true); return; }
                let md = '# AI 写作历史记录备份\\n\\n';
                history.forEach((item, idx) => {
                    md += `## ${idx+1}. ${item.topic} (${item.timestamp})\\n`;
                    md += `- 语气: ${item.tone || '通用'}\\n`;
                    md += `- Token: ${item.tokens || 0}\\n`;
                    md += `\\n${item.content}\\n\\n---\\n\\n`;
                });
                const blob = new Blob([md], { type: 'text/markdown;charset=utf-8' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `历史记录备份_${new Date().toISOString().slice(0,10)}.md`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
                setStatus('📤 全部历史导出成功！');
                setTimeout(() => setStatus(''), 2000);
            }

            // ===== 核心生成逻辑 =====
            async function doGenerate(topic, tone, temp, numPoints, append = false) {
                if (isGenerating) return;
                isGenerating = true;
                genBtn.disabled = true;
                regenerateBtn.disabled = true;

                if (!append) {
                    output.className = '';
                    output.innerHTML = '⏳ 连接后端...';
                    currentText = '';
                    updateButtons(false);
                } else {
                    output.innerHTML += '\\n\\n--- 追加生成 ---\\n';
                }
                setStatus('正在生成...');
                let fullText = append ? currentText : '';

                try {
                    const resp = await fetch('/generate', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ topic, tone, temperature: temp, num_points: numPoints })
                    });
                    const reader = resp.body.getReader();
                    const decoder = new TextDecoder();
                    if (!append) output.innerHTML = '';

                    while (true) {
                        const { value, done } = await reader.read();
                        if (done) break;
                        const chunk = decoder.decode(value);
                        const lines = chunk.split('\\n');
                        for (const line of lines) {
                            if (line.startsWith('data: ')) {
                                const data = line.slice(6);
                                if (data === '[DONE]') continue;
                                try {
                                    const parsed = JSON.parse(data);
                                    if (parsed.type === 'usage') {
                                        updateTokenDisplay(parsed.input, parsed.output);
                                        continue;
                                    }
                                } catch (e) {}
                                fullText += data;
                                output.innerHTML = fullText;
                            }
                        }
                    }

                    currentText = fullText;
                    currentTopic = topic;
                    currentTone = tone;
                    currentTemp = temp;
                    currentNum = numPoints;

                    if (currentText.trim()) {
                        updateButtons(true);
                        setStatus(`✅ 生成完成，共 ${currentText.length} 字符`);
                    } else {
                        output.innerHTML = '⚠️ 生成内容为空，请重试';
                    }
                } catch (e) {
                    output.className = 'error';
                    output.innerHTML = '❌ 错误: ' + e.message;
                    setStatus('生成失败', true);
                } finally {
                    isGenerating = false;
                    genBtn.disabled = false;
                    regenerateBtn.disabled = false;
                }
            }

            function generate() {
                const topic = topicInput.value.trim() || '未命名主题';
                const tone = toneSelect.value;
                const temp = parseFloat(tempSlider.value);
                const numPoints = parseInt(numPointsSelect.value);
                doGenerate(topic, tone, temp, numPoints, false);
            }

            function regenerate() {
                if (!currentTopic) { setStatus('请先生成一次内容', true); return; }
                doGenerate(currentTopic, currentTone, currentTemp, currentNum, false);
            }

            function clearOutput() {
                currentText = '';
                output.innerHTML = '✨ 内容已清空';
                output.className = '';
                updateButtons(false);
                setStatus('');
            }

            async function copyContent() {
                if (!currentText.trim()) return;
                try {
                    await navigator.clipboard.writeText(currentText);
                    setStatus('✅ 已复制到剪贴板！');
                    setTimeout(() => setStatus(''), 2000);
                } catch {
                    const textarea = document.createElement('textarea');
                    textarea.value = currentText;
                    document.body.appendChild(textarea);
                    textarea.select();
                    document.execCommand('copy');
                    document.body.removeChild(textarea);
                    setStatus('✅ 已复制（兼容模式）！');
                    setTimeout(() => setStatus(''), 2000);
                }
            }

            function exportMD() {
                if (!currentText.trim()) return;
                const blob = new Blob([currentText], { type: 'text/markdown;charset=utf-8' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `文案_${new Date().toISOString().slice(0,10)}.md`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
                setStatus('✅ 导出成功！');
                setTimeout(() => setStatus(''), 2000);
            }

            // ===== 事件绑定 =====
            genBtn.addEventListener('click', generate);
            regenerateBtn.addEventListener('click', regenerate);
            clearBtn.addEventListener('click', clearOutput);
            copyBtn.addEventListener('click', copyContent);
            exportBtn.addEventListener('click', exportMD);
            saveHistoryBtn.addEventListener('click', saveHistory);
            clearAllHistoryBtn.addEventListener('click', clearAllHistory);
            exportAllHistoryBtn.addEventListener('click', exportAllHistory);
            topicInput.addEventListener('keypress', (e) => { if (e.key === 'Enter') generate(); });

            // ===== 初始化 =====
            updateButtons(false);
            renderHistory();
            setStatus('就绪，输入主题开始生成');
            updateTokenDisplay(0, 0);
        })();

        // ============================================================
        // 文档问答模块（项目二）
        // ============================================================
        (function() {
            const fileInput = document.getElementById('fileInput');
            const fileName = document.getElementById('fileName');
            const uploadBtn = document.getElementById('uploadBtn');
            const uploadStatus = document.getElementById('uploadStatus');
            const askBtn = document.getElementById('askBtn');
            const question = document.getElementById('question');
            const output = document.getElementById('ragOutput');

            fileInput.addEventListener('change', () => {
                fileName.textContent = fileInput.files[0] ? fileInput.files[0].name : '未选择文件';
            });

            uploadBtn.addEventListener('click', async () => {
                if (!fileInput.files.length) { uploadStatus.innerHTML = '⚠️ 请选择PDF'; return; }
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);
                uploadStatus.innerHTML = '⏳ 上传解析中...';
                uploadBtn.disabled = true;
                try {
                    const resp = await fetch('/upload_pdf', { method: 'POST', body: formData });
                    const data = await resp.json();
                    uploadStatus.innerHTML = data.message;
                } catch(e) {
                    uploadStatus.innerHTML = '❌ 错误: ' + e.message;
                } finally {
                    uploadBtn.disabled = false;
                }
            });

            askBtn.addEventListener('click', async () => {
                askBtn.disabled = true;
                output.innerHTML = '⏳ 检索中...';
                try {
                    const resp = await fetch('/ask', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ question: question.value })
                    });
                    const reader = resp.body.getReader();
                    const decoder = new TextDecoder();
                    let text = '';
                    output.innerHTML = '';
                    while (true) {
                        const { value, done } = await reader.read();
                        if (done) break;
                        const chunk = decoder.decode(value);
                        const lines = chunk.split('\\n');
                        for (const line of lines) {
                            if (line.startsWith('data: ')) {
                                const data = line.slice(6);
                                if (data === '[DONE]') continue;
                                text += data;
                                output.innerHTML = text;
                            }
                        }
                    }
                } catch(e) {
                    output.innerHTML = '❌ 错误: ' + e.message;
                } finally {
                    askBtn.disabled = false;
                }
            });
        })();
    </script>
</body>
</html>
"""

@app.get("/")
async def root():
    return HTMLResponse(HTML)

# ========== 启动 ==========
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")