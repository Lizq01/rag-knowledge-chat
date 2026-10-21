"""Web 服务：FastAPI + 单文件页面（不依赖任何 CDN，断网也能用）

工程上刻意做的几件事：
1. **启动即加载索引与模型**（lifespan），避免第一个请求卡十几秒；
2. **会话有上限与 TTL**，长时间运行不会被会话对象慢慢吃满内存；
3. **并发背压**：检索+重排是 CPU 密集活，用信号量限制同时在跑的重活数量，超出直接返回 503，
   而不是让请求无限堆积把机器拖垮；
4. **输入校验与错误收敛**：限制提问长度、非法会话 id，异常只回可读信息，不吐堆栈。
"""
from __future__ import annotations

import logging
import threading
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from . import config, store
from .memory import SessionStore
from .rag import answer_turn
from .retriever import HybridRetriever

logger = logging.getLogger(__name__)

# 同时在跑的重活数量上限：检索/重排都是 CPU 密集，超过就快速失败而不是排队堆积
MAX_CONCURRENT_JOBS = 4
JOB_ACQUIRE_TIMEOUT = 5.0

_STATE: dict = {}
_JOBS = threading.Semaphore(MAX_CONCURRENT_JOBS)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=config.MAX_QUESTION_CHARS)
    session_id: str | None = None
    rerank: str | None = None
    rewrite: str | None = None


def get_retriever() -> HybridRetriever:
    return _STATE["retriever"]


def get_sessions() -> SessionStore:
    return _STATE["sessions"]


def create_app() -> FastAPI:
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        # 启动即构建/加载索引：索引缺失时服务照常启动，但 /api/ask 会给出明确提示
        try:
            _STATE["retriever"] = HybridRetriever()
            _STATE["sessions"] = SessionStore()
            _STATE["ready"] = True
            _STATE["error"] = None
            logger.info("索引已加载：%s", _STATE["retriever"].backend)
        except Exception as e:  # noqa: BLE001 —— 启动失败不应让进程直接退出
            _STATE["ready"] = False
            _STATE["error"] = str(e)
            logger.error("索引加载失败：%s", e)
        yield
        _STATE.clear()

    app = FastAPI(title="知识库 RAG Agent", version="0.3.0", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse(_PAGE)

    @app.get("/api/health")
    def health() -> dict:
        meta = store.index_meta() or {}
        return {
            "status": "ok" if _STATE.get("ready") else "degraded",
            "error": _STATE.get("error"),
            "vector_backend": _STATE["retriever"].backend if _STATE.get("ready") else None,
            "chunks": len(_STATE["retriever"].chunks) if _STATE.get("ready") else 0,
            "rerank_backend": config.RERANK_BACKEND,
            "reindex_hint": store.check_freshness(),
            "sessions": _STATE["sessions"].stats() if _STATE.get("ready") else None,
            "index_meta": meta,
        }

    @app.post("/api/ask")
    def ask(payload: AskRequest) -> JSONResponse:
        if not _STATE.get("ready"):
            raise HTTPException(
                status_code=503,
                detail=f"索引未就绪：{_STATE.get('error')}。请先运行 python main.py ingest")

        question = payload.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="问题不能为空")

        acquired = _JOBS.acquire(timeout=JOB_ACQUIRE_TIMEOUT)
        if not acquired:
            raise HTTPException(status_code=503, detail="服务繁忙，请稍后重试")
        try:
            sessions = get_sessions()
            memory = sessions.get(payload.session_id)
            started = time.perf_counter()
            result = answer_turn(
                question,
                retriever=get_retriever(),
                memory=memory,
                rerank_backend=payload.rerank,
                rewrite_backend=payload.rewrite,
                store=sessions,
            )
            result["wall_ms"] = round((time.perf_counter() - started) * 1000)
            return JSONResponse(result)
        except HTTPException:
            raise
        except Exception as e:  # noqa: BLE001 —— 内部错误只回可读摘要
            logger.exception("处理提问失败")
            raise HTTPException(status_code=500,
                                detail=f"处理失败：{type(e).__name__}: {e}") from e
        finally:
            _JOBS.release()

    @app.get("/api/sessions/{session_id}")
    def get_session(session_id: str) -> dict:
        memory = get_sessions().get(session_id)
        return {"session_id": memory.session_id, "summary": memory.summary,
                "turns": [t.to_dict() for t in memory.turns]}

    @app.delete("/api/sessions/{session_id}")
    def delete_session(session_id: str) -> dict:
        return {"deleted": get_sessions().delete(session_id)}

    return app


app = create_app()


# ---------------------------------------------------------------- 页面

_PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>知识库 RAG Agent</title>
<style>
  :root { --bg:#f6f7f9; --card:#fff; --line:#e3e6ea; --text:#1f2328; --muted:#6b7280;
          --brand:#2563eb; --brand-soft:#eff4ff; --ok:#0f7b46; --warn:#9a6700; --err:#b42318; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text);
         font:14px/1.6 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif; }
  header { background:var(--card); border-bottom:1px solid var(--line); padding:10px 18px;
           display:flex; align-items:center; gap:12px; flex-wrap:wrap; position:sticky; top:0; z-index:5; }
  header h1 { font-size:15px; margin:0; font-weight:600; }
  .badge { font-size:12px; color:var(--muted); background:var(--bg); border:1px solid var(--line);
           border-radius:999px; padding:2px 10px; }
  .spacer { flex:1; }
  button { font:inherit; cursor:pointer; border:1px solid var(--line); background:var(--card);
           color:var(--text); border-radius:6px; padding:5px 12px; }
  button:hover { border-color:var(--brand); color:var(--brand); }
  button.primary { background:var(--brand); border-color:var(--brand); color:#fff; }
  button.primary:hover { opacity:.9; color:#fff; }
  main { max-width:920px; margin:0 auto; padding:18px 18px 140px; }
  .msg { margin:14px 0; display:flex; }
  .msg.user { justify-content:flex-end; }
  .bubble { max-width:82%; padding:10px 14px; border-radius:10px; background:var(--card);
            border:1px solid var(--line); white-space:pre-wrap; word-break:break-word; }
  .msg.user .bubble { background:var(--brand-soft); border-color:#d7e3ff; }
  .bubble p { margin:6px 0; }
  .bubble ul { margin:6px 0; padding-left:20px; }
  .bubble code { background:#f0f2f5; padding:1px 5px; border-radius:4px; font-size:13px; }
  .bubble pre { background:#f0f2f5; padding:10px; border-radius:6px; overflow:auto; }
  .bubble h3,.bubble h4 { margin:10px 0 4px; font-size:14px; }
  sup.cite { color:var(--brand); font-weight:600; padding:0 1px; }
  .meta { margin-top:8px; font-size:12px; color:var(--muted); border-top:1px dashed var(--line); padding-top:6px; }
  .meta .row { display:flex; gap:10px; flex-wrap:wrap; }
  .meta .warn { color:var(--warn); }
  details.src { margin-top:8px; }
  details.src summary { cursor:pointer; color:var(--muted); font-size:12px; }
  details.src ol { margin:6px 0; padding-left:20px; font-size:12px; color:var(--muted); }
  .empty { color:var(--muted); text-align:center; padding:60px 0; }
  footer { position:fixed; bottom:0; left:0; right:0; background:var(--card);
           border-top:1px solid var(--line); padding:12px 18px; }
  .composer { max-width:920px; margin:0 auto; display:flex; gap:10px; align-items:flex-end; }
  textarea { flex:1; resize:none; height:60px; padding:9px 12px; border:1px solid var(--line);
             border-radius:8px; font:inherit; outline:none; }
  textarea:focus { border-color:var(--brand); }
  .thinking { color:var(--muted); font-size:13px; padding:8px 0; }
  .err { color:var(--err); font-size:13px; }
</style>
</head>
<body>
<header>
  <h1>知识库 RAG Agent</h1>
  <span class="badge" id="status">加载中…</span>
  <span class="badge" id="backend"></span>
  <span class="spacer"></span>
  <button id="newSession">新会话</button>
  <button id="clearView">清空显示</button>
</header>

<main id="chat"><div class="empty">加载中…</div></main>

<footer>
  <div class="composer">
    <textarea id="input" placeholder="输入问题，Enter 发送，Shift+Enter 换行"></textarea>
    <button class="primary" id="send">发送</button>
  </div>
</footer>

<script>
const chat = document.getElementById('chat');
const input = document.getElementById('input');
let sessionId = null;
let busy = false;

function escapeHtml(s) {
  return s.replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
}

// 极简 Markdown 渲染：只处理回答里真正会出现的语法，不引第三方库
function render(text) {
  const lines = escapeHtml(text).split('\\n');
  let out = '', inCode = false, inList = false;
  for (const line of lines) {
    if (line.trim().startsWith('```')) {
      out += inCode ? '</pre>' : '<pre>';
      inCode = !inCode;
      continue;
    }
    if (inCode) { out += line + '\\n'; continue; }
    const head = line.match(/^(#{2,4})\\s+(.*)$/);
    if (head) { out += `<h${head[1].length}>${inline(head[2])}</h${head[1].length}>`; continue; }
    const item = line.match(/^\\s*[-*]\\s+(.*)$/);
    if (item) {
      if (!inList) { out += '<ul>'; inList = true; }
      out += `<li>${inline(item[1])}</li>`;
      continue;
    }
    if (inList) { out += '</ul>'; inList = false; }
    if (line.trim()) out += `<p>${inline(line)}</p>`;
  }
  if (inCode) out += '</pre>';
  if (inList) out += '</ul>';
  return out;
}

function inline(s) {
  return s.replace(/`([^`]+)`/g, '<code>$1</code>')
          .replace(/\\*\\*([^*]+)\\*\\*/g, '<b>$1</b>')
          .replace(/\\[(\\d+)\\]/g, '<sup class="cite">[$1]</sup>');
}

function metaLine(r) {
  const t = r.timings || {};
  const rw = r.rewrite || {};
  const rr = r.rerank || {};
  const parts = [
    `向量 ${r.vector_backend || '-'}`,
    `重排 ${rr.backend || '-'}`,
    `改写 ${rw.backend || '-'}`
  ];
  if (rw.queries && rw.queries.length > 1) {
    parts.push('改写查询：' + rw.queries.slice(1).map(q => `「${q}」`).join(' '));
  }
  const times = `检索 ${Math.round(t.retrieve_ms||0)}ms · 重排 ${Math.round(t.rerank_ms||0)}ms · 生成 ${Math.round(t.llm_ms||0)}ms · 总计 ${Math.round(t.total_ms||0)}ms`;
  const warns = [];
  if (rr.degraded_from) warns.push(`重排降级自 ${rr.degraded_from}`);
  if (rw.degraded_from) warns.push(`改写降级自 ${rw.degraded_from}`);
  if ((r.memory||{}).backend === 'drop') warns.push('记忆压缩降级为丢弃');
  return `<div class="meta"><div class="row">${parts.map(p=>`<span>${escapeHtml(p)}</span>`).join('')}</div>`
       + `<div>${escapeHtml(times)}</div>`
       + (warns.length ? `<div class="warn">${escapeHtml(warns.join('；'))}</div>` : '') + '</div>';
}

function sourcesHtml(citations) {
  if (!citations || !citations.length) return '';
  const items = citations.map(c =>
    `<li><b>[${c.id}]</b> 《${escapeHtml(c.doc)}》第 ${c.position} 段 — ${escapeHtml(c.text.slice(0,140))}${c.text.length>140?'…':''}</li>`
  ).join('');
  return `<details class="src"><summary>引用来源 ${citations.length} 条</summary><ol>${items}</ol></details>`;
}

function addMessage(role, html, extra) {
  const wrap = document.createElement('div');
  wrap.className = 'msg ' + role;
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.innerHTML = html + (extra || '');
  wrap.appendChild(bubble);
  chat.appendChild(wrap);
  chat.scrollTop = chat.scrollHeight;
  return wrap;
}

async function send() {
  const question = input.value.trim();
  if (!question || busy) return;
  busy = true;
  document.getElementById('send').disabled = true;
  input.value = '';
  addMessage('user', render(question));
  const thinking = addMessage('bot', '<span class="thinking">检索并生成中…</span>');

  try {
    const resp = await fetch('/api/ask', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question, session_id: sessionId})
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || ('请求失败 ' + resp.status));
    sessionId = data.session_id || sessionId;
    thinking.querySelector('.bubble').innerHTML =
      render(data.answer) + sourcesHtml(data.citations) + metaLine(data);
  } catch (e) {
    thinking.querySelector('.bubble').innerHTML =
      '<span class="err">出错了：' + escapeHtml(String(e.message || e)) + '</span>';
  } finally {
    busy = false;
    document.getElementById('send').disabled = false;
    input.focus();
  }
}

async function newSession() {
  if (sessionId) {
    try { await fetch('/api/sessions/' + sessionId, {method:'DELETE'}); } catch (e) {}
  }
  sessionId = null;
  chat.innerHTML = '<div class="empty">新会话已开始，随便问点什么。</div>';
  input.focus();
}

async function loadStatus() {
  try {
    const r = await fetch('/api/health');
    const d = await r.json();
    document.getElementById('status').textContent =
      d.status === 'ok' ? `已就绪 · ${d.chunks} 个片段` : '未就绪：' + (d.error || '未知');
    document.getElementById('backend').textContent =
      `${d.vector_backend || '-'} · 重排 ${d.rerank_backend || '-'}`;
  } catch (e) {
    document.getElementById('status').textContent = '无法连接服务';
  }
}

document.getElementById('send').addEventListener('click', send);
document.getElementById('newSession').addEventListener('click', newSession);
document.getElementById('clearView').addEventListener('click', () => {
  chat.innerHTML = '<div class="empty">已清空显示（会话记忆保留）。</div>';
});
input.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});

loadStatus();
chat.innerHTML = '<div class="empty">问点什么吧。多轮追问可以直接说「那它的 k 是多少？」。</div>';
input.focus();
</script>
</body>
</html>
"""
