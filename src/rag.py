"""RAG 问答主流程：查询改写 → 混合检索 → 重排 → 带引用来源的回答生成

两条入口各有分工：
- `answer()`      单轮、无记忆，M1/M2 的评测基线入口（不引入任何额外变量）
- `answer_turn()` 多轮：带会话记忆、做查询改写与多路检索

刻意保持 `answer()` 行为不变，这样 M3 的效果可以拿 M2 的基线直接对比。
"""
import time

from . import llm
from .config import RERANK_TOP_N, TOP_K
from .memory import ConversationMemory
from .reranker import rerank
from .retriever import HybridRetriever
from .rewrite import rewrite

_ANSWER_PROMPT = """你是一个严谨的知识库问答助手，请仅依据「参考资料」回答问题：

1. 用中文回答，先给直接结论，再展开要点；
2. 每个事实性内容后用 [编号] 标注来源，例如 [1][3]；
3. 若参考资料不足以回答，请明确说明「知识库中没有相关信息」，不要编造。

参考资料：
{context}

用户问题：{question}"""

_TURN_PROMPT = """你是一个严谨的知识库问答助手，请仅依据「参考资料」回答用户在这一轮提出的问题：

1. 用中文回答，先给直接结论，再展开要点；
2. 每个事实性内容后用 [编号] 标注来源，例如 [1][3]；
3. 用户的问题可能依赖上文（比如用了「它」「这个」「第二种」），请结合对话历史理解，
   但如果历史里没有对应信息，就按字面意思回答；
4. 若参考资料不足以回答，请明确说明「知识库中没有相关信息」，不要编造。

参考资料：
{context}

用户这一轮的问题：{question}"""

# 原始问题权重给满；改写变体只用于补召回，权重低一些防止改写跑偏
_REWRITE_WEIGHT = 0.7


def _build_context(used: list[dict]) -> str:
    return "\n\n".join(
        f"[{i}] 来源《{c['doc']}》第 {c['position']} 段：\n{c['text']}"
        for i, c in enumerate(used, start=1)
    )


def _citations_of(used: list[dict]) -> list[dict]:
    return [{"id": i, "doc": c["doc"], "position": c["position"], "text": c["text"]}
            for i, c in enumerate(used, start=1)]


def answer(question: str, retriever: HybridRetriever | None = None,
           use_rerank: bool = True, rerank_backend: str | None = None) -> dict:
    """单轮问答（无记忆、不改写），返回回答文本、引用来源与各阶段耗时

    参数：
    - use_rerank=False 等价于 rerank_backend="none"（保留旧接口，便于 A/B 对比）
    - rerank_backend 为 None 时使用 config.RERANK_BACKEND

    返回结构：
    {
      "answer":    带 [编号] 引用的回答文本,
      "citations": 编号 → 来源（文档名 / 段号 / 原文）,
      "candidates": 混合检索的完整候选（便于调试与评测）,
      "used":       实际送入 LLM 的片段,
      "rerank":     重排执行信息（实际后端 / 是否降级 / 原因）,
      "timings":    各阶段耗时（毫秒）,
    }
    """
    retriever = retriever or HybridRetriever()
    backend = rerank_backend if rerank_backend is not None else (
        None if use_rerank else "none")

    candidates, stats = retriever.search_with_stats(question, top_k=TOP_K)

    started = time.perf_counter()
    used, rerank_info = rerank(question, candidates, top_n=RERANK_TOP_N, backend=backend)
    stats["rerank_ms"] = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    text = llm.chat(
        [{"role": "user",
          "content": _ANSWER_PROMPT.format(context=_build_context(used), question=question)}],
        temperature=0.2, max_tokens=1500,
    )
    stats["llm_ms"] = (time.perf_counter() - started) * 1000
    stats["total_ms"] = stats["retrieve_ms"] + stats["rerank_ms"] + stats["llm_ms"]

    return {"answer": text, "citations": _citations_of(used),
            "candidates": candidates, "used": used,
            "rerank": rerank_info, "timings": stats,
            "vector_backend": retriever.backend}


def retrieve_turn(question: str, retriever: HybridRetriever | None = None,
                  history: list[dict] | None = None,
                  rewrite_backend: str | None = None,
                  rerank_backend: str | None = None
                  ) -> tuple[list[dict], list[dict], dict, dict, dict]:
    """多轮检索链路：查询改写 → 多路检索 → 重排（不含生成）

    单独拆出来是为了让评测能只跑检索层——回答质量评测要花 token，
    而「改写有没有用」本身在检索层就能量化。
    """
    retriever = retriever or HybridRetriever()
    history = history or []

    started = time.perf_counter()
    rewrite_info = rewrite(question, history=history, backend=rewrite_backend)
    rewrite_ms = (time.perf_counter() - started) * 1000
    queries = rewrite_info["queries"]

    weights = None
    if len(queries) > 1:
        # 每路查询贡献两条排序（BM25 + 向量），原始问题权重给满，改写变体降权
        weights = [1.0 if i == 0 else _REWRITE_WEIGHT
                   for i in range(len(queries)) for _ in range(2)]
        candidates, stats = retriever.search_multi_with_stats(
            queries, top_k=TOP_K, weights=weights)
    else:
        candidates, stats = retriever.search_with_stats(queries[0], top_k=TOP_K)
    stats["rewrite_ms"] = rewrite_ms

    started = time.perf_counter()
    # 用原始问题打分：改写只是召回手段，相关性判定仍以用户原话为准
    used, rerank_info = rerank(question, candidates, top_n=RERANK_TOP_N,
                               backend=rerank_backend)
    stats["rerank_ms"] = (time.perf_counter() - started) * 1000
    return candidates, used, stats, rewrite_info, rerank_info


def answer_turn(question: str, retriever: HybridRetriever | None = None,
                memory: ConversationMemory | None = None,
                history: list[dict] | None = None,
                rewrite_backend: str | None = None,
                rerank_backend: str | None = None,
                store=None) -> dict:
    """多轮问答：查询改写 → 多路检索 → 重排 → 带历史的生成 → 回写记忆

    - memory 为空时退化为「有改写、无记忆」的单轮（用 history 提供上下文）
    - store 传入时，本轮结束后自动落盘并按需压缩记忆

    返回在 answer() 基础上追加：
      "rewrite":   改写信息（实际后端 / 查询列表 / 是否降级）
      "memory":    记忆压缩信息（是否压缩 / 用了哪种方式）
      "session_id"
    """
    retriever = retriever or HybridRetriever()
    history = history if history is not None else (
        memory.history_dicts() if memory else [])

    candidates, used, stats, rewrite_info, rerank_info = retrieve_turn(
        question, retriever=retriever, history=history,
        rewrite_backend=rewrite_backend, rerank_backend=rerank_backend)

    # ---- 生成（带历史）----
    messages = memory.messages() if memory else []
    messages = messages + [{"role": "user", "content": _TURN_PROMPT.format(
        context=_build_context(used), question=question)}]

    started = time.perf_counter()
    text = llm.chat(messages, temperature=0.2, max_tokens=1500)
    stats["llm_ms"] = (time.perf_counter() - started) * 1000
    stats["total_ms"] = (stats["retrieve_ms"] + stats["rerank_ms"]
                         + stats["llm_ms"] + stats["rewrite_ms"])

    citations = _citations_of(used)

    # ---- 回写记忆 ----
    memory_info = {"compressed": False, "backend": "none"}
    if memory is not None:
        memory.add("user", question)
        memory.add("assistant", text, citations=citations)
        memory_info = memory.compress()
        if store is not None:
            store.save(memory)

    return {"answer": text, "citations": citations,
            "candidates": candidates, "used": used,
            "rerank": rerank_info, "timings": stats,
            "rewrite": rewrite_info, "memory": memory_info,
            "vector_backend": retriever.backend,
            "session_id": memory.session_id if memory else None}
