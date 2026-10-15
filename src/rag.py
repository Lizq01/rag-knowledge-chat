"""RAG 问答主流程：混合检索 → 重排（Cross-Encoder / LLM）→ 带引用来源的回答生成"""
import time

from . import llm
from .config import RERANK_TOP_N, TOP_K
from .reranker import rerank
from .retriever import HybridRetriever

_ANSWER_PROMPT = """你是一个严谨的知识库问答助手，请仅依据「参考资料」回答问题：

1. 用中文回答，先给直接结论，再展开要点；
2. 每个事实性内容后用 [编号] 标注来源，例如 [1][3]；
3. 若参考资料不足以回答，请明确说明「知识库中没有相关信息」，不要编造。

参考资料：
{context}

用户问题：{question}"""


def answer(question: str, retriever: HybridRetriever | None = None,
           use_rerank: bool = True, rerank_backend: str | None = None) -> dict:
    """完整问答链路，返回回答文本、引用来源与各阶段耗时

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

    started = time.perf_counter()
    candidates, stats = retriever.search_with_stats(question, top_k=TOP_K)

    started = time.perf_counter()
    used, rerank_info = rerank(question, candidates, top_n=RERANK_TOP_N, backend=backend)
    stats["rerank_ms"] = (time.perf_counter() - started) * 1000

    context = "\n\n".join(
        f"[{i}] 来源《{c['doc']}》第 {c['position']} 段：\n{c['text']}"
        for i, c in enumerate(used, start=1)
    )

    started = time.perf_counter()
    text = llm.chat(
        [{"role": "user",
          "content": _ANSWER_PROMPT.format(context=context, question=question)}],
        temperature=0.2, max_tokens=1500,
    )
    stats["llm_ms"] = (time.perf_counter() - started) * 1000
    stats["total_ms"] = (stats["retrieve_ms"] + stats["rerank_ms"] + stats["llm_ms"])

    citations = [
        {"id": i, "doc": c["doc"], "position": c["position"], "text": c["text"]}
        for i, c in enumerate(used, start=1)
    ]
    return {"answer": text, "citations": citations,
            "candidates": candidates, "used": used,
            "rerank": rerank_info, "timings": stats,
            "vector_backend": retriever.backend}
