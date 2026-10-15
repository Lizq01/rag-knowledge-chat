"""RAG 问答主流程：混合检索 → LLM 重排 → 带引用来源的回答生成"""
from . import llm
from .config import RERANK_TOP_N, TOP_K
from .reranker import llm_rerank
from .retriever import HybridRetriever

_ANSWER_PROMPT = """你是一个严谨的知识库问答助手，请仅依据「参考资料」回答问题：

1. 用中文回答，先给直接结论，再展开要点；
2. 每个事实性内容后用 [编号] 标注来源，例如 [1][3]；
3. 若参考资料不足以回答，请明确说明「知识库中没有相关信息」，不要编造。

参考资料：
{context}

用户问题：{question}"""


def answer(question: str, retriever: HybridRetriever | None = None,
           use_rerank: bool = True) -> dict:
    """完整问答链路，返回回答文本与引用来源

    返回结构：
    {
      "answer":    带 [编号] 引用的回答文本,
      "citations": 编号 → 来源（文档名 / 段号 / 原文）,
      "candidates": 混合检索的完整候选（便于调试与评测）,
      "used":       实际送入 LLM 的片段,
    }
    """
    retriever = retriever or HybridRetriever()

    candidates = retriever.search(question, top_k=TOP_K)
    if use_rerank and candidates:
        used = llm_rerank(question, candidates, top_n=RERANK_TOP_N)
    else:
        used = candidates[:RERANK_TOP_N]

    context = "\n\n".join(
        f"[{i}] 来源《{c['doc']}》第 {c['position']} 段：\n{c['text']}"
        for i, c in enumerate(used, start=1)
    )

    text = llm.chat(
        [{"role": "user",
          "content": _ANSWER_PROMPT.format(context=context, question=question)}],
        temperature=0.2, max_tokens=1500,
    )

    citations = [
        {"id": i, "doc": c["doc"], "position": c["position"], "text": c["text"]}
        for i, c in enumerate(used, start=1)
    ]
    return {"answer": text, "citations": citations,
            "candidates": candidates, "used": used}
