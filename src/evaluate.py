"""评测：检索命中率（Recall@K）+ 回答要点覆盖率（LLM 裁判）

评测集格式（evals/eval_set.jsonl，每行一条）：
{"question": "...", "expected_doc": "xxx.md", "key_points": ["...", "..."]}
"""
import json
import re

from . import llm, rag
from .config import EVALS_DIR, TOP_K
from .reranker import llm_rerank
from .retriever import HybridRetriever

_JUDGE_PROMPT = """请检查回答是否覆盖了「要点清单」中的每一条（语义等价即可）。

只输出 JSON，不要输出其他内容，格式：
{{"covered": ["要点1"], "missing": ["要点2"], "score": 0.5}}
其中 score = covered 数量 / 总要点数（保留两位小数）。

要点清单：
{key_points}

回答：
{answer}"""


def load_eval_set() -> list[dict]:
    """读取评测集"""
    path = EVALS_DIR / "eval_set.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"评测集不存在：{path}")
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


# ---------- 第一层：检索质量 ----------

def eval_retrieval(retriever: HybridRetriever, items: list[dict],
                   top_k: int = TOP_K) -> list[dict]:
    """对每条问题做混合检索，检查期望文档是否命中及其排名"""
    details = []
    for item in items:
        results = retriever.search(item["question"], top_k=top_k)
        docs = [r["doc"] for r in results]
        expected = item["expected_doc"]
        rank = next((i + 1 for i, d in enumerate(docs) if d == expected), None)
        details.append({
            "question": item["question"],
            "expected_doc": expected,
            "hit": rank is not None,
            "rank": rank,
            "top_docs": docs[:5],
        })
    return details


# ---------- 第二层：回答质量 ----------

def _judge(key_points: list[str], answer_text: str) -> dict:
    """LLM 裁判：判断回答覆盖了哪些要点"""
    prompt = _JUDGE_PROMPT.format(
        key_points="\n".join(f"- {p}" for p in key_points),
        answer=answer_text,
    )
    try:
        raw = llm.chat([{"role": "user", "content": prompt}],
                       temperature=0.0, max_tokens=600)
        match = re.search(r"\{.*\}", raw, re.S)
        data = json.loads(match.group(0)) if match else {}
        score = float(data.get("score", 0.0))
        return {"score": max(0.0, min(1.0, score)),
                "covered": data.get("covered", []),
                "missing": data.get("missing", [])}
    except Exception:  # noqa: BLE001 —— 裁判失败按 0 分记录，不中断评测
        return {"score": 0.0, "covered": [], "missing": key_points}


def eval_answers(items: list[dict], retriever: HybridRetriever,
                 use_rerank: bool = True) -> list[dict]:
    """对每条问题跑完整问答链路，用 LLM 裁判评估要点覆盖率"""
    details = []
    for item in items:
        result = rag.answer(item["question"], retriever=retriever,
                            use_rerank=use_rerank)
        judge = _judge(item["key_points"], result["answer"])
        details.append({
            "question": item["question"],
            "answer": result["answer"],
            "citations": result["citations"],
            "judge": judge,
        })
    return details
