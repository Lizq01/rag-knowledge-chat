"""评测：检索命中率（Recall@K）+ 重排质量（Hit@N / MRR）+ 回答要点覆盖率（LLM 裁判）

评测集格式（evals/eval_set.jsonl，每行一条）：
{"question": "...", "expected_doc": "xxx.md", "key_points": ["...", "..."]}

三层指标各自的职责：
1. Recall@K    —— 召回能力：期望文档有没有进候选池（重排救不回没召回的东西）
2. Hit@N / MRR —— 排序能力：重排后期望文档是否排到前面（区分重排后端的关键指标）
3. 要点覆盖率   —— 端到端质量：最终回答是否把关键点说全
"""
import json
import re
import time

from . import llm, rag
from .config import EVALS_DIR, RERANK_TOP_N, TOP_K
from .reranker import rerank
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


def load_multi_turn_set() -> list[dict]:
    """读取多轮评测集（每条含 history + 追问 + 期望片段）"""
    path = EVALS_DIR / "multi_turn_set.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"多轮评测集不存在：{path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def percentile(values: list[float], pct: float) -> float:
    """最近秩百分位数（P50 / P95）；输入为空返回 0"""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(round(pct / 100 * (len(ordered) - 1)))))
    return ordered[idx]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# ---------- 第一层：检索质量 ----------

def _expected_rank(results: list[dict], item: dict) -> int | None:
    """期望片段在结果中的名次（1 起）；未命中返回 None

    用 expected_text（答案所在片段的特征子串）而非文档名做判定：
    文档级判定太粗（一篇文档有 4~5 个片段，几乎总能进前 5，指标会饱和）；
    片段级才能反映「排在最前面的是不是真正包含答案的那一段」。
    """
    needle = item.get("expected_text", "")
    if needle:
        for i, chunk in enumerate(results, start=1):
            if needle in chunk["text"]:
                return i
        return None
    return next((i + 1 for i, c in enumerate(results)
                 if c["doc"] == item["expected_doc"]), None)


def eval_retrieval(retriever: HybridRetriever, items: list[dict],
                   top_k: int = TOP_K) -> list[dict]:
    """对每条问题做混合检索，检查期望片段/文档是否命中及其排名"""
    details = []
    for item in items:
        results = retriever.search(item["question"], top_k=top_k)
        docs = [r["doc"] for r in results]
        rank = _expected_rank(results, item)
        doc_rank = next((i + 1 for i, d in enumerate(docs)
                         if d == item["expected_doc"]), None)
        details.append({
            "question": item["question"],
            "expected_doc": item["expected_doc"],
            "hit": rank is not None,
            "rank": rank,
            "doc_rank": doc_rank,
            "top_docs": docs[:5],
        })
    return details


# ---------- 第二层：重排质量 ----------

def eval_rerank(retriever: HybridRetriever, items: list[dict], backend: str,
                top_k: int = TOP_K, top_n: int = RERANK_TOP_N) -> dict:
    """评估单个重排后端：重排后 top_n 的片段命中率、MRR 与耗时分位

    MRR（Mean Reciprocal Rank）= 1/rank 的均值，同时反映「有没有命中」与「排得多靠前」，
    比单纯的命中率更能区分不同重排后端。
    片段级指标是主判据，文档级作为参考（文档级通常已接近饱和）。
    """
    chunk_hits: list[float] = []
    doc_hits: list[float] = []
    reciprocal_ranks: list[float] = []
    latencies: list[float] = []
    degraded = 0
    used_backend = backend

    for item in items:
        candidates = retriever.search(item["question"], top_k=top_k)
        started = time.perf_counter()
        picked, info = rerank(item["question"], candidates, top_n=top_n, backend=backend)
        latencies.append((time.perf_counter() - started) * 1000)

        used_backend = info.get("backend", backend)
        if info.get("degraded_from"):
            degraded += 1

        rank = _expected_rank(picked, item)
        chunk_hits.append(1.0 if rank else 0.0)
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
        doc_hits.append(1.0 if any(c["doc"] == item["expected_doc"] for c in picked) else 0.0)

    return {
        "backend": backend,
        "used_backend": used_backend,
        "degraded": degraded,
        "hit_at_n": _mean(chunk_hits),
        "doc_hit_at_n": _mean(doc_hits),
        "mrr": _mean(reciprocal_ranks),
        "latency_ms_p50": percentile(latencies, 50),
        "latency_ms_p95": percentile(latencies, 95),
        "top_n": top_n,
    }


def compare_backends(retriever: HybridRetriever, items: list[dict],
                     backends: list[str] | None = None, top_k: int = TOP_K,
                     top_n: int = RERANK_TOP_N) -> list[dict]:
    """对多个重排后端跑同一份评测集，输出可直接对比的指标表"""
    return [eval_rerank(retriever, items, b, top_k=top_k, top_n=top_n)
            for b in (backends or ["none", "llm", "cross-encoder"])]


# ---------- 第二层（多轮）：查询改写效果 ----------

def eval_multi_turn(retriever: HybridRetriever, items: list[dict],
                    rewrite_backend: str, rerank_backend: str = "auto",
                    top_n: int = RERANK_TOP_N) -> dict:
    """评估某个改写后端在多轮追问下的检索质量（只跑检索+重排，不生成回答）

    不生成回答的原因：改写影响的是「召回什么」，在检索层就能量化，
    生成与否不影响这个判断，却能省下大量 token。
    """
    hits: list[float] = []
    reciprocal_ranks: list[float] = []
    latencies: list[float] = []
    query_counts: list[int] = []
    degraded = 0
    used_backend = rewrite_backend

    for item in items:
        started = time.perf_counter()
        _candidates, used, _stats, rewrite_info, _rerank = rag.retrieve_turn(
            item["question"], retriever=retriever, history=item.get("history", []),
            rewrite_backend=rewrite_backend, rerank_backend=rerank_backend)
        latencies.append((time.perf_counter() - started) * 1000)

        used_backend = rewrite_info.get("backend", rewrite_backend)
        query_counts.append(len(rewrite_info["queries"]))
        if rewrite_info.get("degraded"):
            degraded += 1

        rank = _expected_rank(used, item)
        hits.append(1.0 if rank else 0.0)
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)

    return {
        "rewrite_backend": rewrite_backend,
        "used_backend": used_backend,
        "degraded": degraded,
        "hit_at_n": _mean(hits),
        "mrr": _mean(reciprocal_ranks),
        "avg_queries": _mean([float(q) for q in query_counts]),
        "latency_ms_p50": percentile(latencies, 50),
        "latency_ms_p95": percentile(latencies, 95),
        "top_n": top_n,
    }


def compare_rewrite(retriever: HybridRetriever, items: list[dict],
                    rerank_backend: str = "auto") -> list[dict]:
    """对比「不改写」与「自动改写」在多轮追问上的差异"""
    return [eval_multi_turn(retriever, items, b, rerank_backend=rerank_backend)
            for b in ("none", "auto")]


# ---------- 第三层：回答质量 ----------

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
                 rerank_backend: str | None = None,
                 use_rerank: bool = True) -> list[dict]:
    """对每条问题跑完整问答链路，用 LLM 裁判评估要点覆盖率"""
    details = []
    for item in items:
        result = rag.answer(item["question"], retriever=retriever,
                            use_rerank=use_rerank, rerank_backend=rerank_backend)
        judge = _judge(item["key_points"], result["answer"])
        details.append({
            "question": item["question"],
            "answer": result["answer"],
            "citations": result["citations"],
            "judge": judge,
            "rerank": result["rerank"],
            "timings": result["timings"],
        })
    return details
