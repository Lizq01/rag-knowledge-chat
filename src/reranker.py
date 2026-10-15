"""候选重排：用 LLM 对「问题-片段」相关性打分并重新排序

说明：生产环境常用 Cross-Encoder 重排模型（如 bge-reranker）做这一步。
本项目默认用 LLM 重排——零额外部署、效果稳定；
接口设计与 Cross-Encoder 兼容，后续可无缝替换。
"""
import json
import re

from . import llm

_PROMPT = """你是检索结果重排器。给定用户问题和若干候选文本片段，请为每个片段与问题的相关性打分（0-10 的整数，10 表示直接相关）。

只输出 JSON 数组，不要输出任何其他内容，格式：
[{{"id": 1, "score": 8}}, {{"id": 2, "score": 3}}]

用户问题：{question}

候选片段：
{candidates}"""


def _parse_scores(raw: str) -> dict[int, int]:
    """从 LLM 输出中稳健地提取 {id: score}"""
    match = re.search(r"\[.*?\]", raw, re.S)
    if not match:
        return {}
    try:
        items = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    scores: dict[int, int] = {}
    for item in items:
        try:
            scores[int(item["id"])] = int(item["score"])
        except (KeyError, TypeError, ValueError):
            continue
    return scores


def llm_rerank(question: str, candidates: list[dict], top_n: int = 5) -> list[dict]:
    """LLM 重排；LLM 输出异常时降级为按原顺序截断（保证流程不中断）"""
    if not candidates:
        return []

    lines = [f"[{i}] {c['text'][:300]}" for i, c in enumerate(candidates, start=1)]
    prompt = _PROMPT.format(question=question, candidates="\n\n".join(lines))

    try:
        raw = llm.chat([{"role": "user", "content": prompt}],
                       temperature=0.0, max_tokens=800)
        scores = _parse_scores(raw)
    except Exception:  # noqa: BLE001 —— 重排失败不应影响主流程
        scores = {}

    if not scores:
        return candidates[:top_n]

    indexed = list(enumerate(candidates, start=1))
    ranked = sorted(indexed, key=lambda pair: scores.get(pair[0], 0), reverse=True)
    return [chunk for _, chunk in ranked[:top_n]]
