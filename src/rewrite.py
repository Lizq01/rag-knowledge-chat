"""查询改写：把「带上下文的追问」变成可独立检索的查询，并做多路扩展

解决两个真实问题：
1. **指代消解**：多轮里用户会问「那它的 k 一般取多少？」，这种问题单独拿去检索几乎必然失败——
   向量模型不知道「它」是什么，BM25 也匹配不到关键词。
2. **词汇错配**：用户用词与资料用词不一致（「怎么把两个检索结果合起来」vs「RRF 融合」）。
   生成 1~2 个语义等价的改写变体做多路检索，能显著提高召回。

降级设计（改写是增强环节，失败不能阻断问答）：
- LLM 不可用 / 输出不可解析 → 退到规则改写（把上一轮问题拼到当前问题前）
- 规则改写也没素材（无历史）→ 直接用原问题
- **无论怎么改写，原始问题一定保留为一路检索**，避免改写跑偏把召回带崩
"""
from __future__ import annotations

import json
import logging
import re

from . import config, llm

logger = logging.getLogger(__name__)

VALID_BACKENDS = ("auto", "llm", "rule", "none")

_PROMPT = """你是查询改写器。请根据「对话历史」把用户的追问改写成可以独立检索的问题。

要求：
1. 把指代（它 / 这个 / 上面那种 / 第二个）替换成历史中明确的实体；
2. 第一个问题必须与用户意图完全一致且能独立理解；
3. 再给出 {variants} 个语义等价但用词不同的改写，用于多路检索补充召回；
4. 只输出 JSON 数组，不要任何解释，例如：["独立问题", "改写1", "改写2"]

对话历史：
{history}

用户追问：{question}"""

_RULE_TEMPLATE = "{prev} {question}"


def _parse_queries(raw: str) -> list[str]:
    """从 LLM 输出中提取字符串数组；格式不符时返回空列表交由上层降级"""
    match = re.search(r"\[.*\]", raw, re.S)
    if not match:
        return []
    try:
        items = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    return [str(item).strip() for item in items
            if isinstance(item, str) and item.strip()]


def _format_history(history: list[dict], limit: int = 4) -> str:
    """把历史轮次压成改写器可读的文本；无历史时返回占位说明"""
    if not history:
        return "（无历史，这是第一轮提问）"
    lines = []
    for turn in history[-limit:]:
        role = "用户" if turn.get("role") == "user" else "助手"
        content = (turn.get("content") or "").strip().replace("\n", " ")
        lines.append(f"{role}：{content[:200]}")
    return "\n".join(lines)


def _llm_rewrite(question: str, history: list[dict], variants: int) -> list[str]:
    """LLM 改写：指代消解 + 多路扩展。异常一律向上抛，由 rewrite() 决定降级"""
    prompt = _PROMPT.format(
        history=_format_history(history),
        question=question,
        variants=max(0, variants - 1),
    )
    raw = llm.chat([{"role": "user", "content": prompt}],
                   temperature=0.0, max_tokens=400, timeout=min(30.0, config.LLM_TIMEOUT))
    queries = _parse_queries(raw)
    if not queries:
        raise RuntimeError("LLM 未返回可解析的改写结果")
    return queries


def _rule_rewrite(question: str, history: list[dict]) -> list[str]:
    """规则兜底：把最近一次用户提问拼到当前问题前，解决最常见的指代缺失

    不追求「理解」，只求把关键实体带进检索式——成本为零、永远可用。
    """
    previous = ""
    for turn in reversed(history or []):
        if turn.get("role") == "user" and turn.get("content"):
            previous = turn["content"].strip()
            break
    if not previous or previous == question:
        return []
    return [_RULE_TEMPLATE.format(prev=previous, question=question)]


def rewrite(question: str, history: list[dict] | None = None,
            backend: str | None = None) -> dict:
    """生成用于检索的查询列表

    返回：
    {
      "queries":  检索用的查询列表（首条恒为原始问题）,
      "backend":  实际生效的后端（llm / rule / none）,
      "degraded": 是否发生了降级,
      "reason":   降级原因,
    }
    """
    backend = (backend or config.REWRITE_BACKEND or "auto").lower()
    if backend not in VALID_BACKENDS:
        raise ValueError(f"不支持的改写后端：{backend}（可选 {' / '.join(VALID_BACKENDS)}）")

    history = history or []
    variants = max(1, config.REWRITE_COUNT)
    # 首轮提问没有指代可消，默认不做多路扩展（收益小、耗时翻倍）
    if not history and not config.REWRITE_ON_FIRST_TURN:
        return {"queries": [question], "backend": "none", "degraded": False}

    result: dict = {"queries": [question], "backend": "none", "degraded": False}
    if backend == "none":
        return result

    chain = ("llm", "rule") if backend in ("auto", "llm") else ("rule",)
    last_error: str | None = None

    for candidate in chain:
        try:
            if candidate == "llm":
                rewritten = _llm_rewrite(question, history, variants)
            else:
                rewritten = _rule_rewrite(question, history)
            if not rewritten:
                last_error = f"{candidate} 改写结果为空"
                continue
            # 原始问题必须保留；其余去重后按需截断
            merged = [question]
            for item in rewritten:
                normalized = item.strip()
                if normalized and normalized != question and normalized not in merged:
                    merged.append(normalized)
            result = {"queries": merged[:variants], "backend": candidate, "degraded": False}
            if candidate != backend and backend != "auto":
                result["degraded"] = True
                result["degraded_from"] = backend
            if last_error and candidate != chain[0]:
                result["degraded"] = True
                result["degraded_from"] = chain[0]
                result["reason"] = last_error
            return result
        except Exception as e:  # noqa: BLE001 —— 改写失败不阻断问答
            last_error = f"{type(e).__name__}: {e}"
            logger.warning("查询改写后端 %s 失败：%s", candidate, last_error)
            continue

    result["degraded"] = True
    result["degraded_from"] = chain[0]
    result["reason"] = last_error or "无可用改写结果"
    return result
