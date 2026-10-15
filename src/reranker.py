"""候选重排：本地 Cross-Encoder（ONNX）与 LLM 打分双后端

后端由 config.RERANK_BACKEND 或调用参数决定：
- cross-encoder：本地 bge-reranker（fastembed + ONNX Runtime），无需网络、无 token 成本，默认首选；
- llm：调用大模型给候选打分（M1 方案），零部署成本，但每次问答有网络往返与 token 开销；
- auto：优先 Cross-Encoder，加载失败自动降级 LLM，再失败降级为按召回序截断；
- none：不重排，直接按召回序截断（A/B 对比时的基线）。

降级原则：重排是「增强」环节，任何失败都不应中断主流程——
宁可用召回序，也不能让问答直接报错。降级原因通过返回值回传，便于观测与告警。
"""
from __future__ import annotations

import json
import logging
import os
import re
import time

from . import config, llm
from .config import RERANK_MAX_CHARS, RERANK_TOP_N

logger = logging.getLogger(__name__)

VALID_BACKENDS = ("auto", "cross-encoder", "llm", "none")

# Cross-Encoder 是「(问题, 文本) 拼接后整体过模型」，精度高于双塔向量模型，但每次都要重算，
# 因此只对召回后的少量候选使用（本项目默认 10 选 5）。
_ce_model = None
_ce_error: str | None = None
_ce_load_ms = 0.0   # 本次进程内加载模型耗时；模型已加载时为 0


# ---------------------------------------------------------------- Cross-Encoder

def reset_cross_encoder_cache() -> None:
    """清空模型缓存与失败记录（warmup 或换模型后重新尝试加载时使用）"""
    global _ce_model, _ce_error
    _ce_model = None
    _ce_error = None


def cross_encoder_available() -> tuple[bool, str]:
    """Cross-Encoder 是否已可直接使用（已下载 / 缓存成功）,返回 (可用?, 说明)"""
    if _ce_model is not None:
        return True, "已加载"
    try:
        _load_cross_encoder()
    except Exception as e:  # noqa: BLE001
        return False, str(e)
    return True, "已加载"


def preload_cross_encoder() -> tuple[bool, str]:
    """预下载并加载重排模型（python main.py warmup 使用），返回 (成功?, 说明)"""
    reset_cross_encoder_cache()
    try:
        _load_cross_encoder()
    except Exception as e:  # noqa: BLE001
        return False, str(e)
    return True, f"{config.RERANK_MODEL} 已就绪"


def _load_cross_encoder():
    """惰性加载重排模型；首次使用需联网下载（约 0.6GB），失败原因会被缓存避免反复重试"""
    global _ce_model, _ce_error, _ce_load_ms
    if _ce_model is not None:
        return _ce_model
    if _ce_error is not None:
        raise RuntimeError(_ce_error)

    started = time.perf_counter()
    try:
        # 先 import config 确保 HF 镜像环境变量已生效，再 import fastembed
        from . import config as _config  # noqa: F401
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        threads = _config.RERANK_THREADS or min(16, os.cpu_count() or 4)
        _ce_model = TextCrossEncoder(_config.RERANK_MODEL, threads=threads)
        _ce_load_ms = (time.perf_counter() - started) * 1000
    except Exception as e:  # noqa: BLE001 —— ImportError / 下载失败 / 显存不足 统一转成可读错误
        _ce_error = (
            f"Cross-Encoder 模型（{config.RERANK_MODEL}）加载失败：{type(e).__name__}: {e}。"
            f"首次使用需联网下载模型，可先执行 `python main.py warmup` 预下载；"
            f"也可在 .env 中设置 RERANK_BACKEND=llm 改用大模型重排")
        raise RuntimeError(_ce_error) from e
    return _ce_model


def cross_encoder_rerank(question: str, candidates: list[dict],
                         top_n: int = RERANK_TOP_N) -> list[dict]:
    """本地 Cross-Encoder 重排，返回带 ce_score 的 top_n 片段"""
    if not candidates:
        return []

    model = _load_cross_encoder()
    texts = [c["text"][:RERANK_MAX_CHARS] for c in candidates]
    scores = list(model.rerank(question, texts, batch_size=config.RERANK_BATCH_SIZE))
    if len(scores) != len(candidates):
        raise RuntimeError(
            f"重排返回的分数数量（{len(scores)}）与候选数量（{len(candidates)}）不一致")

    scored = []
    for chunk, score in zip(candidates, scores):
        item = dict(chunk)
        item["ce_score"] = round(float(score), 4)
        scored.append(item)
    scored.sort(key=lambda c: c["ce_score"], reverse=True)
    return scored[:top_n]


# ---------------------------------------------------------------- LLM 重排

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


def _llm_scores(question: str, candidates: list[dict]) -> dict[int, int]:
    """调用 LLM 打分；失败或无有效输出时抛异常，由上层决定降级路径"""
    lines = [f"[{i}] {c['text'][:300]}" for i, c in enumerate(candidates, start=1)]
    prompt = _PROMPT.format(question=question, candidates="\n\n".join(lines))
    raw = llm.chat([{"role": "user", "content": prompt}],
                   temperature=0.0, max_tokens=800)
    scores = _parse_scores(raw)
    if not scores:
        raise RuntimeError("LLM 未返回可解析的打分结果")
    return scores


def _apply_scores(candidates: list[dict], scores: dict[int, int],
                  top_n: int) -> list[dict]:
    indexed = list(enumerate(candidates, start=1))
    ranked = sorted(indexed, key=lambda pair: scores.get(pair[0], 0), reverse=True)
    return [dict(chunk) for _, chunk in ranked[:top_n]]


def llm_rerank(question: str, candidates: list[dict], top_n: int = RERANK_TOP_N) -> list[dict]:
    """LLM 重排；LLM 输出异常时降级为按原顺序截断（保证流程不中断）"""
    if not candidates:
        return []
    try:
        scores = _llm_scores(question, candidates)
    except Exception as e:  # noqa: BLE001 —— 重排失败不应影响主流程
        logger.warning("LLM 重排失败，降级为按召回序截断：%s", e)
        return candidates[:top_n]
    return _apply_scores(candidates, scores, top_n)


# ---------------------------------------------------------------- 统一入口

def rerank(question: str, candidates: list[dict], top_n: int = RERANK_TOP_N,
           backend: str | None = None) -> tuple[list[dict], dict]:
    """按后端重排，返回 (重排后的片段, 执行信息)

    执行信息 info 字段：
    - backend        实际生效的后端（cross-encoder / llm / none）
    - requested      请求的目标后端
    - degraded_from  若发生降级，记录原本想用的后端
    - reason         降级原因
    """
    backend = (backend or config.RERANK_BACKEND or "auto").lower()
    if backend not in VALID_BACKENDS:
        raise ValueError(f"不支持的重排后端：{backend}（可选 {' / '.join(VALID_BACKENDS)}）")
    if not candidates:
        return [], {"backend": "none", "requested": backend}

    chain = {
        "auto": ("cross-encoder", "llm"),
        "cross-encoder": ("cross-encoder",),
        "llm": ("llm",),
        "none": (),
    }[backend]

    last_error: str | None = None
    for candidate_backend in chain:
        try:
            if candidate_backend == "cross-encoder":
                picked = cross_encoder_rerank(question, candidates, top_n=top_n)
            else:
                picked = _apply_scores(candidates, _llm_scores(question, candidates), top_n)
        except Exception as e:  # noqa: BLE001
            last_error = f"{type(e).__name__}: {e}"
            logger.warning("重排后端 %s 失败：%s", candidate_backend, last_error)
            continue

        info = {"backend": candidate_backend, "requested": backend}
        if candidate_backend == "cross-encoder" and _ce_load_ms:
            # 首次使用要加载 ONNX 模型（数秒），单独记出来，避免被误读成重排算法本身慢
            info["model_load_ms"] = round(_ce_load_ms)
        if candidate_backend != backend and backend != "auto":
            info["degraded_from"] = backend
            info["reason"] = last_error
        elif last_error:  # auto 模式下首个后端失败但后续后端成功，同样留痕
            info["degraded_from"] = chain[0]
            info["reason"] = last_error
        return picked, info

    # 全链路失败（或显式指定 none）：按召回序截断，保证问答不中断
    info = {"backend": "none", "requested": backend}
    if last_error:
        info["degraded_from"] = chain[0]
        info["reason"] = last_error
        logger.warning("重排全部后端均失败，按召回序截断：%s", last_error)
    return candidates[:top_n], info
