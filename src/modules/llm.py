"""LLM 客户端封装（OpenAI 兼容接口，兼容 DeepSeek / OpenAI 等）

工程约束：
- 超时必须显式给出（SDK 默认值过长，重排这类同步调用会把整个问答拖死）；
- 重试必须带退避与抖动，避免多个请求同时失败后一起重试造成雪崩；
- SDK 自带的 max_retries 关掉，改由本模块统一控制重试语义与日志。
"""
import logging
import random
import time

from openai import OpenAI

from .config import (LLM_MAX_RETRIES, LLM_TIMEOUT, MODEL_NAME, OPENAI_API_KEY,
                     OPENAI_BASE_URL)

logger = logging.getLogger(__name__)

_client = None


def get_client() -> OpenAI:
    """惰性初始化客户端"""
    global _client
    if _client is None:
        if not OPENAI_API_KEY:
            raise RuntimeError("未配置 OPENAI_API_KEY，请检查项目根目录的 .env 文件")
        _client = OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            timeout=LLM_TIMEOUT,
            max_retries=0,  # 重试由 chat() 统一处理，避免两层重试叠加
        )
    return _client


def chat(messages: list[dict], temperature: float = 0.3, max_tokens: int = 2000,
         retries: int = LLM_MAX_RETRIES, timeout: float = LLM_TIMEOUT) -> str:
    """调用 LLM，失败时按指数退避 + 抖动重试"""
    last_err: Exception | None = None

    for attempt in range(retries + 1):
        try:
            resp = get_client().chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:  # noqa: BLE001 —— 网络 / 限流 / 5xx 统一重试
            last_err = e
            if attempt >= retries:
                break
            # 退避：1s、2s、4s...，叠加 0~30% 抖动打散重试时刻
            delay = (2 ** attempt) * (1.0 + random.random() * 0.3)
            logger.warning("LLM 调用失败（第 %d 次，%.1fs 后重试）：%s: %s",
                           attempt + 1, delay, type(e).__name__, e)
            time.sleep(delay)

    raise RuntimeError(f"LLM 调用失败（已重试 {retries} 次）：{last_err}")
