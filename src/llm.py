"""LLM 客户端封装（OpenAI 兼容接口，兼容 DeepSeek / OpenAI 等）"""
from openai import OpenAI

from .config import MODEL_NAME, OPENAI_API_KEY, OPENAI_BASE_URL

_client = None


def get_client() -> OpenAI:
    """惰性初始化客户端"""
    global _client
    if _client is None:
        if not OPENAI_API_KEY:
            raise RuntimeError("未配置 OPENAI_API_KEY，请检查项目根目录的 .env 文件")
        _client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    return _client


def chat(messages: list[dict], temperature: float = 0.3, max_tokens: int = 2000,
         retries: int = 2) -> str:
    """调用 LLM，失败时自动重试"""
    last_err = None
    for _ in range(retries + 1):
        try:
            resp = get_client().chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:  # noqa: BLE001 —— 网络类异常统一重试
            last_err = e
    raise RuntimeError(f"LLM 调用失败（已重试 {retries} 次）：{last_err}")
