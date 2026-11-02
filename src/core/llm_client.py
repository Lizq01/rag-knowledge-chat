"""core · LLM 客户端接口。

统一契约：吃一组 OpenAI 格式的 messages，吐一段纯文本。
services 层只认这个接口，不感知底层是 DeepSeek / OpenAI / Kimi / Qwen 还是本地模型。
"""
from abc import ABC, abstractmethod


class BaseLLMClient(ABC):
    """LLM 客户端抽象：屏蔽厂商差异，只暴露 chat 能力。"""

    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 2000,
        retries: int = 2,
        timeout: float = 60.0,
    ) -> str:
        """发起一次对话，返回模型输出的纯文本。

        实现侧需保证两点：
        - 超时必须显式给出（SDK 默认值过长，同步调用会把整条问答链路拖死）；
        - 重试带指数退避 + 抖动，且关掉 SDK 自带重试，避免两层重试叠加。
        """
