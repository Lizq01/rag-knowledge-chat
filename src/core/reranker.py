"""core · 重排器接口。

召回追求「不漏」（返回十几条候选），重排追求「精准」（精排出 top N）。
输入输出都是 chunk 字典列表，方便串进 services 的问答链路。

降级原则：重排是「增强」环节，任何失败都不应中断主流程——
宁可用召回序，也不能让问答直接报错。失败原因通过 info 回传，便于观测与告警。
"""
from abc import ABC, abstractmethod


class BaseReranker(ABC):
    """重排器抽象：对召回候选做精排，不改变 chunk 的数据结构。"""

    @property
    @abstractmethod
    def name(self) -> str:
        """重排器标识（如 bge-reranker-v2-m3 / llm），用于日志与评测归因。"""

    @abstractmethod
    def rerank(
        self, question: str, candidates: list[dict], top_n: int
    ) -> tuple[list[dict], dict]:
        """按与 question 的相关度重排 candidates，返回 (top_n 片段, 执行信息)。

        执行信息至少含 backend / requested；发生降级时补 degraded_from 与 reason。
        """
