"""core · 向量库接口。

契约刻意只留「建 / 查 / 存 / 规模」四件事：
FAISS（flat 精确 / hnsw 近似）、NumPy 暴力检索、云端向量库都能用同一张脸面对上层。

约定：向量写入前已做 L2 归一化，因此内积即余弦相似度，取值域 [-1, 1]。
"""
from abc import ABC, abstractmethod

import numpy as np


class BaseVectorStore(ABC):
    """向量库抽象：可替换的检索后端。"""

    backend: str   # 形如 "faiss:flat" / "faiss:hnsw" / "numpy"
    dim: int       # 向量维度
    ntotal: int    # 索引中的向量条数

    @abstractmethod
    def search(self, query_vec: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        """检索最相似的 top_k 条，返回 [(chunk 下标, 相似度), ...]，按相似度降序。"""

    @abstractmethod
    def save(self, path) -> None:
        """持久化索引；内存态实现可抛 NotImplementedError。"""

    @classmethod
    @abstractmethod
    def load(cls, path) -> "BaseVectorStore":
        """从磁盘恢复索引；内存态实现可抛 NotImplementedError。"""
