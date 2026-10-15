"""向量索引抽象层：FAISS（精确 flat / 近似 hnsw）与 NumPy 暴力检索的可替换实现

为什么抽象一层？
- 检索后端是「可替换的工程决策」，不该散落在 retriever 里；
- FAISS 可能装不上（平台无 wheel、离线环境），必须能自动退回 NumPy 保证链路可用；
- 两种后端对外契约完全一致：search(query_vec, top_k) -> [(chunk 下标, 相似度)]，按相似度降序。

约定：向量写入前已做 L2 归一化（见 embedder），因此内积即余弦相似度，取值域 [-1, 1]。
"""
from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

from . import config

logger = logging.getLogger(__name__)


class VectorIndex(Protocol):
    """向量索引统一契约（结构子类型，无需显式继承）"""

    backend: str   # 形如 "faiss:flat" / "faiss:hnsw" / "numpy"
    dim: int       # 向量维度
    ntotal: int    # 索引中的向量条数

    def search(self, query_vec: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        ...


# ---------------------------------------------------------------- NumPy 后端

class NumpyIndex:
    """NumPy 暴力检索：无外部依赖的兜底实现，结果精确

    用 argpartition 取 top_k 而非全量 argsort，避免 N 很大时做 O(N log N) 全排序。
    """

    backend = "numpy"

    def __init__(self, vectors: np.ndarray):
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if self.vectors.ndim != 2:
            raise ValueError(f"向量矩阵必须是二维的，实际形状 {self.vectors.shape}")
        self.dim = int(self.vectors.shape[1])
        self.ntotal = int(self.vectors.shape[0])

    def search(self, query_vec: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        k = min(int(top_k), self.ntotal)
        if k <= 0:
            return []
        sims = self.vectors @ np.asarray(query_vec, dtype=np.float32).reshape(-1)
        # 先 O(N) 划出 top_k 候选，再只对这 k 个排序
        cand = np.argpartition(sims, -k)[-k:]
        cand = cand[np.argsort(sims[cand])[::-1]]
        return [(int(i), float(sims[i])) for i in cand]


# ---------------------------------------------------------------- FAISS 后端

class FaissIndex:
    """FAISS 检索：flat 为精确内积检索，hnsw 为近似检索

    flat 与 NumPy 结果一致（同为精确检索），只是把矩阵乘法交给 C++ 实现；
    hnsw 用图索引换亚线性查询复杂度，召回率由 efSearch 控制（越大越准、越慢）。
    """

    def __init__(self, index, index_type: str = "flat"):
        self.index = index
        self.backend = f"faiss:{index_type}"
        self.dim = int(index.d)
        self.ntotal = int(index.ntotal)

    @classmethod
    def build(cls, vectors: np.ndarray, index_type: str = "flat") -> "FaissIndex":
        import faiss  # 延迟导入：缺依赖时由调用方决定降级还是报错

        matrix = np.ascontiguousarray(vectors, dtype=np.float32)
        dim = int(matrix.shape[1])

        if index_type == "flat":
            index = faiss.IndexFlatIP(dim)          # 精确内积检索，无需训练
        elif index_type == "hnsw":
            index = faiss.IndexHNSWFlat(dim, config.FAISS_HNSW_M)
            index.hnsw.efConstruction = config.FAISS_HNSW_M * 2
        else:
            raise ValueError(f"不支持的 FAISS 索引类型：{index_type}（可选 flat / hnsw）")

        index.add(matrix)
        return cls(index, index_type)

    def search(self, query_vec: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        k = min(int(top_k), self.ntotal)
        if k <= 0:
            return []
        query = np.ascontiguousarray(query_vec, dtype=np.float32).reshape(1, -1)
        scores, ids = self.index.search(query, k)
        # FAISS 在候选不足时返回 -1，需过滤
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]

    def save(self, path) -> None:
        import faiss

        faiss.write_index(self.index, str(path))

    @classmethod
    def load(cls, path) -> "FaissIndex":
        import faiss

        index = faiss.read_index(str(path))
        index_type = "hnsw" if "HNSW" in type(index).__name__ else "flat"
        if index_type == "hnsw":
            # efSearch 是查询期参数，不随索引持久化，需每次加载后重设
            index.hnsw.efSearch = config.FAISS_HNSW_EF_SEARCH
        return cls(index, index_type)


# ---------------------------------------------------------------- 工厂函数

def faiss_available() -> tuple[bool, str]:
    """FAISS 是否可用，返回 (可用?, 说明)"""
    try:
        import faiss  # noqa: F401
    except Exception as e:  # noqa: BLE001 —— ImportError / OSError(动态库缺失) 都要接住
        return False, f"{type(e).__name__}: {e}"
    return True, "ok"


def build_vector_index(vectors: np.ndarray, backend: str | None = None,
                       index_type: str | None = None) -> VectorIndex:
    """按配置构建向量索引；auto 模式在 FAISS 不可用时自动退回 NumPy"""
    backend = (backend or config.VECTOR_BACKEND or "auto").lower()
    index_type = (index_type or config.FAISS_INDEX_TYPE or "flat").lower()

    if backend in ("auto", "faiss"):
        try:
            return FaissIndex.build(vectors, index_type)
        except ImportError as e:
            if backend == "faiss":      # 显式指定 faiss 时不静默降级
                raise RuntimeError(
                    f"VECTOR_BACKEND=faiss 但未安装 faiss-cpu：{e}。"
                    f"请执行 pip install faiss-cpu，或改用 VECTOR_BACKEND=numpy") from e
            logger.warning("FAISS 不可用，退回 NumPy 暴力检索：%s", e)
        except ValueError as e:
            # 索引类型写错等配置类错误：auto 模式也直接抛出，避免静默走错后端
            if backend == "faiss":
                raise
            logger.warning("FAISS 索引构建失败，退回 NumPy：%s", e)

    return NumpyIndex(vectors)
