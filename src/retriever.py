"""混合检索：BM25（稀疏/关键词）+ 语义向量（稠密）→ RRF 排名融合

为什么用混合检索？
- BM25 擅长精确匹配（术语、编号、专有名词），但不懂同义改写；
- 向量检索理解语义，但对精确术语不敏感；
- 两者互补。RRF（Reciprocal Rank Fusion）按「排名倒数」融合两路结果：
      score(d) = Σ 1 / (k + rank_i(d))     （k 取标准值 60）
  只依赖排名、不依赖分数尺度，无需调参，工业界常用且效果稳定。

向量检索的底层实现由 vectorstore 决定（FAISS 或 NumPy），本模块不感知具体后端。
"""
import time

import numpy as np

from . import embedder, store
from .vectorstore import VectorIndex

RRF_K = 60


class HybridRetriever:
    """加载索引后的统一检索入口"""

    def __init__(self):
        self.chunks, self.vector_index, self.bm25 = store.load_index()

    @property
    def backend(self) -> str:
        """当前向量检索后端（faiss:flat / faiss:hnsw / numpy）"""
        return self.vector_index.backend

    @property
    def vector_index_obj(self) -> VectorIndex:
        return self.vector_index

    # ---------- 单路检索 ----------

    def search_bm25(self, query: str, top_k: int) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(store.tokenize(query))
        order = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in order]

    def search_vector(self, query: str, top_k: int) -> list[tuple[int, float]]:
        query_vec = embedder.embed_query(query)
        return self.vector_index.search(query_vec, top_k)

    # ---------- 融合检索 ----------

    def search_with_stats(self, query: str, top_k: int = 10) -> tuple[list[dict], dict]:
        """混合检索，额外返回各阶段耗时（毫秒），用于可观测性与性能回归"""
        stats: dict[str, float] = {}

        started = time.perf_counter()
        query_vec = embedder.embed_query(query)
        stats["embed_ms"] = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        bm25_hits = self.search_bm25(query, top_k)
        stats["bm25_ms"] = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        vector_hits = self.vector_index.search(query_vec, top_k)
        stats["vector_ms"] = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        results = self._fuse([bm25_hits, vector_hits], top_k)
        stats["fuse_ms"] = (time.perf_counter() - started) * 1000
        stats["retrieve_ms"] = stats["embed_ms"] + stats["bm25_ms"] + stats["vector_ms"] + stats["fuse_ms"]
        return results, stats

    def search_multi_with_stats(self, queries: list[str], top_k: int = 10,
                                weights: list[float] | None = None
                                ) -> tuple[list[dict], dict]:
        """多路检索：对每个改写查询各做一次混合检索，再对全部排序结果做 RRF 融合

        权重用于表达「哪一路更可信」：原始问题最贴近用户意图，权重默认给满，
        改写变体只是补充召回，权重略低，避免改写跑偏时把结果带坏。
        """
        queries = [q for q in queries if q and q.strip()]
        if not queries:
            return [], {"retrieve_ms": 0.0}
        if len(queries) == 1:
            return self.search_with_stats(queries[0], top_k)

        stats = {"embed_ms": 0.0, "bm25_ms": 0.0, "vector_ms": 0.0, "fuse_ms": 0.0}
        rankings: list[list[tuple[int, float]]] = []
        per_query_weights: list[float] = []

        for i, query in enumerate(queries):
            started = time.perf_counter()
            query_vec = embedder.embed_query(query)
            stats["embed_ms"] += (time.perf_counter() - started) * 1000

            started = time.perf_counter()
            bm25_hits = self.search_bm25(query, top_k)
            stats["bm25_ms"] += (time.perf_counter() - started) * 1000

            started = time.perf_counter()
            vector_hits = self.vector_index.search(query_vec, top_k)
            stats["vector_ms"] += (time.perf_counter() - started) * 1000

            weight = weights[i] if weights and i < len(weights) else 1.0
            rankings.extend([bm25_hits, vector_hits])
            per_query_weights.extend([weight, weight])

        started = time.perf_counter()
        results = self._fuse(rankings, top_k, weights=per_query_weights)
        stats["fuse_ms"] = (time.perf_counter() - started) * 1000
        stats["queries"] = len(queries)
        stats["retrieve_ms"] = (stats["embed_ms"] + stats["bm25_ms"]
                                + stats["vector_ms"] + stats["fuse_ms"])
        return results, stats

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """混合检索：返回带 RRF 融合分数的 chunk 列表（按相关性降序）"""
        results, _ = self.search_with_stats(query, top_k)
        return results

    def _fuse(self, rankings: list[list[tuple[int, float]]], top_k: int,
              weights: list[float] | None = None) -> list[dict]:
        """RRF 融合：只利用各路排名，规避稀疏/稠密分数尺度不一致的问题"""
        fused: dict[int, float] = {}
        for list_index, hits in enumerate(rankings):
            weight = weights[list_index] if weights and list_index < len(weights) else 1.0
            for rank, (idx, _score) in enumerate(hits):
                fused[idx] = fused.get(idx, 0.0) + weight / (RRF_K + rank + 1)

        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        results = []
        for idx, score in ranked:
            item = dict(self.chunks[idx])
            item["rrf_score"] = round(score, 6)
            results.append(item)
        return results
