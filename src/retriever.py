"""混合检索：BM25（稀疏/关键词）+ 语义向量（稠密）→ RRF 排名融合

为什么用混合检索？
- BM25 擅长精确匹配（术语、编号、专有名词），但不懂同义改写；
- 向量检索理解语义，但对精确术语不敏感；
- 两者互补。RRF（Reciprocal Rank Fusion）按「排名倒数」融合两路结果：
      score(d) = Σ 1 / (k + rank_i(d))     （k 取标准值 60）
  只依赖排名、不依赖分数尺度，无需调参，工业界常用且效果稳定。
"""
import numpy as np

from . import embedder, store

RRF_K = 60


class HybridRetriever:
    """加载索引后的统一检索入口"""

    def __init__(self):
        self.chunks, self.vectors, self.bm25 = store.load_index()

    # ---------- 单路检索 ----------

    def search_bm25(self, query: str, top_k: int) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(store.tokenize(query))
        order = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in order]

    def search_vector(self, query: str, top_k: int) -> list[tuple[int, float]]:
        query_vec = embedder.embed_query(query)
        sims = self.vectors @ query_vec  # 向量已归一化：内积 = 余弦相似度
        order = np.argsort(sims)[::-1][:top_k]
        return [(int(i), float(sims[i])) for i in order]

    # ---------- 融合检索 ----------

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """混合检索：返回带 RRF 融合分数的 chunk 列表（按相关性降序）"""
        bm25_hits = self.search_bm25(query, top_k)
        vector_hits = self.search_vector(query, top_k)

        fused: dict[int, float] = {}
        for hits in (bm25_hits, vector_hits):
            for rank, (idx, _score) in enumerate(hits):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)

        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        results = []
        for idx, score in ranked:
            item = dict(self.chunks[idx])
            item["rrf_score"] = round(score, 6)
            results.append(item)
        return results
