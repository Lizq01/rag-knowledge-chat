"""知识库服务层：面向 Agent 的只读检索能力。

这一层刻意不依赖 MCP：它只关心「怎么查」，不关心「怎么传」。
以后要加 HTTP + SSE 传输、或给内部服务直接 import，都只需换掉 server.py 的外壳。

三个能力的边界：
- search_knowledge：语义入口，按问题召回片段（返回 chunk_id，供后续精读）；
- get_chunk       ：按 chunk_id 取单条完整片段（模型核实细节时用，比重新检索便宜）；
- list_sources    ：枚举知识库里有什么文档（模型判断「这个问题值不值得查」时用）。

失败分两类：
- KnowledgeError：可预期失败（索引没建、chunk_id 不存在），信息可以原样回给 Client；
- 其他异常：程序缺陷，只回传类型与简短信息，细节写 stderr。
两类出口都过 scrub()，避免环境变量里的凭据随异常栈泄漏到 Client。
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

from mcp_server.config import (
    DEFAULT_TOP_K,
    MAX_TEXT_CHARS,
    MAX_TOP_K,
    RERANK_CHOICES,
    SEARCH_MODE_CHOICES,
)

# 环境变量名里命中这些词，其值视为凭据，禁止出现在任何回传给 Client 的文本里
_SECRET_NAME = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|AUTH)", re.I)


class KnowledgeError(RuntimeError):
    """可预期的失败：索引缺失/损坏、chunk_id 不存在、参数越界。"""


def scrub(text: str) -> str:
    """抹掉文本中出现过的敏感环境变量值。

    工具结果会经 JSON-RPC 回到 Client，并可能被塞进模型上下文；
    traceback、配置回显、第三方库报错都可能带上 Key，所以出口统一过滤。
    """
    if not text:
        return text
    for name, value in os.environ.items():
        if len(value) < 8 or not _SECRET_NAME.search(name):
            continue
        if value in text:
            text = text.replace(value, "***")
    return text


@dataclass
class _Index:
    """一次进程内的索引快照：索引对象 + 按 id 的倒排 + 元信息。"""

    retriever: Any
    chunks: list[dict]
    by_id: dict[str, dict]
    backend: str
    meta: dict = field(default_factory=dict)


class KnowledgeService:
    """索引懒加载 + 只读查询。

    懒加载的理由：MCP Server 是常驻子进程，启动即握手。
    如果 import 阶段就加载 FAISS 索引与 BM25 语料，Client 的连接会一直等；
    放到第一次真正查询时再做，握手稳定在毫秒级。
    """

    def __init__(self, default_top_k: int = DEFAULT_TOP_K,
                 max_text_chars: int = MAX_TEXT_CHARS):
        self._default_top_k = default_top_k
        self._max_text_chars = max_text_chars
        self._index: _Index | None = None

    # ------------------------------------------------------------ 索引

    def _get_index(self) -> _Index:
        if self._index is None:
            self._index = self._load_index()
        return self._index

    @staticmethod
    def _load_index() -> _Index:
        from src.modules.retriever import HybridRetriever
        from src.modules.store import index_meta

        try:
            retriever = HybridRetriever()
        except FileNotFoundError as exc:
            raise KnowledgeError(
                f"知识库索引不存在，请先在服务端执行 `python main.py ingest`。原因：{exc}"
            ) from exc
        except (RuntimeError, ValueError) as exc:
            raise KnowledgeError(f"知识库索引加载失败：{exc}") from exc

        chunks = list(retriever.chunks)
        return _Index(
            retriever=retriever,
            chunks=chunks,
            by_id={c["id"]: c for c in chunks},
            backend=retriever.backend,
            meta=index_meta() or {},
        )

    # ------------------------------------------------------------ 工具实现

    def search(self, query: str, top_k: int | None = None, mode: str = "hybrid",
               rerank_mode: str = "none") -> dict:
        """混合检索，返回带 chunk_id 的片段列表。"""
        index = self._get_index()
        query = (query or "").strip()
        top_k = top_k or self._default_top_k

        # Schema 已在协议层拦过一次，这里再校验是因为服务层也可能被直接调用
        if not query:
            raise KnowledgeError("query 不能为空")
        if not 1 <= top_k <= MAX_TOP_K:
            raise KnowledgeError(f"top_k 必须在 1~{MAX_TOP_K} 之间，收到 {top_k}")
        if mode not in SEARCH_MODE_CHOICES:
            raise KnowledgeError(f"mode 必须是 {'/'.join(SEARCH_MODE_CHOICES)} 之一，收到 {mode!r}")
        if rerank_mode not in RERANK_CHOICES:
            raise KnowledgeError(
                f"rerank 必须是 {'/'.join(RERANK_CHOICES)} 之一，收到 {rerank_mode!r}")

        started = time.perf_counter()
        if mode == "hybrid":
            hits = [self._as_hit(h, h.get("rrf_score"), "rrf")
                    for h in index.retriever.search(query, top_k=top_k)]
        elif mode == "bm25":
            hits = [self._as_hit(index.chunks[i], score, "bm25")
                    for i, score in index.retriever.search_bm25(query, top_k)]
        else:
            hits = [self._as_hit(index.chunks[i], score, "cosine")
                    for i, score in index.retriever.search_vector(query, top_k)]
        retrieve_ms = (time.perf_counter() - started) * 1000

        rerank_info: dict = {"requested": rerank_mode, "backend": "none", "degraded": False}
        rerank_ms = 0.0
        if rerank_mode == "cross-encoder" and hits:
            started = time.perf_counter()
            try:
                from src.modules.reranker import rerank

                picked, info = rerank(query, hits, top_n=top_k, backend="cross-encoder")
                hits = [self._as_hit(h, h.get("ce_score"), "cross-encoder") for h in picked]
                rerank_info = {
                    "requested": rerank_mode,
                    "backend": info.get("backend", "none"),
                    # 重排是增强项：模型缺失时降级为召回序，服务本身不失败
                    "degraded": info.get("backend") != "cross-encoder",
                    "reason": info.get("reason"),
                }
            except Exception as exc:  # noqa: BLE001 —— 重排失败不影响检索结果可用性
                rerank_info.update({"degraded": True, "reason": scrub(f"{type(exc).__name__}: {exc}")})
            rerank_ms = (time.perf_counter() - started) * 1000

        results = []
        for hit in hits[:top_k]:
            text, truncated = self._clip(hit["text"])
            score = hit.get("score")
            results.append({
                "chunk_id": hit["id"],
                "doc": hit["doc"],
                "position": hit["position"],
                "score": None if score is None else round(float(score), 4),
                "score_kind": hit.get("score_kind"),
                "chars": len(hit["text"]),
                "text": text,
                "truncated": truncated,
            })

        return {
            "query": query,
            "mode": mode,
            "top_k": top_k,
            "returned": len(results),
            "results": results,
            "index": {"backend": index.backend, "chunks": len(index.chunks)},
            "rerank": rerank_info,
            "timings_ms": {"retrieve": round(retrieve_ms, 1), "rerank": round(rerank_ms, 1)},
        }

    def get_chunk(self, chunk_id: str) -> dict:
        """按 chunk_id 取单条片段；附带前后相邻片段 id，便于模型按需顺藤摸瓜。"""
        index = self._get_index()
        chunk = index.by_id.get(chunk_id)
        if chunk is None:
            raise KnowledgeError(
                f"chunk_id 不存在：{chunk_id!r}。chunk_id 形如 '<文档名>::<段号>'"
                f"（例：rag-notes.md::3）；可用 list_sources() 看有哪些文档，"
                f"或用 search_knowledge() 拿到合法 chunk_id。"
            )

        doc, position = chunk["doc"], int(chunk["position"])
        text, truncated = self._clip(chunk["text"])
        prev_id, next_id = f"{doc}::{position - 1}", f"{doc}::{position + 1}"
        return {
            "chunk_id": chunk_id,
            "doc": doc,
            "position": position,
            "chars": len(chunk["text"]),
            "text": text,
            "truncated": truncated,
            "neighbors": {
                "prev": prev_id if prev_id in index.by_id else None,
                "next": next_id if next_id in index.by_id else None,
            },
        }

    def list_sources(self) -> dict:
        """列出知识库里的文档与规模。

        只回文档名与统计，不回磁盘路径：Client 不需要知道服务端目录结构。
        """
        index = self._get_index()

        docs: dict[str, dict] = {}
        for chunk in index.chunks:
            entry = docs.setdefault(
                chunk["doc"], {"doc": chunk["doc"], "chunks": 0, "chars": 0})
            entry["chunks"] += 1
            entry["chars"] += len(chunk["text"])

        warnings: list[str] = []
        try:
            from src.modules.store import check_freshness

            warnings = [scrub(w) for w in check_freshness()]
        except Exception as exc:  # noqa: BLE001 —— 新鲜度检查属增值信息，失败不能影响主功能
            warnings = [f"索引新鲜度检查失败：{type(exc).__name__}"]

        return {
            "total_docs": len(docs),
            "total_chunks": len(index.chunks),
            "docs": sorted(docs.values(), key=lambda d: d["doc"]),
            "index": {
                "backend": index.backend,
                "built_at": index.meta.get("created_at"),
                "embedding_model": index.meta.get("embedding_model"),
            },
            "warnings": warnings,
        }

    # ------------------------------------------------------------ 内部

    @staticmethod
    def _as_hit(chunk: dict, score: Any, kind: str) -> dict:
        item = dict(chunk)
        item["score"] = score
        item["score_kind"] = kind
        return item

    def _clip(self, text: str) -> tuple[str, bool]:
        limit = self._max_text_chars
        if len(text) <= limit:
            return text, False
        return text[:limit] + "…", True
