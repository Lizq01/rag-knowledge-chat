"""索引构建与持久化：chunks + 向量索引 + BM25 语料 + 索引元信息

索引文件（index/ 目录，不提交 git）：
- chunks.json    切分后的文本块及其来源元信息
- vectors.npy    原始向量矩阵（始终写入：切换后端 / 重建索引时使用）
- faiss.index    FAISS 检索索引（仅后端为 faiss 时写入）
- meta.json      索引元信息（后端、维度、条数、模型、切分参数、文档指纹）

BM25 在加载时对 chunk 文本即时重建（几千块仅需 1~2 秒），无需额外持久化。

meta.json 有两个作用：
1. 加载时校验向量条数是否与 chunk 数一致，不一致说明索引损坏，要求重建；
2. 记录当时的文档指纹与模型/切分参数，加载时可发现「文档已改但未重建索引」这类
   隐性故障——这是 RAG 系统最常见的线上事故之一（答案看起来正常，其实检索的旧内容）。
"""
import hashlib
import json
from datetime import datetime
from pathlib import Path

import jieba
import numpy as np
from rank_bm25 import BM25Okapi

from . import config, embedder, ingest
from .config import CHUNK_OVERLAP, CHUNK_SIZE, DATA_DIR, EMBEDDING_MODEL, INDEX_DIR
from .vectorstore import FaissIndex, NumpyIndex, VectorIndex, build_vector_index

CHUNKS_FILE = "chunks.json"
VECTORS_FILE = "vectors.npy"
FAISS_FILE = "faiss.index"
META_FILE = "meta.json"


def tokenize(text: str) -> list[str]:
    """中文分词（供 BM25 使用）：jieba 精确模式，英文/数字统一小写"""
    return [tok.lower() for tok in jieba.lcut(text) if tok.strip()]


# ---------------------------------------------------------------- 文档指纹

def docs_fingerprint(data_dir: Path | None = None) -> str:
    """对 data/docs 下的文档（名称 + 大小 + 修改时间）取指纹

    用文件元信息而非全文哈希：避免每次加载索引都把全部文档（含 PDF 解析）读一遍。
    """
    data_dir = data_dir or DATA_DIR
    if not data_dir.exists():
        return ""
    digest = hashlib.sha1()
    for path in sorted(data_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in ingest.SUPPORTED_SUFFIXES:
            stat = path.stat()
            digest.update(f"{path.name}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8"))
    return digest.hexdigest()[:16]


# ---------------------------------------------------------------- 构建

def build_index(data_dir: Path | None = None, backend: str | None = None,
                index_type: str | None = None) -> dict:
    """扫描文档 → 切分 → 向量化 → 构建向量索引 → 持久化，返回索引统计信息"""
    chunks = ingest.build_chunks(data_dir)
    if not chunks:
        raise RuntimeError("data/docs/ 目录下没有可用文档（支持 .md / .txt / .pdf）")

    vectors = embedder.embed_texts([c["text"] for c in chunks])
    vector_index = build_vector_index(vectors, backend=backend, index_type=index_type)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(INDEX_DIR / VECTORS_FILE, vectors)
    (INDEX_DIR / CHUNKS_FILE).write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")

    saved = [CHUNKS_FILE, VECTORS_FILE]
    if vector_index.backend.startswith("faiss"):
        vector_index.save(INDEX_DIR / FAISS_FILE)   # type: ignore[attr-defined]
        saved.append(FAISS_FILE)

    meta = {
        "backend": vector_index.backend,
        "dim": vector_index.dim,
        "ntotal": vector_index.ntotal,
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "rerank_model": config.RERANK_MODEL,
        "docs_fingerprint": docs_fingerprint(data_dir),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    (INDEX_DIR / META_FILE).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    saved.append(META_FILE)

    return {
        "chunks": len(chunks),
        "docs": len({c["doc"] for c in chunks}),
        "dim": vector_index.dim,
        "backend": vector_index.backend,
        "saved": saved,
    }


# ---------------------------------------------------------------- 加载

def load_index(backend: str | None = None) -> tuple[list[dict], VectorIndex, BM25Okapi]:
    """载入索引，返回 (chunks, 向量索引, bm25)

    后端选择：显式 VECTOR_BACKEND=numpy 时忽略已存在的 FAISS 索引（便于 A/B 对比与故障回退），
    其余情况优先使用磁盘上的 FAISS 索引。
    """
    chunks_path = INDEX_DIR / CHUNKS_FILE
    vectors_path = INDEX_DIR / VECTORS_FILE
    if not chunks_path.exists() or not vectors_path.exists():
        raise FileNotFoundError("索引不存在，请先运行：python main.py ingest")

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    vectors = np.load(vectors_path)

    faiss_path = INDEX_DIR / FAISS_FILE
    if (backend or config.VECTOR_BACKEND).lower() != "numpy" and faiss_path.exists():
        try:
            vector_index: VectorIndex = FaissIndex.load(faiss_path)
        except Exception as e:  # noqa: BLE001 —— 索引文件损坏时退回 NumPy，不阻断问答
            raise RuntimeError(
                f"FAISS 索引读取失败：{e}。可删除 index/faiss.index 后重新 ingest，"
                f"或设置 VECTOR_BACKEND=numpy") from e
    else:
        vector_index = NumpyIndex(vectors)

    if vector_index.ntotal != len(chunks):
        raise RuntimeError(
            f"索引不一致：向量 {vector_index.ntotal} 条 vs 文本块 {len(chunks)} 条，"
            f"请重新运行 python main.py ingest")

    bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
    return chunks, vector_index, bm25


def index_meta() -> dict | None:
    """读取索引元信息；索引不存在或损坏时返回 None"""
    path = INDEX_DIR / META_FILE
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def check_freshness() -> list[str]:
    """对比当前文档与配置，返回索引过期风险提示（空列表表示无风险）"""
    meta = index_meta()
    if not meta:
        return []

    warnings: list[str] = []
    if meta.get("docs_fingerprint") != docs_fingerprint():
        warnings.append("data/docs/ 内容与构建索引时不一致，建议重新 ingest")
    if meta.get("embedding_model") != EMBEDDING_MODEL:
        warnings.append(
            f"向量模型已变更（索引时 {meta.get('embedding_model')} → 当前 {EMBEDDING_MODEL}），必须重新 ingest")
    if meta.get("chunk_size") != CHUNK_SIZE or meta.get("chunk_overlap") != CHUNK_OVERLAP:
        warnings.append(
            f"切分参数已变更（索引时 {meta.get('chunk_size')}/{meta.get('chunk_overlap')}"
            f" → 当前 {CHUNK_SIZE}/{CHUNK_OVERLAP}），建议重新 ingest")
    return warnings
