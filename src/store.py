"""索引构建与持久化：chunks + 归一化向量 + BM25 语料

索引文件（index/ 目录，不提交 git）：
- chunks.json  切分后的文本块及其来源元信息
- vectors.npy  对应的归一化向量矩阵
BM25 在加载时对 chunk 文本即时重建（几千块仅需 1~2 秒），无需额外持久化。
"""
import json
from pathlib import Path

import jieba
import numpy as np
from rank_bm25 import BM25Okapi

from . import embedder, ingest
from .config import INDEX_DIR


def tokenize(text: str) -> list[str]:
    """中文分词（供 BM25 使用）：jieba 精确模式，英文/数字统一小写"""
    return [tok.lower() for tok in jieba.lcut(text) if tok.strip()]


def build_index(data_dir: Path | None = None) -> dict:
    """扫描文档 → 切分 → 向量化 → 持久化，返回索引统计信息"""
    chunks = ingest.build_chunks(data_dir)
    if not chunks:
        raise RuntimeError("data/docs/ 目录下没有可用文档（支持 .md / .txt / .pdf）")

    vectors = embedder.embed_texts([c["text"] for c in chunks])

    INDEX_DIR.mkdir(exist_ok=True)
    np.save(INDEX_DIR / "vectors.npy", vectors)
    (INDEX_DIR / "chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "chunks": len(chunks),
        "docs": len({c["doc"] for c in chunks}),
        "dim": int(vectors.shape[1]),
    }


def load_index() -> tuple[list[dict], np.ndarray, BM25Okapi]:
    """载入索引，返回 (chunks, vectors, bm25)"""
    chunks_path = INDEX_DIR / "chunks.json"
    vectors_path = INDEX_DIR / "vectors.npy"
    if not chunks_path.exists() or not vectors_path.exists():
        raise FileNotFoundError("索引不存在，请先运行：python main.py ingest")

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    vectors = np.load(vectors_path)
    bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
    return chunks, vectors, bm25
