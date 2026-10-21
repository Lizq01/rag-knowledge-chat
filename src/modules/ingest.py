"""文档加载与切分（支持 .md / .txt / .pdf）

切分策略：
1. 先按空行切出自然段；
2. 段落合并到不超过 chunk_size 字符；
3. 超长段落按句子边界（。！？；等）二次切分；
4. 相邻块之间保留 overlap 字符重叠，避免边界处语义丢失。
"""
import re
from pathlib import Path

from .config import CHUNK_OVERLAP, CHUNK_SIZE, DATA_DIR

SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}


# ---------- 加载 ----------

def _read_txt_or_md(path: Path) -> str:
    for encoding in ("utf-8", "gbk"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


def load_documents(data_dir: Path | None = None) -> list[tuple[str, str]]:
    """加载目录下所有支持的文档，返回 [(文件名, 全文), ...]"""
    data_dir = data_dir or DATA_DIR
    if not data_dir.exists():
        raise FileNotFoundError(f"文档目录不存在：{data_dir}")

    docs: list[tuple[str, str]] = []
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        text = _read_pdf(path) if path.suffix.lower() == ".pdf" else _read_txt_or_md(path)
        if text.strip():
            docs.append((path.name, text))
    return docs


# ---------- 切分 ----------

_SENTENCE_RE = re.compile(r"(?<=[。！？；!?;])")


def _split_long_paragraph(para: str, chunk_size: int) -> list[str]:
    """超长段落按句子边界切成多块；单句仍超长则硬切"""
    pieces: list[str] = []
    buf = ""
    for sent in _SENTENCE_RE.split(para):
        if not sent:
            continue
        while len(sent) > chunk_size:  # 罕见的超长单句，硬切兜底
            if buf:
                pieces.append(buf)
                buf = ""
            pieces.append(sent[:chunk_size])
            sent = sent[chunk_size:]
        if len(buf) + len(sent) <= chunk_size:
            buf += sent
        else:
            if buf:
                pieces.append(buf)
            buf = sent
    if buf:
        pieces.append(buf)
    return pieces


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    """把长文本切分成带重叠的语义块"""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buf = ""

    for para in paragraphs:
        if len(para) > chunk_size:
            if buf:
                chunks.append(buf)
                buf = buf[-overlap:] if overlap else ""
            for piece in _split_long_paragraph(para, chunk_size):
                chunks.append(piece)
            continue

        if not buf:
            buf = para
        elif len(buf) + len(para) + 1 <= chunk_size:
            buf = f"{buf}\n{para}"
        else:
            chunks.append(buf)
            tail = buf[-overlap:] if overlap else ""
            buf = f"{tail}\n{para}".strip() if tail else para

    if buf:
        chunks.append(buf)
    return chunks


def build_chunks(data_dir: Path | None = None) -> list[dict]:
    """加载 + 切分，输出带元信息的 chunk 列表"""
    chunks: list[dict] = []
    for doc_name, text in load_documents(data_dir):
        for i, piece in enumerate(chunk_text(text), start=1):
            chunks.append({
                "id": f"{doc_name}::{i}",
                "doc": doc_name,
                "position": i,
                "text": piece,
            })
    return chunks
