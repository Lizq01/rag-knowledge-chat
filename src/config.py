"""全局配置：读取 .env、路径常量、检索参数"""
import os
from pathlib import Path

from dotenv import load_dotenv

# ---------- 路径 ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "docs"
INDEX_DIR = PROJECT_ROOT / "index"
EVALS_DIR = PROJECT_ROOT / "evals"

# ---------- 环境变量 ----------
load_dotenv(PROJECT_ROOT / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-chat")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")

# LLM 调用：显式超时（避免默认长连接挂死）+ 重试退避
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "60"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))

TOP_K = int(os.getenv("TOP_K", "10"))
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "5"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))

# ---------- 向量索引后端（M2）----------
# auto = 优先 FAISS，装不上自动退回 NumPy；faiss = 强制（装不上直接报错）；numpy = 强制
VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "auto").strip().lower()
# flat = IndexFlatIP 精确检索（与 NumPy 暴力检索结果一致，仅更快）
# hnsw = IndexHNSWFlat 近似检索（牺牲少量召回换亚线性查询，数据量大时启用）
FAISS_INDEX_TYPE = os.getenv("FAISS_INDEX_TYPE", "flat").strip().lower()
FAISS_HNSW_M = int(os.getenv("FAISS_HNSW_M", "32"))
FAISS_HNSW_EF_SEARCH = int(os.getenv("FAISS_HNSW_EF_SEARCH", "64"))

# ---------- 重排后端（M2）----------
# auto = Cross-Encoder 优先，不可用时降级 LLM；cross-encoder / llm / none
RERANK_BACKEND = os.getenv("RERANK_BACKEND", "auto").strip().lower()
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3-int8")
RERANK_BATCH_SIZE = int(os.getenv("RERANK_BATCH_SIZE", "16"))
# ONNX Runtime 推理线程数；0 = 自动（取 min(16, CPU 核数)）。
# 实测（32 核 Windows）：默认线程下 411ms/候选，显式 16 线程 316ms/候选，线程过多反而变慢。
RERANK_THREADS = int(os.getenv("RERANK_THREADS", "0"))
# Cross-Encoder 输入有长度上限，超长片段截断以控制 CPU 耗时（正常 chunk 约 500 字，不会触发）
RERANK_MAX_CHARS = int(os.getenv("RERANK_MAX_CHARS", "1500"))

# 国内网络下载本地模型时自动走镜像（在加载 fastembed 之前设置）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# hf-mirror 镜像不支持 Xet 传输协议，强制走传统下载通道
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
