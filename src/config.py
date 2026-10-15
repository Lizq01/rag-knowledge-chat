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

TOP_K = int(os.getenv("TOP_K", "10"))
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "5"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))

# 国内网络下载本地模型时自动走镜像（在加载 fastembed 之前设置）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# hf-mirror 镜像不支持 Xet 传输协议，强制走传统下载通道
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
