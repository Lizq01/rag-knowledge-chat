"""MCP 服务端配置。

原则：**服务端自己的配置由服务端自己读环境变量**，不通过工具参数从 Client 传进来。
- Client 不该知道检索阈值、路径、模型名，更不该知道 API Key；
- 参数收窄（top_k 上限等）写在代码里的 Schema 与这里的环境变量上，Client 只能在这个范围内选。

导入 `src.config` 会加载 rag-knowledge-chat/.env —— 这是服务端凭据的唯一来源。
"""
import os

import mcp_server  # noqa: F401  —— 触发 sys.path 引导

# 加载项目 .env：LLM / 向量模型的凭据只存在于服务端进程
import src.config  # noqa: F401,E402

SERVER_NAME = os.getenv("MCP_SERVER_NAME", "rag-knowledge")
SERVER_VERSION = os.getenv("MCP_SERVER_VERSION", "1.0.0")

# ---- 工具参数边界（同时写进 JSON Schema，Client 传越界值会被直接拒绝）----
DEFAULT_TOP_K = int(os.getenv("MCP_DEFAULT_TOP_K", "5"))
MAX_TOP_K = max(1, int(os.getenv("MCP_MAX_TOP_K", "20")))
MIN_QUERY_CHARS = int(os.getenv("MCP_MIN_QUERY_CHARS", "2"))
MAX_QUERY_CHARS = int(os.getenv("MCP_MAX_QUERY_CHARS", "200"))

# 单条片段回传给 Client 的字符上限：避免一次检索把上下文撑爆
MAX_TEXT_CHARS = int(os.getenv("MCP_MAX_TEXT_CHARS", "1200"))

# 重排是 CPU 开销大头（首次还要加载 ONNX 模型），默认关闭，按需开启
RERANK_CHOICES = ("none", "cross-encoder")
DEFAULT_RERANK = os.getenv("MCP_DEFAULT_RERANK", "none").strip().lower()
if DEFAULT_RERANK not in RERANK_CHOICES:
    # 配置写错不该让服务端起不来：Schema 的默认值必须是枚举内的合法值
    DEFAULT_RERANK = "none"

SEARCH_MODE_CHOICES = ("hybrid", "bm25", "vector")
