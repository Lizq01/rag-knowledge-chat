"""MCP Server 定义：把 KnowledgeService 的三个能力注册成 MCP 工具。

三条设计约束（也是 MCP 与传统 Function Calling 的关键差别所在）：

1. **Schema 是契约，不是提示**。参数类型、枚举、取值范围全部由类型标注 + pydantic
   Field 生成 JSON Schema 并随 tools/list 下发；Client 传越界值时，服务端在进入业务
   代码之前就拒绝，模型拿到的是结构化错误，而不是一段自然语言报错。
2. **凭据只在服务端**。工具不收 token、不收路径、不收模型名；服务端自己读环境变量。
   stdio 传输下 Client 只传白名单环境变量（见 SDK 的 DEFAULT_INHERITED_ENV_VARS），
   因此 Client 进程里的 Key 根本不会进入服务端进程。
3. **stdout 是协议通道**。stdio 服务端的 stdout 只能走 JSON-RPC 帧，
   任何 print() 都会让 Client 解析失败；本模块所有日志走 stderr。

启动：python -m mcp_server（由 Client 以子进程方式拉起）
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Annotated, Literal

# 允许 `python mcp_server/server.py` 这种直接以脚本方式启动（Client 配置里常这么写）
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ToolAnnotations
from pydantic import Field

from mcp_server.config import (
    DEFAULT_RERANK,
    DEFAULT_TOP_K,
    MAX_QUERY_CHARS,
    MAX_TOP_K,
    MIN_QUERY_CHARS,
    SERVER_NAME,
    SERVER_VERSION,
)
from mcp_server.knowledge import KnowledgeError, KnowledgeService, scrub

log = logging.getLogger("mcp_server")

# 三个工具都是只读：声明出去，Client 侧（以及 inspector）可以据此做权限控制与 UI 提示
_READ_ONLY = ToolAnnotations(read_only_hint=True, idempotent_hint=True, open_world_hint=False)

INSTRUCTIONS = """\
本服务提供本地知识库（AI Agent / RAG / 向量库 / 提示工程 / 微调 主题的技术笔记）的只读检索。

推荐调用顺序：
1. list_sources —— 先确认知识库覆盖哪些文档，判断用户问题是否在内；
2. search_knowledge —— 用自然语言问题检索，返回按相关性排序的片段，每条带 chunk_id；
3. get_chunk —— 需要完整上下文时，按 chunk_id 取单条片段；可顺 neighbors.prev/next 扩展。

引用规范：回答时用「《文档名》第 N 段」标注出处，N 取自结果里的 position。
边界：知识库只覆盖上述主题；问题超出范围时直接说明，不要用检索结果硬凑。
"""

service = KnowledgeService()

server = MCPServer(
    name=SERVER_NAME,
    version=SERVER_VERSION,
    instructions=INSTRUCTIONS,
)

# ===== 工具入参类型（JSON Schema 的唯一来源）=====

Query = Annotated[
    str,
    Field(
        min_length=MIN_QUERY_CHARS,
        max_length=MAX_QUERY_CHARS,
        description="检索问题，用完整问句或关键词短语均可；中文优先。过短（<2 字）或过长（>200 字）会被拒绝。",
    ),
]
TopK = Annotated[
    int,
    Field(ge=1, le=MAX_TOP_K, description=f"返回片段数，取值 1~{MAX_TOP_K}。事实核实用 2~3，综述用 5~8。"),
]
Mode = Annotated[
    Literal["hybrid", "bm25", "vector"],
    Field(
        description=(
            "检索通道。hybrid=BM25+向量经 RRF 融合（默认，综合效果最好）；"
            "bm25=纯关键词，适合精确术语/编号；vector=纯语义，适合同义改写。"
        ),
    ),
]
Rerank = Annotated[
    Literal["none", "cross-encoder"],
    Field(
        description=(
            "是否用本地 Cross-Encoder 重排。none=按召回序返回（快，默认）；"
            "cross-encoder=更准但每次多耗 CPU（首次调用还要加载模型）。"
        ),
    ),
]
ChunkId = Annotated[
    str,
    Field(
        pattern=r"^[^#\s]{1,180}\.(?:md|txt|pdf)::\d{1,4}$",
        max_length=200,
        description="片段 id，格式 '<文档名>::<段号>'，例如 rag-notes.md::3。只能来自 search_knowledge 的返回或 list_sources 的文档名。",
    ),
]


# ===== 工具注册 =====


@server.tool(
    name="search_knowledge",
    title="知识库检索",
    description=(
        "在本地知识库中做语义+关键词混合检索，返回按相关性排序的文本片段。\n\n"
        "何时用：需要查证概念、对比方案、找具体做法或数字时，先用它拿候选片段。\n"
        "何时不用：只是想确认知识库有哪些文档 → 用 list_sources；"
        "已经拿到 chunk_id、要读完整片段 → 用 get_chunk。\n\n"
        "返回：query/mode/returned/results[]，每条含 chunk_id、doc、position、score、"
        "score_kind（rrf|bm25|cosine|cross-encoder）、text（超长会截断并置 truncated=true）。\n"
        "检索不到内容是正常结果（returned=0），换个说法或改用 bm25 通道再试，不要臆造出处。"
    ),
    annotations=_READ_ONLY,
)
def search_knowledge(query: Query, top_k: TopK = DEFAULT_TOP_K,
                     mode: Mode = "hybrid", rerank: Rerank = DEFAULT_RERANK) -> dict:  # type: ignore[assignment]
    try:
        return service.search(query, top_k=top_k, mode=mode, rerank_mode=rerank)
    except KnowledgeError as exc:
        raise ToolError(scrub(str(exc))) from exc


@server.tool(
    name="get_chunk",
    title="读取知识库片段",
    description=(
        "按 chunk_id 取回单条片段的完整文本，并给出前后相邻片段 id。\n\n"
        "何时用：search_knowledge 返回的片段被截断（truncated=true）、"
        "或需要完整上下文核对细节时；也可用 neighbors 顺次翻阅同一文档。\n"
        "何时不用：还没有 chunk_id → 先调 search_knowledge，不要凭猜测构造 id。\n\n"
        "返回：chunk_id/doc/position/text/chars/truncated/neighbors{prev,next}（无相邻段时为 null）。"
    ),
    annotations=_READ_ONLY,
)
def get_chunk(chunk_id: ChunkId) -> dict:
    try:
        return service.get_chunk(chunk_id)
    except KnowledgeError as exc:
        raise ToolError(scrub(str(exc))) from exc


@server.tool(
    name="list_sources",
    title="列出知识库文档",
    description=(
        "列出知识库收录的文档清单与规模（文档名、片段数、字符数），以及索引构建信息。\n\n"
        "何时用：回答前先判断知识库是否覆盖该主题；或需要向用户说明「依据哪些资料」。\n"
        "何时不用：问题已明确落在知识库范围内 → 直接 search_knowledge，省一次往返。\n\n"
        "返回：total_docs/total_chunks/docs[]/index{backend,built_at,embedding_model}/"
        "warnings[]（索引与源文档不一致时会给出提示）。不含服务端磁盘路径。"
    ),
    annotations=_READ_ONLY,
)
def list_sources() -> dict:
    try:
        return service.list_sources()
    except KnowledgeError as exc:
        raise ToolError(scrub(str(exc))) from exc


# ===== 启动 =====


def main() -> None:
    logging.basicConfig(
        level=os.getenv("MCP_LOG_LEVEL", "WARNING").upper(),
        stream=sys.stderr,  # 绝不能是 stdout
        format="[mcp_server] %(levelname)s %(message)s",
    )
    # 第三方库在 INFO 级别会刷屏（faiss 加载、jieba 分词、hf 下载），只保留告警以上
    for noisy in ("faiss", "jieba", "httpx", "httpcore", "huggingface_hub", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if sys.stdout.isatty():
        log.warning(
            "stdout 是终端而非管道：stdio 服务端应由 MCP Client 拉起，手动运行会一直等待输入。"
            "自检请用 python -m mcp_server.selfcheck，或用 inspector 调试。"
        )
    log.info("%s v%s 启动（stdio）", SERVER_NAME, SERVER_VERSION)
    server.run("stdio")


if __name__ == "__main__":
    main()
