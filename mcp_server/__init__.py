"""MCP 服务端：把 rag-knowledge-chat 的检索能力封装成标准 MCP 工具。

模块分工：
- config.py    ：服务端配置，全部从环境变量读，不接受 Client 传入；
- knowledge.py ：KnowledgeService，索引加载 / 混合检索 / 片段查询 / 来源清单；
- server.py    ：MCP Server 定义（工具注册、参数 Schema、stdio 启动）；
- selfcheck.py ：端到端自检（tools/list、正常调用、非法参数、凭据泄漏）。

启动（stdio，由 Client 以子进程方式拉起）：
    python -m mcp_server

被拉起时 stdout 是 JSON-RPC 通道，任何 print() 都会污染协议流 —— 日志一律走 stderr。
"""
import sys
from pathlib import Path

# 允许从任意工作目录启动（Client 的 cwd 未必是项目根目录）
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

__version__ = "1.0.0"
