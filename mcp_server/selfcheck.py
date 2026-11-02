"""MCP 服务端端到端自检：用真实 stdio Client 拉起服务端，逐项验收。

覆盖四类检查：
1. 握手与工具发现：initialize → tools/list，工具数量、名称、必填项、枚举、取值范围；
2. 正常调用：search_knowledge → get_chunk → list_sources，校验返回结构；
3. 参数校验：越界、枚举外取值、缺必填、chunk_id 格式错误，必须被服务端拒绝；
4. 凭据隔离：Client 进程的 Key 不进服务端子进程，工具输出里不出现任何敏感值。

运行：python -m mcp_server.selfcheck
退出码 0 表示全部通过，1 表示有失败项。
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from mcp_server import PROJECT_ROOT
from mcp_server.config import MAX_TOP_K as CONFIG_MAX_TOP_K
from mcp_server.knowledge import scrub

EXPECTED_TOOLS = {"search_knowledge", "get_chunk", "list_sources"}

# 只放行进程启动必需的环境变量。
# 这是「凭据不进 Client → Server 通道」的第一道闸门：OPENAI_API_KEY 不在其中。
ALLOWED_ENV = (
    "PATH", "PATHEXT", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC",
    "TEMP", "TMP", "APPDATA", "LOCALAPPDATA", "PROGRAMDATA",
    "USERPROFILE", "HOMEDRIVE", "HOMEPATH", "USERNAME",
    "PROCESSOR_ARCHITECTURE", "NUMBER_OF_PROCESSORS",
    "PYTHONIOENCODING", "PYTHONUTF8", "HF_ENDPOINT", "HF_HUB_DISABLE_XET",
)

SAMPLE_QUERY = "混合检索和重排分别解决什么问题"


class Report:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0

    def ok(self, msg: str) -> None:
        self.passed += 1
        print(f"  [PASS] {msg}")

    def fail(self, msg: str) -> None:
        self.failed += 1
        print(f"  [FAIL] {msg}")

    def check(self, cond: bool, msg: str, detail: str = "") -> bool:
        if cond:
            self.ok(msg)
        else:
            self.fail(f"{msg}{(' —— ' + detail) if detail else ''}")
        return cond


def _server_params() -> StdioServerParameters:
    env = {k: os.environ[k] for k in ALLOWED_ENV if k in os.environ}
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_server"],
        cwd=str(PROJECT_ROOT),
        env=env,
    )


async def _call(session: ClientSession, name: str, arguments: dict[str, Any]):
    """调工具，返回 (ok, payload)；参数校验失败时 SDK 走 ToolError → isError 结果。"""
    result = await session.call_tool(name, arguments)
    if result.is_error:
        text = "".join(getattr(c, "text", "") for c in result.content)
        return False, text
    if result.structured_content is not None:
        return True, result.structured_content
    text = "".join(getattr(c, "text", "") for c in result.content)
    try:
        return True, json.loads(text)
    except json.JSONDecodeError:
        return True, text


async def _expect_rejected(session: ClientSession, name: str,
                           arguments: dict[str, Any]) -> str:
    """非法参数必须被拒绝：MCP 规范里两种表现都合法（isError 结果 / JSON-RPC 错误）。"""
    try:
        ok, payload = await _call(session, name, arguments)
    except Exception as exc:  # JSON-RPC 层错误
        return f"JSON-RPC 拒绝：{type(exc).__name__}: {str(exc)[:90]}"
    if ok:
        return ""
    return str(payload)[:110].replace("\n", " ")


async def run() -> int:
    report = Report()
    outputs: list[str] = []

    params = _server_params()
    print(f"连接：{params.command} -m mcp_server  (cwd={params.cwd})")

    # 服务端 stderr 收到临时文件：通过时丢弃（含索引加载、分词器初始化等噪音），
    # 失败时回放，避免把真正的原因埋在日志里。
    errlog = tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace")

    async with stdio_client(params, errlog=errlog) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            info = init.server_info
            print(f"服务端：{info.name} v{info.version}"
                  f" | 协议版本 {init.protocol_version}\n")

            # ---------- 1. 工具发现 ----------
            print("[1/4] 工具发现（tools/list）")
            listed = await session.list_tools()
            names = {t.name for t in listed.tools}
            report.check(names == EXPECTED_TOOLS, f"工具集合 = {sorted(names)}",
                         f"期望 {sorted(EXPECTED_TOOLS)}")

            for tool in listed.tools:
                schema = tool.input_schema or {}
                props = schema.get("properties", {})
                required = schema.get("required", [])
                bits = []
                for pname, spec in props.items():
                    enum = f" enum={spec['enum']}" if "enum" in spec else ""
                    rng = ""
                    if "minimum" in spec or "maximum" in spec:
                        rng = f" 范围[{spec.get('minimum')},{spec.get('maximum')}]"
                    pat = " pattern" if "pattern" in spec else ""
                    bits.append(f"{pname}:{spec.get('type')}{enum}{rng}{pat}")
                print(f"  · {tool.name}  必填={required or '无'}")
                for bit in bits:
                    print(f"      {bit}")
                report.check(bool(tool.description and len(tool.description) > 80),
                             f"{tool.name} description 足够具体（{len(tool.description or '')} 字）")

            # 枚举/边界必须真的写进了 Schema
            sk = next((t for t in listed.tools if t.name == "search_knowledge"), None)
            props = (sk.input_schema or {}).get("properties", {}) if sk else {}
            report.check(props.get("mode", {}).get("enum") == ["hybrid", "bm25", "vector"],
                         "search_knowledge.mode 枚举已下发")
            report.check(props.get("top_k", {}).get("maximum") == CONFIG_MAX_TOP_K
                         and props.get("top_k", {}).get("minimum") == 1,
                         f"search_knowledge.top_k 取值范围已下发（1~{CONFIG_MAX_TOP_K}）")
            report.check((sk.input_schema or {}).get("required") == ["query"],
                         "search_knowledge 必填项只有 query")

            # ---------- 2. 正常调用 ----------
            print("\n[2/4] 正常调用")
            ok, search = await _call(session, "search_knowledge",
                                     {"query": SAMPLE_QUERY, "top_k": 3})
            outputs.append(json.dumps(search, ensure_ascii=False))
            report.check(ok, "search_knowledge 调用成功")
            if ok and isinstance(search, dict):
                results = search.get("results", [])
                report.check(len(results) == 3, f"返回 {len(results)} 条片段")
                first = results[0]
                need = {"chunk_id", "doc", "position", "score", "text"}
                report.check(need <= set(first), f"片段字段完整：{sorted(need)}")
                print(f"       Top1: {first['chunk_id']} (score={first['score']}, "
                      f"{first['score_kind']}) {first['text'][:40].replace(chr(10), ' ')}…")

                chunk_id = first["chunk_id"]
                ok2, chunk = await _call(session, "get_chunk", {"chunk_id": chunk_id})
                outputs.append(json.dumps(chunk, ensure_ascii=False))
                report.check(ok2 and isinstance(chunk, dict)
                             and chunk.get("chunk_id") == chunk_id,
                             f"get_chunk({chunk_id}) 命中")
                if ok2 and isinstance(chunk, dict):
                    report.check(chunk.get("text", "")[:30] in first["text"]
                                 or first["text"].startswith(chunk.get("text", "")[:30]),
                                 "get_chunk 内容与检索片段一致")
                    print(f"       相邻片段：prev={chunk['neighbors']['prev']} "
                          f"next={chunk['neighbors']['next']}")

            ok3, sources = await _call(session, "list_sources", {})
            outputs.append(json.dumps(sources, ensure_ascii=False))
            report.check(ok3 and isinstance(sources, dict) and sources.get("total_docs", 0) > 0,
                         f"list_sources 返回 {sources.get('total_docs') if ok3 else '?'} 个文档")
            if ok3 and isinstance(sources, dict):
                print(f"       文档：{', '.join(d['doc'] for d in sources['docs'])}")
                report.check(not any("/" in d["doc"] or "\\" in d["doc"]
                                     for d in sources["docs"]),
                             "文档清单不含磁盘路径")

            # ---------- 3. 参数校验 ----------
            print("\n[3/4] 非法参数必须被拒绝")
            cases = [
                ("search_knowledge", {"query": SAMPLE_QUERY, "top_k": 999}, "top_k 超上限"),
                ("search_knowledge", {"query": SAMPLE_QUERY, "top_k": 0}, "top_k 低于下限"),
                ("search_knowledge", {"query": SAMPLE_QUERY, "mode": "fuzzy"}, "mode 不在枚举内"),
                ("search_knowledge", {"query": "a"}, "query 过短"),
                ("search_knowledge", {}, "缺少必填 query"),
                ("get_chunk", {"chunk_id": "not a chunk id"}, "chunk_id 格式不符"),
                ("get_chunk", {"chunk_id": "rag-notes.md::9999"}, "chunk_id 不存在"),
            ]
            for name, args, note in cases:
                detail = await _expect_rejected(session, name, args)
                report.check(bool(detail), f"{note} → 已拒绝", "服务端接受了非法参数")
                if detail:
                    print(f"       {detail}")

            # ---------- 4. 凭据隔离 ----------
            print("\n[4/4] 凭据隔离")
            leaked_env = [k for k in params.env or {} if "KEY" in k.upper() or "TOKEN" in k.upper()]
            report.check(not leaked_env, "Client 传给服务端的 env 不含任何 KEY/TOKEN",
                         f"泄漏了 {leaked_env}")
            print(f"       放行环境变量：{', '.join(sorted(params.env or {}))}")

            blob = "\n".join(outputs)
            secrets = {k: v for k, v in os.environ.items()
                       if len(v) >= 8 and any(w in k.upper()
                                              for w in ("KEY", "TOKEN", "SECRET", "PASSWORD"))}
            hit = [k for k, v in secrets.items() if v in blob]
            report.check(not hit, f"工具输出不含敏感值（检查了 {len(secrets)} 个变量）",
                         f"命中 {hit}")
            os.environ["MCP_SELFCHECK_FAKE_KEY"] = "sk-selfcheck-1234567890"
            masked = scrub("token=sk-selfcheck-1234567890")
            del os.environ["MCP_SELFCHECK_FAKE_KEY"]
            report.check(masked == "token=***", "scrub() 能抹掉敏感值", masked)

    print(f"\n结果：{report.passed} 通过 / {report.failed} 失败")

    if report.failed:
        errlog.seek(0)
        tail = errlog.read().strip().splitlines()[-25:]
        if tail:
            print("\n--- 服务端 stderr（末尾 25 行）---")
            print("\n".join(tail))
    errlog.close()
    return 0 if report.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run()))
