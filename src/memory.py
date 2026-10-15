"""多轮会话记忆：短期滑动窗口 + 超阈值摘要压缩 + 会话持久化

分层策略（与 RAG 里的「短期/长期记忆」是同一套思路）：
- **短期**：最近 N 轮原文进上下文。多轮问答的答案正确性高度依赖最近几轮，原文最可靠。
- **中期**：更早的轮次压成一段摘要（LLM 生成），避免上下文随对话长度线性膨胀。
- **长期**：整段对话落盘（JSON），进程重启后可恢复会话。

必须处理的两个工程风险：
1. **上下文无限膨胀**——不做压缩的话，几十轮之后必然撑爆窗口或被静默截断；
2. **服务端内存被会话吃满**——要有会话数上限与 TTL 淘汰，否则长时间运行就是一次 OOM。

压缩失败时降级为「直接丢弃较早轮次」而不是报错：少记几轮历史不会让问答失败，但抛异常会。
"""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from . import config, llm

logger = logging.getLogger(__name__)

SESSIONS_DIR = config.PROJECT_ROOT / "data" / "sessions"

_SUMMARY_PROMPT = """请把下面的对话压缩成一段不超过 300 字的摘要。

要求：保留用户问过的话题、已确认的关键结论与术语；不要保留客套话；用中文陈述句。

待压缩对话：
{turns}"""

_APPEND_SUMMARY_PROMPT = """已有摘要：
{summary}

新增对话：
{turns}

请把以上内容合并成一段不超过 300 字的中文摘要，保留话题、已确认结论与关键术语。"""


@dataclass
class Turn:
    """一轮对话（一问或一答）"""

    role: str                      # user / assistant
    content: str
    ts: float = field(default_factory=time.time)
    citations: list[dict] | None = None

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content, "ts": self.ts,
                "citations": self.citations}

    @classmethod
    def from_dict(cls, data: dict) -> "Turn":
        return cls(role=data.get("role", "user"), content=data.get("content", ""),
                   ts=data.get("ts", time.time()), citations=data.get("citations"))


class ConversationMemory:
    """单个会话的记忆"""

    def __init__(self, session_id: str | None = None, turns: list[Turn] | None = None,
                 summary: str = ""):
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.turns: list[Turn] = turns or []
        self.summary = summary
        self.updated_at = time.time()

    # ---------- 写入 ----------

    def add(self, role: str, content: str, citations: list[dict] | None = None) -> None:
        self.turns.append(Turn(role=role, content=content, citations=citations))
        self.updated_at = time.time()
        # 兜底：即便压缩失败，轮数也不能无限增长
        if len(self.turns) > config.MEMORY_MAX_TURNS:
            dropped = len(self.turns) - config.MEMORY_MAX_TURNS
            self.turns = self.turns[-config.MEMORY_MAX_TURNS:]
            logger.warning("会话 %s 超出最大轮数，丢弃最早的 %d 轮",
                           self.session_id, dropped)

    # ---------- 读取 ----------

    @property
    def chars(self) -> int:
        return sum(len(t.content) for t in self.turns)

    def recent(self, turns: int | None = None) -> list[Turn]:
        """最近 N 轮（默认取配置值）"""
        n = turns or config.MEMORY_TURNS
        return self.turns[-n:] if n > 0 else []

    def history_dicts(self, turns: int | None = None) -> list[dict]:
        """给改写器用的历史（字典形式，不含当前问题）"""
        return [{"role": t.role, "content": t.content} for t in self.recent(turns)]

    def messages(self, turns: int | None = None) -> list[dict]:
        """给 LLM 用的消息列表：摘要 + 最近 N 轮原文"""
        messages: list[dict] = []
        if self.summary:
            messages.append({"role": "system",
                             "content": f"以下是更早对话的摘要，供你理解上下文：\n{self.summary}"})
        for turn in self.recent(turns):
            messages.append({"role": turn.role, "content": turn.content})
        return messages

    # ---------- 压缩 ----------

    def should_compress(self) -> bool:
        """是否需要压缩

        两个触发条件任一满足即可：
        1. 历史字符数超阈值 —— 再不压，上下文就要撑爆 LLM 窗口；
        2. 轮数逼近硬上限 —— add() 的兜底裁剪马上生效，必须先摘要再丢。
           少了这条，短答案的长对话会永远走不到摘要，直接被静默截断。

        反过来，短期窗口还没填满时不压：压了也只是把 2 轮原文换成 2 轮摘要，没有收益。
        """
        if len(self.turns) <= config.MEMORY_TURNS:
            return False
        return (self.chars > config.MEMORY_SUMMARY_CHARS
                or len(self.turns) >= config.MEMORY_MAX_TURNS - config.MEMORY_TURNS)

    def compress(self) -> dict:
        """把较早轮次压成摘要；LLM 失败则降级为直接丢弃，不抛异常

        返回 {"compressed": 是否压缩, "backend": llm/drop, "reason": 失败原因}
        """
        if not self.should_compress():
            return {"compressed": False, "backend": "none"}

        keep = config.MEMORY_TURNS
        old_turns, self.turns = self.turns[:-keep], self.turns[-keep:]
        text = "\n".join(f"{'用户' if t.role == 'user' else '助手'}：{t.content[:400]}"
                         for t in old_turns)

        try:
            prompt = (_APPEND_SUMMARY_PROMPT.format(summary=self.summary, turns=text)
                      if self.summary else _SUMMARY_PROMPT.format(turns=text))
            self.summary = llm.chat([{"role": "user", "content": prompt}],
                                    temperature=0.0, max_tokens=600,
                                    timeout=min(45.0, config.LLM_TIMEOUT)).strip()
            backend = "llm"
            reason = None
        except Exception as e:  # noqa: BLE001 —— 摘要失败不能让整轮问答失败
            backend = "drop"
            reason = f"{type(e).__name__}: {e}"
            self.summary = (f"{self.summary} （较早对话已省略）" if self.summary
                            else "（较早对话已省略）")
            logger.warning("会话 %s 摘要压缩失败，降级为丢弃较早轮次：%s",
                           self.session_id, reason)

        info = {"compressed": True, "backend": backend,
                "dropped_turns": len(old_turns)}
        if reason:
            info["reason"] = reason
        return info

    # ---------- 持久化 ----------

    def to_dict(self) -> dict:
        return {"session_id": self.session_id, "summary": self.summary,
                "updated_at": self.updated_at,
                "turns": [t.to_dict() for t in self.turns]}

    @classmethod
    def from_dict(cls, data: dict) -> "ConversationMemory":
        return cls(session_id=data.get("session_id"),
                   turns=[Turn.from_dict(t) for t in data.get("turns", [])],
                   summary=data.get("summary", ""))


class SessionStore:
    """会话仓库：内存缓存 + 落盘 + TTL/数量淘汰

    FastAPI 的同步接口跑在线程池里，因此所有共享状态用锁保护。
    """

    def __init__(self, directory: Path | None = None,
                 ttl_minutes: int | None = None,
                 max_active: int | None = None):
        self.directory = directory or SESSIONS_DIR
        self.ttl_minutes = ttl_minutes if ttl_minutes is not None else config.SESSION_TTL_MINUTES
        self.max_active = max_active if max_active is not None else config.SESSION_MAX_ACTIVE
        self._sessions: dict[str, ConversationMemory] = {}
        self._lock = threading.Lock()

    def _path(self, session_id: str) -> Path:
        # 会话 id 由服务端生成（uuid 十六进制），这里再做一次白名单校验，避免路径穿越
        if not session_id or not all(c.isalnum() for c in session_id):
            raise ValueError(f"非法会话 id：{session_id!r}")
        return self.directory / f"{session_id}.json"

    def get(self, session_id: str | None = None) -> ConversationMemory:
        """取会话；不存在时优先从磁盘恢复，都没有则新建"""
        with self._lock:
            if session_id and session_id in self._sessions:
                memory = self._sessions[session_id]
                memory.updated_at = time.time()
                return memory

            if session_id:
                path = self._path(session_id)
                if path.exists():
                    try:
                        memory = ConversationMemory.from_dict(
                            json.loads(path.read_text(encoding="utf-8")))
                        self._sessions[memory.session_id] = memory
                        self._prune_locked()
                        return memory
                    except (json.JSONDecodeError, OSError) as e:
                        logger.warning("会话 %s 读取失败，按新会话处理：%s", session_id, e)

            memory = ConversationMemory(session_id=session_id)
            self._sessions[memory.session_id] = memory
            self._prune_locked()
            return memory

    def save(self, memory: ConversationMemory) -> None:
        with self._lock:
            self._sessions[memory.session_id] = memory
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            self._path(memory.session_id).write_text(
                json.dumps(memory.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as e:  # 落盘失败不影响当前会话继续
            logger.warning("会话 %s 落盘失败：%s", memory.session_id, e)

    def delete(self, session_id: str) -> bool:
        with self._lock:
            existed = self._sessions.pop(session_id, None) is not None
        try:
            path = self._path(session_id)
            if path.exists():
                path.unlink()
                existed = True
        except (OSError, ValueError) as e:
            logger.warning("删除会话 %s 失败：%s", session_id, e)
        return existed

    def stats(self) -> dict:
        with self._lock:
            return {"active": len(self._sessions),
                    "turns": sum(len(m.turns) for m in self._sessions.values()),
                    "ttl_minutes": self.ttl_minutes,
                    "max_active": self.max_active}

    def _prune_locked(self) -> None:
        """淘汰过期与超量的会话（调用方需持有锁）"""
        if self.ttl_minutes > 0:
            deadline = time.time() - self.ttl_minutes * 60
            expired = [sid for sid, m in self._sessions.items() if m.updated_at < deadline]
            for sid in expired:
                self._sessions.pop(sid, None)
            if expired:
                logger.info("清理过期会话 %d 个", len(expired))

        if self.max_active > 0 and len(self._sessions) > self.max_active:
            ordered = sorted(self._sessions.items(), key=lambda kv: kv[1].updated_at)
            for sid, _ in ordered[:len(self._sessions) - self.max_active]:
                self._sessions.pop(sid, None)
