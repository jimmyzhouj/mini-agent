"""
SessionManager — append-only JSONL tree for conversation persistence.

Each record has id + parent_id, forming a tree that supports branching.
The current active branch is determined by leaf_id.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class SessionEntry(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    parent_id: str | None = None
    type: Literal["user", "assistant", "tool_result", "meta"]
    data: dict
    timestamp: float = Field(default_factory=time.time)
    model: str | None = None


class SessionManager:
    """
    JSONL tree-based session management. Mirrors Pi's SessionManager.

    Key concepts:
    - Every entry has id and parent_id, forming a tree.
    - The current active branch is determined by leaf_id.
      Tracing from leaf to root gives the current conversation.
    - branch(entry_id) moves leaf to that entry; subsequent appends fork from there.
    - The file is append-only — crash at most loses one line.
    """

    def __init__(self, path: str | None = None) -> None:
        self._path: Path | None = Path(path) if path else None
        self._entries: dict[str, SessionEntry] = {}  # id → entry
        self._leaf_id: str | None = None

        if self._path and self._path.exists():
            self._load()

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @staticmethod
    def in_memory() -> SessionManager:
        """Create an in-memory session (lost on process exit)."""
        return SessionManager(path=None)

    @staticmethod
    def create(session_dir: str) -> SessionManager:
        """Create a new JSONL file in session_dir (filename = timestamp)."""
        d = Path(session_dir)
        d.mkdir(parents=True, exist_ok=True)
        filename = f"session_{int(time.time() * 1000)}.jsonl"
        return SessionManager(path=str(d / filename))

    @staticmethod
    def open(path: str) -> SessionManager:
        """Open an existing session file."""
        return SessionManager(path=path)

    @staticmethod
    def continue_recent(session_dir: str) -> SessionManager:
        """
        Open the most recently modified JSONL in session_dir.
        Creates a new session if none exist.
        """
        d = Path(session_dir)
        if d.exists():
            files = sorted(d.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
            if files:
                return SessionManager(path=str(files[0]))
        return SessionManager.create(session_dir)

    @staticmethod
    def list_sessions(session_dir: str) -> list[dict]:
        """List all session files in directory."""
        d = Path(session_dir)
        if not d.exists():
            return []
        result = []
        for p in sorted(d.glob("*.jsonl"), key=lambda f: f.stat().st_mtime, reverse=True):
            stat = p.stat()
            # Count valid lines
            try:
                lines = [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
            except Exception:
                lines = []
            result.append({
                "path": str(p),
                "modified_time": stat.st_mtime,
                "entry_count": len(lines),
            })
        return result

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def append(
        self,
        entry_type: str,
        data: dict,
        model: str | None = None,
    ) -> SessionEntry:
        """
        Append a new entry.
        - Auto-generates id
        - parent_id = current leaf_id
        - Writes to JSONL file if path is set
        - Updates leaf_id
        """
        entry = SessionEntry(
            parent_id=self._leaf_id,
            type=entry_type,  # type: ignore[arg-type]
            data=data,
            model=model,
        )
        self._entries[entry.id] = entry
        self._leaf_id = entry.id

        if self._path:
            self._write_entry(entry)

        return entry

    def branch(self, entry_id: str) -> None:
        """
        Fork. Set leaf_id to entry_id.
        Subsequent appends will use this entry as parent.
        No data is deleted (append-only).
        """
        if entry_id not in self._entries:
            raise ValueError(f"Entry {entry_id!r} not found")
        self._leaf_id = entry_id

    def build_context(self) -> list[dict]:
        """
        Trace from current leaf up to root, build ordered messages list.
        Returns Anthropic API format messages.
        """
        if not self._leaf_id:
            return []

        # Walk up the parent chain
        chain: list[SessionEntry] = []
        current_id: str | None = self._leaf_id
        while current_id:
            entry = self._entries.get(current_id)
            if not entry:
                break
            chain.append(entry)
            current_id = entry.parent_id

        chain.reverse()  # root → leaf

        messages: list[dict] = []
        for entry in chain:
            if entry.type == "meta":
                continue  # compaction metadata, skip

            msg = _entry_to_message(entry)
            if msg:
                messages.append(msg)

        return messages

    def get_leaf(self) -> SessionEntry | None:
        if not self._leaf_id:
            return None
        return self._entries.get(self._leaf_id)

    def get_entries(self) -> list[SessionEntry]:
        return list(self._entries.values())

    def get_tree(self) -> dict:
        """Return complete tree structure as nested dict."""
        # Build children map
        children: dict[str | None, list[str]] = {}
        for entry in self._entries.values():
            pid = entry.parent_id
            children.setdefault(pid, []).append(entry.id)

        def _node(eid: str) -> dict:
            e = self._entries[eid]
            data_preview = str(e.data)[:80]
            return {
                "id": eid,
                "type": e.type,
                "data_preview": data_preview,
                "children": [_node(c) for c in children.get(eid, [])],
            }

        roots = children.get(None, [])
        return {"roots": [_node(r) for r in roots]}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load entries from JSONL file. Skips malformed lines (crash recovery)."""
        assert self._path is not None
        try:
            text = self._path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return

        last_id: str | None = None
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                entry = SessionEntry.model_validate(raw)
                self._entries[entry.id] = entry
                last_id = entry.id
            except Exception:
                # Skip malformed line (crash recovery)
                continue

        self._leaf_id = last_id

    def _write_entry(self, entry: SessionEntry) -> None:
        """Append one entry to the JSONL file."""
        assert self._path is not None
        self._path.parent.mkdir(parents=True, exist_ok=True)
        line = entry.model_dump_json() + "\n"
        with self._path.open("a", encoding="utf-8") as f:
            f.write(line)


# ---------------------------------------------------------------------------
# Entry → Anthropic message conversion
# ---------------------------------------------------------------------------

def _entry_to_message(entry: SessionEntry) -> dict | None:
    """Convert a SessionEntry to an Anthropic API message dict."""
    if entry.type == "user":
        content = entry.data.get("content", "")
        return {"role": "user", "content": content}

    elif entry.type == "assistant":
        content = entry.data.get("content", [])
        return {"role": "assistant", "content": content}

    elif entry.type == "tool_result":
        # tool_results are user-role messages with tool_result blocks
        results = entry.data.get("results", [])
        if not results:
            return None
        return {"role": "user", "content": results}

    return None
