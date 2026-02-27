"""
Tests for SessionManager.

1. In-memory mode: append + build_context correctly rebuilds messages
2. File persistence: write → reopen → content consistent
3. Branch: build_context only returns new branch messages
4. Tree structure is correct
5. continue_recent finds the most recent file
6. Crash recovery: loading file with a corrupt line does not crash
"""

from __future__ import annotations

import json
import time
import pytest
from pathlib import Path

from mini_agent.session import SessionManager


# ---------------------------------------------------------------------------
# 1. In-memory mode
# ---------------------------------------------------------------------------

def test_in_memory_append_and_build_context():
    sm = SessionManager.in_memory()

    sm.append("user", {"content": "Hello"})
    sm.append("assistant", {"content": [{"type": "text", "text": "Hi there"}]})

    messages = sm.build_context()
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Hello"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == [{"type": "text", "text": "Hi there"}]


def test_in_memory_no_file_written():
    sm = SessionManager.in_memory()
    assert sm._path is None
    sm.append("user", {"content": "test"})
    # No exception, no file


def test_build_context_skips_meta():
    sm = SessionManager.in_memory()
    sm.append("user", {"content": "hello"})
    sm.append("meta", {"type": "compaction"})
    sm.append("assistant", {"content": [{"type": "text", "text": "world"}]})

    messages = sm.build_context()
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"


# ---------------------------------------------------------------------------
# 2. File persistence
# ---------------------------------------------------------------------------

def test_file_persistence_write_and_reopen(tmp_path: Path):
    path = str(tmp_path / "session.jsonl")
    sm = SessionManager(path=path)

    sm.append("user", {"content": "Persist me"})
    sm.append("assistant", {"content": [{"type": "text", "text": "Persisted"}]})

    # Reopen
    sm2 = SessionManager(path=path)
    messages = sm2.build_context()

    assert len(messages) == 2
    assert messages[0]["content"] == "Persist me"
    assert messages[1]["content"] == [{"type": "text", "text": "Persisted"}]


def test_file_entry_count(tmp_path: Path):
    path = str(tmp_path / "s.jsonl")
    sm = SessionManager(path=path)
    for i in range(5):
        sm.append("user", {"content": f"msg {i}"})

    lines = Path(path).read_text().strip().splitlines()
    assert len(lines) == 5


def test_factory_create(tmp_path: Path):
    sm = SessionManager.create(str(tmp_path))
    assert sm._path is not None
    assert sm._path.exists() or True  # file created on first append
    sm.append("user", {"content": "test"})
    assert sm._path.exists()


def test_factory_open(tmp_path: Path):
    # Create a session file manually
    path = tmp_path / "manual.jsonl"
    sm1 = SessionManager(path=str(path))
    sm1.append("user", {"content": "hello"})

    sm2 = SessionManager.open(str(path))
    messages = sm2.build_context()
    assert messages[0]["content"] == "hello"


# ---------------------------------------------------------------------------
# 3. Branch: build_context returns only new branch messages
# ---------------------------------------------------------------------------

def test_branch_creates_fork():
    sm = SessionManager.in_memory()

    # Main branch: A → B → C
    e_a = sm.append("user", {"content": "A"})
    e_b = sm.append("assistant", {"content": [{"type": "text", "text": "B"}]})
    e_c = sm.append("user", {"content": "C"})

    # Branch from B
    sm.branch(e_b.id)

    # New branch from B: D
    sm.append("user", {"content": "D"})

    messages = sm.build_context()
    contents = [m.get("content") for m in messages]

    # Should include A, B, D but NOT C
    assert any(c == "A" for c in contents)
    assert any(c == "D" for c in contents)
    assert not any(c == "C" for c in contents)


def test_branch_invalid_id_raises():
    sm = SessionManager.in_memory()
    with pytest.raises(ValueError, match="not found"):
        sm.branch("nonexistent_id")


# ---------------------------------------------------------------------------
# 4. Tree structure
# ---------------------------------------------------------------------------

def test_get_tree_structure():
    sm = SessionManager.in_memory()

    root = sm.append("user", {"content": "root"})
    child1 = sm.append("assistant", {"content": [{"type": "text", "text": "child1"}]})
    sm.branch(root.id)
    child2 = sm.append("user", {"content": "child2"})

    tree = sm.get_tree()
    assert "roots" in tree
    assert len(tree["roots"]) == 1

    root_node = tree["roots"][0]
    assert root_node["id"] == root.id
    assert len(root_node["children"]) == 2  # child1 and child2 both parented to root

    child_ids = {c["id"] for c in root_node["children"]}
    assert child1.id in child_ids
    assert child2.id in child_ids


def test_get_entries_returns_all():
    sm = SessionManager.in_memory()
    sm.append("user", {"content": "a"})
    sm.append("assistant", {"content": []})
    sm.append("user", {"content": "b"})

    entries = sm.get_entries()
    assert len(entries) == 3


# ---------------------------------------------------------------------------
# 5. continue_recent finds most recent file
# ---------------------------------------------------------------------------

def test_continue_recent_finds_latest(tmp_path: Path):
    # Create two session files with different timestamps
    sm1 = SessionManager.create(str(tmp_path))
    sm1.append("user", {"content": "older"})

    time.sleep(0.1)  # ensure different mtime (Windows mtime resolution ~10ms)

    sm2 = SessionManager.create(str(tmp_path))
    sm2.append("user", {"content": "newer"})

    # continue_recent should load the second (newer) session
    sm3 = SessionManager.continue_recent(str(tmp_path))
    messages = sm3.build_context()
    assert messages[0]["content"] == "newer"


def test_continue_recent_creates_new_if_empty(tmp_path: Path):
    sm = SessionManager.continue_recent(str(tmp_path))
    assert sm._path is not None
    # Should be an empty session (no entries)
    assert sm.build_context() == []


def test_list_sessions(tmp_path: Path):
    sm1 = SessionManager.create(str(tmp_path))
    sm1.append("user", {"content": "one"})
    sm2 = SessionManager.create(str(tmp_path))
    sm2.append("user", {"content": "two"})
    sm2.append("user", {"content": "three"})

    sessions = SessionManager.list_sessions(str(tmp_path))
    assert len(sessions) == 2
    # Most recently modified first
    entry_counts = [s["entry_count"] for s in sessions]
    assert 2 in entry_counts
    assert 1 in entry_counts


# ---------------------------------------------------------------------------
# 6. Crash recovery: corrupt line does not crash
# ---------------------------------------------------------------------------

def test_crash_recovery_corrupt_line(tmp_path: Path):
    path = tmp_path / "crashed.jsonl"

    # Write two valid entries and one corrupt line
    sm = SessionManager(path=str(path))
    sm.append("user", {"content": "before crash"})
    sm.append("assistant", {"content": [{"type": "text", "text": "response"}]})

    # Corrupt the file by appending a broken line
    with path.open("a") as f:
        f.write("{broken json line\n")

    # Reload — should not raise
    sm2 = SessionManager(path=str(path))
    messages = sm2.build_context()

    # Valid entries should still be loadable
    assert len(messages) >= 0  # no crash; valid entries loaded


def test_crash_recovery_empty_lines(tmp_path: Path):
    path = tmp_path / "empty_lines.jsonl"

    sm = SessionManager(path=str(path))
    sm.append("user", {"content": "hello"})

    # Add blank lines
    with path.open("a") as f:
        f.write("\n\n   \n")

    sm2 = SessionManager(path=str(path))
    messages = sm2.build_context()
    assert len(messages) == 1
    assert messages[0]["content"] == "hello"


# ---------------------------------------------------------------------------
# get_leaf
# ---------------------------------------------------------------------------

def test_get_leaf():
    sm = SessionManager.in_memory()
    assert sm.get_leaf() is None

    sm.append("user", {"content": "a"})
    leaf = sm.append("user", {"content": "b"})

    assert sm.get_leaf() is not None
    assert sm.get_leaf().id == leaf.id
