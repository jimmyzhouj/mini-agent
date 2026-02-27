"""
Tests for tool execution logic. Uses real file system (tmp_path).

bash: normal, non-zero exit, timeout, truncation
read: text file, missing file, offset/limit, large file
write: new file, overwrite, auto-create parent dirs
edit: normal replace, old_text missing, old_text non-unique
"""

from __future__ import annotations

import pytest
from pathlib import Path

from mini_agent.tools.bash import BashTool, BashParams
from mini_agent.tools.read import ReadTool, ReadParams, DEFAULT_READ_LINES
from mini_agent.tools.write import WriteTool, WriteParams
from mini_agent.tools.edit import EditTool, EditParams


# ---------------------------------------------------------------------------
# bash
# ---------------------------------------------------------------------------

class TestBashTool:
    @pytest.fixture
    def bash(self, tmp_path: Path) -> BashTool:
        return BashTool(working_dir=str(tmp_path))

    async def test_normal_command(self, bash: BashTool):
        result = await bash.execute(BashParams(command='python -c "print(\'hello\')"'))
        assert result.is_error is False
        assert "hello" in result.output
        assert result.details["exit_code"] == 0

    async def test_nonzero_exit_code(self, bash: BashTool):
        result = await bash.execute(BashParams(
            command='python -c "import sys; sys.exit(42)"'
        ))
        assert result.is_error is False  # non-zero exit is NOT a system error
        assert "Exit code: 42" in result.output
        assert result.details["exit_code"] == 42

    async def test_timeout(self, bash: BashTool):
        result = await bash.execute(BashParams(
            command='python -c "import time; time.sleep(10)"',
            timeout=1,
        ))
        assert result.is_error is True
        assert "timed out" in result.output.lower()

    async def test_output_truncation(self, bash: BashTool):
        # Generate > 2000 lines of output
        result = await bash.execute(BashParams(
            command='python -c "print(\'\\n\'.join(str(i) for i in range(3000)))"',
        ))
        assert result.is_error is False
        assert result.details["truncated"] is True
        assert "truncated" in result.output

    async def test_stderr_captured(self, bash: BashTool):
        result = await bash.execute(BashParams(
            command='python -c "import sys; sys.stderr.write(\'err msg\\n\')"'
        ))
        assert "err msg" in result.output

    async def test_working_dir_used(self, bash: BashTool, tmp_path: Path):
        (tmp_path / "sentinel.txt").write_text("found")
        result = await bash.execute(BashParams(
            command='python -c "import os; print(os.listdir(\'.\'))"'
        ))
        assert "sentinel.txt" in result.output


# ---------------------------------------------------------------------------
# read
# ---------------------------------------------------------------------------

class TestReadTool:
    @pytest.fixture
    def read(self, tmp_path: Path) -> ReadTool:
        return ReadTool(working_dir=str(tmp_path))

    @pytest.fixture
    def sample_file(self, tmp_path: Path) -> Path:
        p = tmp_path / "sample.txt"
        lines = [f"line {i}\n" for i in range(1, 21)]  # 20 lines
        p.write_text("".join(lines))
        return p

    async def test_read_text_file(self, read: ReadTool, sample_file: Path):
        result = await read.execute(ReadParams(path=str(sample_file)))
        assert result.is_error is False
        assert "File:" in result.output
        assert "20 lines" in result.output
        assert "1: line 1" in result.output
        assert "20: line 20" in result.output

    async def test_file_not_found(self, read: ReadTool, tmp_path: Path):
        result = await read.execute(ReadParams(path=str(tmp_path / "nope.txt")))
        assert result.is_error is True
        assert "not found" in result.output.lower()

    async def test_offset_and_limit(self, read: ReadTool, sample_file: Path):
        result = await read.execute(ReadParams(
            path=str(sample_file),
            offset=5,
            limit=3,
        ))
        assert result.is_error is False
        # offset=5 (1-indexed) → lines 5, 6, 7
        assert "5: line 5" in result.output
        assert "6: line 6" in result.output
        assert "7: line 7" in result.output
        assert "1: line 1" not in result.output
        assert "8: line 8" not in result.output

    async def test_default_limit_applied(self, read: ReadTool, tmp_path: Path):
        big = tmp_path / "big.txt"
        big.write_text("".join(f"x{i}\n" for i in range(DEFAULT_READ_LINES + 100)))
        result = await read.execute(ReadParams(path=str(big)))
        assert result.is_error is False
        assert result.details["shown_lines"] == DEFAULT_READ_LINES

    async def test_relative_path(self, read: ReadTool, tmp_path: Path):
        (tmp_path / "rel.txt").write_text("relative content")
        result = await read.execute(ReadParams(path="rel.txt"))
        assert result.is_error is False
        assert "relative content" in result.output

    async def test_large_file_byte_truncation(self, read: ReadTool, tmp_path: Path):
        # 100 lines × 1020 bytes each = ~100KB, exceeds 50KB limit
        big = tmp_path / "fat.txt"
        big.write_text("".join("x" * 1020 + "\n" for _ in range(100)))
        result = await read.execute(ReadParams(path=str(big)))
        assert result.is_error is False
        assert result.details["truncated"] is True


# ---------------------------------------------------------------------------
# write
# ---------------------------------------------------------------------------

class TestWriteTool:
    @pytest.fixture
    def write(self, tmp_path: Path) -> WriteTool:
        return WriteTool(working_dir=str(tmp_path))

    async def test_write_new_file(self, write: WriteTool, tmp_path: Path):
        path = str(tmp_path / "new.txt")
        result = await write.execute(WriteParams(path=path, content="hello"))
        assert result.is_error is False
        assert result.details["created"] is True
        assert Path(path).read_text() == "hello"

    async def test_overwrite_existing_file(self, write: WriteTool, tmp_path: Path):
        path = tmp_path / "existing.txt"
        path.write_text("old content")
        result = await write.execute(WriteParams(path=str(path), content="new content"))
        assert result.is_error is False
        assert result.details["created"] is False
        assert path.read_text() == "new content"

    async def test_auto_create_parent_dirs(self, write: WriteTool, tmp_path: Path):
        path = str(tmp_path / "a" / "b" / "c" / "deep.txt")
        result = await write.execute(WriteParams(path=path, content="deep"))
        assert result.is_error is False
        assert Path(path).read_text() == "deep"

    async def test_bytes_written_reported(self, write: WriteTool, tmp_path: Path):
        content = "abc"
        result = await write.execute(WriteParams(
            path=str(tmp_path / "b.txt"),
            content=content,
        ))
        assert result.details["bytes_written"] == len(content.encode("utf-8"))


# ---------------------------------------------------------------------------
# edit
# ---------------------------------------------------------------------------

class TestEditTool:
    @pytest.fixture
    def edit(self, tmp_path: Path) -> EditTool:
        return EditTool(working_dir=str(tmp_path))

    @pytest.fixture
    def sample_file(self, tmp_path: Path) -> Path:
        p = tmp_path / "code.py"
        p.write_text("def hello():\n    return 'world'\n")
        return p

    async def test_normal_replace(self, edit: EditTool, sample_file: Path):
        result = await edit.execute(EditParams(
            path=str(sample_file),
            old_text="return 'world'",
            new_text="return 'earth'",
        ))
        assert result.is_error is False
        assert "earth" in sample_file.read_text()
        assert "world" not in sample_file.read_text()

    async def test_old_text_not_found(self, edit: EditTool, sample_file: Path):
        result = await edit.execute(EditParams(
            path=str(sample_file),
            old_text="this does not exist",
            new_text="replacement",
        ))
        assert result.is_error is True
        assert "not found" in result.output.lower()

    async def test_old_text_multiple_matches(self, edit: EditTool, tmp_path: Path):
        p = tmp_path / "dup.txt"
        p.write_text("foo\nfoo\nbar\n")
        result = await edit.execute(EditParams(
            path=str(p),
            old_text="foo",
            new_text="baz",
        ))
        assert result.is_error is True
        assert "2" in result.output  # found 2 times

    async def test_file_not_found(self, edit: EditTool, tmp_path: Path):
        result = await edit.execute(EditParams(
            path=str(tmp_path / "ghost.py"),
            old_text="x",
            new_text="y",
        ))
        assert result.is_error is True
        assert "not found" in result.output.lower()

    async def test_output_shows_diff_summary(self, edit: EditTool, sample_file: Path):
        result = await edit.execute(EditParams(
            path=str(sample_file),
            old_text="return 'world'",
            new_text="return 'universe'",
        ))
        assert result.is_error is False
        # Output should contain both old and new text (possibly with ↵ for newlines)
        assert "return" in result.output
        assert "Edited" in result.output
