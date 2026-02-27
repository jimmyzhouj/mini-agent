"""Read tool — read file contents."""

from __future__ import annotations

import base64
import os
from pathlib import Path

from pydantic import BaseModel, Field

from mini_agent.tools.base import Tool
from mini_agent.types import ToolResult

DEFAULT_READ_LINES = 2000
MAX_OUTPUT_BYTES = 50 * 1024  # 50 KB

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".gif", ".webp"}


class ReadParams(BaseModel):
    path: str = Field(description="Path to the file to read (relative or absolute)")
    offset: int | None = Field(default=None, description="Line number to start reading from (1-indexed)")
    limit: int | None = Field(default=None, description="Maximum number of lines to read")


class ReadTool(Tool[ReadParams]):
    name = "read"
    description = (
        "Read the contents of a file. For text files, defaults to first 2000 lines. "
        "Use offset/limit for large files."
    )
    parameters = ReadParams

    def __init__(self, working_dir: str | None = None) -> None:
        self.working_dir = working_dir or os.getcwd()

    async def execute(self, params: ReadParams) -> ToolResult:
        path = Path(params.path)
        if not path.is_absolute():
            path = Path(self.working_dir) / path
        path = path.resolve()

        if not path.exists():
            return ToolResult(
                output=f"File not found: {path}",
                is_error=True,
                details={"path": str(path)},
            )

        if not path.is_file():
            return ToolResult(
                output=f"Not a file: {path}",
                is_error=True,
                details={"path": str(path)},
            )

        # Image handling
        if path.suffix.lower() in IMAGE_SUFFIXES:
            raw = path.read_bytes()
            data = base64.b64encode(raw).decode()
            mime = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".gif": "image/gif",
                ".webp": "image/webp",
            }.get(path.suffix.lower(), "image/png")
            return ToolResult(
                output=f"Image file: {path} ({len(raw)} bytes, {mime})",
                details={"path": str(path), "type": "image", "mime_type": mime, "data": data},
            )

        # Text file
        raw_bytes = path.read_bytes()
        text = raw_bytes.decode("utf-8", errors="replace")
        all_lines = text.splitlines(keepends=True)
        total_lines = len(all_lines)

        offset = (params.offset or 1) - 1  # convert to 0-indexed
        offset = max(0, offset)
        limit = params.limit or DEFAULT_READ_LINES

        selected = all_lines[offset: offset + limit]
        shown_lines = len(selected)

        content = "".join(selected)

        # Enforce byte limit
        truncated = shown_lines < (total_lines - offset)
        encoded = content.encode("utf-8")
        if len(encoded) > MAX_OUTPUT_BYTES:
            content = encoded[:MAX_OUTPUT_BYTES].decode("utf-8", errors="replace")
            truncated = True

        # Add line numbers
        numbered_lines = []
        for i, line in enumerate(content.splitlines(keepends=True), start=offset + 1):
            numbered_lines.append(f"{i}: {line}")
        numbered = "".join(numbered_lines)

        header = f"File: {path} ({total_lines} lines)\n"
        output = header + numbered

        return ToolResult(
            output=output,
            details={
                "path": str(path),
                "total_lines": total_lines,
                "shown_lines": shown_lines,
                "truncated": truncated,
            },
        )
