"""Write tool — write content to a file."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field

from mini_agent.tools.base import Tool
from mini_agent.types import ToolResult


class WriteParams(BaseModel):
    path: str = Field(description="Path to the file to write (relative or absolute)")
    content: str = Field(description="Content to write to the file")


class WriteTool(Tool[WriteParams]):
    name = "write"
    description = (
        "Write content to a file. Creates the file if it doesn't exist, "
        "overwrites if it does. Automatically creates parent directories."
    )
    parameters = WriteParams

    def __init__(self, working_dir: str | None = None) -> None:
        self.working_dir = working_dir or os.getcwd()

    async def execute(self, params: WriteParams) -> ToolResult:
        path = Path(params.path)
        if not path.is_absolute():
            path = Path(self.working_dir) / path
        path = path.resolve()

        created = not path.exists()

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            raw = params.content.encode("utf-8")
            path.write_bytes(raw)
        except Exception as e:
            return ToolResult(
                output=f"Failed to write file: {e}",
                is_error=True,
                details={"path": str(path), "error": str(e)},
            )

        n = len(raw)
        return ToolResult(
            output=f"Wrote {n} bytes to {path}",
            details={"path": str(path), "bytes_written": n, "created": created},
        )
