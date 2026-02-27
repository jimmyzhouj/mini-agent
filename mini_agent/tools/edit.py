"""Edit tool — surgical text replacement in files."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field

from mini_agent.tools.base import Tool
from mini_agent.types import ToolResult


class EditParams(BaseModel):
    path: str = Field(description="Path to the file to edit (relative or absolute)")
    old_text: str = Field(description="Exact text to find and replace (must match exactly)")
    new_text: str = Field(description="New text to replace the old text with")


class EditTool(Tool[EditParams]):
    name = "edit"
    description = (
        "Edit a file by replacing exact text. The old_text must match exactly "
        "(including whitespace). Use this for precise, surgical edits."
    )
    parameters = EditParams

    def __init__(self, working_dir: str | None = None) -> None:
        self.working_dir = working_dir or os.getcwd()

    async def execute(self, params: EditParams) -> ToolResult:
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

        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return ToolResult(
                output=f"Failed to read file: {e}",
                is_error=True,
                details={"path": str(path), "error": str(e)},
            )

        count = content.count(params.old_text)
        if count == 0:
            return ToolResult(
                output=f"old_text not found in {path}",
                is_error=True,
                details={"path": str(path)},
            )
        if count > 1:
            return ToolResult(
                output=f"old_text found {count} times in {path}, must be unique",
                is_error=True,
                details={"path": str(path), "occurrences": count},
            )

        new_content = content.replace(params.old_text, params.new_text, 1)

        try:
            path.write_text(new_content, encoding="utf-8")
        except Exception as e:
            return ToolResult(
                output=f"Failed to write file: {e}",
                is_error=True,
                details={"path": str(path), "error": str(e)},
            )

        # Build a brief diff summary for output
        old_preview = params.old_text[:200].replace("\n", "↵")
        new_preview = params.new_text[:200].replace("\n", "↵")
        output = f"Edited {path}:\n- {old_preview}\n+ {new_preview}"

        return ToolResult(
            output=output,
            details={
                "path": str(path),
                "old_text": params.old_text,
                "new_text": params.new_text,
            },
        )
