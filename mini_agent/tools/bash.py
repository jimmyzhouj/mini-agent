"""Bash tool — execute shell commands."""

from __future__ import annotations

import asyncio
import os

from pydantic import BaseModel, Field

from mini_agent.tools.base import Tool
from mini_agent.types import ToolResult

MAX_OUTPUT_LINES = 2000
MAX_OUTPUT_BYTES = 50 * 1024  # 50 KB


class BashParams(BaseModel):
    command: str = Field(description="Bash command to execute")
    timeout: int | None = Field(default=None, description="Timeout in seconds")


class BashTool(Tool[BashParams]):
    name = "bash"
    description = "Execute a bash command in the working directory. Returns stdout and stderr."
    parameters = BashParams

    def __init__(self, working_dir: str | None = None) -> None:
        self.working_dir = working_dir or os.getcwd()

    async def execute(self, params: BashParams) -> ToolResult:
        try:
            proc = await asyncio.create_subprocess_shell(
                params.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.working_dir,
            )
        except Exception as e:
            return ToolResult(
                output=f"Failed to start process: {e}",
                is_error=True,
                details={"error": str(e)},
            )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=params.timeout,
            )
        except asyncio.TimeoutError:
            try:
                proc.kill()
                await proc.communicate()
            except Exception:
                pass
            return ToolResult(
                output=f"Command timed out after {params.timeout}s",
                is_error=True,
                details={"exit_code": -1, "stdout": "", "stderr": "", "truncated": False},
            )

        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")
        exit_code = proc.returncode

        stdout, truncated = _truncate(stdout)
        stderr, _ = _truncate(stderr)

        if exit_code == 0:
            output = f"stdout:\n{stdout}\nstderr:\n{stderr}"
        else:
            output = f"Exit code: {exit_code}\nstdout:\n{stdout}\nstderr:\n{stderr}"

        return ToolResult(
            output=output,
            is_error=False,  # non-zero exit code is not a system error; let LLM decide
            details={
                "exit_code": exit_code,
                "stdout": stdout,
                "stderr": stderr,
                "truncated": truncated,
            },
        )


def _truncate(text: str) -> tuple[str, bool]:
    """Truncate text to MAX_OUTPUT_LINES lines or MAX_OUTPUT_BYTES, whichever is smaller."""
    lines = text.splitlines(keepends=True)
    total = len(lines)

    # Enforce line limit
    if len(lines) > MAX_OUTPUT_LINES:
        lines = lines[-MAX_OUTPUT_LINES:]
        truncated = True
    else:
        truncated = False

    result = "".join(lines)

    # Enforce byte limit
    encoded = result.encode("utf-8")
    if len(encoded) > MAX_OUTPUT_BYTES:
        # Trim to MAX_OUTPUT_BYTES from the end
        result = encoded[-MAX_OUTPUT_BYTES:].decode("utf-8", errors="replace")
        truncated = True

    if truncated:
        result = f"[truncated: showing last {MAX_OUTPUT_LINES} of {total} lines]\n" + result

    return result, truncated
