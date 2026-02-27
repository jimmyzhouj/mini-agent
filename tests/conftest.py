"""Shared pytest fixtures."""

from __future__ import annotations

import pytest
from pathlib import Path
from pydantic import BaseModel, Field

from mini_agent.tools.base import Tool, ToolRegistry
from mini_agent.types import AssistantMessage, AgentEvent, EventType, ToolResult


# ---------------------------------------------------------------------------
# Simple mock tool for loop tests
# ---------------------------------------------------------------------------

class EchoParams(BaseModel):
    message: str = Field(description="Message to echo back")


class EchoTool(Tool[EchoParams]):
    """Echoes the message param. Used in loop tests."""
    name = "echo"
    description = "Echo a message"
    parameters = EchoParams

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def execute(self, params: EchoParams) -> ToolResult:
        self.calls.append(params.message)
        return ToolResult(output=f"Echo: {params.message}", details={"message": params.message})


class FailingTool(Tool[EchoParams]):
    """Always raises an exception. Used to test tool execution errors."""
    name = "fail"
    description = "Always fails"
    parameters = EchoParams

    async def execute(self, params: EchoParams) -> ToolResult:
        raise RuntimeError("intentional failure")


# ---------------------------------------------------------------------------
# Mock provider
# ---------------------------------------------------------------------------

class MockProvider:
    """
    Fake provider that returns pre-canned AssistantMessage objects in sequence.
    Satisfies the duck-typed interface expected by agent_loop / _call_llm.
    """

    def __init__(self, responses: list[AssistantMessage]) -> None:
        self._queue = list(responses)
        self.call_count = 0
        self.received_messages: list[list[dict]] = []  # messages snapshot per call

    async def stream_response(
        self,
        messages: list[dict],
        system_prompt: str,
        tools,
        model=None,
        on_event=None,
    ) -> AssistantMessage:
        self.call_count += 1
        self.received_messages.append(list(messages))
        msg = self._queue.pop(0)
        # Emit TEXT_DELTA for any text blocks (simulates streaming)
        if on_event:
            for block in msg.content:
                if hasattr(block, "text"):
                    on_event(AgentEvent(type=EventType.TEXT_DELTA, data={"text": block.text}))
        return msg

    async def complete(
        self,
        messages: list[dict],
        system_prompt: str,
        tools=None,
        model=None,
    ) -> AssistantMessage:
        self.call_count += 1
        self.received_messages.append(list(messages))
        return self._queue.pop(0)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def echo_tool() -> EchoTool:
    return EchoTool()


@pytest.fixture
def echo_registry(echo_tool) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(echo_tool)
    return registry


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    return tmp_path
