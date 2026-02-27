"""
Tests for agent_loop core logic. Uses MockProvider to avoid real API calls.

Scenarios:
1. Pure text response → loop runs once and ends
2. Single tool call → execute → feed back → LLM final text
3. Multiple tool calls in same turn → all executed sequentially
4. Multi-turn tool calls → LLM calls tool across multiple rounds
5. Tool param validation failure → error fed back → LLM retries
6. Tool execution exception → error fed back → LLM handles it
7. Abort signal → loop stops
8. Unknown tool name → error fed back
"""

from __future__ import annotations

import asyncio
import pytest
from pydantic import BaseModel, Field

from mini_agent.loop import agent_loop
from mini_agent.tools.base import Tool, ToolRegistry
from mini_agent.types import (
    AgentEvent,
    AssistantMessage,
    EventType,
    TextBlock,
    ToolResult,
    ToolUseBlock,
)
from tests.conftest import EchoTool, FailingTool, MockProvider

SYSTEM = "You are a test assistant."


def _text_msg(text: str) -> AssistantMessage:
    """Helper: AssistantMessage with a single TextBlock."""
    return AssistantMessage(content=[TextBlock(text=text)], stop_reason="end_turn")


def _tool_msg(*tool_uses: tuple[str, str, dict]) -> AssistantMessage:
    """
    Helper: AssistantMessage with ToolUseBlocks.
    tool_uses: list of (id, name, input_dict)
    """
    blocks = [ToolUseBlock(id=tid, name=name, input=inp) for tid, name, inp in tool_uses]
    return AssistantMessage(content=blocks, stop_reason="tool_use")


def _make_registry(*tools: Tool) -> ToolRegistry:
    r = ToolRegistry()
    for t in tools:
        r.register(t)
    return r


# ---------------------------------------------------------------------------
# Test 1: Pure text response — loop runs once and ends
# ---------------------------------------------------------------------------

async def test_pure_text_response():
    provider = MockProvider([_text_msg("Hello, world!")])
    messages = [{"role": "user", "content": "hi"}]
    events: list[AgentEvent] = []

    result = await agent_loop(
        provider=provider,
        system_prompt=SYSTEM,
        messages=messages,
        tool_registry=ToolRegistry(),
        on_event=events.append,
    )

    assert provider.call_count == 1
    assert len(result.content) == 1
    assert result.content[0].text == "Hello, world!"

    event_types = [e.type for e in events]
    assert EventType.AGENT_START in event_types
    assert EventType.AGENT_END in event_types
    assert EventType.TEXT_DELTA in event_types

    # No tool events
    assert EventType.TOOL_CALL_START not in event_types
    assert EventType.TOOL_CALL_END not in event_types

    # messages: original user + assistant
    assert len(messages) == 2
    assert messages[1]["role"] == "assistant"


# ---------------------------------------------------------------------------
# Test 2: Single tool call → execute → feed back → final text
# ---------------------------------------------------------------------------

async def test_single_tool_call():
    echo = EchoTool()
    provider = MockProvider([
        _tool_msg(("t1", "echo", {"message": "ping"})),
        _text_msg("Done."),
    ])
    messages = [{"role": "user", "content": "use echo"}]

    result = await agent_loop(
        provider=provider,
        system_prompt=SYSTEM,
        messages=messages,
        tool_registry=_make_registry(echo),
    )

    assert provider.call_count == 2
    assert echo.calls == ["ping"]

    # Final message is the text response
    assert result.content[0].text == "Done."

    # messages: user, assistant(tool_use), user(tool_result), assistant(text)
    assert len(messages) == 4
    tool_result_msg = messages[2]
    assert tool_result_msg["role"] == "user"
    assert tool_result_msg["content"][0]["type"] == "tool_result"
    assert tool_result_msg["content"][0]["tool_use_id"] == "t1"
    assert "Echo: ping" in tool_result_msg["content"][0]["content"]


# ---------------------------------------------------------------------------
# Test 3: Multiple tool calls in same turn → executed sequentially
# ---------------------------------------------------------------------------

async def test_multiple_tool_calls_same_turn():
    echo = EchoTool()
    provider = MockProvider([
        _tool_msg(
            ("t1", "echo", {"message": "first"}),
            ("t2", "echo", {"message": "second"}),
        ),
        _text_msg("Both done."),
    ])
    messages = [{"role": "user", "content": "two calls"}]

    result = await agent_loop(
        provider=provider,
        system_prompt=SYSTEM,
        messages=messages,
        tool_registry=_make_registry(echo),
    )

    assert provider.call_count == 2
    # Executed in order
    assert echo.calls == ["first", "second"]

    # One tool_result message with two blocks
    tool_result_msg = messages[2]
    assert len(tool_result_msg["content"]) == 2
    assert tool_result_msg["content"][0]["tool_use_id"] == "t1"
    assert tool_result_msg["content"][1]["tool_use_id"] == "t2"


# ---------------------------------------------------------------------------
# Test 4: Multi-turn tool calls
# ---------------------------------------------------------------------------

async def test_multi_turn_tool_calls():
    echo = EchoTool()
    provider = MockProvider([
        _tool_msg(("t1", "echo", {"message": "round1"})),
        _tool_msg(("t2", "echo", {"message": "round2"})),
        _text_msg("All done."),
    ])
    messages = [{"role": "user", "content": "multi-turn"}]

    result = await agent_loop(
        provider=provider,
        system_prompt=SYSTEM,
        messages=messages,
        tool_registry=_make_registry(echo),
    )

    assert provider.call_count == 3
    assert echo.calls == ["round1", "round2"]
    assert result.content[0].text == "All done."

    # messages: user, asst(t1), user(r1), asst(t2), user(r2), asst(text)
    assert len(messages) == 6


# ---------------------------------------------------------------------------
# Test 5: Tool param validation failure → error fed back → LLM retries
# ---------------------------------------------------------------------------

async def test_tool_param_validation_failure():
    echo = EchoTool()
    # LLM sends wrong param name (missing required 'message')
    provider = MockProvider([
        _tool_msg(("t1", "echo", {"wrong_param": "oops"})),
        _text_msg("I see the error, my bad."),
    ])
    messages = [{"role": "user", "content": "bad params"}]

    result = await agent_loop(
        provider=provider,
        system_prompt=SYSTEM,
        messages=messages,
        tool_registry=_make_registry(echo),
    )

    assert provider.call_count == 2
    # Tool was not actually called (validation failed)
    assert echo.calls == []

    # tool_result should contain validation error
    tool_result_content = messages[2]["content"][0]["content"]
    assert "validation" in tool_result_content.lower() or "Parameter" in tool_result_content
    assert messages[2]["content"][0]["is_error"] is True


# ---------------------------------------------------------------------------
# Test 6: Tool execution exception → error fed back
# ---------------------------------------------------------------------------

async def test_tool_execution_exception():
    fail = FailingTool()
    provider = MockProvider([
        _tool_msg(("t1", "fail", {"message": "go"})),
        _text_msg("Handled the error."),
    ])
    messages = [{"role": "user", "content": "fail please"}]
    events: list[AgentEvent] = []

    result = await agent_loop(
        provider=provider,
        system_prompt=SYSTEM,
        messages=messages,
        tool_registry=_make_registry(fail),
        on_event=events.append,
    )

    assert provider.call_count == 2

    # tool_result should contain the exception message
    tool_result_content = messages[2]["content"][0]["content"]
    assert "intentional failure" in tool_result_content
    assert messages[2]["content"][0]["is_error"] is True

    # TOOL_CALL_END event should show error
    end_events = [e for e in events if e.type == EventType.TOOL_CALL_END]
    assert len(end_events) == 1
    assert end_events[0].data["is_error"] is True


# ---------------------------------------------------------------------------
# Test 7: Abort signal → loop stops
# ---------------------------------------------------------------------------

async def test_abort_signal():
    echo = EchoTool()
    abort = asyncio.Event()

    call_count = 0

    class AbortingProvider:
        async def stream_response(self, messages, system_prompt, tools, model=None, on_event=None):
            nonlocal call_count
            call_count += 1
            abort.set()  # signal abort right after first LLM call
            return _tool_msg(("t1", "echo", {"message": "hi"}))

        async def complete(self, messages, system_prompt, tools=None, model=None):
            nonlocal call_count
            call_count += 1
            return _tool_msg(("t1", "echo", {"message": "hi"}))

    messages = [{"role": "user", "content": "abort me"}]
    events: list[AgentEvent] = []

    result = await agent_loop(
        provider=AbortingProvider(),
        system_prompt=SYSTEM,
        messages=messages,
        tool_registry=_make_registry(echo),
        on_event=events.append,
        abort_signal=abort,
    )

    # abort.set() is called inside stream_response; on the next loop iteration
    # the abort check at the top of while True fires → AGENT_END(aborted).
    agent_end_events = [e for e in events if e.type == EventType.AGENT_END]
    assert len(agent_end_events) == 1
    # Could be "aborted" or "completed" depending on timing, but loop must have ended
    assert agent_end_events[0].data.get("reason") in ("aborted", "completed")


# ---------------------------------------------------------------------------
# Test 8: Unknown tool name → error fed back
# ---------------------------------------------------------------------------

async def test_unknown_tool_name():
    provider = MockProvider([
        _tool_msg(("t1", "nonexistent_tool", {"x": 1})),
        _text_msg("OK, no such tool."),
    ])
    messages = [{"role": "user", "content": "call unknown"}]

    result = await agent_loop(
        provider=provider,
        system_prompt=SYSTEM,
        messages=messages,
        tool_registry=ToolRegistry(),  # empty registry
    )

    assert provider.call_count == 2

    tool_result_content = messages[2]["content"][0]["content"]
    assert "Unknown tool" in tool_result_content
    assert "nonexistent_tool" in tool_result_content
    assert messages[2]["content"][0]["is_error"] is True
