"""
Agent loop — the heart of mini-agent.

Implements the Pi agentLoop() pattern: while True → call LLM → execute tools
→ feed results back → repeat until no tool calls.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable

import anthropic as anthropic_sdk

from mini_agent.providers.anthropic import AnthropicProvider
from mini_agent.tools.base import ToolRegistry
from mini_agent.types import (
    AgentEvent,
    AssistantMessage,
    EventType,
    TextBlock,
    ToolResult,
    ToolUseBlock,
)

logger = logging.getLogger(__name__)


async def agent_loop(
    provider: AnthropicProvider,
    system_prompt: str,
    messages: list[dict],
    tool_registry: ToolRegistry,
    model: str | None = None,
    on_event: Callable[[AgentEvent], None] | None = None,
    get_queued_messages: Callable[[], list[dict]] | None = None,
    abort_signal: asyncio.Event | None = None,
) -> AssistantMessage:
    """
    Core agent loop.

    Args:
        provider: Anthropic provider instance
        system_prompt: system prompt text
        messages: current conversation context (mutated in-place)
        tool_registry: registered tools
        model: model ID (optional, overrides provider default)
        on_event: event callback, called once per event
        get_queued_messages: called after each turn, returns externally injected messages
        abort_signal: when set(), abort the loop

    Returns:
        The last AssistantMessage from the LLM.
    """

    def emit(event_type: EventType, data: dict | None = None) -> None:
        if on_event:
            on_event(AgentEvent(type=event_type, data=data or {}))

    emit(EventType.AGENT_START)
    last_response: AssistantMessage | None = None

    while True:
        # Check abort
        if abort_signal and abort_signal.is_set():
            emit(EventType.AGENT_END, {"reason": "aborted"})
            break

        emit(EventType.TURN_START)

        # ① Call LLM
        try:
            assistant_message = await _call_llm(
                provider, messages, system_prompt, tool_registry, model, on_event
            )
        except Exception as e:
            emit(EventType.ERROR, {"error": str(e)})
            emit(EventType.AGENT_END, {"reason": "error"})
            raise

        last_response = assistant_message

        # ② Append assistant message to context
        messages.append(_to_api_format(assistant_message))

        # ③ Extract tool_use blocks
        tool_uses = [b for b in assistant_message.content if isinstance(b, ToolUseBlock)]

        # ④ No tool calls → end (or check for queued messages)
        if not tool_uses:
            emit(EventType.TURN_END)
            if get_queued_messages:
                queued = get_queued_messages()
                if queued:
                    for msg in queued:
                        messages.append(msg)
                    continue  # resume loop with queued messages
            emit(EventType.AGENT_END, {"reason": "completed"})
            break

        # ⑤ Execute each tool call sequentially
        tool_results = []
        for tool_use in tool_uses:
            # Check abort between tool calls
            if abort_signal and abort_signal.is_set():
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": "Aborted by user",
                    "is_error": True,
                })
                continue

            emit(EventType.TOOL_CALL_START, {"name": tool_use.name, "input": tool_use.input})

            tool = tool_registry.get(tool_use.name)
            if not tool:
                result = ToolResult(output=f"Unknown tool: {tool_use.name}", is_error=True)
            else:
                validated = tool.validate_params(tool_use.input)
                if isinstance(validated, ToolResult):
                    # Validation failed — feed error back to LLM
                    result = validated
                else:
                    try:
                        result = await tool.execute(validated)
                    except Exception as e:
                        result = ToolResult(output=f"Tool execution error: {e}", is_error=True)

            emit(EventType.TOOL_CALL_END, {
                "name": tool_use.name,
                "output": result.output[:500],
                "is_error": result.is_error,
            })
            emit(EventType.TOOL_RESULT, {"name": tool_use.name, "details": result.details})

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": result.output,
                "is_error": result.is_error,
            })

        # ⑥ Feed tool results back into context
        messages.append({"role": "user", "content": tool_results})

        emit(EventType.TURN_END)

        # ⑦ Check for queued messages (steer scenario)
        if get_queued_messages:
            queued = get_queued_messages()
            if queued:
                for msg in queued:
                    messages.append(msg)

        # ⑧ Loop back to top → call LLM again

    return last_response  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _call_llm(
    provider: AnthropicProvider,
    messages: list[dict],
    system_prompt: str,
    tool_registry: ToolRegistry,
    model: str | None,
    on_event: Callable[[AgentEvent], None] | None,
) -> AssistantMessage:
    """
    Call the LLM. Uses streaming if on_event is provided, otherwise non-streaming.
    """
    tools = tool_registry.list_tools()

    # Try streaming first; fall back to complete() only for streaming-protocol failures.
    # Real API errors (4xx/5xx) are re-raised immediately.
    try:
        return await provider.stream_response(
            messages=messages,
            system_prompt=system_prompt,
            tools=tools,
            model=model,
            on_event=on_event,
        )
    except anthropic_sdk.APIError:
        raise  # propagate real API errors (auth, rate-limit, bad request, etc.)
    except Exception:
        # Fallback to non-streaming (e.g., internal endpoints that don't support SSE)
        logger.debug("Streaming failed, falling back to complete()")
        return await provider.complete(
            messages=messages,
            system_prompt=system_prompt,
            tools=tools,
            model=model,
        )


def _to_api_format(assistant_message: AssistantMessage) -> dict:
    """Convert internal AssistantMessage to Anthropic API dict format."""
    content = []
    for block in assistant_message.content:
        if isinstance(block, TextBlock):
            content.append({"type": "text", "text": block.text})
        elif isinstance(block, ToolUseBlock):
            content.append({
                "type": "tool_use",
                "id": block.id,
                "name": block.name,
                "input": block.input,
            })
        else:
            # ImageBlock or other — serialize via model_dump
            content.append(block.model_dump())
    return {"role": "assistant", "content": content}
