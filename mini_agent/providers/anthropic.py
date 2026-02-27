"""Anthropic API provider — the only module that talks to the LLM."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Callable

import anthropic

from mini_agent.types import (
    AgentEvent,
    AssistantMessage,
    ContentBlock,
    EventType,
    ImageBlock,
    ProviderConfig,
    TextBlock,
    TokenUsage,
    ToolUseBlock,
)

if TYPE_CHECKING:
    from mini_agent.tools.base import Tool

logger = logging.getLogger(__name__)


class AnthropicProvider:
    """Wraps the Anthropic SDK. Everything else uses this to talk to the LLM."""

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self._client = anthropic.Anthropic(
            api_key=config.api_key,          # None → SDK reads ANTHROPIC_API_KEY
            base_url=config.base_url,        # None → official endpoint
            timeout=config.timeout,
            default_headers=config.default_headers or {},
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_tools(self, tools: list[Tool]) -> list[dict]:
        """Convert Tool objects to Anthropic SDK format."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.parameters.model_json_schema(),
            }
            for t in tools
        ]

    def _parse_content(self, raw_content: list) -> list[ContentBlock]:
        """Convert Anthropic SDK content blocks to internal types."""
        blocks: list[ContentBlock] = []
        for block in raw_content:
            if block.type == "text":
                blocks.append(TextBlock(text=block.text))
            elif block.type == "tool_use":
                blocks.append(
                    ToolUseBlock(
                        id=block.id,
                        name=block.name,
                        input=block.input,
                    )
                )
            elif block.type == "image":
                # pass-through image blocks (rare in assistant responses)
                source = block.source
                blocks.append(
                    ImageBlock(
                        data=source.data if hasattr(source, "data") else "",
                        mime_type=source.media_type if hasattr(source, "media_type") else "image/png",
                    )
                )
        return blocks

    def _parse_usage(self, usage) -> TokenUsage:
        if usage is None:
            return TokenUsage()
        return TokenUsage(
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def complete(
        self,
        messages: list[dict],
        system_prompt: str,
        tools: list[Tool] | None = None,
        model: str | None = None,
    ) -> AssistantMessage:
        """Non-streaming call. Used for compaction and Phase 1 agent loop."""
        kwargs: dict = {
            "model": model or self.config.model,
            "max_tokens": self.config.max_tokens,
            "system": system_prompt,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = self._build_tools(tools)

        # Run the blocking SDK call in a thread to stay async-friendly
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._client.messages.create(**kwargs),
        )

        content = self._parse_content(response.content)
        usage = self._parse_usage(response.usage)

        return AssistantMessage(
            content=content,
            stop_reason=response.stop_reason,
            model=response.model,
            usage=usage,
        )

    async def stream_response(
        self,
        messages: list[dict],
        system_prompt: str,
        tools: list[Tool],
        model: str | None = None,
        on_event: "Callable[[AgentEvent], None] | None" = None,
    ) -> AssistantMessage:
        """
        Streaming call. Yields TEXT_DELTA events via on_event callback,
        then returns the complete AssistantMessage.

        Phase 2 implementation — uses client.messages.stream().
        """
        kwargs: dict = {
            "model": model or self.config.model,
            "max_tokens": self.config.max_tokens,
            "system": system_prompt,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = self._build_tools(tools)

        def _emit(event: AgentEvent) -> None:
            if on_event:
                on_event(event)

        # Accumulate content blocks
        text_blocks: dict[int, str] = {}   # index → accumulated text
        tool_blocks: dict[int, dict] = {}  # index → {id, name, input_json}
        stop_reason: str | None = None
        usage: TokenUsage = TokenUsage()
        response_model: str | None = None

        loop = asyncio.get_running_loop()

        # Run the synchronous SDK stream in a thread pool executor
        final_message_holder: list = []

        def _run_stream():
            nonlocal stop_reason, usage, response_model
            with self._client.messages.stream(**kwargs) as stream:
                for event in stream:
                    _process_event(event)
                final = stream.get_final_message()
                final_message_holder.append(final)

        def _process_event(event):
            nonlocal stop_reason, usage, response_model
            etype = event.type

            if etype == "content_block_start":
                idx = event.index
                block = event.content_block
                if block.type == "text":
                    text_blocks[idx] = ""
                elif block.type == "tool_use":
                    tool_blocks[idx] = {"id": block.id, "name": block.name, "input_json": ""}
                    _emit(AgentEvent(
                        type=EventType.TOOL_CALL_START,
                        data={"name": block.name, "input": {}},
                    ))

            elif etype == "content_block_delta":
                idx = event.index
                delta = event.delta
                if delta.type == "text_delta":
                    text_blocks[idx] = text_blocks.get(idx, "") + delta.text
                    _emit(AgentEvent(
                        type=EventType.TEXT_DELTA,
                        data={"text": delta.text},
                    ))
                elif delta.type == "input_json_delta":
                    if idx in tool_blocks:
                        tool_blocks[idx]["input_json"] += delta.partial_json

            elif etype == "message_delta":
                if hasattr(event, "delta") and hasattr(event.delta, "stop_reason"):
                    stop_reason = event.delta.stop_reason
                if hasattr(event, "usage"):
                    usage = self._parse_usage(event.usage)

            elif etype == "message_start":
                if hasattr(event, "message"):
                    response_model = getattr(event.message, "model", None)
                    usage = self._parse_usage(getattr(event.message, "usage", None))

        try:
            await loop.run_in_executor(None, _run_stream)
        except anthropic.APIError as e:
            _emit(AgentEvent(type=EventType.ERROR, data={"error": str(e)}))
            raise
        except asyncio.CancelledError:
            raise

        # Parse final message from holder if available
        if final_message_holder:
            final = final_message_holder[0]
            content = self._parse_content(final.content)
            usage = self._parse_usage(final.usage)
            stop_reason = final.stop_reason
            response_model = final.model
        else:
            # Reconstruct from accumulated blocks
            content_blocks: list[ContentBlock] = []
            # Rebuild in index order
            all_indices = sorted(set(list(text_blocks.keys()) + list(tool_blocks.keys())))
            for idx in all_indices:
                if idx in text_blocks:
                    content_blocks.append(TextBlock(text=text_blocks[idx]))
                elif idx in tool_blocks:
                    tb = tool_blocks[idx]
                    try:
                        parsed_input = json.loads(tb["input_json"]) if tb["input_json"] else {}
                    except Exception:
                        parsed_input = {}
                    content_blocks.append(
                        ToolUseBlock(id=tb["id"], name=tb["name"], input=parsed_input)
                    )
            content = content_blocks

        return AssistantMessage(
            content=content,
            stop_reason=stop_reason,
            model=response_model,
            usage=usage,
        )
