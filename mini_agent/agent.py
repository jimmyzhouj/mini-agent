"""
Agent class — high-level wrapper over agent_loop.

Manages state, events, steer/followUp queues, and optional session persistence.
Mirrors Pi's Agent class design.
"""

from __future__ import annotations

import asyncio
from typing import Callable

from mini_agent.compact import compact_messages
from mini_agent.loop import agent_loop
from mini_agent.providers.anthropic import AnthropicProvider
from mini_agent.session import SessionManager
from mini_agent.tools.base import Tool, ToolRegistry
from mini_agent.tools.bash import BashTool
from mini_agent.tools.edit import EditTool
from mini_agent.tools.read import ReadTool
from mini_agent.tools.write import WriteTool
from mini_agent.types import (
    AgentEvent,
    AssistantMessage,
    EventType,
    ProviderConfig,
    TextBlock,
    TokenUsage,
    ToolResultBlock,
    ToolUseBlock,
)

DEFAULT_SYSTEM_PROMPT = """You are an expert coding assistant. You help users with coding tasks by reading files, executing commands, editing code, and writing new files.

Available tools:
- read: Read file contents
- bash: Execute bash commands
- edit: Make surgical edits to files (old text must match exactly)
- write: Create or overwrite files

Guidelines:
- Use bash for file operations like ls, grep, find
- Read files before editing to understand context
- Use edit for precise changes, write only for new files or complete rewrites
- Be concise in your responses
"""

# Compaction threshold: ~100K estimated tokens (chars / 4)
_COMPACT_CHAR_THRESHOLD = 400_000


class Agent:
    """
    High-level agent. Provides prompt(), steer(), follow_up(), abort(), subscribe().
    """

    def __init__(
        self,
        provider_config: ProviderConfig,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        tools: list[Tool] | None = None,
        session_manager: SessionManager | None = None,
    ) -> None:
        self.provider = AnthropicProvider(provider_config)
        self._config = provider_config
        self.system_prompt = system_prompt
        self.session_manager = session_manager

        # Tool registry
        self._registry = ToolRegistry()
        if tools is None:
            tools = [BashTool(), ReadTool(), WriteTool(), EditTool()]
        for t in tools:
            self._registry.register(t)

        # Conversation state
        self._messages: list[dict] = []

        # Queues for steer / follow_up
        self._steer_queue: list[dict] = []
        self._followup_queue: list[dict] = []

        # Abort signal
        self._abort_signal = asyncio.Event()

        # Event subscribers: list of (callback, unsubscribe_fn)
        self._subscribers: list[Callable[[AgentEvent], None]] = []

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    async def prompt(self, message: str) -> AssistantMessage:
        """
        Send a user message and run the agent loop until completion.
        """
        # Reset abort signal for this run
        self._abort_signal.clear()

        # Build user message
        user_msg: dict = {"role": "user", "content": message}
        self._messages.append(user_msg)

        # Persist user message
        if self.session_manager:
            self.session_manager.append("user", {"content": message})

        # Record where new messages will start
        snapshot_len = len(self._messages)

        # Run the loop
        result = await agent_loop(
            provider=self.provider,
            system_prompt=self.system_prompt,
            messages=self._messages,
            tool_registry=self._registry,
            model=self._config.model,
            on_event=self._emit,
            get_queued_messages=self._drain_queues,
            abort_signal=self._abort_signal,
        )

        # Persist only messages added by agent_loop
        if self.session_manager and result:
            new_messages = self._messages[snapshot_len:]
            _persist_new_messages(self.session_manager, new_messages, result)

        # Check compaction
        await self._maybe_compact()

        return result

    def steer(self, message: str) -> None:
        """
        Interrupt current execution and inject a message.
        The message is injected after the current tool finishes, before the next LLM call.
        """
        self._steer_queue.append({"role": "user", "content": message})

    def follow_up(self, message: str) -> None:
        """
        Queue a message to inject after the agent naturally finishes.
        Does not interrupt current execution.
        """
        self._followup_queue.append({"role": "user", "content": message})

    def abort(self) -> None:
        """Abort the current agent loop."""
        self._abort_signal.set()

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def set_model(self, model: str) -> None:
        """Switch model. Takes effect on the next prompt()."""
        self._config = self._config.model_copy(update={"model": model})

    def set_system_prompt(self, prompt: str) -> None:
        """Update the system prompt."""
        self.system_prompt = prompt

    def set_tools(self, tools: list[Tool]) -> None:
        """Replace the tool set."""
        self._registry = ToolRegistry()
        for t in tools:
            self._registry.register(t)

    def get_messages(self) -> list[dict]:
        """Return a read-only copy of the current message list."""
        return list(self._messages)

    def replace_messages(self, messages: list[dict]) -> None:
        """Replace the message list (used after compaction)."""
        self._messages = messages

    # ------------------------------------------------------------------
    # Event subscription
    # ------------------------------------------------------------------

    def subscribe(self, callback: Callable[[AgentEvent], None]) -> Callable[[], None]:
        """
        Subscribe to agent events. Returns an unsubscribe function.
        """
        self._subscribers.append(callback)

        def unsubscribe() -> None:
            try:
                self._subscribers.remove(callback)
            except ValueError:
                pass

        return unsubscribe

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _drain_queues(self) -> list[dict]:
        """
        Return queued messages. Priority: steer > followup.
        Steer clears the followup queue (user intervened).
        """
        if self._steer_queue:
            msgs = list(self._steer_queue)
            self._steer_queue.clear()
            self._followup_queue.clear()  # user intervened, drop pending followups
            return msgs
        if self._followup_queue:
            msgs = list(self._followup_queue)
            self._followup_queue.clear()
            return msgs
        return []

    def _emit(self, event: AgentEvent) -> None:
        """Broadcast event to all subscribers."""
        for cb in list(self._subscribers):
            try:
                cb(event)
            except Exception:
                pass  # never let subscriber errors crash the agent

    async def _maybe_compact(self) -> None:
        """Trigger compaction if estimated token count exceeds threshold."""
        estimated_chars = sum(len(str(m)) for m in self._messages)
        if estimated_chars < _COMPACT_CHAR_THRESHOLD:
            return

        estimated_tokens = estimated_chars // 4
        self._emit(AgentEvent(
            type=EventType.COMPACTION,
            data={"estimated_tokens": estimated_tokens},
        ))

        self._messages = await compact_messages(
            provider=self.provider,
            messages=self._messages,
            system_prompt=self.system_prompt,
            model=self._config.model,
        )

        if self.session_manager:
            self.session_manager.append("meta", {
                "type": "compaction",
                "estimated_tokens_before": estimated_tokens,
            })


# ---------------------------------------------------------------------------
# Session persistence helper
# ---------------------------------------------------------------------------

def _persist_new_messages(
    session_manager: SessionManager,
    messages: list[dict],
    last_response: AssistantMessage,
) -> None:
    """
    Persist assistant messages and tool results that were added during agent_loop.
    We look for assistant messages and tool_result messages at the end of the list
    (after the user message we already persisted).

    This is a best-effort scan — we persist assistant + tool_result pairs.
    """
    # Find consecutive assistant/tool-result messages from the end
    # (the user message was already persisted before agent_loop ran)
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")

        if role == "assistant":
            # Serialize content blocks for storage
            if isinstance(content, list):
                serialized = content  # already dicts
            else:
                serialized = content
            session_manager.append(
                "assistant",
                {"content": serialized},
                model=last_response.model,
            )

        elif role == "user" and isinstance(content, list):
            # Check if this is a tool_result message
            if content and isinstance(content[0], dict) and content[0].get("type") == "tool_result":
                session_manager.append("tool_result", {"results": content})
