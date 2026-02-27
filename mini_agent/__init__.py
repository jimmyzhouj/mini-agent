"""Mini Agent — minimal AI coding agent framework inspired by Pi."""

__version__ = "0.1.0"

from mini_agent.types import (
    TextBlock,
    ImageBlock,
    ToolUseBlock,
    ToolResultBlock,
    ContentBlock,
    UserMessage,
    AssistantMessage,
    TokenUsage,
    Message,
    ToolResult,
    EventType,
    AgentEvent,
    AgentState,
    ProviderConfig,
)
from mini_agent.agent import Agent, DEFAULT_SYSTEM_PROMPT
from mini_agent.session import SessionManager
from mini_agent.loop import agent_loop

__all__ = [
    # types
    "TextBlock",
    "ImageBlock",
    "ToolUseBlock",
    "ToolResultBlock",
    "ContentBlock",
    "UserMessage",
    "AssistantMessage",
    "TokenUsage",
    "Message",
    "ToolResult",
    "EventType",
    "AgentEvent",
    "AgentState",
    "ProviderConfig",
    # high-level
    "Agent",
    "DEFAULT_SYSTEM_PROMPT",
    "SessionManager",
    "agent_loop",
]
