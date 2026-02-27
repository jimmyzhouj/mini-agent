"""Tool base class and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Type, TypeVar

from pydantic import BaseModel, ValidationError

from mini_agent.types import ToolResult

T = TypeVar("T", bound=BaseModel)


class Tool(ABC, Generic[T]):
    """
    Base class for all tools. Modelled after Pi's AgentTool interface.

    Subclasses must define:
    - name: str
    - description: str
    - parameters: Type[T]  (a Pydantic BaseModel class)
    - execute(params: T) -> ToolResult
    """

    name: str
    description: str
    parameters: Type[T]

    def get_schema(self) -> dict:
        """Return the Anthropic-format tool definition."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters.model_json_schema(),
        }

    def validate_params(self, raw_input: dict) -> T | ToolResult:
        """
        Validate params with Pydantic.
        Success → returns validated Pydantic object.
        Failure → returns ToolResult(is_error=True) with detailed error message.

        Key Pi design: validation failure does NOT raise — the error message is
        fed back to the LLM as a tool_result so the model can self-correct.
        """
        try:
            return self.parameters.model_validate(raw_input)
        except ValidationError as e:
            return ToolResult(
                output=f"Parameter validation failed:\n{e}",
                is_error=True,
            )

    @abstractmethod
    async def execute(self, params: T) -> ToolResult:
        """Execute the tool. Implemented by subclasses."""
        ...


class ToolRegistry:
    """Tool registry. Manages all available tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def get_schemas(self) -> list[dict]:
        """Return all tool schemas in Anthropic format."""
        return [t.get_schema() for t in self._tools.values()]
