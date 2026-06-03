from typing import Any
from agent.tools.base import Tool


class ToolRegistry:
    """Central registry: register tools once, dispatch by name, export provider formats."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' is not registered.")
        return self._tools[name]

    def dispatch(self, name: str, inputs: dict) -> Any:
        tool = self.get(name)
        validated = tool.input_schema.model_validate(inputs)
        return tool.handler(validated)

    def all_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def to_anthropic_format(self) -> list[dict]:
        return [t.to_anthropic_format() for t in self._tools.values()]

    def to_openai_format(self) -> list[dict]:
        return [t.to_openai_format() for t in self._tools.values()]
