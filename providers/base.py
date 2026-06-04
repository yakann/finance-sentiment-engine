import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from agent.tools.base import Tool

logger = logging.getLogger(__name__)


class LLMUsage(BaseModel):
    input_tokens: int
    cached_tokens: int = 0
    output_tokens: int
    reasoning_tokens: int = 0
    total_tokens: int


class ToolCall(BaseModel):
    id: str
    name: str
    input: dict[str, Any]


class NextAction(BaseModel):
    type: Literal["text", "tool_calls"]
    tool_calls: list[ToolCall] = []


class LLMResponse(BaseModel):
    text: str
    usage: LLMUsage
    next_action: NextAction = Field(default_factory=lambda: NextAction(type="text"))
    # Provider-specific content blocks for rebuilding conversation history
    raw_assistant_content: Any = None


class LLMProvider(ABC):
    # Subclasses can override to tune per-provider safe concurrency
    default_concurrency: int = 10

    @abstractmethod
    def generate(
        self,
        messages,
        system=None,
        tools: list["Tool"] | None = None,
    ) -> LLMResponse:
        ...

    @abstractmethod
    def generate_structured(self, messages, schema: type[BaseModel]) -> BaseModel:
        ...

    @abstractmethod
    async def agenerate_structured(
        self,
        messages,
        schema: type[BaseModel],
        system: str | None = None,
    ) -> tuple[BaseModel, LLMUsage]:
        ...

    def extend_messages_with_assistant_turn(
        self, messages: list, response: LLMResponse
    ) -> None:
        """Append the assistant's tool-use turn to the message history (Anthropic format)."""
        messages.append({"role": "assistant", "content": response.raw_assistant_content})

    def extend_messages_with_tool_results(
        self,
        messages: list,
        tool_results: list[dict],
    ) -> None:
        """Append tool results to the message history.

        tool_results: list of {"tool_call_id": str, "result": str}
        Default format: Anthropic tool_result blocks.
        """
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tr["tool_call_id"],
                    "content": tr["result"],
                }
                for tr in tool_results
            ],
        })

    def _log_usage(self, usage: LLMUsage, provider: str, model: str) -> None:
        logger.info(
            "%s/%s — input: %d, cached: %d, output: %d, total: %d tokens",
            provider, model,
            usage.input_tokens, usage.cached_tokens,
            usage.output_tokens, usage.total_tokens,
        )
