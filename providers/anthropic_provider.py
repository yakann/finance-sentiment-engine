import logging
from typing import TYPE_CHECKING
from pydantic import BaseModel
from providers.base import LLMProvider, LLMUsage, LLMResponse, ToolCall, NextAction
from dotenv import load_dotenv
from pathlib import Path
import anthropic

if TYPE_CHECKING:
    from agent.tools.base import Tool

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

logger = logging.getLogger(__name__)


class AnthropicProvider(LLMProvider):
    def __init__(self, model: str):
        self.model = model
        self.client = anthropic.Anthropic()
        self._async_client: anthropic.AsyncAnthropic | None = None

    @property
    def _aclient(self) -> anthropic.AsyncAnthropic:
        if self._async_client is None:
            self._async_client = anthropic.AsyncAnthropic()
        return self._async_client

    def generate(
        self,
        messages,
        system=None,
        tools: list["Tool"] | None = None,
    ) -> LLMResponse:
        kwargs: dict = dict(model=self.model, max_tokens=1024, messages=messages)
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = [t.to_anthropic_format() for t in tools]
        message = self.client.messages.create(**kwargs)

        text = ""
        tool_calls: list[ToolCall] = []
        raw_content: list[dict] = []

        for block in message.content:
            if block.type == "text":
                text = block.text
                raw_content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, input=dict(block.input))
                )
                raw_content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": dict(block.input),
                })

        next_action = NextAction(
            type="tool_calls" if tool_calls else "text",
            tool_calls=tool_calls,
        )
        usage = LLMUsage(
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
            total_tokens=message.usage.input_tokens + message.usage.output_tokens,
        )
        result = LLMResponse(
            text=text,
            usage=usage,
            next_action=next_action,
            raw_assistant_content=raw_content,
        )
        self._log_usage(result.usage, "anthropic", self.model)
        return result

    def generate_structured(self, messages, schema: type[BaseModel]) -> BaseModel:
        tool = {
            "name": "structured_output",
            "description": "Return structured data matching the required schema.",
            "input_schema": schema.model_json_schema(),
        }
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            tools=[tool],
            tool_choice={"type": "tool", "name": "structured_output"},
            messages=messages,
        )
        tool_block = next(b for b in response.content if b.type == "tool_use")
        return schema.model_validate(tool_block.input)

    async def agenerate_structured(
        self,
        messages,
        schema: type[BaseModel],
        system: str | None = None,
    ) -> tuple[BaseModel, LLMUsage]:
        tool = {
            "name": "structured_output",
            "description": "Return structured data matching the required schema.",
            "input_schema": schema.model_json_schema(),
        }
        kwargs: dict = dict(
            model=self.model,
            max_tokens=1024,
            tools=[tool],
            tool_choice={"type": "tool", "name": "structured_output"},
            messages=list(messages),
        )
        if system:
            kwargs["system"] = system
        response = await self._aclient.messages.create(**kwargs)
        tool_block = next(b for b in response.content if b.type == "tool_use")
        usage = LLMUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
        )
        self._log_usage(usage, "anthropic", self.model)
        return schema.model_validate(tool_block.input), usage
