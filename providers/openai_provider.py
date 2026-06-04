import json
import logging
from typing import TYPE_CHECKING
from pydantic import BaseModel
from providers.base import LLMProvider, LLMUsage, LLMResponse, ToolCall, NextAction
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv
from pathlib import Path

if TYPE_CHECKING:
    from agent.tools.base import Tool

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    def __init__(self, model: str):
        self.model = model
        self.client = OpenAI()
        self._async_client: AsyncOpenAI | None = None

    @property
    def _aclient(self) -> AsyncOpenAI:
        if self._async_client is None:
            self._async_client = AsyncOpenAI()
        return self._async_client

    def generate(
        self,
        messages,
        system=None,
        tools: list["Tool"] | None = None,
    ) -> LLMResponse:
        kwargs: dict = dict(model=self.model, input=messages, instructions=system)
        if tools:
            kwargs["tools"] = [t.to_openai_responses_format() for t in tools]
        response = self.client.responses.create(**kwargs)

        text = response.output_text or ""
        tool_calls: list[ToolCall] = []
        raw_content: list[dict] = []

        for item in response.output:
            if getattr(item, "type", None) == "function_call":
                tool_calls.append(ToolCall(
                    id=item.call_id,
                    name=item.name,
                    input=json.loads(item.arguments),
                ))
                raw_content.append({
                    "type": "function_call",
                    "id": item.id,
                    "call_id": item.call_id,
                    "name": item.name,
                    "arguments": item.arguments,
                })

        next_action = NextAction(
            type="tool_calls" if tool_calls else "text",
            tool_calls=tool_calls,
        )
        usage = response.usage
        result = LLMResponse(
            text=text,
            usage=LLMUsage(
                input_tokens=usage.input_tokens,
                cached_tokens=usage.input_tokens_details.cached_tokens,
                output_tokens=usage.output_tokens,
                reasoning_tokens=usage.output_tokens_details.reasoning_tokens,
                total_tokens=usage.total_tokens,
            ),
            next_action=next_action,
            raw_assistant_content=raw_content,
        )
        self._log_usage(result.usage, "openai", self.model)
        return result

    def extend_messages_with_assistant_turn(
        self, messages: list, response: LLMResponse
    ) -> None:
        """Responses API: extend the flat input list with function_call items."""
        messages.extend(response.raw_assistant_content)

    def extend_messages_with_tool_results(
        self,
        messages: list,
        tool_results: list[dict],
    ) -> None:
        """Responses API: append function_call_output items to the flat input list."""
        for tr in tool_results:
            messages.append({
                "type": "function_call_output",
                "call_id": tr["tool_call_id"],
                "output": tr["result"],
            })

    def generate_structured(self, messages, schema: type[BaseModel]) -> BaseModel:
        response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=messages,
            response_format=schema,
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            raise ValueError("OpenAI returned no parsed output")
        return parsed

    async def agenerate_structured(
        self,
        messages,
        schema: type[BaseModel],
        system: str | None = None,
    ) -> tuple[BaseModel, LLMUsage]:
        all_messages = (
            [{"role": "system", "content": system}] if system else []
        ) + list(messages)
        response = await self._aclient.beta.chat.completions.parse(
            model=self.model,
            messages=all_messages,
            response_format=schema,
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            raise ValueError("OpenAI returned no parsed output")
        u = response.usage
        cached = 0
        if u.prompt_tokens_details and hasattr(u.prompt_tokens_details, "cached_tokens"):
            cached = u.prompt_tokens_details.cached_tokens or 0
        usage = LLMUsage(
            input_tokens=u.prompt_tokens,
            cached_tokens=cached,
            output_tokens=u.completion_tokens,
            total_tokens=u.total_tokens,
        )
        self._log_usage(usage, "openai", self.model)
        return parsed, usage
