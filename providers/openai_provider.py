import logging
from pydantic import BaseModel
from providers.base import LLMProvider, LLMUsage, LLMResponse
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv
from pathlib import Path

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

    def generate(self, messages, system=None) -> LLMResponse:
        response = self.client.responses.create(
            model=self.model,
            input=messages,
            instructions=system,
        )
        usage = response.usage
        result = LLMResponse(
            text=response.output_text,
            usage=LLMUsage(
                input_tokens=usage.input_tokens,
                cached_tokens=usage.input_tokens_details.cached_tokens,
                output_tokens=usage.output_tokens,
                reasoning_tokens=usage.output_tokens_details.reasoning_tokens,
                total_tokens=usage.total_tokens,
            ),
        )
        self._log_usage(result.usage, "openai", self.model)
        return result

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
