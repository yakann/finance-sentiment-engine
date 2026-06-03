import logging
import os
import random
from typing import TYPE_CHECKING
from pydantic import BaseModel
from providers.base import LLMProvider, LLMUsage, LLMResponse
from openai import OpenAI, AsyncOpenAI, RateLimitError
from tenacity import retry, stop_after_attempt, retry_if_exception_type
from dotenv import load_dotenv
from pathlib import Path

if TYPE_CHECKING:
    from agent.tools.base import Tool

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

logger = logging.getLogger(__name__)


def _groq_retry_wait(retry_state) -> float:
    """Honour Groq's retry-after header when present; otherwise random-exponential backoff."""
    exc = retry_state.outcome.exception()
    if isinstance(exc, RateLimitError) and exc.response is not None:
        header = exc.response.headers.get("retry-after")
        if header:
            try:
                return float(header) + random.uniform(0.1, 0.5)
            except (ValueError, TypeError):
                pass
    # Random-exponential with jitter to prevent thundering-herd retries
    cap = min(30.0, 2 ** retry_state.attempt_number)
    return random.uniform(1.0, cap)


class GroqProvider(LLMProvider):
    # Groq free tier: 6 K–12 K TPM; concurrency=3 keeps us well under the limit
    default_concurrency: int = 3

    def __init__(self, model: str):
        self.model = model
        self._groq_kwargs = dict(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY"),
        )
        self.client = OpenAI(**self._groq_kwargs)
        self._async_client: AsyncOpenAI | None = None

    @property
    def _aclient(self) -> AsyncOpenAI:
        if self._async_client is None:
            self._async_client = AsyncOpenAI(**self._groq_kwargs)
        return self._async_client

    def generate(
        self,
        messages,
        system=None,
        tools: list["Tool"] | None = None,
    ) -> LLMResponse:
        all_messages = []
        if system:
            all_messages.append({"role": "system", "content": system})
        all_messages.extend(messages)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=all_messages,
        )
        usage = response.usage
        result = LLMResponse(
            text=response.choices[0].message.content,
            usage=LLMUsage(
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
            ),
        )
        self._log_usage(result.usage, "groq", self.model)
        return result

    def generate_structured(self, messages, schema: type[BaseModel]) -> BaseModel:
        all_messages = [
            {
                "role": "system",
                "content": f"Respond only with valid JSON matching this schema: {schema.model_json_schema()}",
            }
        ] + list(messages)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=all_messages,
            response_format={"type": "json_object"},
        )
        return schema.model_validate_json(response.choices[0].message.content)

    @retry(
        retry=retry_if_exception_type(RateLimitError),
        wait=_groq_retry_wait,
        stop=stop_after_attempt(6),
        reraise=True,
    )
    async def agenerate_structured(
        self,
        messages,
        schema: type[BaseModel],
        system: str | None = None,
    ) -> tuple[BaseModel, LLMUsage]:
        system_content = f"Respond only with valid JSON matching this schema: {schema.model_json_schema()}"
        if system:
            system_content = f"{system}\n\n{system_content}"
        all_messages = [{"role": "system", "content": system_content}] + list(messages)
        response = await self._aclient.chat.completions.create(
            model=self.model,
            messages=all_messages,
            response_format={"type": "json_object"},
        )
        parsed = schema.model_validate_json(response.choices[0].message.content)
        u = response.usage
        usage = LLMUsage(
            input_tokens=u.prompt_tokens,
            output_tokens=u.completion_tokens,
            total_tokens=u.total_tokens,
        )
        self._log_usage(usage, "groq", self.model)
        return parsed, usage
