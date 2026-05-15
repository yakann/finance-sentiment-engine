import logging
from pydantic import BaseModel
from providers.base import LLMProvider, LLMUsage, LLMResponse
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    def __init__(self, model: str):
        self.model = model
        self.client = OpenAI()

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
