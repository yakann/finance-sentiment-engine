import logging
import os
from pydantic import BaseModel
from providers.base import LLMProvider, LLMUsage, LLMResponse
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

logger = logging.getLogger(__name__)


class GroqProvider(LLMProvider):
    def __init__(self, model: str):
        self.model = model
        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY"),
        )

    def generate(self, messages, system=None) -> LLMResponse:
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
