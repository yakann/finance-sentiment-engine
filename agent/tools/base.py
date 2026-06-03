from typing import Any, Callable
from pydantic import BaseModel, ConfigDict


class Tool(BaseModel):
    """Provider-agnostic tool definition. Converts to OpenAI or Anthropic wire format on demand."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    input_schema: type[BaseModel]
    handler: Callable[..., Any]

    def to_anthropic_format(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema.model_json_schema(),
        }

    def to_openai_format(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema.model_json_schema(),
            },
        }
