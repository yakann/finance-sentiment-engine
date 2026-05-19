import logging
from abc import ABC, abstractmethod
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class LLMUsage(BaseModel):
    input_tokens: int
    cached_tokens: int = 0
    output_tokens: int
    reasoning_tokens: int = 0
    total_tokens: int


class LLMResponse(BaseModel):
    text: str
    usage: LLMUsage


class LLMProvider(ABC):
    # Subclasses can override to tune per-provider safe concurrency
    default_concurrency: int = 10

    @abstractmethod
    def generate(self, messages, system=None) -> LLMResponse:
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

    def _log_usage(self, usage: LLMUsage, provider: str, model: str) -> None:
        logger.info(
            "%s/%s — input: %d, cached: %d, output: %d, total: %d tokens",
            provider, model,
            usage.input_tokens, usage.cached_tokens,
            usage.output_tokens, usage.total_tokens,
        )
