from providers.base import LLMProvider


def get_provider(name: str, model: str) -> LLMProvider:
    if name == "openai":
        from providers.openai_provider import OpenAIProvider
        return OpenAIProvider(model=model)
    elif name == "anthropic":
        from providers.anthropic_provider import AnthropicProvider
        return AnthropicProvider(model=model)
    elif name == "groq":
        from providers.groq_provider import GroqProvider
        return GroqProvider(model=model)
    else:
        raise ValueError(f"Unsupported provider: {name}")
