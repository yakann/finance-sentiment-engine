import logging
from pydantic import BaseModel
from providers.factory import get_provider

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

PROMPT = [{"role": "user", "content": "Say hello in exactly 5 words."}]

PROVIDERS = [
    ("openai", "gpt-4.1-mini"),
    ("anthropic", "claude-haiku-4-5-20251001"),
    ("groq", "llama-3.1-8b-instant"),
]


class Greeting(BaseModel):
    message: str
    word_count: int


def test_generate():
    for name, model in PROVIDERS:
        provider = get_provider(name, model)
        response = provider.generate(PROMPT)
        print(f"\n[{name}/{model}]\n  text: {response.text}\n  usage: {response.usage}")
        assert response.text, f"{name} returned empty text"
        assert response.usage.total_tokens > 0, f"{name} returned zero tokens"


def test_generate_structured():
    for name, model in PROVIDERS:
        provider = get_provider(name, model)
        result = provider.generate_structured(
            messages=[{"role": "user", "content": "Say 'Hello world' and count the words."}],
            schema=Greeting,
        )
        print(f"\n[{name}/{model}] structured: {result}")
        assert isinstance(result, Greeting), f"{name} did not return a Greeting"
        assert result.message, f"{name} returned empty message"


if __name__ == "__main__":
    print("=== generate ===")
    test_generate()
    print("\n=== generate_structured ===")
    test_generate_structured()
    print("\nAll providers OK.")
