# Finance Sentiment Engine

LLM provider abstraction layer — OpenAI, Anthropic ve Groq'u ortak bir interface üzerinden kullanır.

## Quickstart

```bash
# Bağımlılıkları yükle
uv sync

# .env dosyasına API key'lerini ekle
cp .env.example .env   # veya doğrudan .env düzenle
# OPENAI_API_KEY=...
# ANTHROPIC_API_KEY=...
# GROQ_API_KEY=...

# Tüm provider'ları test et
python tests/test_providers.py
```

## Kullanım

```python
from providers.factory import get_provider

provider = get_provider("openai", "gpt-4.1-mini")
response = provider.generate([{"role": "user", "content": "Merhaba!"}])
print(response.text)
print(response.usage)  # input/output/total tokens
```

### Structured output

```python
from pydantic import BaseModel
from providers.factory import get_provider

class Sentiment(BaseModel):
    label: str   # positive / negative / neutral
    score: float # 0.0 – 1.0

provider = get_provider("anthropic", "claude-haiku-4-5-20251001")
result = provider.generate_structured(
    messages=[{"role": "user", "content": "Apple stock surged 10% today."}],
    schema=Sentiment,
)
print(result.label, result.score)
```

## Provider'lar

| Provider  | `generate`            | `generate_structured`         |
|-----------|-----------------------|-------------------------------|
| OpenAI    | Responses API         | `beta.chat.completions.parse` |
| Anthropic | Messages API          | Tool use + `tool_choice`      |
| Groq      | Chat Completions API  | JSON mode                     |
