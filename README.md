# Finance Sentiment Engine

Finansal haberleri LLM ile analiz eden ve günlük önem sıralamalı Markdown brief üreten pipeline. OpenAI, Anthropic ve Groq'u ortak bir interface üzerinden kullanır.

## Pipeline Özeti

```
scraper/ → cache/news.jsonl → analyzer/ → cache/analysis_*.jsonl → report/ → reports/{date}.md
```

**Üretim modeli:** `openai/gpt-4.1-mini` — urgency, sentiment ve key_event sınıflandırmasında en iyi denge.

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

## Haber Analizi

```bash
# Tek model ile analiz
python main.py --provider openai --model gpt-4.1-mini

# Tüm modelleri karşılaştır (benchmark)
python main.py --benchmark

# İlk 5 haberi test et
python main.py --limit 5
```

Sonuçlar `cache/analysis_{provider}_{model}.jsonl` dosyasına kaydedilir.

## Günlük Brief Üretimi

```bash
# Bugünün briefingi (cache'ten okur, reports/ klasörüne yazar)
python -m report.daily

# Belirli bir tarih için
python -m report.daily --date 2026-05-18

# Farklı model veya top-n
python -m report.daily --date 2026-05-18 --top-n 10 --model gpt-4.1-mini
```

Çıktı hem `stdout`'a basılır hem `reports/{date}.md` olarak kaydedilir.

### Örnek Çıktı

```markdown
# Daily Finance Brief — 2026-05-18
## Top 5 Urgent Movements

### TSLA · ⚪ neutral · medium
Tesla raised U.S. Model Y prices for the first time in two years...

[Full article](https://...)
```

## Provider Karşılaştırması (36 etiketli örnek üzerinde)

| Provider | Model | sentiment_acc | urgency_acc | key_event_acc | cost/run |
|----------|-------|:---:|:---:|:---:|:---:|
| openai | gpt-4.1-mini | 83% | 67% | 78% | $0.016 |
| openai | gpt-4.1-nano | 69% | 75% | 58% | $0.004 |
| groq | llama-3.3-70b | 56% | 50% | 44% | $0.022 |

Detaylı analiz: [`eval/results.md`](eval/results.md)

## Provider'lar

| Provider  | `generate`            | `generate_structured`         |
|-----------|-----------------------|-------------------------------|
| OpenAI    | Responses API         | `beta.chat.completions.parse` |
| Anthropic | Messages API          | Tool use + `tool_choice`      |
| Groq      | Chat Completions API  | JSON mode                     |

## Low-level Kullanım

```python
from providers.factory import get_provider

provider = get_provider("openai", "gpt-4.1-mini")
response = provider.generate([{"role": "user", "content": "Merhaba!"}])
print(response.text)
print(response.usage)  # input/output/total tokens
```
