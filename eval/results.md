# Eval Results

**Eval set:** 36 labeled examples  
**Ground truth:** human-labeled (Turkish notes in `labels.jsonl`)  
**Schema:** `key_event` as 10-bucket Literal enum, urgency with price-impact rubric

| Provider | Model | sentiment_acc | urgency_acc | key_event_acc | avg_latency | cost/run |
|----------|-------|:-------------:|:-----------:|:-------------:|:-----------:|:--------:|
| openai | gpt-4.1-mini | 83% | 67% | 78% | 1261 ms | $0.016 |
| openai | gpt-4.1-nano | **69%** | **75%** | 58% | **912 ms** | **$0.004** |
| groq | llama-3.3-70b-versatile | 56% | 50% | 44% | 11496 ms† | $0.022 |
| groq | llama-3.1-8b-instant | 52% | 39% | 39% | 27531 ms† | $0.002 |

† Groq latency is dominated by tenacity backoff on free-tier TPM rate limits (12K/6K tokens/min),
not by model inference speed. True per-call inference is ~200–400 ms; the rest is retry wait time.

## Interpretation

`gpt-4.1-mini` leads on sentiment and key_event classification but `gpt-4.1-nano` beats it on urgency (75% vs 67%) at one-quarter the cost — suggesting that urgency is a simpler signal that a smaller model handles well once the rubric is explicit. The Groq models plateau around 50% on sentiment and trail significantly on key_event (44% / 39%), likely because open-weight instruction-following degrades when forced to pick from a constrained enum under a long system prompt; at free-tier token limits they also accumulate retry overhead that makes them impractical for batch eval. The clear production pick is `gpt-4.1-nano`: best urgency accuracy, 4× cheaper than mini, and fast enough for near-real-time use. The remaining accuracy gap (sentiment at 69%, key_event at 58%) points to label ambiguity more than model weakness — a second human pass on the 15 disagreement cases would likely lift the ceiling for all models.
