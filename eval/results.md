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

## Yorum

`gpt-4.1-mini` sentiment ve key_event sınıflandırmasında önde giderken, urgency'de `gpt-4.1-nano` onu geçiyor (75% vs 67%) — üstelik dörtte biri kadar maliyetle. Bu, urgency'nin rubric net tanımlandığında küçük bir modelin rahatlıkla çözebildiği, görece basit bir sinyal olduğunu gösteriyor. Groq modelleri sentiment'te ~50% civarında takılı kalıyor ve key_event'te belirgin biçimde geride (44% / 39%); uzun bir system prompt altında kısıtlı enum'dan seçim yaptırıldığında açık ağırlıklı modellerin instruction-following kalitesinin düştüğü görülüyor. Ücretsiz katmandaki token limitleri (12K/6K TPM) de batch eval için pratik kullanımı zorlaştırıyor. **Üretim için net seçim `gpt-4.1-nano`**: en iyi urgency doğruluğu, mini'den 4× ucuz, gerçek zamanlı kullanım için yeterince hızlı. Kalan doğruluk açığı (sentiment 69%, key_event 58%) model zayıflığından çok etiket belirsizliğine işaret ediyor — 15 anlaşmazlık vakasına ikinci bir insan bakışı tüm modellerin tavanını yükseltir.
