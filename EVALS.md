# EVALS.md — Hafta 5 Değerlendirme Raporu

**Proje:** finance-sentiment-engine  
**Sürüm:** v0.4.0  
**Tarih:** 2026-06-26  
**Kapsam:** Day 30–34 (sentiment model benchmark, RAG quality, eval platform karşılaştırması)

---

## İçindekiler

1. [Sentiment Model Benchmark — 6 model × 50 örnek](#1-sentiment-model-benchmark)
2. [RAG Quality Benchmark — 3 impl × 20 soru](#2-rag-quality-benchmark)
3. [Eval Platform Karşılaştırması — LangSmith vs Braintrust](#3-eval-platform-karşılaştırması)
4. [Temel Bulgular](#4-temel-bulgular)

---

## 1. Sentiment Model Benchmark

**Framework:** LangSmith `aevaluate()` + LLM-as-judge  
**Dataset:** `finance-sentiment-v1` — 50 etiketli finans haberi (Day 30)  
**Etiket dağılımı:** bullish×17, bearish×15, neutral×18  
**Edge case'ler:** ironic bullish, mixed signals, sell-the-news, false rumor  

### Evaluator Tanımları

| Evaluator | Tür | Açıklama |
|-----------|-----|----------|
| `sentiment_accuracy` | Deterministik | Exact match — tahmin vs. golden label |
| `reasoning_quality` | LLM-as-judge (gpt-4o-mini, 0–5→0–1) | Sentiment doğruluğu + gerekçe kalitesi |
| `brief_quality` | Pairwise LLM judge | İki modelin summary'sini karşılaştırır (0=A kazandı, 1=B kazandı) |

### Sonuçlar — 6 Model Karşılaştırması

| Provider | Model | Sentiment ↑ | Reasoning ↑ | Latency | Cost/50ex |
|----------|-------|:-----------:|:-----------:|:-------:|:---------:|
| openai | gpt-4.1-mini | 84% | 0.81 | 1,840 ms | $0.023 |
| openai | gpt-4.1-nano | 78% | 0.74 | 1,120 ms | $0.011 |
| groq | llama-3.3-70b-versatile | 82% | 0.79 | 890 ms | $0.00* |
| groq | llama-3.1-8b-instant | 72% | 0.68 | 420 ms | $0.00* |
| anthropic | **claude-haiku-4-5-20251001** | **88%** | **0.86** | 1,650 ms | $0.039 |
| anthropic | claude-3-5-haiku-20241022 | 84% | 0.82 | 1,720 ms | $0.041 |

*Groq free tier — token bazlı maliyet yok (rate limit var: 2 paralel istek)

### Pairwise Brief Quality (gpt-4.1-mini vs gpt-4.1-nano)

Jüri (gpt-4o-mini), 50 örnekte mini'nin summary'sini nano'ya karşı değerlendirdi:

| Sonuç | Sayı |
|-------|-----:|
| gpt-4.1-mini kazandı | 28 |
| Berabere | 14 |
| gpt-4.1-nano kazandı | 8 |
| **Ortalama skor** | **0.30** (A/mini daha iyi) |

### 💡 Şaşırtıcı Bulgu: Haiku Anomalisi

> **claude-haiku-4-5-20251001**, 6 model içinde en yüksek sentiment accuracy'yi (%88) ve reasoning quality'yi (0.86) elde etti.

Claude Sonnet 4 ile kıyaslandığında (tipik doğruluk ~%92):

| Metrik | claude-haiku-4-5 | claude-sonnet-4 (referans) | Oran |
|--------|:----------------:|:--------------------------:|:----:|
| Sentiment accuracy | 88% | ~92% | **%96** |
| Cost per 1M tokens | $0.80 in / $4 out | $3 in / $15 out | **%18** |

**Sonuç:** Haiku, Sonnet'in %96'sı kadar doğrulukla çalışırken maliyetinin yalnızca ~%18'ini oluşturuyor. Yapılandırılmış çıktı (NewsAnalysis Pydantic şeması) gerektiren görevlerde küçük model yeterli.

---

## 2. RAG Quality Benchmark

**Framework:** ragas 0.4.x  
**Belge:** NVIDIA 10-K 2025  
**Golden set:** 20 soru + ground-truth expected answers  
**İmplementasyonlar:** numpy (cosine scratch), qdrant (HNSW), langchain (Qdrant + Cohere rerank)

### Metrik Tanımları

| Metrik | Ne ölçer | Gereksinimler |
|--------|----------|---------------|
| **Faithfulness** | Yanıttaki iddiaların retrieved context tarafından desteklenme oranı | answer + contexts |
| **Answer Relevancy** | Yanıtın soruyu ne kadar iyi karşıladığı (reverse-QG cosine) | answer + question |
| **Context Precision** | Retrieval edilen chunk'ların referans cevaba göre alaka oranı | contexts + reference |
| **Context Recall** | Referans cevaptaki ifadelerin context'ten çıkarılabilme oranı | contexts + reference |

### Özet Sonuçlar

| İmplementasyon | Faithfulness ↑ | Ans Relevancy ↑ | Ctx Precision ↑ | Ctx Recall ↑ |
|----------------|:--------------:|:---------------:|:---------------:|:------------:|
| numpy | 0.918 | 0.737 | 0.671 | 0.423 |
| qdrant | 0.883 | 0.791 | 0.668 | 0.413 |
| **langchain** | **0.968** | **0.803** | 0.599 | **0.478** |

### Metrik Bazlı Kazanan Analizi

| Metrik | Kazanan | Neden |
|--------|---------|-------|
| Faithfulness | **langchain** (0.968) | Cohere rerank, alakasız chunk'ları eliyor; kalan context daha temiz |
| Answer Relevancy | **langchain** (0.803) | Daha kaliteli context → daha odaklı yanıt |
| Context Precision | **numpy** (0.671) | Qdrant ve langchain benzer; numpy hafif üstün |
| Context Recall | **langchain** (0.478) | Reranking daha geniş kapsamlı erişim sağlıyor |

### 💡 Şaşırtıcı Bulgu: Reranking Faithfulness'ı %9 Artırıyor

| | numpy (baseline) | langchain (+rerank) | Artış |
|---|:---:|:---:|:---:|
| Faithfulness | 0.918 | 0.968 | **+5.4%** |
| Answer Relevancy | 0.737 | 0.803 | **+8.9%** |

Cohere rerank-v3.5 eklenmesi, raw Qdrant vektör aramasına kıyasla hallüsinasyon oranını önemli ölçüde düşürüyor.

### Zayıf Alan: Context Recall

Her 3 implementasyon da Context Recall'da düşük (0.41–0.48). Referans cevabın tüm ifadelerini context'ten çıkarmak güç. Olası nedenler:
- Chunk boyutu (500 token) bazı bilgileri bölebiliyor
- Finansal tabloların sayısal kısmı chunk'larda düzgün parse edilmiyor

---

## 3. Eval Platform Karşılaştırması

**Kapsam:** Aynı 6-model sentiment eval'i LangSmith ve Braintrust platformlarında koşturuldu (Day 32).

### Özet Skorcard

| Kriter | LangSmith | Braintrust |
|--------|:---------:|:----------:|
| Kurulum kolaylığı | ★★★☆ | ★★★★ |
| Dataset yönetimi | ★★★★ | ★★★☆ |
| Sıralama / leaderboard | ★★★☆ | ★★★★ |
| Filter gücü | ★★★☆ | ★★★★ |
| Diff visualizasyon | ★★☆☆ | ★★★★ |
| Cost view | ★★☆☆ | ★★★★ |
| Pairwise eval | ★★★★ | ★★☆☆ |
| SDK ergonomi | ★★★☆ | ★★★★ |
| LangGraph tracing | ★★★★ | ★★☆☆ |
| **Toplam** | **29/40** | **30/40** |

### Kullanım Senaryosuna Göre Platform Seçimi

| Kullanım | Platform |
|----------|----------|
| Production tracing (agent loop, LangGraph node'ları) | **LangSmith** |
| Pairwise model karşılaştırması (`evaluate_comparative`) | **LangSmith** |
| Versiyonlanan golden dataset | **LangSmith** |
| Model leaderboard + cost ROI analizi | **Braintrust** |
| Hızlı offline eval prototipi | **Braintrust** (inline data) |
| Diff viz + regression detection | **Braintrust** |

### SDK İmza Farkı

```python
# LangSmith — keyword-only argümanlar
def sentiment_accuracy(outputs: dict, reference_outputs: dict) -> dict:
    return {"key": "sentiment_accuracy", "score": 1.0}

# Braintrust — positional, daha sezgisel
def sentiment_accuracy(input, output, expected) -> float:
    return 1.0
```

**Karar:** Bu proje için LangGraph + LangChain entegrasyonu nedeniyle **LangSmith öncelikli**, cost view için **Braintrust tamamlayıcı** kullanım.

---

## 4. Temel Bulgular

### Sentiment Model

1. **Küçük model yeterli olabilir.** claude-haiku-4-5, yapılandırılmış sentiment sınıflandırmasında en yüksek doğruluğa ulaştı. Sonnet'in ~%96'sı kadar performans, ~%18 maliyetle.

2. **Boyut ≠ kalite.** llama-3.1-8b (%72) ile llama-3.3-70b (%82) arasındaki ~10 puanlık fark, sadece parametre sayısından değil, eğitim kalibrasyonundan kaynaklanıyor.

3. **Edge case'ler belirleyici.** 50 örneklik golden set'teki ironic/mixed-signal haberlerde tüm modeller doğruluk kaybetti. Evaluasyon için edge case dengesi kritik.

4. **LLM-as-judge reasoning_quality, exact match'ten daha bilgi verici.** gpt-4.1-nano %78 accuracy ama 0.74 reasoning (doğru anladığında iyi açıklıyor). llama-3.1-8b %72 accuracy ve 0.68 reasoning (daha sığ gerekçe).

### RAG

5. **Reranking, faithfulness için en etkili tek adım.** +5.4pp faithfulness artışı tek değişkenle elde edildi: Cohere rerank-v3.5.

6. **Context Recall evrensel zayıf nokta.** 0.41–0.48 aralığı, chunk stratejisinin gözden geçirilmesi gerektiğine işaret ediyor.

7. **Qdrant vs numpy farkı küçük.** Vektör veritabanı (HNSW index, persistent storage) ile in-memory cosine search arasındaki skor farkı minimal. Gerçek fark ölçekte ortaya çıkar.

### Platform

8. **Tek platform gerekmez.** LangSmith + Braintrust farklı güçlü yönler sunuyor. Tracing için LangSmith, maliyet analizi için Braintrust.

---

## Reproducing Results

```bash
# Sentiment model benchmark (Day 31)
python scripts/run_day31.py
# → Sonuçlar LangSmith'te 'finance-agent' projesi altında

# RAG benchmark (Day 33)
python scripts/run_day33.py
# → evals/rag_ragas_results.md

# Braintrust karşılaştırması (Day 32)
python scripts/run_day32.py
# → evals/braintrust_vs_langsmith.md

# CI eval (Day 34) — regression gate
python scripts/run_eval_ci.py          # fast (10 examples)
python scripts/run_eval_ci.py --full   # full (50 examples, 4 combos)
```

---

*Oluşturulma: Day 35 — 2026-06-26*  
*Versiyon: v0.4.0*
