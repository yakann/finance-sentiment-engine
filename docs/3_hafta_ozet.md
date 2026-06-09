# 3 Hafta, 21 Gün — Ne Öğrendik?

**Proje:** `finance-sentiment-engine`  
**Süre:** 21 gün (Hafta 1–3)  
**Bugün:** v0.2.0 — production-ready agent sistemi

---

## Büyük Resim: Ne İnşa Ettik?

```
Başlangıç (Gün 1):           Şu An (Gün 21):
─────────────────            ──────────────────────────────────────────
"LLM API'yi nasıl            brief NVDA
 çağırırım?"          →      → 4 tool paralel çalışır
                             → 10-K RAG + haber sentimentı + fiyat verisi
                             → briefs/NVDA.md (2 sayfalık yatırım raporu)
                             → traces/run_xxx.json (tam audit trail)
```

Üç haftada **3 katmanlı bir AI sistemi** inşa ettik:

| Katman | Ne Yaptı |
|--------|----------|
| **Hafta 1** — LLM Temelleri | Ham API çağrısından structured output ve multi-provider'a |
| **Hafta 2** — RAG Sistemi | Vektör aramasından iki aşamalı reranking pipeline'ına |
| **Hafta 3** — Agent Sistemi | Tek tool'dan 4-tool multi-turn agent loop'una |

---

## Hafta 1 (Gün 1–7): LLM ile Structured Thinking

### Ne Öğrendik?

**Problem:** RSS'ten gelen ham haber başlığını "Bu haber NVDA için bullish mı bearish mi?" sorusuna dönüştür.

Bu soruyu çözerken şunu fark ettik: LLM'den tutarlı veri almak için ona hem format vermek hem de bir **çıktı kontratı** (schema) tanımlamak gerekiyor.

**Temel kavramlar:**

```
Ham metin → Prompt → LLM → JSON → Pydantic → Typed Python nesnesi
                              ↑
                    Her provider bunu farklı yapıyor!
```

- **OpenAI:** `beta.chat.completions.parse()` → Pydantic'e doğrudan parse eder
- **Anthropic:** Tool call mekanizması → `input` alanından çıkarılır
- **Groq:** `response_format={"type":"json_object"}` → manuel `model_validate()`

**Neden aynı promptu 3 provider'a gönderdik?**  
Çünkü "LLM çalışıyor" = "API cevap verdi" değildir. Sentiment accuracy, urgency accuracy, key_event accuracy → bunlar ölçülmeden model seçimi tahmindir.

**Öğrenilen acı ders:**  
Groq'un free tier'ında 12K TPM limiti var. Basit bir batch işlemi 400ms'lik bir modeli 27 saniyeye çıkardı. Çözüm: `tenacity` ile exponential backoff + per-provider concurrency cap.

### Kullanılan Teknolojiler

| Teknoloji | Ne İçin |
|-----------|---------|
| `openai` SDK | Responses API, structured output |
| `anthropic` SDK | Messages API, tool-use tabanlı JSON çıkarma |
| `groq` SDK | Llama 3.x, JSON mode |
| `pydantic` | Output schema doğrulama |
| `feedparser` | Yahoo Finance RSS parse |
| `tenacity` | Rate limit backoff |
| `asyncio` | Paralel batch analiz |

### Hafta 1 Sonunda Sistem Ne Yapabiliyordu?

```bash
python main.py --provider openai --model gpt-4.1-nano
# → cache/analysis_openai_gpt-4.1-nano.jsonl
# → Her haber için: sentiment + urgency + key_event + summary

python -m report.daily
# → reports/2026-05-18.md (öncelik sıralı Markdown brief)
```

---

## Hafta 2 (Gün 8–14): RAG — Dış Belgeyi LLM'e Bağlama

### Ne Öğrendik?

**Problem:** NVDA'nın 10-K dosyasındaki risk faktörlerini sormak istiyoruz. Belge 200 sayfa. Tamamını context'e atamayız.

**Çözüm:** RAG (Retrieval-Augmented Generation)

```
10-K PDF
    │
    ▼ chunk (2000 karakter parçalar)
[Chunk 1][Chunk 2]...[Chunk N]
    │
    ▼ embed (her chunk → 1536 boyutlu vektör)
[0.12, -0.33, ...]  [0.88, 0.21, ...]  ...
    │
    ▼ store
Qdrant (vektör veritabanı)
    │
    ▼ query: "What are NVDA's AI risks?"
cosine similarity → top-10 chunk
    │
    ▼ rerank (Cohere cross-encoder)
top-5 chunk (daha hassas)
    │
    ▼ LLM
"NVDA'nın AI riskleri şunlardır: ..."
```

**Neden sıfırdan yazdık önce?**  
Çünkü `rag_numpy.py`'ı sıfırdan yazmadan LangChain'in ne yaptığını bilemezdik. Sıfırdan yazınca şunu anladık:

- `QdrantVectorStore.from_documents()` = bizim `upsert()` loop'umuz
- `ContextualCompressionRetriever` = bizim `rerank_cohere()` fonksiyonumuz
- `RecursiveCharacterTextSplitter` = bizim sliding window chunk mantığımız (ama token değil char bazlı — bu bir precision kaybı)

**İki aşamalı retrieval neden önemli?**

| Aşama | Yöntem | Hız | Kalite |
|-------|--------|-----|--------|
| 1. Aşama | Cosine similarity (ANN) | ⚡ çok hızlı | iyi (geometrik benzerlik) |
| 2. Aşama | Cohere cross-encoder | yavaş | mükemmel (derin anlam) |

Trick: İlk aşamada 10 chunk al (hız), ikinci aşamada 5'e indir (kalite). Sadece küçük sete cross-encoder uygulayınca maliyet kabul edilebilir olur.

**LangChain dersi:**  
56% daha az kod. Ama: rerank skorları gizleniyor, chunk boundary'leri farklı, her adımı debug edemiyorsun. **Kendi yazdıktan sonra framework'ü kullanmak** — doğru sıra bu.

### Evaluation (Gün 14)

Sadece "çalışıyor" demek yetmez. **Recall@5 + LLM-as-judge** ile ölçtük:

| Implementasyon | Recall@5 | Faithfulness | Answer Relevance |
|----------------|----------|--------------|------------------|
| NumPy (sıfırdan) | 0.627 | 0.300 | 0.833 |
| Qdrant | 0.627 | 0.367 | **0.900** |
| LangChain | 0.560 | **0.500** | 0.800 |

Sonuç: Cohere reranking faithfulness'ı artırıyor (LLM context'e daha sadık). Ama recall düşüyor (char-based chunking).

### Kullanılan Teknolojiler

| Teknoloji | Ne İçin |
|-----------|---------|
| `qdrant-client` | Vektör veritabanı (HNSW indexing) |
| `openai` embeddings | `text-embedding-3-small` (1536 dim) |
| `tiktoken` | Token-accurate chunking |
| `cohere` | Cross-encoder reranking |
| `langchain` + LCEL | RAG pipeline orchestration |
| `langchain-qdrant` | QdrantVectorStore entegrasyonu |
| `numpy` | Cosine similarity (sıfırdan) |
| `sentence-transformers` | BGE local reranker |

### Hafta 2 Sonunda Sistem Ne Yapabiliyordu?

```bash
# Qdrant başlat
docker run -p 6333:6333 qdrant/qdrant

# NVDA 10-K'yı indexle ve sorgula
python rag_langchain.py
# Soru: "What is NVIDIA's strategy in autonomous vehicles?"
# → 5 reranked chunk + LLM yanıtı + kaynaklar

# Tüm implementasyonları değerlendir
python eval/rag_eval.py
# → 15 sorgu × 3 implementasyon = 45 satır sonuç
```

---

## Hafta 3 (Gün 15–21): Agent Sistemi — LLM'e Araç Kutusu Ver

### Ne Öğrendik?

**Problem:** RAG ve sentiment analizi ayrı ayrı çalışıyor. Bir kullanıcı hem NVDA'nın fiyatını hem haberlerini hem de 10-K risklerini sormak isterse ne olur?

**Çözüm:** Tool-use agent — LLM'e "hangi tool'u ne zaman çağıracağına" karar verme yetkisi ver.

```
Kullanıcı sorusu
      │
      ▼
   LLM (orchestrator)
   "Bu soruyu cevaplamak için get_stock_data ve query_10k lazım"
      │
      ├──→ get_stock_data("NVDA")    → {price: 205, market_cap: 4.97T, ...}
      ├──→ analyze_news_sentiment    → [{bullish, ...}, {neutral, ...}, ...]
      ├──→ web_search(...)           → [{url, snippet, ...}, ...]
      └──→ query_10k("NVDA", "...")  → {answer: "...", sources: [...]}
            │
            ▼
         LLM (synthesizer)
         "Tüm bu verilerden birleşik bir yanıt üreteyim"
            │
            ▼
      Structured Markdown brief
```

**Önemli kararlar:**

**1. Tool abstraction (Gün 15)**  
Her tool aynı interface'i implement eder: `input_schema (Pydantic) + handler`. Provider'lar bunu kendi formatlarına (`to_openai_format()`, `to_anthropic_format()`) dönüştürür. Yeni tool yazmak = sadece bir Pydantic class + bir fonksiyon.

**2. Multi-turn loop (Gün 17)**  
LLM tek iterasyonda cevap vermeyebilir. Loop: LLM → tool call → sonuç → LLM → ... → text yanıt. Mesaj geçmişi büyür, provider bazında farklı formatlarda tutulur.

**3. Error recovery (Gün 20)**  
Tool exception fırlattığında sistemi çökertme — hatayı structured JSON olarak LLM'e gönder. LLM "bu tool başarısız, farklı input dene veya atla" yapabilir.

**4. Token budget + tracing (Gün 20)**  
100k token sonrası dur (maliyet kontrolü). Her çalışma için `traces/run_xxx.json` yaz (audit trail, debug için).

**5. Brief CLI (Gün 21)**  
Tüm sistemi tek komutla çalıştır: `brief NVDA`. Desteklenmeyen ticker için graceful fallback: `query_10k` hata verince LLM otomatik `web_search`'e geçer.

### Kullanılan Teknolojiler

| Teknoloji | Ne İçin |
|-----------|---------|
| `yfinance` | Gerçek zamanlı hisse fiyatı ve piyasa verisi |
| `tavily-python` | Web arama API (structured search results) |
| `structlog` | JSON formatında yapılandırılmış loglama |
| `uuid` | Her agent çalışması için benzersiz run_id |
| `argparse` | `brief <TICKER>` CLI arayüzü |

### Hafta 3 Sonunda Sistem Ne Yapabiliyordu?

```bash
# Tek komutla 2 sayfalık yatırım raporu
brief NVDA          # → briefs/NVDA.md
brief TSLA --print  # → briefs/TSLA.md + stdout
brief MSFT          # → briefs/MSFT.md

# Her brief'te:
# 1. Company Snapshot  (fiyat, piyasa değeri, 1-aylık getiri)
# 2. Recent News       (5 haber, emoji sentiment, genel yorum)
# 3. Key Risk Factors  (10-K RAG veya web fallback ile 5 risk)
# 4. Analyst Verdict   (Buy/Hold/Watch + tek cümle özet)
# 5. Sources           (tüm kaynaklar)
```

---

## Teknoloji Haritası: 21 Günde Ne Öğrendik?

```
┌─────────────────────────────────────────────────────────────┐
│                    LLM PROVIDERS                            │
│  OpenAI (Responses API)  │  Anthropic  │  Groq (Llama 3.x) │
│  Structured output       │  Tool use   │  JSON mode         │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                    AGENT LAYER                              │
│  ToolRegistry  │  run_agent() loop  │  Error recovery       │
│  4 tools       │  Multi-turn        │  Token budget         │
│  Brief CLI     │  Parallel calls    │  Structured tracing   │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                    RAG PIPELINE                             │
│  Chunking (tiktoken)  │  Embeddings (ada-3-small)           │
│  Qdrant (HNSW)        │  Cohere Rerank (cross-encoder)      │
│  LangChain / LCEL     │  Recall@5 + LLM-as-judge eval       │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                    DATA LAYER                               │
│  Yahoo Finance RSS  │  yfinance  │  Tavily  │  SEC EDGAR    │
│  feedparser         │  OHLCV     │  Web     │  10-K JSON    │
└─────────────────────────────────────────────────────────────┘
```

---

## Öğrenilen Kalıcı Dersler

**1. "Çalışıyor" ≠ "İyi çalışıyor"**  
Her şeyi ölçtük: sentiment accuracy, Recall@5, faithfulness, latency, token cost. Ölçmeden optimize edemezsin.

**2. Soyutlamayı kullanmadan önce altını anla**  
`rag_numpy.py` → `rag_qdrant.py` → `rag_langchain.py` sırası kasıtlıydı. LangChain'in ne yaptığını biliyoruz çünkü önce elle yazdık. Framework kullananlar debug'da çaresiz kalır; altyapıyı bilenler çözer.

**3. Provider abstraction kritik**  
`LLMProvider` interface'i sayesinde OpenAI → Anthropic → Groq geçişi tek satır. Olmadan her değişiklik N yerde değişiklik gerektirir.

**4. Hata mesajını kullanıcıya değil, LLM'e ver**  
Tool exception'ı fırlatmak yerine structured JSON olarak LLM'e dönmek, sistemi kırılgan değil dirençli yapar. LLM kendisi recovery stratejisi üretir.

**5. Token = para**  
100k token budget guard yazmak üretim sistemlerde zorunlu. Sonsuz loop yazan bir agent API faturanızı patlatabilir.

---

## Şu An Sistem Neler Yapabiliyor?

| Yetenek | Komut / Kod |
|---------|------------|
| Haber sentiment analizi (3 provider) | `python main.py --provider openai` |
| Günlük Markdown brief (tüm haberler) | `python -m report.daily` |
| 10-K RAG sorgusu (NVDA) | `python rag_langchain.py` |
| RAG kalite değerlendirmesi | `python eval/rag_eval.py` |
| 4-tool agent sorusu | `run_agent(query, provider, registry)` |
| 2 sayfalık yatırım raporu | `brief NVDA` |
| Yapılandırılmış trace | `traces/run_xxx.json` (otomatik) |

---

## 4. Hafta İçin Zemin Hazır

21 günde inşa ettiğimiz sistem şu problemleri çözmeye hazır:

- **Streaming:** Kullanıcı cevabı beklerken token token görmek isterse?
- **Memory:** Agent önceki soruları hatırlamalı mı?
- **Multi-agent:** Bir agent diğer agent'ı tool olarak çağırabilir mi?
- **Evaluation:** Agent'ın "iyi iş çıkardığını" nasıl ölçeriz?
- **Deployment:** Bu sistemi bir API olarak nasıl sunarız?

Bunların hepsi, var olan `agent/loop.py` + `agent/tools/` + `providers/` altyapısı üzerine inşa edilebilir.
