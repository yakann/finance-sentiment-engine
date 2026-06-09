# Day 19 — RAG as Tool: 10-K Agent Entegrasyonu

## Ne Yaptık?

Hafta 2'de (Day 14) kurulan LangChain RAG pipeline'ını, agent'ın kendi kararıyla çağırabileceği bir **tool'a dönüştürdük**.

Önceden RAG pipeline standalone'du — sadece direkt çağırılabiliyordu. Bugünden itibaren agent, kullanıcı sorusuna bakarak "bunu resmi 10-K belgesinden sormam lazım" diyip `query_10k` tool'unu **kendi kararıyla** tetikleyebiliyor.

---

## Önceki Günle (Day 18) Karşılaştırma

| | Day 18 | Day 19 |
|---|---|---|
| Eklenen tool | `analyze_news_sentiment` | `query_10k` |
| Veri kaynağı | Yahoo Finance RSS (güncel haberler) | SEC 10-K yıllık raporu |
| Cevaplanan soru türü | "Piyasa bu hisse hakkında ne düşünüyor?" | "Şirketin strateji/risk/gelir planı nedir?" |
| Veri karakteri | Anlık, değişken | Statik, resmi, kaynak gösterilebilir |
| Agent tool sayısı | 3 | **4** |

---

## query_10k Tool'u Nasıl Çalışır?

```
query_10k(ticker="NVDA", question="autonomous vehicle strategy")
         │
         ▼
1. Qdrant'ta soruya en yakın 10 chunk'ı vektör aramasıyla bulur
   (OpenAI text-embedding-3-small, cosine similarity)
         │
         ▼
2. Cohere reranker bu 10 chunk'ı cross-encoder ile sıralar
   → en alakalı 5 chunk kalır
         │
         ▼
3. gpt-4o-mini bu 5 chunk'ı context olarak alır, soruyu cevaplar
         │
         ▼
4. {answer: "...", sources: [{section, snippet}, ...]} döner
```

Sadece cevap değil, **hangi 10-K bölümünden** geldiği de döner:
`Item 1 - Business`, `Item 1A - Risk Factors`, `Item 7 - MD&A` vb.

---

## Sentiment Tool vs RAG Tool (Teknik Fark)

| | `analyze_news_sentiment` | `query_10k` |
|---|---|---|
| Veri çekme | Her çağrıda live RSS | Qdrant'ta önceden index'li |
| İlk çalışma | Direkt | Embedding yapılır (1 kez) |
| Sonraki çağrılar | Her seferinde RSS + LLM | Sadece vektör arama → hızlı |
| Deterministik? | Hayır (haberler değişir) | Evet (aynı 10-K, aynı vektörler) |
| Kaynak gösterimi | Haber başlığı | 10-K bölüm adı + metin snippet |

---

## 4-Tool Agent — Büyük Resim

```
Kullanıcı sorusu
       │
       ▼
  Agent (LLM)
       │
  ┌────┴──────────────────────────────────────┐
  │              │              │             │
web_search  get_stock_data  analyze_news   query_10k
                             _sentiment
(güncel      (fiyat/hacim/   (piyasa       (resmi SEC
 haberler)    market cap)     duygusu)      belgesi)
```

Dört tool birlikte bir "finansal araştırma asistanı"nın temelini oluşturuyor:
- **Anlık piyasa verisi** → `get_stock_data`
- **Güncel haberler** → `web_search`
- **Piyasa duygusu** → `analyze_news_sentiment`
- **Resmi belge analizi** → `query_10k`

---

## Dosya Yapısı

```
agent/
└── tools/
    ├── base.py          # Tool dataclass, provider format adapters
    ├── builtins.py      # get_current_time (dummy)
    ├── finance.py       # get_stock_data (yfinance)
    ├── search.py        # web_search (Tavily)
    ├── sentiment.py     # analyze_news_sentiment (Week-1 analyzer)
    └── rag.py           # query_10k (Week-2 LangChain RAG) ← DAY 19
```

---

## Desteklenen Ticker'lar

Şu anda `NVDA` destekleniyor (`nvda_10k_lc` Qdrant koleksiyonu, 2025 10-K).

Yeni ticker eklemek için `agent/tools/rag.py` içindeki iki dict'e satır eklenir:

```python
_TICKER_COLLECTIONS = {
    "NVDA": "nvda_10k_lc",
    "MSFT": "msft_10k_lc",   # örnek
}

_TICKER_DATA = {
    "NVDA": Path(...) / "data" / "nvda_10k_2025.json",
    "MSFT": Path(...) / "data" / "msft_10k_2025.json",   # örnek
}
```

Koleksiyon yoksa `_get_or_build_vectorstore()` otomatik index'ler.

---

## Test

```bash
python3 test_tools_day19.py
```

Üç test çalışır:
1. `query_10k` direkt dispatch — NVDA otonom araç stratejisi
2. Desteklenmeyen ticker (TSLA) — graceful error dict
3. 4-tool agent loop — agent'ın `query_10k`'ı kendi seçmesi
