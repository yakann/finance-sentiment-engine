# Day 29 — LangSmith Tracing: Kapsamlı Teknik Rapor

**Tarih:** 9 Haziran 2026 | **Süre:** ~1.5 saat | **Proje:** finance-sentiment-engine

---

## 1. Gün Özeti: Ne Yaptık?

Bugün projeye **LangSmith observability (gözlemlenebilirlik) katmanı** ekledik. Artık her LLM çağrısı, her tool çağrısı ve tüm agent akışları bir web dashboard üzerinden izlenebilir, filtrelenebilir ve analiz edilebilir hale geldi.

### Değiştirilen / Oluşturulan Dosyalar

| Dosya | İşlem | Amaç |
|-------|-------|-------|
| `.env` | Güncellendi | 3 LangSmith env değişkeni eklendi |
| `pyproject.toml` | Güncellendi | `langsmith>=0.3.0` explicit dependency |
| `providers/openai_provider.py` | Güncellendi | `generate()` → `@traceable(run_type="llm")` |
| `providers/anthropic_provider.py` | Güncellendi | `generate()` → `@traceable(run_type="llm")` |
| `providers/groq_provider.py` | Güncellendi | `generate()` → `@traceable(run_type="llm")` |
| `agent/loop.py` | Güncellendi | `@traceable`, `_dispatch_tool`, `metadata` param |
| `scripts/run_day29.py` | Oluşturuldu | 5 trace demosu (3 Week-3 + 2 Week-4) |
| `tests/test_graph_day29.py` | Oluşturuldu | 11 test — hepsi geçti |

---

## 2. Neden Yaptık? (Motivasyon)

### Sorun: "Kara Kutu" Sendromu

Day 17–28 boyunca çok katmanlı bir sistem inşa ettik:

- **Week-3:** Multi-turn tool-use loop (`agent/loop.py`)
- **Week-4:** LangGraph state machine (`research_subgraph → analyze_sentiment → deep_analysis → draft`)

Ama bu sistemde şu sorular yanıtsız kalıyordu:

> *"LLM tam olarak hangi prompt'u gördü?"*  
> *"`get_stock_data` tool'u kaç ms sürdü?"*  
> *"TSLA için mi NVDA için mi daha fazla token harcandı?"*  
> *"3. iterasyonda ne döndü?"*

Lokal `structlog` ve `traces/run_*.json` dosyaları bunlara kısmen cevap veriyordu — ama geri dönüp karşılaştırmak, filtrelemek veya zaman serisi çizmek mümkün değildi.

### Çözüm: Observability-as-a-Service

LangSmith bu sorunları bir web dashboard + API ile çözer. Her "run" (çalışma) kaydedilir, hiyerarşik olarak gösterilir ve metadata ile etiketlenip filtrelenebilir.

---

## 3. Öğrenilen Kavramlar

### 3.1 Observability vs Logging

| Kavram | structlog (Day 20) | LangSmith (Day 29) |
|--------|-------------------|-------------------|
| Çıktı | Terminal + JSON dosya | Web dashboard |
| Kapsam | Tek çalışma | Tüm çalışmalar (geçmiş) |
| LLM içeriği | Token sayısı | Tam prompt + completion |
| Karşılaştırma | Manuel | Filtreleme + grafik |
| Tool detayı | duration_ms, status | Input + output + latency |

> **Öğreni:** Logging "ne oldu"yu söyler; observability "neden öyle oldu"yu görselleştirir.

---

### 3.2 `@traceable` Dekoratörü

Python'da dekoratör pattern'inin pratik bir kullanımı:

```python
# providers/openai_provider.py — ÖNCE
def generate(self, messages, system=None, tools=None) -> LLMResponse:
    response = self.client.responses.create(...)
    return result

# providers/openai_provider.py — SONRA
@traceable(run_type="llm", name="openai_generate")   # ← tek satır eklendi
def generate(self, messages, system=None, tools=None) -> LLMResponse:
    response = self.client.responses.create(...)
    return result
```

`@traceable` dekoratörü şunları yapar:

1. Fonksiyon çağrılınca LangSmith'te yeni bir "run" açar
2. Fonksiyon argümanlarını (inputs) kaydeder
3. Return değerini (outputs) kaydeder
4. Latency'i (ms) ölçer
5. `LANGCHAIN_TRACING_V2=false` ise **tamamen no-op** — mevcut kodu bozmaz

---

### 3.3 Run Type Hiyerarşisi

LangSmith'te her trace bir ağaç yapısıdır. `run_type` bu ağaçta hangi kategoride gösterileceğini belirler:

```
finance_agent_run  [chain]          ← run_agent()
├── openai_generate  [llm]          ← provider.generate() — iterasyon 1
│   └── (prompt + completion içeriği)
├── _dispatch_tool  [tool]          ← get_stock_data çağrısı
│   └── (args: {ticker: "NVDA"}, output: {price: 134.5, ...})
├── openai_generate  [llm]          ← provider.generate() — iterasyon 2
└── _dispatch_tool  [tool]          ← analyze_news_sentiment çağrısı
```

Üç tip run:

- **`chain`**: Genel iş akışı (agent run, LangGraph node)
- **`llm`**: LLM API çağrısı — prompt + completion görünür
- **`tool`**: External tool çağrısı — input/output görünür

---

### 3.4 Custom Metadata

Her trace'e bağlam bilgisi ekleme:

```python
# agent/loop.py — Week-3 agent
result = run_agent(
    query="What is NVDA price?",
    provider=provider,
    registry=registry,
    metadata={
        "ticker": "NVDA",        # Hangi hisse?
        "provider": "openai",    # Hangi LLM sağlayıcı?
        "model": "gpt-4o-mini",  # Hangi model?
        "day": 29,               # Öğrenme günü
    }
)

# LangGraph graph — config içinde metadata (Week-4)
config = {
    "configurable": {"thread_id": "day29-NVDA-1234"},
    "metadata": {
        "ticker": "NVDA",
        "provider": "openai",
        "model": "gpt-4o-mini",
        "week": 4,
    }
}
graph.invoke({"ticker": "NVDA", "messages": []}, config)
```

Dashboard'da bu metadata ile filtreleme yapılabilir:

- "Tüm NVDA trace'lerini göster"
- "gpt-4o-mini vs gpt-4o latency karşılaştır"
- "Week-4'te kaç token harcandı?"

---

### 3.5 `langsmith.trace()` vs `@traceable`

| Yöntem | Ne Zaman Kullanılır | Sözdizimi |
|--------|-------------------|-----------|
| `@traceable` | Fonksiyon her zaman izlenecekse | Dekoratör — bir kez yaz |
| `langsmith.trace()` | Belirli bir kod bloğunu izlemek için | Context manager — dinamik |

```python
# @traceable — kalıcı, fonksiyon seviyesi
@traceable(run_type="chain")
def run_agent(...):
    ...

# langsmith.trace() — geçici, blok seviyesi
with langsmith.trace("my_operation", metadata={"key": "val"}):
    do_something()
```

---

## 4. Mimari Akış Diyagramları

### 4.1 LangSmith Trace Hiyerarşisi

```
LangSmith Dashboard — "finance-agent" project
│
├── [chain] finance_agent_run                      ← run_agent() çağrısı
│   │   metadata: {ticker: "NVDA", provider: "openai", model: "gpt-4o-mini"}
│   │   duration: ~3500ms | tokens: 1240
│   │
│   ├── [llm] openai_generate  (iter 1)            ← provider.generate()
│   │       input:  [{role: "user", content: "What is NVDA price?"}]
│   │       output: tool_calls: [get_stock_data, analyze_news_sentiment]
│   │       tokens: 420 | latency: 890ms
│   │
│   ├── [tool] _dispatch_tool → get_stock_data     ← yfinance çağrısı
│   │       input:  {ticker: "NVDA", period: "1mo"}
│   │       output: {price: 134.5, pct_change: 8.2, market_cap: 3.3e12}
│   │       latency: 210ms
│   │
│   ├── [tool] _dispatch_tool → analyze_news_sentiment
│   │       input:  {ticker: "NVDA", top_n: 5}
│   │       output: [{sentiment: "bullish", summary: "..."}×5]
│   │       latency: 1800ms
│   │
│   └── [llm] openai_generate  (iter 2)            ← final yanıt
│           input:  [user msg + tool results]
│           output: "NVDA is currently trading at $134.5..."
│           tokens: 820 | latency: 650ms
│
└── [chain] LangGraph: finance_graph               ← graph.invoke()
        metadata: {ticker: "TSLA", week: 4, thread_id: "day29-TSLA-xxx"}
        │
        ├── [chain] collect_news (research_subgraph)
        │   ├── [llm] openai_generate
        │   └── [tool] _dispatch_tool → analyze_news_sentiment
        │
        ├── [chain] analyze_sentiment
        ├── [chain] deep_analysis  (BULLISH yol)
        ├── [chain] fetch_price
        └── [chain] draft
```

---

### 4.2 Kod Değişikliklerinin Tam Akışı

```
BEFORE (Day 28):
┌─────────────────────────────────────────────┐
│  run_agent()                                │
│    │                                        │
│    ├── provider.generate()   [kara kutu]   │
│    │                                        │
│    ├── registry.dispatch()   [kara kutu]   │
│    │                                        │
│    └── _flush_trace() → traces/run_*.json  │  lokal dosya
└─────────────────────────────────────────────┘

AFTER (Day 29):
┌──────────────────────────────────────────────────────────┐
│  @traceable("chain") run_agent()                        │
│    │   └── metadata: {ticker, provider, model}          │  ──► LangSmith
│    │                                                     │
│    ├── @traceable("llm") provider.generate()            │  ──► LangSmith
│    │       prompt + completion görünür                   │
│    │                                                     │
│    ├── @traceable("tool") _dispatch_tool()              │  ──► LangSmith
│    │       input/output görünür                          │
│    │                                                     │
│    └── _flush_trace() → traces/run_*.json               │  lokal dosya (aynı)
└──────────────────────────────────────────────────────────┘
          │
          ▼
    LangGraph graph.invoke(config={metadata: {...}})
          │
    Her node otomatik trace edilir (LangGraph–LangSmith entegrasyonu)
```

---

### 4.3 Env Değişkeni Kontrol Akışı

```
.env                            LangSmith Davranışı
──────────────────────────────────────────────────────
LANGCHAIN_TRACING_V2=false  →  @traceable: no-op
                                (kod çalışır, veri gönderilmez)

LANGCHAIN_TRACING_V2=true
  LANGSMITH_API_KEY=boş     →  ConnectionError
                                (uyarı loglanır, trace düşer)
  LANGSMITH_API_KEY=dolu    →  Tüm trace'ler dashboard'a iletilir ✓
  LANGSMITH_PROJECT=xxx     →  Dashboard'da "xxx" projesi altında görünür
```

---

## 5. Test Sonuçları

```
tests/test_graph_day29.py — 11 test
══════════════════════════════════════════════════════
✓ test_openai_provider_generate_is_traceable
✓ test_anthropic_provider_generate_is_traceable
✓ test_groq_provider_generate_is_traceable
✓ test_traceable_import_in_providers
✓ test_run_agent_accepts_metadata_param
✓ test_dispatch_tool_is_traceable
✓ test_env_file_has_langsmith_fields
✓ test_env_langsmith_project_value
✓ test_langgraph_config_accepts_metadata
✓ test_run_agent_metadata_does_not_break_execution
✓ test_langsmith_importable

11 passed, 3 warnings — 5.76s
```

---

## 6. LangSmith'i Aktif Etme (Son Adım)

Bugün yaptıklarımız tamamen hazır; tek eksik gerçek API key:

```bash
# Adım 1: https://smith.langchain.com/settings → "Create API Key"

# Adım 2: .env dosyasını güncelle
LANGSMITH_API_KEY=lsv2_pt_xxxxxxxxxxxx   # key yapıştır
LANGCHAIN_TRACING_V2=true               # false → true yap

# Adım 3: Demo çalıştır (5 trace gönderir)
python scripts/run_day29.py

# Adım 4: Dashboard'da kontrol et
# https://smith.langchain.com → "finance-agent" projesi
```

---

## 7. Bu Günün Projeye Katkısı

| Özellik | Day 28 (Öncesi) | Day 29 (Sonrası) |
|---------|----------------|-----------------|
| LLM prompt içeriği | Görünmez | LangSmith'te tam görünür |
| Tool input/output | JSON dosyasında | Dashboard'da aranabilir |
| Multi-run karşılaştırma | Yok | Metadata filtresiyle |
| Latency grafiği | Yok | Otomatik |
| Hata analizi | Terminal logu | Dashboard'da trace ağacı |
| Token harcaması | Toplam sayı | Per-call breakdown |
| Geriye dönük analiz | `traces/*.json` elle inceleme | Dashboard'da arama |

---

## 8. Commit

```
commit 5202327
feat: langsmith tracing

- @traceable(run_type="llm") added to all 3 provider generate() methods
- run_agent() gains metadata param + @traceable(run_type="chain")
  + _dispatch_tool helper @traceable(run_type="tool")
- .env: LANGSMITH_API_KEY / LANGSMITH_PROJECT / LANGCHAIN_TRACING_V2 keys
- pyproject.toml: langsmith>=0.3.0 explicit dependency
- scripts/run_day29.py: 5 traces (3 Week-3 agent + 2 Week-4 LangGraph)
- tests/test_graph_day29.py: 11 tests — 11 passed
```

---

> **Özet:** Bugün sistemin "ne yaptığını" izlemekten "nasıl yaptığını" anlayabilir hale getirdik. LangSmith, LLM uygulamalarında observability katmanının ne olduğunu ve neden üretim sistemlerinde zorunlu olduğunu somut olarak gösterdi. `@traceable` dekoratörü tek satır eklemenin tüm iç akışı görünür kıldığını, `metadata` ise aynı kodu farklı bağlamlarda (ticker, model, hafta) etiketlemenin ne kadar kolay olduğunu öğretti.
