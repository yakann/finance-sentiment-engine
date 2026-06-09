# Day 22 — LangGraph: Nedir, Ne Çözer?

## LangGraph Nedir?

`agent/loop.py`'de şöyle bir döngü vardı:

```
LLM → tool çağır → sonucu al → LLM → tool çağır → ... → cevap ver
```

Bu döngüyü elle kontrol ettik. "Kaç iterasyon?", "Ne zaman dur?", "Hangi tool önce?" — bunların hepsini `while` loop + `if/else` ile yönettik.

**LangGraph** diyor ki: "Bu akışı bir **graf** olarak tanımla, ben çalıştırayım."

---

## Graf Ne Demek?

```
Düğümler (nodes)  = ne yapılacak  → Python fonksiyonu
Kenarlar (edges)  = sıra          → hangi node'dan sonra hangisi gelir
State             = verinin taşındığı çanta → her node bunu okur/yazar
```

`graph/hello.py`'de yaptığımız:

```
START → [greet] → [reply] → END

State: { messages: [...] }

greet:  messages'a "Hello! I received: ..." ekle
reply:  messages'a "Graph traversal complete..." ekle
```

Her node aynı imzayı takip etti:

```python
def greet(state: State) -> State:
    # state'i oku, yeni state döndür
```

LangGraph node'ları sırayla çalıştırdı, state'i aralarında taşıdı.

---

## Eski Loop'tan Farkı

| | `agent/loop.py` (elle yazılan) | LangGraph |
|---|---|---|
| Akış kontrolü | `while True` + `if` | Graf kenarları |
| Durum taşıma | `messages` listesini elle büyüttük | `State` TypedDict, her node günceller |
| Dal (branching) | `if action == "tool_call"` | `add_conditional_edges()` |
| Görselleştirme | Yok | `draw_mermaid_png()` → PNG |
| Paralel node | Zor | Yerleşik destek |

---

## Gerçek Dünya Senaryosu: "NVDA Almalı mıyım?"

### Şu Anki `brief NVDA` — Düz Çizgi

```
soru → LLM → 4 tool paralel → LLM → cevap
```

### Gerçek Bir Analist Nasıl Çalışır?

```
1. Soruyu anla       → "Kısa vadeli mi uzun vadeli mi soruyorsun?"
2. Fiyata bak        → Pahalı mı ucuz mu? (P/E ratio)
3. Haberlere bak     → Kötü haber varsa devam etmeye değer mi?
        ↓
   KÖTÜ HABER VARSA → daha derin araştır, riski ölç
   İYİ HABER VARSA  → 10-K'ya geç, uzun vadeli riski bak
        ↓
4. Rakiplerle karşılaştır → AMD, Intel nasıl?
5. Özet yaz
6. Özeti kendisi gözden geçir → "Mantıklı mı?" → hayırsa tekrar yaz
```

### Bunu `loop.py` ile Yazmaya Çalışsan

```python
result = run_agent("NVDA almalı mıyım?")

if "bad news" in result:
    result2 = run_agent("NVDA riskleri neler?")
    
if result2["risk"] == "high":
    # rakip analizi yap
    ...

review = run_agent(f"Bu özet mantıklı mı: {result['brief']}")
if "hayır" in review:
    # tekrar yaz — ama hangi adımdan?
```

Her yeni "eğer şuysa şunu yap" bir `if` bloğu daha. Kod hızla spagetti olur.

### LangGraph ile Aynı Akış

```
START
  │
  ▼
[fiyat_ve_haber]          ← paralel 2 tool
  │
  ▼
[haber_değerlendir]       ← sentiment negatif mi?
  │
  ├─ negatif ──→ [derin_risk_analizi] ──┐
  │                                      │
  └─ pozitif ──→ [10k_sorgula]    ──────┤
                                         │
                                         ▼
                                   [rakip_karşılaştır]
                                         │
                                         ▼
                                   [özet_yaz]
                                         │
                                         ▼
                                   [özet_kontrol]
                                         │
                                   ├─ yetersiz ──→ [özet_yaz] (döngü!)
                                   │
                                   └─ yeterli ──→ END
```

Bu akışı LangGraph'te tanımlamak:

```python
builder.add_conditional_edges(
    "haber_değerlendir",
    sentiment_router,          # fonksiyon: state'e bakıp karar verir
    {
        "negatif": "derin_risk_analizi",
        "pozitif": "10k_sorgula"
    }
)

builder.add_conditional_edges(
    "özet_kontrol",
    quality_check,
    {
        "yetersiz": "özet_yaz",   # geri döner!
        "yeterli": END
    }
)
```

---

## LangGraph'in Çözdüğü Problemler

| Problem | `loop.py`'de ne olur | LangGraph'te |
|---|---|---|
| "Şarta göre farklı yol" | İç içe `if/else` | `add_conditional_edges` |
| "Yanlışsa tekrar yap" | Manuel `while` + sayaç | Graf döngüsü, node'a geri edge |
| "Bu iki şeyi paralel yap" | `asyncio` elle | Paralel node desteği |
| "Nerede takıldım?" | `print` ile debug | Her node'da state snapshot |
| "Akışı birine anlat" | Kodu oku | PNG çiz, göster |

---

## Özet

`loop.py` düz bir yolda çalışır.
LangGraph **kavşakları, geri dönüşleri ve çatalları** olan bir yol haritası çizer.

Bu, Hafta 4'teki asıl hedefin (memory + multi-agent) altyapısı.
`brief NVDA` yerine "önce planla, sonra araştır, sonra sentezle, yanlışsa tekrar araştır" diyebilmek için gereken yapı bu.
