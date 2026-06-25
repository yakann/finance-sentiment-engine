# Test Sonuçlarında Keşfedilen Sorunları Handle Etmek

Bu belge iki soruyu yanıtlar:
1. Test sonuçlarını okuyunca **hangi sorunu** gördüğünü nasıl anlarsın?
2. O sorunu **nasıl çözersin?**

Tüm örnekler bu projeden alınmıştır (Day 14 + Day 33 RAG eval, Day 29-32 Agent eval).

---

## Genel Yaklaşım

Sorun buldun → hemen kodu değiştirme. Önce şunu sor:

```
1. Semptom nedir?     (metrik değerleri)
2. Kök neden nedir?   (neden bu değer çıktı?)
3. Kaçıncı katman?    (chunking? retrieval? LLM? agent loop?)
4. Maliyet/etki ne?   (ne kadar soru etkileniyor, kritiklik nedir?)
5. En kolay fix ne?   (önce onu dene)
```

Doğru katmanı bulmadan fix yapmak başka şeyleri bozar.

---

# BÖLÜM 1 — RAG Sorunları

## Problem 1: Answer Relevancy = 0.000

**Semptom:** Belirli sorular için AnsRel tamamen sıfır.

**Bu projeden örnek (Day 33):**
```
Q2  "What were NVIDIA's total revenues in FY2025?"
    numpy  → AnsRel=0.000, CtxPrec=0.000, CtxRec=0.000
    qdrant → AnsRel=0.000, CtxPrec=0.000, CtxRec=0.000

Q7  "What is NVIDIA's R&D expenditure?"
    numpy  → AnsRel=0.000, CtxPrec=0.000, CtxRec=0.000

Q13 "What is NVIDIA's gross margin trend?"
    numpy  → AnsRel=0.000, CtxPrec=0.000
```

**Kök neden teşhisi:**
Üç metrik birlikte sıfırsa genellikle **cevap çok kısa veya sayısal tablodan gelmiş**.
10-K'daki finansal tablolar şöyle görünür:

```
Revenue  | FY2025    | FY2024
---------|-----------|--------
Total    | $130,497  | $60,922
```

Bu tablo tiktoken ile chunk'landığında anlamsız parçalara ayrılır.
LLM cevap üretir ama ragas'ın reverse-QG (geriye soru üretme) adımı
kısa sayısal cevaplardan anlamlı soru türetemez → cosine similarity = 0.

**Çözüm stratejileri (kolay → zor):**

```
1. HIZLI FIX: top_k artır (5→10)
   Tablo chunk'larının gelme ihtimali artar.

2. ORTA: Finansal tabloları ayrı işle
   JSON'daki tablo satırlarını ayrı document olarak yükle:
   "Item 9 > Revenue Table > FY2025 Total: $130.5B"

3. İDEAL: Hibrit chunking
   Metin → token-bazlı chunk
   Tablo → satır bazında ayrı chunk (her satır kendi chunk'ı)
```

---

## Problem 2: Context Recall Sistematik Olarak Düşük

**Semptom:** Tüm sorularda recall ~0.4, precision iyi ama recall kötü.

**Bu projeden örnek (Day 33):**
```
Q5  "Data center revenue and growth?"
    numpy  → CtxPrec=0.200, CtxRec=0.000
    qdrant → CtxPrec=0.325, CtxRec=0.000

Q6  "Supply chain risks?"
    numpy  → CtxPrec=0.917, CtxRec=0.000   ← precision yüksek ama recall sıfır
    qdrant → CtxPrec=0.917, CtxRec=0.000
```

**Kök neden:**
Cevap için gereken bilgi **birden fazla 10-K bölümünde** dağılmış.
Q5 için gelir rakamı Item 9'da, büyüme açıklaması Item 7'de.
Cosine search her ikisini aynı anda getiremiyor — top-5 dolup bitiyor.

**Çözüm stratejileri:**

```
1. HIZLI FIX: top_k artır (5→10 veya 20)
   Daha fazla chunk → farklı bölümlerden materyal gelme şansı artar.
   Maliyet: daha fazla token, biraz daha yavaş.

2. ORTA: Section-aware retrieval
   Her bölümden en az 1 chunk gelmesini zorla:
   
   # Her expected_section için ayrı arama yap
   results = []
   for section in ["Item 7", "Item 9"]:
       results += qdrant_search(query, section_filter=section, top_k=3)
   deduplicate(results)

3. İDEAL: Multi-query retrieval
   Tek sorudan birden fazla arama sorgusu türet:
   "NVIDIA data center revenue" → ["data center revenue FY2025",
                                   "data center growth percentage",
                                   "segment revenue breakdown"]
   Hepsini ara, birleştir, rerank et.
```

---

## Problem 3: Faithfulness < 1.0 (Kısmi Hallucination)

**Semptom:** Belirli sorularda faithfulness 0.5-0.8 arasında.

**Bu projeden örnek (Day 33):**
```
Q4  "Who are NVIDIA's main competitors?"
    numpy  → Faith=0.786
    qdrant → Faith=0.769
    
Q17 "CUDA platform competitive significance?"
    numpy  → Faith=0.750
```

**Kök neden:**
LLM retrieved chunk'lara dayanırken bazı iddiaları kendiliğinden ekliyor.
"Qualcomm ve MediaTek mobile'da rakip" bilgisi chunk'larda yoksa
ama LLM bunu genel bilgisinden ekliyorsa faithfulness düşer.

**Çözüm stratejileri:**

```
1. HIZLI FIX: System prompt'u sıkılaştır
   Mevcut: "Answer using ONLY the provided context"
   Güçlendirilmiş:
   "Answer ONLY using the provided context.
    If information is not in the context, say 'Not mentioned in the document.'
    Do NOT use your general knowledge."

2. ORTA: Temperature düşür (0.2 → 0.0)
   Daha deterministik çıktı → daha az "yaratıcılık".

3: İDEAL: Citation zorunlu kıl
   LLM'i chunk ID ile citation yazmaya zorla:
   "NVIDIA competes with AMD [Chunk 3] and Intel [Chunk 1]."
   Citation olmayan cümle → otomatik filtre.
```

---

## Problem 4: Context Precision Düşük (Gereksiz Chunk'lar)

**Semptom:** Üst sıralarda ilgisiz chunk'lar var.

**Bu projeden örnek (Day 33):**
```
Q5  "Data center revenue?"
    numpy  → CtxPrec=0.200  ← 5 chunk'tan sadece 1'i işe yarıyor

Q9  "Gaming segment performance?"
    numpy  → CtxPrec=0.333  ← 5 chunk'tan 2'si işe yarıyor
    qdrant → CtxPrec=0.333
```

**Kök neden:**
Cosine similarity semantik olarak alakalı ama **cevap için gerekli olmayan**
chunk'ları da getiriyor. "Data center revenue" sorusu GPU mimari açıklamalarını
da getirebilir çünkü vektör uzayında yakınlar.

**Çözüm stratejileri:**

```
1. HIZLI FIX: numpy ve qdrant'a reranking ekle
   LangChain implementasyonunda zaten Cohere reranking var (Day 12/13).
   numpy ve qdrant'ta yok — onlara da eklemek precision'ı artırır.
   Not: LangChain'in Day 33 precision'ı (0.599) numpy/qdrant'tan düşük
   çünkü farklı (character-based, daha büyük) chunking kullanıyor —
   reranking bu farkı tam telafi edemiyor.

2. ORTA: MMR (Maximal Marginal Relevance)
   Hem alakalı hem birbirinden farklı chunk'lar getir.
   QdrantVectorStore.as_retriever(search_type="mmr")

3: İDEAL: Metadata filtering
   Soru tipine göre section filtresi:
   Finansal soru → Item 7, Item 9'dan ara
   Risk sorusu   → Item 1A'dan ara
   Bu projede Qdrant metadata filtresi zaten var (Day 10).
```

---

## RAG Sorun → Çözüm Özet Tablosu

| Metrik Paterni | Kök Neden | İlk Deneyeceğin Fix |
|----------------|-----------|---------------------|
| AnsRel=0, CtxPrec=0, CtxRec=0 (hepsi sıfır) | Tablo/sayısal veri chunk'landı | top_k artır, tablo satırlarını ayrı işle |
| CtxRec=0, CtxPrec yüksek | Cross-section bilgi dağınık | top_k artır veya section-aware retrieval |
| Faith < 0.8 | LLM genel bilgisinden ekleme yapıyor | System prompt sıkılaştır, temperature=0 |
| CtxPrec düşük, CtxRec iyi | Gereksiz chunk'lar üst sıralarda | Reranking ekle veya MMR kullan |
| Tüm metrikler orta (~0.5) | Chunk size yanlış | Chunk size ve overlap dene |

---

# BÖLÜM 2 — Agent Sorunları

## Problem 5: Tool Selection Hatası

**Semptom:** Agent yanlış tool'u çağırıyor.

**Bu projeden örnek:**
Braintrust ve LangSmith eval'lerinde bazı ticker'lar için
`query_10k` yerine `web_search` çağrıldı (Day 19-21).
NVDA için 10-K var ama TSLA için yok → agent bunu öğrenmeli.

**Teşhis yöntemi:**
```python
# LangSmith trace'den çek
for trace in traces:
    tools_called = [t.name for t in trace.tool_calls]
    expected_tools = golden_set[trace.id]["expected_tools"]
    if tools_called != expected_tools:
        log_failure(trace.id, tools_called, expected_tools)
```

**Çözüm stratejileri:**

```
1. HIZLI FIX: Tool description'ını güçlendir
   Mevcut: "Query the 10-K document"
   Güçlendirilmiş:
   "Query NVIDIA 10-K filing. Use ONLY for NVDA ticker.
    For other tickers, use web_search instead."

2. ORTA: Tool availability check
   Registry'e ticker → available_tools mapping ekle:
   tools_for("NVDA") → [query_10k, get_stock_data, ...]
   tools_for("TSLA") → [web_search, get_stock_data, ...]

3. İDEAL: Few-shot prompt
   System prompt'a doğru tool seçimi örnekleri ekle.
   "When asked about NVDA strategy: use query_10k.
    When asked about TSLA news: use web_search."
```

---

## Problem 6: Agent Loop Bitmeyip Dönüyor (Infinite Loop)

**Semptom:** Agent max_iterations'a takılıyor, anlamlı cevap üretemiyor.

**Bu projeden örnek (Day 17-20):**
Agent loop `max_iterations=10` ile sınırlandırıldı.
Bazı durumlarda tool error → LLM retry → aynı error döngüsü.
Day 20'de fix: broken tool recovery + structured error JSON.

**Teşhis yöntemi:**
```python
# AgentResult.trace içinde bak
if result.iterations == max_iterations:
    # Loop bitmedi → neden?
    last_tool = result.trace[-1].tool_call
    last_error = result.trace[-1].error
```

**Çözüm stratejileri:**

```
1. HIZLI FIX: Tool error'ı LLM'e structured olarak gönder
   # Day 20'de yapılan fix:
   error_msg = {
       "tool": tool_name,
       "error": str(exc),
       "suggestion": "Try web_search as fallback"
   }
   # Bu JSON mesajla LLM alternatif tool seçer

2. ORTA: Fallback chain tanımla
   query_10k fail → web_search
   web_search fail → "Bilgi bulunamadı" ile tamamla

3. İDEAL: Tool retry budgeti
   Her tool için max 2 deneme hakkı.
   2 fail → otomatik skip ve devam et.
```

---

## Problem 7: Sentiment Accuracy Düşük

**Semptom:** LLM-as-judge evaluator'da sentiment doğruluğu ~0.6-0.7.

**Bu projeden örnek (Day 31):**
6 model kombinasyonunda sentiment accuracy değişkenlik gösterdi.
Edge case'ler (ironic bullish, mixed signals) tüm modellerde zorladı.

**Teşhis yöntemi:**
```python
# Hangi örnek tipi hata yapıyor?
errors_by_type = defaultdict(list)
for ex in wrong_predictions:
    errors_by_type[ex["category"]].append(ex)

# Çıktı: {"ironic_bullish": 8, "mixed_signals": 5, "clear_bullish": 1}
# → Model ironic ve mixed'da zorlanıyor
```

**Çözüm stratejileri:**

```
1. HIZLI FIX: Prompt'a edge case örnekleri ekle
   "Layoff announcement + stock rally = ironic bullish (not bullish, not bearish)"
   "Beat earnings + guidance cut = mixed (score: neutral)"

2. ORTA: Chain-of-thought zorunlu kıl
   "First explain WHY this is bullish/bearish/neutral, then give label."
   Düşünce süreci görmek hataları azaltır.

3. İDEAL: Confidence score ekle
   Model belirsizse düşük confidence dönsün.
   Düşük confidence → "uncertain" etiketi veya human review.
```

---

## Problem 8: Latency Yüksek (10+ saniye)

**Semptom:** Bazı sorgular 10 saniyenin üzerinde süriyor.

**Bu projeden örnek (Day 33):**
```
Q8  "Export control restrictions?"
    numpy   → 10.3s
    qdrant  →  6.6s
    langchain → 6.2s
```

**Kök neden:**
- numpy: her sorguda embedding API çağrısı (query embed = 0.2s) + O(N) arama
- Uzun context + çok chunk → LLM generation süresi artar

**Çözüm stratejileri:**

```
1. HIZLI FIX: Query embedding cache
   Aynı soru tekrar gelirse embed etme:
   @lru_cache(maxsize=256)
   def embed_query_cached(query: str) -> list[float]: ...

2. ORTA: Async paralel tool çağrısı
   # Day 17'de zaten yapıldı: parallel tool calls
   # Birden fazla tool → aynı anda çağır

3. İDEAL: Streaming response
   LLM ilk token'ı üretince kullanıcıya gönder.
   Toplam latency aynı ama perceived latency düşer.
```

---

## Agent Sorun → Çözüm Özet Tablosu

| Gözlem | Kök Neden | İlk Deneyeceğin Fix |
|--------|-----------|---------------------|
| Yanlış tool çağrılıyor | Tool description belirsiz | Tool description'ı örnekle güçlendir |
| Loop max_iterations'a takılıyor | Tool error döngüsü | Structured error JSON + fallback chain |
| Sentiment accuracy düşük | Edge case'ler prompt'ta yok | Few-shot örnekler + CoT |
| Latency > 5s | Seri API çağrıları | Async parallel + embedding cache |
| Ara sıra tutarsız cevap | Temperature yüksek | Temperature = 0 dene |
| Belirli ticker'larda hata | Tool availability | Ticker-based tool routing |

---

# BÖLÜM 3 — Önceliklendirme Çerçevesi

Tüm sorunları aynı anda çözmek gerekmez. Şu matrisi kullan:

```
           ETKİ YÜKSEK     ETKİ DÜŞÜK
           ──────────────────────────────
KOLAY   │  ① Hemen yap   │  ③ Boş vakitte
        │  (quick win)   │
ZORU    │  ② Plan yap    │  ④ Yapma
```

**Bu projede öncelik sırası:**

| Sıra | Sorun | Etki | Kolaylık | Aksiyon |
|:----:|-------|:----:|:--------:|---------|
| ① | System prompt sıkılaştır (Faith için) | Yüksek | Kolay | Hemen |
| ① | top_k artır 5→10 (CtxRec için) | Yüksek | Kolay | Hemen |
| ② | Tablo satırlarını ayrı chunk yap | Yüksek | Zor | Sprint planı |
| ② | Section-aware retrieval | Orta | Orta | Sprint planı |
| ③ | MMR retrieval | Orta | Kolay | Boş vakitte |
| ④ | Hibrit chunking mimarisi | Düşük (tek belge) | Çok Zor | Yapma |

---

# BÖLÜM 4 — Sürekli İyileştirme Döngüsü

Tek seferlik fix yetmez. Production'da bu döngü çalışmalı:

```
        ┌─────────────────────────────────┐
        │                                 │
        ▼                                 │
   DEPLOY          ──────────────────►  MONITOR
   (yeni versiyon)                    (LangSmith trace)
        │                                 │
        │                                 ▼
   ITERATE          ◄──────────────  EVALUATE
   (fix + test)                      (Ragas / LLM-judge)
        │                                 │
        │                                 ▼
        └─────────────────  TRIAGE (hangi sorun önce?)
```

**Pratik adımlar:**

**1. Threshold belirle — deployment öncesi geçmesi gereken minimum:**
```python
MINIMUM_THRESHOLDS = {
    "faithfulness":      0.85,  # hallucination sınırı
    "answer_relevancy":  0.75,  # odak sınırı
    "context_recall":    0.50,  # kapsam sınırı
}

# CI/CD'de:
if any(score < threshold for score, threshold in ...):
    raise Exception("RAG quality gate failed — do not deploy")
```

**2. Regression testi — yeni fix öncekini bozmadı mı?**
```python
# Her değişiklikten sonra golden set'i koştur
# Braintrust baseline karşılaştırması burada çok işe yarıyor
baseline = load_scores("v1.0")
current  = run_ragas(golden_set)
regressions = [q for q in current if current[q] < baseline[q] - 0.05]
```

**3. Golden set büyüt — yeni hata bulunca ekle:**
```python
# Bir kullanıcı şikayet etti → o soruyu golden set'e ekle
# Bir edge case keşfedildi → ekle
# Golden set zamanla sorunlu alanları temsil eder hale gelir
```

**4. Hata logla, pattern bul:**
```
Ayda bir bak:
- Hangi sorular sürekli düşük skor alıyor?
- Hangi bölümler (Item 7, Item 1A) daha zor?
- Hangi model kombinasyonu en stabil?
```

---

## Son Söz

Test bulguları sana iki şey söyler:
- **Ne** çalışmıyor (metrik değerleri)
- **Nerede** çalışmıyor (hangi katman)

Ama **neden** çalışmıyor ve **nasıl** düzeltilir soruları
her zaman biraz araştırma gerektirir.
Bu belgede görüldüğü üzere: çoğu sorunun hızlı bir fix'i var.
İdeal çözümler daha pahalı ama daha sağlam.
Önce hızlı fix ile başla, ölçüm yap, gerekirse ideale ilerle.

---

*Day 33 — Proje: finance-sentiment-engine*
