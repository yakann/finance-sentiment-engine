# Gün 12 — Reranking: İki Aşamalı Retrieval

**Hedef:** Vector search'ün top-10 adayını cross-encoder ile rerank edip top-5'e indir. Quality artar.

---

## 1. Problem: Vector Search Neden Yeterli Değil?

Qdrant'ta cosine similarity araması şunu sorar:
> "Bu query vektörüne geometrik olarak en yakın chunk'lar hangileri?"

Bu **hızlıdır** ama bir sorunu var. Model query ve passage'ı **ayrı ayrı** embed ettiği için aralarındaki derin anlam bağını kaçırabilir.

Örnek:
```
Query:   "What are NVIDIA's AI risks?"

Chunk A: "NVIDIA faces risks from AI regulation..."   → cosine: 0.82  ✅ alakalı
Chunk B: "AI chip demand drives revenue growth..."    → cosine: 0.81  ⚠️  alakasız ama yapısal benzerlik var
Chunk C: "Export controls may limit AI chip sales..." → cosine: 0.79  ✅ alakalı ama geride kaldı
```

Vector search B'yi C'nin önüne koydu — ama B asıl soruyu yanıtlamıyor.

---

## 2. Çözüm: İki Aşamalı Retrieval

```
[Stage 1]  Vector Search   →  geniş ağ, hızlı      →  top-10 candidate
[Stage 2]  Reranker        →  hassas, yavaş         →  top-5 final
```

Stage 1 (bi-encoder): Qdrant, ~0.1ms içinde 10 aday bulur.  
Stage 2 (cross-encoder): Her aday için query ile birlikte okunur ve yeniden sıralanır.

---

## 3. Bi-encoder vs Cross-encoder

### Bi-encoder (vector search)

```
query   ──→ [BERT] ──→ vector_q  ─┐
                                   ├─ cosine(vq, vp) → skor
passage ──→ [BERT] ──→ vector_p  ─┘
```

- Query ve passage **birbirini görmez**, ayrı ayrı embed edilir.
- Embeddingler önceden hesaplanabildiği için çok hızlıdır.
- Tüm anlam 768 sayıya sıkıştırılır → **bilgi kaybı** olur.

### Cross-encoder (reranker)

```
"[CLS] query [SEP] passage [SEP]"  ──→  [BERT]  ──→  1 adet relevance skoru
```

- Query ve passage **aynı anda** transformer'a verilir.
- Her token diğer tüm tokenlara dikkat (attention) verebilir.
- Sıkıştırma adımı yoktur; model doğrudan "ne kadar alakalı?" sorusunu cevaplar.

---

## 4. Teorik Temel: Self-Attention

Cross-encoder'ın gücü transformer'daki **self-attention** mekanizmasından gelir.

```
Input: [CLS] "What are AI risks?" [SEP] "NVIDIA faces export control risks..." [SEP]

Her token diğer tüm tokenlara dikkat verir:
  "risks" (query) ←──→ "risks"  (passage)          yüksek attention ✅
  "AI"    (query) ←──→ "export control" (passage)  orta attention
  "What"  (query) ←──→ "faces"  (passage)          düşük attention
```

Query'deki "risks" kelimesi, passage'daki "risks" kelimesini **doğrudan görebilir**. Bu çapraz dikkat sayesinde iki metin arasındaki anlam köprüsü kurulur.

---

## 5. Information Bottleneck Problemi

```
Bi-encoder:
  Tüm passage anlamı → 768 float → query ile karşılaştır

  Problem: 768 boyutlu vektör "bu passage bu spesifik query için
  ne kadar alakalı?" sorusunu cevaplamak için tasarlanmamış.
  Genel anlamı taşır, spesifik alaka bilgisini kaçırır.

Cross-encoder:
  (query, passage) → transformer → 1 skor

  Model tam olarak "bu ikisi ne kadar alakalı?" sorusunu
  cevaplamak için eğitilmiş. Sıkıştırma adımı yoktur.
```

**Sezgi:** Seni 768 sayıyla tarif etmem gerekse çok şey kaybolur. Ama seni başka biriyle **karşılaştırarak** tarif etsem çok daha hassas olur.

---

## 6. Eğitim Farkı

Bu teorik farkı pratikte mümkün kılan şey, iki modelin **farklı görevler için eğitilmiş** olmasıdır.

**Bi-encoder** eğitimi (contrastive learning):
> "Benzer cümlelerin vektörleri yakın olsun."

**Cross-encoder** eğitimi (pairwise classification):
> "Bu (query, passage) çifti alakalı mı? Evet / Hayır."

BGE reranker modeli, milyonlarca `(query, alakalı_passage, alakasız_passage)` üçlüsü ile eğitildi. Yani doğrudan **"bu ikisi birbirine ne kadar uyuyor?"** sorusunu cevaplamayı öğrendi.

---

## 7. Neden "Semantic Similarity ≠ Relevance"?

```
Query:   "NVIDIA revenue 2024"

Chunk A: "Apple revenue in 2024 reached $391B..."
         → bi-encoder skoru: 0.81  (yapısal benzerlik! aynı cümle kalıbı)

Chunk B: "NVIDIA data center segment Q4 income surpassed..."
         → bi-encoder skoru: 0.78  (daha uzak ama çok daha alakalı)

Cross-encoder:
  ("NVIDIA revenue 2024", Chunk A) → 0.12  ❌ farklı şirket
  ("NVIDIA revenue 2024", Chunk B) → 0.91  ✅ aynı konu
```

Bi-encoder "benzer cümle yapısı" görür. Cross-encoder "anlam ilişkisi" görür.

---

## 8. Projedeki Implementasyonlar

### 8a. Cohere Rerank API (cloud)

```python
co = cohere.ClientV2(api_key=COHERE_API_KEY)
response = co.rerank(
    model="rerank-v3.5",
    query=query,
    documents=[c["text"] for c in candidates],  # top-10 chunk metni
    top_n=5,
)
# response.results → relevance_score ile sıralı top-5
```

- Tek HTTP çağrısı, sıfır GPU gereksinimi.
- Cohere'in kendi cross-encoder modeli (kapalı kaynak, SOTA).
- Free tier: 1000 istek/ay.

### 8b. BAAI/bge-reranker-large (local)

```python
from sentence_transformers import CrossEncoder

model  = CrossEncoder("BAAI/bge-reranker-large", max_length=512)
pairs  = [(query, chunk["text"]) for chunk in candidates]  # 10 çift
scores = model.predict(pairs)  # her çift için 1 skor

# Skora göre büyükten küçüğe sırala → top-5
```

- Tamamen yerel, internet gerekmez.
- İlk çalıştırmada ~1.1 GB model indirir.
- `max_length=512`: query + passage birleşik 512 token ile sınırlanır.

---

## 9. Recall@5 Metriği

**Recall@5**, reranking kalitesini ölçmek için kullanılır.

```
Recall@5(method) = |method_top5 ∩ oracle_top5| / 5
```

- **Oracle** = Cohere top-5 (endüstri standardı cross-encoder)
- **No-rerank** = vector search'ün ilk 5'i
- **BGE** = local cross-encoder'ın top-5'i

```
Örnek:
  Cohere top-5:    [A, B, C, D, E]
  No-rerank top-5: [A, C, F, G, H]  → 2 örtüşme → Recall@5 = 0.40
  BGE top-5:       [A, B, C, D, F]  → 4 örtüşme → Recall@5 = 0.80
```

---

## 10. Trade-off Özeti

| | Vector Only | + Cohere | + BGE |
|---|---|---|---|
| Recall@5 | ~0.60–0.70 | 1.00 (oracle) | ~0.85–0.95 |
| Latency | ~50ms | ~200ms | ~500ms–2s (CPU) |
| Maliyet | Düşük | 1000 req/ay ücretsiz | Ücretsiz |
| Privacy | ✅ | ❌ veri cloud'a gider | ✅ tamamen yerel |
| GPU | Gerekmez | Gerekmez | Gerekmez (CPU yeterli) |

**Ne zaman hangisi?**
- **Cohere** → Hız önemliyse, veri hassasiyeti yoksa, prototipler için.
- **BGE** → Veri gizliyse (finansal belgeler!), production offline ortamı için.
- **Vector only** → Real-time uygulamalar, latency kritik ise.

---

## 11. Çalıştırma

```bash
# Qdrant'ı başlat (Day 10'dan hâlâ çalışıyorsa skip)
docker run -p 6333:6333 qdrant/qdrant

# Cohere API key'ini .env'e ekle (opsiyonel)
echo "COHERE_API_KEY=your_key" >> .env

# Reranking çalıştır
uv run python rerank.py
```

Cohere API key yoksa otomatik olarak **BGE-only modda** çalışır.

---

## İlgili Dosyalar

| Dosya | Açıklama |
|---|---|
| `rerank.py` | Day 12 — Cohere + BGE implementasyonları |
| `rag_qdrant.py` | Day 10 — Qdrant vector search (Stage 1 altyapısı) |
| `rag_numpy.py` | Day 9 — NumPy ile RAG temelleri |
| `docs/day09_rag_from_scratch.md` | RAG teorisi ve NumPy implementasyonu |
