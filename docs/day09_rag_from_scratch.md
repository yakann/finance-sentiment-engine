# Gün 9 — RAG From Scratch with Pure NumPy

**Hedef:** "Vector DB ne yapıyor?" sorusunu kendi kodunla cevapla.

---

## 1. Büyük Resim: RAG Nedir?

**RAG (Retrieval-Augmented Generation)** = "Önce ilgili metni bul, sonra LLM'e ver."

Neden lazım?
- LLM'lerin bilgisi training cut-off ile sınırlı.
- 10-K gibi özel/kurumsal belgeler LLM'in içinde yok.
- Tüm belgeyi context'e atmak hem pahalı hem mümkün değil (token limit).

Çözüm: Belgeyi parçala → her parçayı sayısal vektöre dönüştür → soruyu da vektöre dönüştür → en yakın parçaları bul → LLM'e sadece onları ver.

---

## 2. Vector DB Gerçekte Ne Yapıyor?

Bugün `numpy.ndarray` ile bir Vector DB sıfırdan yazdık. Şema:

```
Ham Metin
    │
    ▼
[Tokenize] ──── tiktoken (cl100k_base)
    │
    ▼
[Chunk]  ──── 500 token / chunk, 50 token overlap
    │
    ▼
[Embed]  ──── OpenAI text-embedding-3-small → 1536-dim float vector
    │
    ▼
[Sakla]  ──── numpy.ndarray shape=(N, 1536)   ← BU = Vector DB'nin özü
    │
    ▼ (query time)
[Ara]    ──── cosine similarity → top-K
    │
    ▼
[Üret]   ──── top-K chunk + soru → LLM → cevap
```

**Pinecone / Weaviate / FAISS ne ekler?**

| Özellik | Bizim numpy | Üretim Vector DB |
|---|---|---|
| Matematik | ✅ Aynı cosine | ✅ Aynı cosine |
| Depolama | RAM (geçici) | Disk (kalıcı) |
| Arama hızı | O(N) linear scan | O(log N) HNSW/IVF indexing |
| Ölçek | ~Binlerce vektör | Milyar+ vektör |
| Filtreler | Yok | Metadata filtering |
| Dağıtık | Yok | Horizontal scaling |

**Sonuç:** Vector DB karmaşık bir şey değil. Özünde bir "benzerlik arama motoru". Biz bugün onu 150 satır Python ile yazdık.

---

## 3. Teknik Detaylar

### 3.1 Tokenizasyon & Chunking

```python
enc = tiktoken.get_encoding("cl100k_base")
tokens = enc.encode(text)
# 500 tokenlik pencereler, 50 token overlap
```

**Neden char ile değil token ile chunk?**
- `"transformer"` → 1 token
- Bazı Türkçe kelimeler → 3-4 token
- Char sayısı yanıltıcı, token sayısı kesin.
- 500 token ≈ 2000 karakter ama variance çok yüksek.

**Neden overlap?**
- Bir cümle chunk sınırında ikiye bölünebilir.
- 50 token overlap ile her chunk komşusunun başını/sonunu görür.

**Sonuç:** 19 bölüm → **134 chunk** → ~61.125 token toplam

### 3.2 Embedding

```python
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=texts  # batch of 100
)
# Her chunk → 1536-boyutlu float32 vektör
```

**Embedding nedir?**
- Metnin anlamını sayısal uzayda bir nokta olarak temsil eder.
- Anlamca benzer metinler bu uzayda birbirine yakın durur.
- "AI risks" ve "artificial intelligence concerns" → çok yakın vektörler.
- "AI risks" ve "quarterly dividends" → uzak vektörler.

**Sonuç:** `ndarray shape=(134, 1536)` — 804 KB RAM

### 3.3 Cosine Similarity

```python
# Formül:
cosine(A, B) = (A · B) / (||A|| × ||B||)

# Kod:
dot_products = embeddings @ q_vec        # (134,) — tüm chunk'larla dot product
emb_norms   = np.linalg.norm(embeddings, axis=1)
q_norm      = np.linalg.norm(q_vec)
scores      = dot_products / (emb_norms * q_norm)
top_5       = np.argsort(scores)[::-1][:5]
```

**Neden cosine, Euclidean değil?**
- Vektör uzunluğundan bağımsız, sadece yön önemli.
- Kısa chunk vs uzun chunk arasında adil karşılaştırma.
- Range: [-1, 1] → 1 = aynı yön = çok benzer.

### 3.4 RAG Sonuçları

| Sorgu | En Yüksek Skorlu Chunk | Score |
|---|---|---|
| "What are the AI-related risks?" | `Item 1A - Risk Factors::chunk_37` | 0.6673 |
| "Revenue from data center segment?" | `Item 7 - MD&A::chunk_8` | 0.5446 |
| "Main competitors?" | `Item 1 - Business::chunk_11` | 0.5688 |

Sistem doğru bölümlere yöneldi — Risk Factors, MD&A, Business — hiç hard-code edilmeden.

---

## 4. Öğrenilen Kavramlar

### Embedding Space (Gömme Uzayı)
LLM'ler metni sayılara dönüştürerek "anlar". Bu sayılar keyfi değil — anlam korunuyor:
`king - man + woman ≈ queen` klasik örneği bunun kanıtı.

### Chunking Stratejisi
Chunk boyutu kritik bir hyperparameter:
- **Çok küçük chunk** → bağlam yok, LLM cevap üretemez
- **Çok büyük chunk** → alakasız bilgi giriyor, noise artar
- **500 token** iyi bir başlangıç noktası

### Batch Embedding
100 chunk'ı tek API çağrısında göndermek, 100 ayrı çağrıdan ~10x daha hızlı ve ucuz.

### Grounding
LLM'e "sadece bu context'i kullan" dediğimizde hallucination riski dramatik düşer. Model "bilmiyorum ama context'te şu var" diyebilir.

---

## 5. Dosya

```
rag_numpy.py
├── load_10k()          → JSON'dan bölümleri yükle
├── chunk_sections()    → tiktoken ile token-accurate chunking
├── embed_chunks()      → batch embedding → ndarray
├── cosine_search()     → manuel cosine similarity hesabı
└── rag_answer()        → retrieval + generation pipeline
```

---

## 6. Bir Sonraki Adım (Gün 10+)

Bu sıfırdan implementasyonu anladıktan sonra üretimde şunlara geçilebilir:

- **ChromaDB / FAISS** → kalıcı depolama + hızlı arama
- **LangChain / LlamaIndex** → pipeline'ı soyutla
- **Reranking** → cross-encoder ile top-5'i yeniden sırala
- **Hybrid Search** → BM25 (keyword) + vector search birleştir
- **Streaming** → LLM cevabını token token aktar
