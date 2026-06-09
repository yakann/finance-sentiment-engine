# RAG Pipeline Map

> RAG bir **slot makinesi** — her slot'un sabit bir rolü var, sen oraya en uygun parçayı takıyorsun.
> Bu projede kullandığımız seçimler **kalın** olarak işaretlenmiştir.

---

## Pipeline Akışı

```
Ham Belge (nvda_10k_2025.json)
        │
        ▼
┌─────────────────────┐
│   SLOT 1: CHUNKING  │  Belgeyi parçalara böl
└──────────┬──────────┘
           │ list[Document]
           ▼
┌──────────────────────┐
│   SLOT 2: EMBEDDING  │  Metni vektöre çevir
└──────────┬───────────┘
           │ list[float]  (1536-dim)
           ▼
┌───────────────────────────┐
│   SLOT 3: VECTOR STORE    │  Vektörleri sakla ve ANN araması yap
└──────────┬────────────────┘
           │ top-K Documents  (cosine similarity)
           ▼
┌────────────────────────┐
│   SLOT 4: RERANKER     │  Cross-encoder ile ince filtreleme
└──────────┬─────────────┘
           │ top-N Documents  (relevance_score)
           ▼
┌────────────────────────┐
│   SLOT 5: PROMPT       │  Bağlam + soruyu LLM'e hazırla
└──────────┬─────────────┘
           │ ChatPromptValue
           ▼
┌────────────────────────┐
│   SLOT 6: LLM          │  Cevabı üret
└──────────┬─────────────┘
           │ AIMessage
           ▼
┌───────────────────────────┐
│   SLOT 7: OUTPUT PARSER   │  Çıktıyı kullanılabilir tipe çevir
└──────────┬────────────────┘
           │ str
           ▼
        Cevap
```

---

## Slot Detayları

### Slot 1 — Chunking

**Rol:** Ham metni modelin işleyebileceği boyuta getir.

| | Araç | Notlar |
|--|------|--------|
| ✅ **Kullandık** | `RecursiveCharacterTextSplitter` | Karakter bazlı, `chunk_size=2000` (~500 token) |
| Alternatif | Tiktoken sliding window | Token-accurate; Day-12 `rerank.py`'de kullandık |
| Alternatif | `SemanticChunker` | Cümle anlamına göre böler, daha akıllı sınır |
| Alternatif | `HTMLHeaderTextSplitter` | Yapılandırılmış HTML belgeler için |

**LCEL karşılığı:**
```python
splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
chunks = splitter.split_documents(docs)
```

---

### Slot 2 — Embedding

**Rol:** Anlamı sayısal uzaya taşı. Semantik benzerlik aramasının temeli.

| | Araç | Notlar |
|--|------|--------|
| ✅ **Kullandık** | `OpenAIEmbeddings` (`text-embedding-3-small`) | 1536 dim, ücretli, yüksek kalite |
| Alternatif | `HuggingFaceEmbeddings` (`BAAI/bge-small`) | Ücretsiz, local, iyi İngilizce performansı |
| Alternatif | `CohereEmbeddings` (`embed-v3`) | Çok dilli, ücretli |
| Alternatif | `OllamaEmbeddings` | Tamamen local, internet gerektirmez |

**LCEL karşılığı:**
```python
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
```

---

### Slot 3 — Vector Store

**Rol:** Vektörleri indexle ve sorgu vektörüne en yakın komşuları bul (ANN).

| | Araç | Notlar |
|--|------|--------|
| ✅ **Kullandık** | `QdrantVectorStore` | HNSW, persistent, Docker; Day-10'dan beri kullanıyoruz |
| Alternatif | `FAISS` | In-memory, persist yok; Day-9 NumPy'a en yakın |
| Alternatif | `Chroma` | Embedded DB, dosya tabanlı, kurulum yok |
| Alternatif | `Pinecone` | Cloud, managed, production-ready |
| Alternatif | `PGVector` | PostgreSQL extension, var olan DB'ye entegre |

**LCEL karşılığı:**
```python
vectorstore = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    url="http://localhost:6333",
    collection_name="nvda_10k_lc",
)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
```

> `as_retriever()` konfigürasyonu tanımlar, çalıştırmaz.
> Query `chain.invoke(query)` anında gelir.

---

### Slot 4 — Reranker

**Rol:** Bi-encoder'ın geniş havuzunu (top-10) cross-encoder ile daralt (top-5).
Bi-encoder hızlıdır ama kabadır; cross-encoder yavaştır ama çok daha hassastır.

| | Araç | Notlar |
|--|------|--------|
| ✅ **Kullandık** | `CohereRerank` (`rerank-v3.5`) | Cloud API, SOTA, 1000 req/ay ücretsiz |
| Alternatif | `BGE CrossEncoder` (`bge-reranker-large`) | Local CPU, ücretsiz; Day-12'de kullandık |
| Alternatif | `FlashrankRerank` | Ultra-hızlı, local, ~%95 Cohere paritesi |
| Alternatif | `LLMChainFilter` | LLM'e "alakalı mı?" diye soran basit yol |

**LCEL karşılığı:**
```python
compressor = CohereRerank(model="rerank-v3.5", top_n=5)
retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever,
)
```

> `ContextualCompressionRetriever` Slot 3 → Slot 4 köprüsü.
> Tek `retriever.invoke(query)` çağrısı her iki aşamayı tetikler.

---

### Slot 5 — Prompt

**Rol:** Reranked context ile kullanıcı sorusunu LLM'in anlayacağı formata getir.

| | Araç | Notlar |
|--|------|--------|
| ✅ **Kullandık** | `ChatPromptTemplate` | System + human mesajları, chat modeller için |
| Alternatif | `PromptTemplate` | Tek string, eski stil completions için |
| Alternatif | `FewShotChatMessagePromptTemplate` | Örneklerle (few-shot) |
| Alternatif | Elle f-string | Day-10 `rag_answer()` içinde yaptığımız |

**LCEL karşılığı:**
```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a financial analyst assistant..."),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])
```

---

### Slot 6 — LLM

**Rol:** Bağlamı okuyup doğal dilde cevap yaz.

| | Araç | Notlar |
|--|------|--------|
| ✅ **Kullandık** | `ChatOpenAI` (`gpt-4o-mini`, `temp=0.2`) | Ücretli, hızlı, yüksek kalite |
| Alternatif | `ChatAnthropic` | Claude; uzun bağlam için güçlü |
| Alternatif | `ChatGroq` | Llama tabanlı, çok hızlı, ücretli |
| Alternatif | `ChatOllama` | Tamamen local, ücretsiz |
| Alternatif | `ChatCohere` | Command R+, retrieval-augmented generation'a optimize |

**LCEL karşılığı:**
```python
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
```

---

### Slot 7 — Output Parser

**Rol:** LLM'in döndürdüğü `AIMessage` objesini kullanılabilir Python tipine çevir.

| | Araç | Çıktı tipi |
|--|------|------------|
| ✅ **Kullandık** | `StrOutputParser` | `str` |
| Alternatif | `JsonOutputParser` | `dict` |
| Alternatif | `PydanticOutputParser` | Pydantic model instance |
| Alternatif | `CommaSeparatedListOutputParser` | `list[str]` |

**LCEL karşılığı:**
```python
| StrOutputParser()
```

---

## Tam LCEL Chain (Tek Bakışta)

```python
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt    # Slot 5
    | llm       # Slot 6
    | StrOutputParser()  # Slot 7
)
# Slot 3+4 retriever içinde gizli, Slot 2 vectorstore içinde gizli

answer = chain.invoke("What are NVIDIA's main AI risks?")
```

---

## Kod ↔ Slot Eşleştirmesi

| Slot | `rag_langchain.py` satırı | Paket |
|------|--------------------------|-------|
| Chunking | `RecursiveCharacterTextSplitter(...)` | `langchain_text_splitters` |
| Embedding | `OpenAIEmbeddings(model="text-embedding-3-small")` | `langchain_openai` |
| Vector Store | `QdrantVectorStore.from_documents(...)` | `langchain_qdrant` |
| Retriever bridge | `vectorstore.as_retriever(k=10)` | `langchain_qdrant` |
| Reranker | `CohereRerank(model="rerank-v3.5", top_n=5)` | `langchain_cohere` |
| Rerank wrapper | `ContextualCompressionRetriever(...)` | `langchain_classic` |
| Prompt | `ChatPromptTemplate.from_messages(...)` | `langchain_core` |
| LLM | `ChatOpenAI(model="gpt-4o-mini")` | `langchain_openai` |
| Output Parser | `StrOutputParser()` | `langchain_core` |

---

## Ham Kod (Day-12) ↔ LangChain Karşılaştırması

| Ham fonksiyon (`rerank.py`) | LangChain karşılığı | Kontrol kaybı |
|-----------------------------|--------------------|----|
| `chunk_sections()` tiktoken | `RecursiveCharacterTextSplitter` | Char-based, ~%5 sınır sapması |
| `embed_texts()` batch loop | `OpenAIEmbeddings` | Batch boyutu, retry görünmez |
| `embed_query()` | Retriever içinde otomatik | Skor görünmez |
| `ensure_qdrant_index()` | `QdrantVectorStore.from_documents()` | Upsert batch kontrolü yok |
| `vector_search()` | `base_retriever` | ✅ Eşdeğer |
| `rerank_cohere()` → dict | `CohereRerank` compressor | Skor `metadata`'ya gömülü |
| `rerank_bge()` local | ❌ port edilmedi | BGE desteği yok (custom gerekir) |
| `recall_at_k()` eval | ❌ port edilmedi | Evaluation harness kayıp |
| `rag_answer()` f-string | `prompt \| llm \| StrOutputParser()` | ✅ Daha temiz |

**Sonuç:** 425 satır → 185 satır (%56 azalma), 2 büyük kayıp: BGE reranker ve Recall@5 eval.
