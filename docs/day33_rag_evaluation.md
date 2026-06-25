# RAG Evaluation — Day 33

## RAG nedir, retrieval nedir?

RAG üç adımdan oluşur:

```
Soru gelir
    ↓
1. RETRIEVAL  → Belgeden ilgili chunk'ları bul ve getir
    ↓
2. AUGMENT    → Bulduklarını prompt'a ekle
    ↓
3. GENERATION → LLM cevap üretsin
```

**Retrieval** sadece ilk adım — soruyu embed et, vektör DB'de ara, en yakın N chunk'ı getir. LLM henüz devrede değil, saf arama işlemi.

---

## Agent eval vs RAG eval farkı

| | Ne test ediyor? | Araçlar |
|--|----------------|---------|
| **Agent eval** (LangSmith, Braintrust) | "Agent doğru kararı veriyor mu?" — tool seçimi, reasoning, sentiment accuracy | Day 29-32 |
| **RAG eval** (Ragas) | "Retrieval doğru bilgiyi buluyor mu?" — chunk relevance, coverage, grounding | Day 33 |

İkisi birbirini tamamlar. Agent iyi karar verse bile yanlış chunk getirirse kötü cevap verir.

---

## 4 Ragas Metriği

### 1. Faithfulness (Sadakat)

**Soru:** Cevaptaki her iddia, getirilen chunk'lardan geliyor mu?

**Nasıl ölçülür:** LLM cevabı cümlelere ayırır, her cümleyi chunk'larla karşılaştırır (NLI).

```
Cevap: "NVIDIA geliri $130B'dı. Merkezi California'dadır."

Chunk'larda var mı?
  ✅ "$130B gelir"      → Chunk 3'te geçiyor
  ❌ "California'dadır" → Hiçbir chunk'ta yok

Faithfulness = 1/2 = 0.50
```

**Neyi yakalar:** Hallucination. LLM chunk'larda olmayan bir şeyi uydurmaya başladıysa bu metrik düşer.

---

### 2. Answer Relevancy (Cevap İlgililiği)

**Soru:** Cevap gerçekten soruyu yanıtlıyor mu, yoksa konu dışına mı çıkıyor?

**Nasıl ölçülür:** LLM cevaptan geriye doğru soru üretir, o soruların orijinal soruya embed benzerliğine bakar. Tek metrik embed kullanan.

```
Orijinal soru: "NVIDIA'nın rakipleri kimler?"
Cevap: "AMD ve Intel rakip. NVIDIA Türkiye'de ofis açtı."

LLM cevaptan soru üretir:
  → "NVIDIA'nın rakipleri kimler?" ✅ (ilgili)
  → "NVIDIA hangi ülkelerde ofis açtı?" ❌ (ilgisiz)

Cosine similarity ortalaması → düşük skor
```

**Neyi yakalar:** Cevap doğru bilgi içerse de soruya odaklanmıyorsa bunu yakalar.

---

### 3. Context Precision (Bağlam Hassasiyeti)

**Soru:** Getirilen chunk'ların kaçı gerçekten işe yarıyor?

**Nasıl ölçülür:** Her chunk'ın referans cevap için gerekli olup olmadığını LLM değerlendirir. Sıralama önemli — üstteki chunk'ların işe yarıyor olması skoru artırır.

```
Soru: "NVIDIA'nın riski nedir?"
Getirilen 5 chunk:

  Rank 1: "Export kontrolleri..."    ✅ işe yarıyor
  Rank 2: "GeForce RTX özellikleri" ❌ ilgisiz
  Rank 3: "Supply chain riski..."    ✅ işe yarıyor
  Rank 4: "Oyun geliri..."           ❌ ilgisiz
  Rank 5: "AI rekabeti..."           ✅ işe yarıyor
```

**Neyi yakalar:** RAG sistemi doğru chunk'ları getiriyor ama aralarına gereksizleri de karıştırıyorsa bunu yakalar. Gereksiz chunk = gereksiz token = para.

---

### 4. Context Recall (Bağlam Kapsamı)

**Soru:** Referans cevabı oluşturmak için gereken tüm bilgiler getirilen chunk'larda var mı?

**Nasıl ölçülür:** Referans cevap cümlelere ayrılır, her cümle chunk'larda aranır (LLM).

```
Referans cevap:
  "Gelir $130B'dı."                  → Chunk'larda var ✅
  "%142 büyüdü."                     → Chunk'larda var ✅
  "Data Center en büyük segment."    → Chunk'larda YOK ❌
  "Blackwell mimarisi sürüyor."      → Chunk'larda YOK ❌

Context Recall = 2/4 = 0.50
```

**Neyi yakalar:** RAG sistemi bazı kritik bilgileri hiç getirememişse bunu yakalar.

**Neden bizim recall'umuz düşüktü (0.4)?**
Token-bazlı chunking bölüm sınırlarına uymak zorunda değil. Bir sorunun cevabı Item 7 ve Item 9'dan birden geliyorsa, RAG her iki bölümden de yeterince chunk getiremeyebiliyor.

---

### Özet

| Metrik | Ölçüm yöntemi | "Nerede hata var?" | Düşükse anlam |
|--------|--------------|-------------------|---------------|
| **Faithfulness** | LLM (NLI) | LLM katmanında | Model uydurmuş |
| **Answer Relevancy** | LLM + **embed** | LLM katmanında | Model konu dışına çıkmış |
| **Context Precision** | LLM | Retrieval katmanında | Gereksiz chunk'lar geliyor |
| **Context Recall** | LLM | Retrieval katmanında | Gerekli chunk'lar gelmiyor |

---

## Production'da hangi metrikleri kullanmalısın?

Hepsini ölçmek zorunda değilsin. Şu soruyu sor:

> "Kullanıcı ne zaman şikayet eder?"

### Karar ağacı

```
Yanlış bilgi verilirse kritik mi? (medikal, hukuki, finans)
  → Evet: Faithfulness zorunlu

Belgenin tamamı taranmazsa kritik mi? (müşteri destek, sözleşme)
  → Evet: Context Recall zorunlu

Gereksiz bilgi karışırsa sorun olur mu? (maliyet, hız önemli)
  → Evet: Context Precision ekle

Cevap odaksız olursa sorun olur mu?
  → Evet: Answer Relevancy ekle
```

### Örnek senaryolar

| Use case | Öncelikli metrik | Neden |
|----------|-----------------|-------|
| Medikal/hukuki RAG | Faithfulness | Yanlış bilgi kritik zarar verir |
| Müşteri destek botu | Context Recall + Precision | Politikanın tamamı doğru gelmeli |
| Araştırma asistanı | Answer Relevancy + Faithfulness | Odak ve doğruluk önemli, eksik tolere edilebilir |
| Agent (tool seçimi) | Ragas yetmez, ayrıca tool accuracy + iteration count ölç | Ragas sadece retrieval ölçer |

---

## Day 33 sonuçları

```
Golden set : 20 NVIDIA 10-K sorusu
Impl       : numpy, qdrant, langchain (Cohere rerank)

               Faith   AnsRel  CtxPrec  CtxRec
  numpy        0.918   0.737   0.671    0.423
  qdrant       0.883   0.791   0.668    0.413
  langchain    0.968   0.803   0.599    0.478  ← Cohere rerank etkisi
```

**Gözlem:** LangChain (Cohere reranking) faithfulness ve relevancy'de en iyi. Context recall tüm implementasyonlarda düşük — token-bazlı chunking cross-section bilgiyi parçalıyor.
