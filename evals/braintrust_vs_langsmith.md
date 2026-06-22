# Braintrust vs LangSmith — Eval Platform Karşılaştırması

Day 32 gözlemi: aynı 6-model sentiment eval'i her iki platformda koşturuldu.
Hedef: hangi platform bu proje için daha uygun?

---

## Setup

| | LangSmith | Braintrust |
|---|---|---|
| Hesap | smith.langchain.com | braintrust.dev |
| SDK | `langsmith>=0.3.0` | `braintrust>=0.0.170` + `autoevals` |
| Auth | `LANGSMITH_API_KEY` | `BRAINTRUST_API_KEY` |
| Env | EU endpoint ayrıca set edilmeli | sadece API key yeterli |

**Gözlem:** LangSmith kurulumu biraz daha fazla env var gerektiriyor
(endpoint + project + tracing flag). Braintrust tek key ile başlıyor.

---

## Dataset Yönetimi

### LangSmith
```python
# Önce dataset upload (run_day30.py):
client.create_dataset("finance-sentiment-v1")
client.create_examples(inputs, outputs, dataset_id=...)

# Sonra eval zamanı — cloud'dan çekilir:
await aevaluate(target, data="finance-sentiment-v1", ...)
```

- Dataset UI'da ayrı bir nesne olarak görünür
- Versiyonlama var; örnek eklenebilir/silinebilir
- Birden fazla experiment aynı dataset'e bağlanır → trend grafikleri

### Braintrust
```python
# Dataset inline — upload adımı yok:
Eval("project", data=lambda: [{input: ..., expected: ...}], ...)
```

- Hızlı prototip için ideal (upload bekleme yok)
- Dataset Braintrust UI'da ayrıca görünmez — sadece experiment içinde
- Uzun vadede: Braintrust `braintrust.init_dataset()` ile cloud dataset da destekler

**Karar:** Paylaşılan, versiyonlanan bir dataset için LangSmith daha iyi.
Hızlı deneme için Braintrust inline yaklaşımı kazanır.

---

## Sıralama (Leaderboard / Model Ranking)

### LangSmith
- "Experiments" sekmesinde her experiment ayrı satır
- Sütun başlıklarına tıklayarak metriğe göre sırala
- Filtreleme: tag, tarih, experiment prefix ile
- Tek tıkla iki experiment seçip "Compare" → diff görünümü

### Braintrust
- "Experiments" dashboard'da otomatik sıralama
- **Baseline karşılaştırma**: bir experiment'ı baseline yap,
  sonraki tüm run'lar ona göre diff skorlar gösterir
  (`SummaryScore.diff` alanı)
- Improvement/regression sayıları otomatik hesaplanır
- Renk kodlaması: yeşil = iyileşme, kırmızı = kötüleşme

**Karar:** Braintrust'un otomatik baseline diff sistemi güçlü.
LangSmith manuel compare gerektiriyor. Braintrust burada kazanır.

---

## Filter (Sonuçları Filtreleme)

### LangSmith
- Sol panel: evaluator > score range slider
- Metadata filtresi: tag, run_type, model adı
- SQL benzeri advanced filter (Pro plan)

### Braintrust
- Score threshold filtresi: "sentiment_accuracy < 0.5 olan örnekleri göster"
- Metadata filter: `input.ticker == "NVDA"` gibi alan bazlı
- Toplu inceleme: "wrong predictions" tek tıkla gruplanır

**Karar:** Braintrust'un alan bazlı filter ifadeleri daha güçlü.
LangSmith yeterli ama daha az granüler.

---

## Diff Visualizasyonu

### LangSmith
- İki experiment seç → "Compare Experiments" → yan yana satırlar
- Hangi örnekte A iyi B kötü? Hızlıca görülür
- Renk yok; sadece tablo

### Braintrust
- Her örnek için otomatik score diff gösterimi
- "Improvements" ve "Regressions" sekmesi ayrı ayrı listelenir
- Output diff: eski çıktı vs yeni çıktı karakter seviyesinde highlight
- Özellikle LLM output metin diff'i için çok kullanışlı

**Karar:** Braintrust diff UI açık ara daha iyi.
LLM output'larını karşılaştırmak için tasarlanmış.

---

## Cost View (Maliyet Takibi)

### LangSmith
- Tracing varsa her LLM call'un token sayısı görünür
- Toplam maliyet hesabı yok — manuel hesaplanmalı
- `@traceable` decorator ile input/output token'lar loglanır

### Braintrust
- Her experiment run'da otomatik token sayımı
- Model bazlı maliyet hesabı (OpenAI/Anthropic fiyat listesiyle)
- Experiment summary'de toplam `$` maliyeti görünür
- Hangi model en ucuza en iyi sonucu veriyor? → tek sayfada

**Karar:** Braintrust maliyet takibinde çok daha iyi.
Model seçimi için ROI analizi Braintrust'ta çok kolay.

---

## Pairwise Evaluation (Brief Quality)

### LangSmith
```python
from langsmith.evaluation import evaluate_comparative

evaluate_comparative([exp_a, exp_b], evaluators=[brief_quality])
```
- SDK'da native pairwise desteği var
- A vs B kazanma sayıları otomatik hesaplanır

### Braintrust
- SDK'da `evaluate_comparative()` eşdeğeri **yok**
- UI'da "Compare" ile iki experiment yan yana açılır
- Manuel inceleme veya custom scorer ile çözülebilir:
  ```python
  # Workaround: baseline'dan A output'u çek, scorer içinde karşılaştır
  ```

**Karar:** Pairwise eval için LangSmith açık ara daha iyi.
Braintrust burada bir boşluk var.

---

## SDK Deneyimi (Developer Ergonomi)

### LangSmith
```python
# Scorer imzası — keyword-only, biraz kafa karıştırıcı
def sentiment_accuracy(outputs: dict, reference_outputs: dict) -> dict:
    return {"key": "sentiment_accuracy", "score": 1.0}

# Async-native, aevaluate() ile
await aevaluate(target, data="dataset", evaluators=[scorer])
```

### Braintrust
```python
# Scorer imzası — positional, daha açık
def sentiment_accuracy(input, output, expected) -> float:
    return 1.0

# Sync call, Braintrust kendi event loop'unu yönetir
Eval("project", data=lambda: [...], task=fn, scores=[scorer])
```

**Karar:** Braintrust scorer API'si daha sezgisel.
`input/output/expected` üçlüsü anlamsal olarak netlik sağlıyor.
LangSmith'in `outputs/reference_outputs` terminolojisi ilk bakışta kafa karıştırıyor.

---

## Tracing Entegrasyonu

### LangSmith
- `@traceable` decorator ile LangGraph/LangChain tam entegrasyon
- Agent loop'ları, tool call'lar, LangGraph node'lar otomatik trace edilir
- Day 29'dan bu yana projede zaten kurulu

### Braintrust
- `@braintrust.traced` decorator ile tracing
- LangChain entegrasyonu var ama daha az seamless
- OpenAI/Anthropic otomatik patch özelliği var

**Karar:** Tracing için LangSmith kazanır — zaten LangGraph ile tam entegre.

---

## Özet Skorcard

| Kriter | LangSmith | Braintrust |
|--------|-----------|------------|
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

---

## Karar

**Bu proje için: LangSmith + Braintrust birlikte kullanım**

| Kullanım | Platform |
|----------|----------|
| Production tracing (agent loop, LangGraph) | LangSmith |
| Pairwise model karşılaştırması | LangSmith `evaluate_comparative()` |
| Versiyonlanan golden dataset | LangSmith |
| Model leaderboard + cost ROI analizi | Braintrust |
| Hızlı offline eval prototip | Braintrust (inline data) |
| Diff viz + regression detection | Braintrust |

**Tek platform seçmek zorunda kalsaydık:**
LangGraph + LangChain ağırlıklı bu projede **LangSmith** tercih edilir.
Eğer framework-agnostic ve maliyet optimizasyonu öncelikliyse Braintrust.

---

*Day 32 — 2026-06-22*
