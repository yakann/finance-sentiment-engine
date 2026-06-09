# Day 27 — Hafta-3 Tool-Use Loop'unu LangGraph Subgraph'a Taşıma

## Ne Yaptık?

Hafta 3'te (`agent/loop.py`) elle yazdığımız `while` döngüsünü
LangGraph'ın node + condition yapısına taşıdık.

Bu "research_subgraph" artık ana `finance_graph`'ın içine **bir node gibi** takılıyor.
`collect_news` node'unu komple değiştiriyor; dışarıdan bakınca hiçbir şey değişmemiş gibi görünüyor,
ama içeride bir LLM döngüsü çalışıyor.

---

## Önceki Günlerle Karşılaştırma

| | Day 22–26 (finance_graph) | Day 27 (+ research_subgraph) |
|---|---|---|
| `collect_news` ne yapıyor? | `_analyze_news_sentiment()` direkt çağırır | LLM + tool döngüsü çalıştırır |
| Haber nasıl geliyor? | Fonksiyon → sonuç | LLM tool call yapıyor → dispatch → sonuç |
| Graf yapısı | Tek düzlemli (flat) graph | **İç içe graph (subgraph)** |
| Hata toleransı | Yok — exception yayılır | dispatch_tools yakalar, loop devam eder |
| Döngü kontrolü | Yok (tek geçiş) | `should_continue` condition ile döngü |

---

## Temel Kavram: Subgraph

LangGraph'te **subgraph**, başka bir graph'ın node'u olarak çalışan derlenmiş bir graph'tır.

```
Ana Graph (finance_graph):

  START
    │
    ▼
  collect_news  ←── bu node içinde aslında şu çalışıyor:
    │
    │   ┌─────────────── research_subgraph ───────────────┐
    │   │  START                                           │
    │   │    │                                             │
    │   │    ▼                                             │
    │   │  call_model ──tool_calls var?── dispatch_tools   │
    │   │    ▲                                    │        │
    │   │    └────────────────────────────────────┘        │
    │   │    │                                             │
    │   │    └──tool_calls yok──▶ END                      │
    │   └─────────────────────────────────────────────────┘
    │
    ▼
  analyze_sentiment
    │
   ...
```

Ana graph açısından `collect_news` sıradan bir node.
Subgraph açısından kendi içinde çalışan tam bir `call_model → dispatch_tools → ...` döngüsü.

---

## State Tasarımı: Neden `loop_messages`?

Bu gün en kritik tasarım kararı buydu.

### Problem

`FinanceState`'in `messages` alanı var:
```python
messages: Annotated[list[BaseMessage], add_messages]   # LangChain BaseMessage
```

`ResearchState`'e de `messages` desek LangGraph **aynı key'i eşleştirir** ve parent'ın
`BaseMessage` listesini subgraph'a enjekte eder. Ama subgraph OpenAI Responses API'nin
**flat dict formatını** bekliyor — tamamen farklı yapı.

### Çözüm

```python
class ResearchState(TypedDict, total=False):
    ticker: str                                         # parent'tan gelir
    loop_messages: Annotated[list[Any], operator.add]   # iç format, çakışmaz
    pending_tool_calls: list[dict]
    news: list                                          # parent'a çıkar
```

`loop_messages` key adıyla:
- Parent'ın `messages` alanıyla çakışma yok
- OpenAI Responses API formatı bozulmadan korunuyor
- `operator.add` sayesinde her node yalnızca **yeni** mesajları döndürüyor, hepsi birikime ekleniyor

### Otomatik State Eşleşmesi

LangGraph subgraph'ı bir node olarak eklenince sadece **eşleşen key'ler** haritalanır:

```
FinanceState.ticker  →  ResearchState.ticker   (input)
ResearchState.news   →  FinanceState.news      (output)

ResearchState.loop_messages      → parent'a gitmez (iç)
ResearchState.pending_tool_calls → parent'a gitmez (iç)
FinanceState.messages            → subgraph'a gelmez (key yok)
```

---

## Node'lar Nasıl Çalışıyor?

### `call_model`

```python
def call_model(state: ResearchState) -> dict:
    loop_messages = list(state.get("loop_messages") or [])

    if not loop_messages:
        # İlk çağrı: kullanıcı mesajı oluştur
        init_msg = {"role": "user", "content": f"Analyze news for {ticker}..."}
        loop_messages = [init_msg]
        new_msgs = [init_msg]
    else:
        new_msgs = []   # geçmiş zaten state'te, sadece yenileri ekle

    response = provider.generate(loop_messages, tools=registry.all_tools())

    # Asistan turn'ünü (function_call item'ları) new_msgs'e ekle
    new_msgs.extend(response.raw_assistant_content or [])

    return {
        "loop_messages": new_msgs,           # operator.add ile birikir
        "pending_tool_calls": [...],         # tool calls varsa dolu, yoksa []
    }
```

**Kritik**: Node, `loop_messages`'e sadece **o iterasyonda oluşan mesajları** döndürüyor.
`operator.add` reducer bunları mevcut listeye ekliyor. Böylece her `call_model` çağrısında
`provider.generate()` tüm geçmişi görüyor.

### `dispatch_tools`

```python
def dispatch_tools(state: ResearchState) -> dict:
    for tc in state["pending_tool_calls"]:
        result = registry.dispatch(tc["name"], tc["input"])

        # analyze_news_sentiment ise → NewsAnalysis'e dönüştür
        if tc["name"] == "analyze_news_sentiment":
            for item in result:
                extracted_news.append(NewsAnalysis(**item))

        # OpenAI Responses API formatında tool result mesajı
        tool_result_msgs.append({
            "type": "function_call_output",
            "call_id": tc["id"],
            "output": json.dumps(result),
        })

    return {
        "loop_messages": tool_result_msgs,   # LLM bir sonraki turda görecek
        "pending_tool_calls": [],            # temizlendi
        "news": extracted_news,              # parent'a çıkacak
    }
```

### `should_continue`

```python
def should_continue(state) -> Literal["dispatch_tools", "__end__"]:
    return "dispatch_tools" if state.get("pending_tool_calls") else "__end__"
```

Bu tek satır, Hafta 3'teki `if response.next_action.type == "text":` mantığının LangGraph karşılığı.

---

## `agent/loop.py` vs `research_subgraph.py` — Yan Yana

| Hafta 3 (`loop.py`) | Bugün (`research_subgraph.py`) |
|---|---|
| `while iteration < max_iterations:` | `call_model → dispatch_tools` kenarı |
| `if response.next_action.type == "text": break` | `should_continue() → "__end__"` |
| `provider.extend_messages_with_assistant_turn(messages, response)` | `operator.add` reducer |
| `provider.extend_messages_with_tool_results(messages, results)` | `dispatch_tools` node'un döndürdüğü `loop_messages` |
| `registry.dispatch(tc.name, tc.input)` | `registry.dispatch(tc["name"], tc["input"])` |
| `AgentResult(answer=..., logs=...)` | LangGraph state geçmişi |

---

## Graf Yapısı — İki Seviye

```
Seviye 1 — finance_graph (ana):

  START → collect_news → analyze_sentiment → [deep_analysis / short_brief]
       → fetch_price → draft → [revise / END]

Seviye 2 — research_subgraph (collect_news içinde):

  START → call_model ──┬──(tool calls)──▶ dispatch_tools ──┐
                       └──(text)──▶ END                     │
                       ▲                                     │
                       └─────────────────────────────────────┘
```

---

## Mesaj Geçmişi — Bir Döngünün İzlediği Yol

```
Adım 1 — call_model (ilk çağrı)
  loop_messages += [
    {"role": "user", "content": "NVDA news analiz et..."},        ← kullanıcı
    {"type": "function_call", "call_id": "c1", "name": "analyze_news_sentiment", ...}  ← LLM
  ]
  pending_tool_calls = [{"id": "c1", "name": "analyze_news_sentiment", ...}]

Adım 2 — dispatch_tools
  registry.dispatch("analyze_news_sentiment", {"ticker": "NVDA"})
  → [{"sentiment": "bullish", ...}, ...]
  loop_messages += [
    {"type": "function_call_output", "call_id": "c1", "output": "[{...}]"}  ← tool sonucu
  ]
  pending_tool_calls = []
  news = [NewsAnalysis(ticker="NVDA", sentiment="bullish", ...)]

Adım 3 — call_model (ikinci çağrı)
  provider.generate(tüm loop_messages)  ← 3 mesaj, LLM geçmişi görüyor
  → metin yanıt ("NVDA for 5 articles, 3 bullish...")
  pending_tool_calls = []

should_continue → "__end__"  → subgraph tamamlandı
FinanceState.news ← [NewsAnalysis(...)]  ← parent'a aktarıldı
```

---

## Testler (8 adet)

| Test | Ne doğruluyor |
|---|---|
| `test_subgraph_compiles` | `build_research_subgraph()` exception olmadan derleniyor |
| `test_should_continue_with_tool_calls` | pending varsa `dispatch_tools` |
| `test_should_continue_without_tool_calls` | pending boşsa `__end__` |
| `test_should_continue_missing_key` | key yoksa `__end__` (güvenli default) |
| `test_dispatch_tools_extracts_news` | Tool sonucu `NewsAnalysis`'e doğru dönüşüyor |
| `test_dispatch_tools_error_tolerance` | Tool exception'ı subgraph'ı durdurmaz |
| `test_main_graph_compiles_with_subgraph` | Ana graph subgraph ile birlikte derleniyor |
| `test_subgraph_invoke_mocked` | call_model→dispatch_tools→call_model döngüsü uçtan uca çalışıyor |

---

## Öğrenilen Kavramlar

### 1. Subgraph — İç İçe Graf
Bir graph, başka bir graph'ın node'u olabilir. Dışarıdan bakınca normal bir node;
içeride tam bir state machine.

### 2. State Key Eşleşmesi
Subgraph ile parent arasında veri akışı otomatik. Kural: **sadece aynı isimdeki key'ler haritalanır**.
Farklı isimler izole kalır. `loop_messages` vs `messages` bunun bilinçli kullanımıydı.

### 3. `operator.add` Reducer
LangGraph'te bir state field'ı `Annotated[list, operator.add]` yapılırsa,
her node yalnızca **yeni eklenen** elemanları döndürebilir. LangGraph bunları mevcut listeye ekler.
`add_messages` de aynı prensibin LangChain-aware versiyonu.

### 4. Tool-Use Döngüsünün Graf Karşılığı
```
while loop    →  edge (dispatch_tools → call_model)
if/break      →  should_continue condition
mesaj geçmişi →  operator.add reducer
```

### 5. Katmanlı Mimari
```
Week 1: tool fonksiyonları (ham Python)
Week 2: RAG pipeline
Week 3: tool-use loop (while + provider)
Week 4: LangGraph (loop → subgraph → ana graph)
```
Her hafta bir öncekini **wrap ediyor**, silmiyor.

---

## Dosya Yapısı (Day 27 itibarıyla)

```
graph/
├── state.py               # FinanceState TypedDict
├── research_subgraph.py   # ← YENİ: tool-use döngüsü subgraph
├── finance_graph.py       # ana 7-node graph; collect_news → subgraph
├── checkpointer.py        # SqliteSaver context manager
└── hello.py               # Day 22 başlangıç örneği
```
