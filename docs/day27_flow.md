# Day 27 — Research Subgraph Akış Diyagramı

## Genel Akış

```
╔══════════════════════════════════════════════════════════════════════════╗
║                        finance_graph (ana)                               ║
║                                                                          ║
║   FinanceState: { ticker: "NVDA", news: [], messages: [], ... }         ║
║                                                                          ║
║   START                                                                  ║
║     │                                                                    ║
║     ▼                                                                    ║
║  ┌─────────────────────────────────────────────────────────────────┐    ║
║  │                   collect_news                                  │    ║
║  │             (= research_subgraph içeride)                       │    ║
║  │                                                                 │    ║
║  │  ResearchState: { ticker: "NVDA", loop_messages: [], ... }      │    ║
║  │                                                                 │    ║
║  │  START                                                          │    ║
║  │    │                                                            │    ║
║  │    ▼                                                            │    ║
║  │  ┌────────────────────────────────────────────────────────┐    │    ║
║  │  │  call_model (1. tur)                                   │    │    ║
║  │  │                                                        │    │    ║
║  │  │  loop_messages boş → kullanıcı mesajı oluştur:         │    │    ║
║  │  │  {"role":"user", "content":"Analyze NVDA news..."}     │    │    ║
║  │  │                                                        │    │    ║
║  │  │  provider.generate([user_msg], tools=[sentiment_tool]) │    │    ║
║  │  │         │                                              │    │    ║
║  │  │         ▼                                              │    │    ║
║  │  │  LLM: "analyze_news_sentiment çağırayım"              │    │    ║
║  │  │                                                        │    │    ║
║  │  │  loop_messages += [user_msg, function_call_item]       │    │    ║
║  │  │  pending_tool_calls = [{"id":"c1","name":"analyze...}] │    │    ║
║  │  └────────────────────────────────────────────────────────┘    │    ║
║  │    │                                                            │    ║
║  │    ▼  should_continue()                                         │    ║
║  │  pending_tool_calls DOLU? ──── EVET ───────────────────┐       │    ║
║  │                                                         │       │    ║
║  │                                                         ▼       │    ║
║  │                              ┌────────────────────────────┐    │    ║
║  │                              │  dispatch_tools            │    │    ║
║  │                              │                            │    │    ║
║  │                              │  registry.dispatch(        │    │    ║
║  │                              │    "analyze_news_sentiment",│    │    ║
║  │                              │    {"ticker":"NVDA","top_n":5}) │    ║
║  │                              │         │                  │    │    ║
║  │                              │         ▼                  │    │    ║
║  │                              │  Yahoo RSS → LLM analiz    │    │    ║
║  │                              │  → [{"sentiment":"bullish",│    │    ║
║  │                              │      "summary":"..."},...]  │    │    ║
║  │                              │                            │    │    ║
║  │                              │  NewsAnalysis(**item) ×5   │    │    ║
║  │                              │  news = [NewsAnalysis, ...] │    │    ║
║  │                              │                            │    │    ║
║  │                              │  loop_messages +=          │    │    ║
║  │                              │   [function_call_output]   │    │    ║
║  │                              │  pending_tool_calls = []   │    │    ║
║  │                              └────────────────────────────┘    │    ║
║  │                                         │                      │    ║
║  │                    ┌────────────────────┘                      │    ║
║  │                    ▼                                            │    ║
║  │  ┌────────────────────────────────────────────────────────┐    │    ║
║  │  │  call_model (2. tur)                                   │    │    ║
║  │  │                                                        │    │    ║
║  │  │  loop_messages = [user_msg, function_call,             │    │    ║
║  │  │                   function_call_output]  ← geçmiş tam  │    │    ║
║  │  │                                                        │    │    ║
║  │  │  provider.generate(3 mesaj, tools=[...])               │    │    ║
║  │  │         │                                              │    │    ║
║  │  │         ▼                                              │    │    ║
║  │  │  LLM: "NVDA 5 haber, 3 bullish..." (metin yanıt)      │    │    ║
║  │  │                                                        │    │    ║
║  │  │  pending_tool_calls = []                               │    │    ║
║  │  └────────────────────────────────────────────────────────┘    │    ║
║  │    │                                                            │    ║
║  │    ▼  should_continue()                                         │    ║
║  │  pending_tool_calls BOŞ? ──── EVET ───▶  END (subgraph bitti)  │    ║
║  │                                                                 │    ║
║  │  State çıktısı: { news: [NewsAnalysis ×5] }                    │    ║
║  └─────────────────────────────────────────────────────────────────┘    ║
║     │                                                                    ║
║     │  LangGraph otomatik eşleşme:                                      ║
║     │  ResearchState.news  ──▶  FinanceState.news                       ║
║     │                                                                    ║
║     ▼                                                                    ║
║  analyze_sentiment  →  [deep_analysis / short_brief]  →  fetch_price    ║
║     →  draft  →  [revise / END]                                          ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
```

---

## Mesaj Geçmişi (loop_messages) Nasıl Büyüyor

```
Başlangıç : []

call_model (1. tur) sonrası:
  [0] {"role": "user",          "content": "Analyze NVDA..."}
  [1] {"type": "function_call", "call_id": "c1", "name": "analyze_news_sentiment"}

dispatch_tools sonrası:
  [0] {"role": "user",                 "content": "Analyze NVDA..."}
  [1] {"type": "function_call",        "call_id": "c1", ...}
  [2] {"type": "function_call_output", "call_id": "c1", "output": "[{sentiment:...}]"}

call_model (2. tur) sonrası:
  [0] {"role": "user",                 ...}
  [1] {"type": "function_call",        ...}
  [2] {"type": "function_call_output", ...}
  [3] {"role": "assistant",            "content": "NVDA 5 haber, 3 bullish..."}
```

Her node sadece **yeni** mesajları döndürür. `operator.add` bunları mevcut listeye ekler.
`provider.generate()` her çağrıda tam geçmişi görür — LLM ne istediğini ve ne sonuç geldiğini bilir.

---

## should_continue Karar Ağacı

```
call_model bitti
     │
     ├── pending_tool_calls = [...]  →  "dispatch_tools"  (döngü devam)
     │
     └── pending_tool_calls = []    →  "__end__"          (subgraph bitti)
```

---

## State Key Eşleşmesi (Parent ↔ Subgraph)

```
FinanceState (ana graph)          ResearchState (subgraph)
─────────────────────────         ──────────────────────────
ticker          ──────────────▶   ticker           (input)
messages        ✗ eşleşmez        loop_messages    (iç, izole)
                                  pending_tool_calls (iç, izole)
news            ◀──────────────   news             (output)
```

`loop_messages` kasıtlı olarak `messages` değil: parent'ın `BaseMessage` listesiyle
çakışmayı önlemek için farklı isim seçildi.

---

## Hafta-3 loop.py vs research_subgraph.py

```
agent/loop.py                      research_subgraph.py
─────────────────────────────────  ────────────────────────────────────
while iteration < max_iterations:  call_model → dispatch_tools kenarı
if action == "text": break         should_continue() → "__end__"
messages.extend(assistant_turn)    operator.add reducer
messages.append(tool_results)      dispatch_tools'un döndürdüğü loop_messages
registry.dispatch(name, input)     registry.dispatch(tc["name"], tc["input"])
AgentResult(answer, logs)          LangGraph state geçmişi
```
