"""
Day 27 — Week-3 Tool-Use Loop, LangGraph Subgraph olarak bağlandı

Day 26'ya ek değişiklik:
    collect_news node'u kaldırıldı; yerini build_research_subgraph() aldı.
    Subgraph kendi içinde call_model → dispatch_tools döngüsü çalıştırır,
    analyze_news_sentiment tool'u çağırır ve news alanını doldurur.

GENEL AKIŞ (Day 27 — research_subgraph dahil):
    START
      │
      ▼
  research_subgraph   ← Hafta-3 tool-use loop: LLM + analyze_news_sentiment tool
      │
      ▼
  analyze_sentiment   ← Bullish/bearish/neutral sayısını özetle
      │
      ├─── bullish / bearish ──▶ deep_analysis  (10-K RAG sorgusu)
      │                                │
      └─── neutral ──────────▶ short_brief      (hızlı özet, RAG yok)
                                       │
                               (her iki dal)
                                       ▼
                               fetch_price       ← yfinance fiyat verisi
                                       │
                                       ▼
                               draft             ← final brief yaz  [INTERRUPT HERE]
                                       │
                               ┌───────┴────────┐
                               │ feedback var?  │
                               │                │
                              YES               NO
                               │                │
                               ▼                ▼
                             revise            END
                               │
                               └──────────────▶ draft (loop)

HITL Protokolü:
    1. graph.compile(checkpointer=saver, interrupt_after=["draft"])
    2. graph.invoke(input, config)  → draft node'unda durur
    3. state = graph.get_state(config)  → draft'ı kullanıcıya göster
    4a. Approve:  graph.invoke(None, config)  → END
    4b. Reject:   graph.update_state(config, {"feedback": "..."})
                  graph.invoke(None, config)  → revise → draft (loop)

STATE'İN YOLCULUĞU:
    Başlangıç             : { ticker: "NVDA", messages: [] }
    collect_news sonrası  : + { news: [NewsAnalysis, ...] }
    analyze_sentiment son.: + { sentiment_summary: "BULLISH — 3 bullish, ..." }
    deep_analysis sonrası : + { risks: ["10-K'dan risk özeti..."] }   (bullish/bearish)
    short_brief sonrası   : + { draft: "[SHORT BRIEF] NVDA — ..." }   (neutral)
    fetch_price sonrası   : + { price_data: { price: 134.5, ... } }
    draft sonrası         : + { draft: "[DRAFT] NVDA Investment Brief..." }  ← INTERRUPT
    revise sonrası        : + { draft: "...+ [REVISION]...", feedback: "" }
"""

from typing import Literal

from langgraph.graph import END, START, StateGraph
from providers.factory import get_provider as _get_provider

from agent.tools.finance import _GetStockDataInput, _get_stock_data
from agent.tools.rag import _Query10KInput, _query_10k
from agent.tools.sentiment import _AnalyzeNewsSentimentInput, _analyze_news_sentiment
from graph.research_subgraph import build_research_subgraph
from graph.state import FinanceState
from schemas import NewsAnalysis


# ---------------------------------------------------------------------------
# NODE 1: collect_news  (Day 23'ten değişmedi)
# ---------------------------------------------------------------------------

def collect_news(state: FinanceState) -> dict:
    """
    Yahoo Finance RSS'den haber başlıklarını çeker ve her biri için
    LLM tabanlı sentiment analizi yapar.
    """
    ticker = state["ticker"]
    raw = _analyze_news_sentiment(_AnalyzeNewsSentimentInput(ticker=ticker, top_n=5))
    news = [NewsAnalysis(**item) for item in raw if "error" not in item]
    return {"news": news}


# ---------------------------------------------------------------------------
# NODE 2: analyze_sentiment  (Day 23'ten değişmedi)
# ---------------------------------------------------------------------------

def analyze_sentiment(state: FinanceState) -> dict:
    """
    Haber listesini okur ve genel piyasa duyarlılığını özetleyen bir metin üretir.

    Döndürdüğü sentiment_summary'ye bakarak route_by_sentiment dallanma kararı verir:
        "BULLISH — ..."  →  deep_analysis
        "BEARISH — ..."  →  deep_analysis
        "NEUTRAL — ..."  →  short_brief
    """
    news: list[NewsAnalysis] = state.get("news", [])

    if not news:
        return {"sentiment_summary": "No news available."}

    counts = {"bullish": 0, "bearish": 0, "neutral": 0}
    for item in news:
        counts[item.sentiment] = counts.get(item.sentiment, 0) + 1

    dominant = max(counts, key=counts.get)
    total = len(news)

    summary = (
        f"{dominant.upper()} — "
        f"{counts['bullish']} bullish, {counts['bearish']} bearish, "
        f"{counts['neutral']} neutral ({total} articles)"
    )

    return {"sentiment_summary": summary}


# ---------------------------------------------------------------------------
# ROUTING FUNCTION  (Day 24 — YENİ)
# ---------------------------------------------------------------------------

def route_by_sentiment(state: FinanceState) -> Literal["deep_analysis", "short_brief"]:
    """
    analyze_sentiment node'undan sonra LangGraph'ın çalıştıracağı routing fonksiyonu.

    Karar mantığı:
        sentiment_summary "BULLISH" veya "BEARISH" ile başlıyor → deep_analysis
        Diğer tüm durumlar (neutral, veri yok, boş)              → short_brief

    LangGraph bu fonksiyonun döndürdüğü string'i node adı olarak yorumlar ve
    add_conditional_edges()'de belirtilen haritaya göre geçiş yapar.
    """
    summary = state.get("sentiment_summary", "").upper()
    if summary.startswith("BULLISH") or summary.startswith("BEARISH"):
        return "deep_analysis"
    return "short_brief"


# ---------------------------------------------------------------------------
# NODE 3a: deep_analysis  (Day 24 — YENİ, bullish/bearish dal)
# ---------------------------------------------------------------------------

def deep_analysis(state: FinanceState) -> dict:
    """
    Bullish veya bearish sentiment sinyali aldığında devreye girer.

    10-K RAG pipeline'ını kullanarak şirketin yıllık raporundan risk faktörleri
    ve stratejik öncelikleri sorgular.

    Neden RAG burada?
        Güçlü bir sentiment (iyi ya da kötü) → yatırımcı "neden?" diye sorar.
        10-K, şirketin kendi ağzından resmi risk ve strateji açıklamasıdır.

    State'e eklediği alan:
        risks: list[str]  — 10-K yanıtı (genellikle tek uzun string) veya
                            desteklenmeyen ticker için hata mesajı

    Desteklenen ticker'lar: NVDA (bkz. agent/tools/rag.py → _TICKER_COLLECTIONS)
    """
    ticker = state["ticker"]

    result = _query_10k(_Query10KInput(
        ticker=ticker,
        question=(
            "What are the key risk factors, competitive threats, and strategic "
            "initiatives mentioned in the 10-K annual report?"
        ),
    ))

    if "error" in result:
        risks = [result["error"]]
    else:
        risks = [result.get("answer", "No answer returned.")]

    return {"risks": risks}


# ---------------------------------------------------------------------------
# NODE 3b: short_brief  (Day 24 — YENİ, neutral dal)
# ---------------------------------------------------------------------------

def short_brief(state: FinanceState) -> dict:
    """
    Neutral sentiment sinyali aldığında devreye girer.

    RAG çağrısı yapmadan, mevcut haber özetlerini ve sentiment özetini birleştirerek
    kısa bir draft üretir.

    Neden RAG yok?
        Neutral → belirsizlik → pahalı 10-K sorgusu mantıklı değil.
        Yatırımcıya "bekleme modunda kal" mesajı iletmek yeterli.

    State'e eklediği alan:
        draft: str  — "[SHORT BRIEF] TICKER — sentiment. Top headlines: ..."
    """
    ticker = state["ticker"]
    sentiment = state.get("sentiment_summary", "No data.")
    news = state.get("news", [])
    headlines = " | ".join(n.summary for n in news[:3]) if news else "No headlines available."
    draft = f"[SHORT BRIEF] {ticker} — {sentiment}. Top headlines: {headlines}"
    return {"draft": draft}


# ---------------------------------------------------------------------------
# NODE 4: fetch_price  (Day 23'ten değişmedi)
# ---------------------------------------------------------------------------

def fetch_price(state: FinanceState) -> dict:
    """
    yfinance ile güncel fiyat verisi çeker.
    Her iki branch (deep_analysis ve short_brief) bu node'da birleşir.
    """
    ticker = state["ticker"]
    price_data = _get_stock_data(_GetStockDataInput(ticker=ticker))
    return {"price_data": price_data}


# ---------------------------------------------------------------------------
# NODE 5: draft  (Day 26 — YENİ, HITL interrupt noktası)
# ---------------------------------------------------------------------------

def write_draft(state: FinanceState) -> dict:
    """
    Tüm toplanan veriden (sentiment, fiyat, riskler) yatırım brief'i taslağı yazar.

    Bu node interrupt_after=["draft"] ile durdurma noktasıdır.
    Human reviewer:
        - Onaylarsa → graph.invoke(None, config)          → END
        - Red ederse → graph.update_state(..., feedback)
                       graph.invoke(None, config)          → revise → draft (loop)

    Risks varsa (deep_analysis dalı) bunları öne çıkarır.
    Yoksa short_brief'ten gelen taslağı veya minimal özeti kullanır.
    """
    # revise_draft'tan geldikten sonra draft zaten güncel; tekrar üretme.
    # "--- REVISION ---" işaretçisi revise_draft tarafından eklenir.
    existing = state.get("draft", "")
    if "--- REVISION ---" in existing:
        return {"draft": existing}

    ticker = state["ticker"]
    sentiment = state.get("sentiment_summary", "N/A")
    price = state.get("price_data", {})
    risks = state.get("risks", [])

    price_val = price.get("price", "N/A")
    pct = price.get("pct_change")
    price_str = f"${price_val}" + (f" ({pct:+.2f}% 1mo)" if pct is not None else "")

    if risks:
        draft = (
            f"[DRAFT] {ticker} Investment Brief\n\n"
            f"Sentiment: {sentiment}\n"
            f"Price: {price_str}\n\n"
            f"Key Risks & Strategy (10-K):\n"
            f"{risks[0][:500]}"
        )
    else:
        short = state.get("draft", "")
        draft = short if short else f"[DRAFT] {ticker} — {sentiment}. Price: {price_str}"

    return {"draft": draft}


# ---------------------------------------------------------------------------
# NODE 6: revise  (Day 26 — YENİ, feedback entegrasyonu)
# ---------------------------------------------------------------------------

def revise_draft(state: FinanceState) -> dict:
    """
    Human feedback'i draft'a entegre eder.

    Neden feedback alanı sıfırlanıyor?
        revise → draft döngüsünde sonsuz tekrarı önlemek için.
        Feedback temizlenince route_after_draft bir sonraki draft
        interrupt'ta "approved" dalına gider.
    """
    draft = state.get("draft", "")
    feedback = state.get("feedback", "")
    revised = (
        f"{draft}\n\n"
        f"--- REVISION ---\n"
        f"Editor feedback: {feedback}\n"
        f"(Content updated based on above feedback.)"
    )
    return {"draft": revised, "feedback": ""}


# ---------------------------------------------------------------------------
# LLM HELPER: reflect  (Day 28 — testlerde patch'lenebilir)
# ---------------------------------------------------------------------------

def _call_reflect_llm(draft: str, ticker: str) -> str:
    provider = _get_provider("openai", "gpt-4o-mini")
    messages = [
        {
            "role": "user",
            "content": (
                f"You are a financial editor reviewing an investment brief for {ticker}.\n\n"
                f"DRAFT:\n{draft}\n\n"
                f"Check for these required elements:\n"
                f"1. Ticker symbol ({ticker}) is mentioned\n"
                f"2. Sentiment (bullish/bearish/neutral) is stated\n"
                f"3. Current price data is included\n"
                f"4. At least one key risk or analysis point\n\n"
                f"If ALL elements are present, return the draft UNCHANGED.\n"
                f"If any element is missing, add it concisely and return the complete improved draft.\n"
                f"Return ONLY the draft text, no commentary."
            ),
        }
    ]
    response = provider.generate(messages)
    return response.text.strip()


# ---------------------------------------------------------------------------
# NODE 6b: reflect  (Day 28 — YENİ, otomatik draft değerlendirme)
# ---------------------------------------------------------------------------

def reflect_draft(state: FinanceState) -> dict:
    """
    Draft'ı LLM ile değerlendirir; eksik eleman varsa otomatik tamamlar.

    Reflection pattern: LLM kendi ürettiği çıktıyı başka bir LLM çağrısıyla
    denetler. Human interrupt'tan ÖNCE çalışır; insan zaten iyileştirilmiş
    bir draft görür.
    """
    draft = state.get("draft", "")
    ticker = state.get("ticker", "UNKNOWN")
    improved = _call_reflect_llm(draft, ticker)
    reflect_note = "auto-reflected" if improved != draft else "no changes"
    return {"draft": improved, "reflect_note": reflect_note}


# ---------------------------------------------------------------------------
# ROUTING: reflect sonrası  (Day 26 logic, Day 28'de reflect'e taşındı)
# ---------------------------------------------------------------------------

def route_after_draft(state: FinanceState) -> Literal["revise", "__end__"]:
    """
    Human onayı/red kararını state'teki feedback alanına göre verir.

        feedback boş (onay) → "__end__" → END
        feedback dolu (red) → "revise"  → revise node
    """
    return "revise" if state.get("feedback", "").strip() else "__end__"


# ---------------------------------------------------------------------------
# GRAF DERLEME
# ---------------------------------------------------------------------------

def build_finance_graph(checkpointer=None, interrupt_after: list[str] | None = None):
    """
    HITL destekli 7 node'lu grafı derler.

    Lineer (Day 23):
        collect_news → analyze_sentiment → fetch_price

    Conditional (Day 24):
        collect_news → analyze_sentiment
                             ├─ bullish/bearish → deep_analysis → fetch_price
                             └─ neutral         → short_brief   → fetch_price

    Checkpointing (Day 25):
        checkpointer=SqliteSaver → her node sonrası state SQLite'a yazılır.

    HITL (Day 26):
        fetch_price → draft  [interrupt_after=["draft"]]
                        ├─ feedback boş → END   (approve)
                        └─ feedback dolu → revise → draft (loop)

    interrupt_after:
        ["draft"] geçilirse graf draft node'undan sonra durur.
        None veya [] geçilirse interrupt olmadan normal çalışır.
    """
    builder = StateGraph(FinanceState)

    # Node'ları kaydet
    # Day 27: collect_news → research_subgraph (Hafta-3 tool-use loop)
    builder.add_node("collect_news", build_research_subgraph())
    builder.add_node("analyze_sentiment", analyze_sentiment)
    builder.add_node("deep_analysis", deep_analysis)
    builder.add_node("short_brief", short_brief)
    builder.add_node("fetch_price", fetch_price)

    # Day 26: HITL node'ları
    builder.add_node("draft", write_draft)
    builder.add_node("revise", revise_draft)

    # Day 28: Reflection node
    builder.add_node("reflect", reflect_draft)

    # Lineer başlangıç
    builder.add_edge(START, "collect_news")
    builder.add_edge("collect_news", "analyze_sentiment")

    # Conditional branch — Day 24
    builder.add_conditional_edges(
        "analyze_sentiment",
        route_by_sentiment,
        {"deep_analysis": "deep_analysis", "short_brief": "short_brief"},
    )

    # Her iki branch'i fetch_price'ta birleştir
    builder.add_edge("deep_analysis", "fetch_price")
    builder.add_edge("short_brief", "fetch_price")

    # Day 26: fetch_price → draft (interrupt noktası)
    builder.add_edge("fetch_price", "draft")

    # Day 28: draft → reflect (LLM otomatik değerlendirme)
    builder.add_edge("draft", "reflect")

    # Day 28: reflect → revise (feedback var) veya END (approved)
    builder.add_conditional_edges(
        "reflect",
        route_after_draft,
        {"revise": "revise", "__end__": END},
    )
    builder.add_edge("revise", "draft")

    return builder.compile(
        checkpointer=checkpointer,
        interrupt_after=interrupt_after or [],
    )


# ---------------------------------------------------------------------------
# ÇALIŞTIRMA (manuel test + Mermaid diagram)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from graph.checkpointer import make_checkpointer

    # Mermaid diagram
    _graph_no_cp = build_finance_graph()
    mermaid_text = _graph_no_cp.get_graph().draw_mermaid()
    print("=== Mermaid Diagram ===")
    print(mermaid_text)

    # PNG olarak kaydet
    try:
        png_bytes = _graph_no_cp.get_graph().draw_mermaid_png()
        with open("graph/finance_graph.png", "wb") as f:
            f.write(png_bytes)
        print("Graph PNG saved → graph/finance_graph.png\n")
    except Exception as e:
        print(f"PNG export skipped: {e}\n")

    # HITL demo: NVDA için interrupt_after=["draft"]
    ticker = sys.argv[1] if len(sys.argv) > 1 else "NVDA"

    with make_checkpointer("hitl_demo.db") as cp:
        graph = build_finance_graph(checkpointer=cp, interrupt_after=["reflect"])
        config = {"configurable": {"thread_id": f"hitl-{ticker}-demo"}}

        print(f"=== HITL Run: {ticker} ===")
        result = graph.invoke({"ticker": ticker, "messages": []}, config)

        print("\n[INTERRUPT] Graf draft node'unda durdu.")
        print(f"Draft önizleme:\n{result.get('draft', '')[:400]}\n")

        # Simüle onay
        answer = input("Approve (a) / Reject with feedback (r)? [a]: ").strip().lower() or "a"

        if answer == "r":
            feedback = input("Feedback: ").strip()
            graph.update_state(config, {"feedback": feedback})
            print("\n[REVISE] Feedback uygulanıyor...")
            revised = graph.invoke(None, config)
            print(f"Revised draft:\n{revised.get('draft', '')[:400]}\n")
            # Revised draft'ı onayla
            graph.invoke(None, config)
        else:
            graph.invoke(None, config)

        print("[DONE] Graf tamamlandı.")
