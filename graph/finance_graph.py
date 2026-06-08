"""
Day 24 — Conditional Routing by Sentiment

GENEL AKIŞ (Conditional Branch'li):
    START
      │
      ▼
  collect_news        ← Yahoo Finance RSS'den haber çek + LLM sentiment analizi
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
                                      END

Routing Mantığı:
    sentiment_summary "BULLISH" veya "BEARISH" ile başlıyorsa → deep_analysis
    Aksi hâlde (neutral / veri yok) → short_brief

STATE'İN YOLCULUĞU:
    Başlangıç             : { ticker: "NVDA", messages: [] }
    collect_news sonrası  : + { news: [NewsAnalysis, ...] }
    analyze_sentiment son.: + { sentiment_summary: "BULLISH — 3 bullish, ..." }
    deep_analysis sonrası : + { risks: ["10-K'dan risk özeti..."] }   (bullish/bearish)
    short_brief sonrası   : + { draft: "[SHORT BRIEF] NVDA — ..." }   (neutral)
    fetch_price sonrası   : + { price_data: { price: 134.5, ... } }
"""

from typing import Literal

from langgraph.graph import END, START, StateGraph

from agent.tools.finance import _GetStockDataInput, _get_stock_data
from agent.tools.rag import _Query10KInput, _query_10k
from agent.tools.sentiment import _AnalyzeNewsSentimentInput, _analyze_news_sentiment
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
# GRAF DERLEME
# ---------------------------------------------------------------------------

def build_finance_graph():
    """
    Conditional routing ile 5 node'lu branch'li grafı derler.

    Lineer (Day 23):
        collect_news → analyze_sentiment → fetch_price

    Conditional (Day 24):
        collect_news → analyze_sentiment
                             ├─ bullish/bearish → deep_analysis → fetch_price
                             └─ neutral         → short_brief   → fetch_price

    add_conditional_edges(source, routing_fn, mapping):
        - routing_fn döndürdüğü string'e göre hedef node seçilir
        - mapping: { routing_fn_output: target_node_name }
        - mapping atlanırsa routing_fn'in döndürdüğü string doğrudan node adı olarak yorumlanır
    """
    builder = StateGraph(FinanceState)

    # Node'ları kaydet
    builder.add_node("collect_news", collect_news)
    builder.add_node("analyze_sentiment", analyze_sentiment)
    builder.add_node("deep_analysis", deep_analysis)
    builder.add_node("short_brief", short_brief)
    builder.add_node("fetch_price", fetch_price)

    # Lineer başlangıç
    builder.add_edge(START, "collect_news")
    builder.add_edge("collect_news", "analyze_sentiment")

    # Conditional branch — Day 24'ün özü
    builder.add_conditional_edges(
        "analyze_sentiment",
        route_by_sentiment,
        {"deep_analysis": "deep_analysis", "short_brief": "short_brief"},
    )

    # Her iki branch'i fetch_price'ta birleştir
    builder.add_edge("deep_analysis", "fetch_price")
    builder.add_edge("short_brief", "fetch_price")

    builder.add_edge("fetch_price", END)

    return builder.compile()


# ---------------------------------------------------------------------------
# ÇALIŞTIRMA (manuel test + Mermaid diagram)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    graph = build_finance_graph()

    # Mermaid diagram metnini stdout'a yaz (branch'ler görünür)
    mermaid_text = graph.get_graph().draw_mermaid()
    print("=== Mermaid Diagram ===")
    print(mermaid_text)

    # NVDA için grafı çalıştır
    result = graph.invoke({"ticker": "NVDA", "messages": []})

    print("\n=== Finance Graph Output ===")
    print(f"Ticker     : {result['ticker']}")
    print(f"News count : {len(result.get('news', []))}")
    print(f"Sentiment  : {result.get('sentiment_summary')}")
    price = result.get("price_data", {})
    print(f"Price      : ${price.get('price')}  ({price.get('pct_change', 0):+.2f}% 1mo)")

    # Branch sonucunu göster
    if result.get("risks"):
        print(f"\n[deep_analysis] Risks (first 200 chars):\n  {result['risks'][0][:200]}...")
    elif result.get("draft"):
        print(f"\n[short_brief] Draft:\n  {result['draft']}")

    # PNG olarak kaydet
    try:
        png_bytes = graph.get_graph().draw_mermaid_png()
        with open("graph/finance_graph.png", "wb") as f:
            f.write(png_bytes)
        print("\nGraph PNG saved → graph/finance_graph.png")
    except Exception as e:
        print(f"\nPNG export skipped: {e}")
