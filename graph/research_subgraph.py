"""
Day 27 — Week-3 Tool-Use Loop as LangGraph Subgraph

Hafta 3'te agent/loop.py'da while döngüsüyle yazılan tool-use akışını
LangGraph node'larına taşır ve ana finance_graph.py'a subgraph olarak bağlanır.

AKIŞ:
    START
      │
      ▼
  call_model     ← ticker → LLM'e gönder, tool call talep eder
      │
      │  should_continue()
      ├── tool_calls var → dispatch_tools → call_model (döngü)
      └── tool_calls yok → END

SUBGRAPH STATE:
    ticker           — parent FinanceState'ten gelir (otomatik eşleşme)
    loop_messages    — OpenAI Responses API flat format; operator.add ile birikir
                       (parent'ın BaseMessage tabanlı messages alanıyla çakışmaz)
    pending_tool_calls — call_model'dan dispatch_tools'a tool call bilgisi taşır
    news             — parent FinanceState.news'e çıkar (otomatik eşleşme)

PARENT BAĞLANTISI (finance_graph.py):
    builder.add_node("collect_news", build_research_subgraph())

    LangGraph otomatik key eşleşmesi:
        Girdi:  FinanceState.ticker → ResearchState.ticker
        Çıktı:  ResearchState.news  → FinanceState.news

HAFTA-3 KARŞILAŞTIRMASI:
    agent/loop.py          →  research_subgraph.py
    ─────────────────────────────────────────────
    while loop             →  call_model node (döngü)
    registry.dispatch()    →  dispatch_tools node
    next_action.type check →  should_continue() condition
    list[IterationLog]     →  LangGraph state geçmişi
"""

import json
import operator
from typing import Annotated, Any, Literal

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from agent.registry import ToolRegistry
from agent.tools.sentiment import analyze_news_sentiment
from providers.factory import get_provider
from schemas import NewsAnalysis


# ---------------------------------------------------------------------------
# SUBGRAPH STATE
# ---------------------------------------------------------------------------

class ResearchState(TypedDict, total=False):
    ticker: str                                          # parent'tan gelir
    loop_messages: Annotated[list[Any], operator.add]   # OpenAI Responses API flat format
    pending_tool_calls: list[dict]                       # [{id, name, input}]; call_model → dispatch_tools
    news: list                                           # list[NewsAnalysis]; parent'a çıkar


# ---------------------------------------------------------------------------
# HELPER
# ---------------------------------------------------------------------------

def _make_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(analyze_news_sentiment)
    return reg


# ---------------------------------------------------------------------------
# NODE 1: call_model
# ---------------------------------------------------------------------------

def call_model(state: ResearchState) -> dict:
    """
    Birikmiş mesaj geçmişini (loop_messages) LLM'e gönderir.

    İlk çağrı: ticker'dan kullanıcı mesajı oluşturur.
    Sonraki çağrılar: tool sonuçlarıyla zenginleşmiş geçmişi kullanır.

    LLM tool call talep ederse → pending_tool_calls doldurulur → dispatch_tools.
    LLM metin yanıt verirse   → pending_tool_calls boş kalır  → END.
    """
    ticker = state.get("ticker", "")
    loop_messages: list = list(state.get("loop_messages") or [])
    new_msgs: list = []

    # İlk çağrı: geçmişte hiç mesaj yok → kullanıcı mesajı oluştur
    if not loop_messages:
        init_msg: dict = {
            "role": "user",
            "content": (
                f"Fetch and analyze recent news sentiment for the stock ticker {ticker}. "
                "Use the analyze_news_sentiment tool to get structured results."
            ),
        }
        new_msgs.append(init_msg)
        loop_messages = [init_msg]

    provider = get_provider("openai", "gpt-4o-mini")
    registry = _make_registry()
    response = provider.generate(loop_messages, tools=registry.all_tools())

    # Asistan turn'ünü kaydet (function_call item'lar veya metin)
    if response.raw_assistant_content:
        for item in response.raw_assistant_content:
            new_msgs.append(item if isinstance(item, dict) else item.model_dump())

    pending = [
        {"id": tc.id, "name": tc.name, "input": tc.input}
        for tc in response.next_action.tool_calls
    ]

    return {
        "loop_messages": new_msgs,
        "pending_tool_calls": pending,
    }


# ---------------------------------------------------------------------------
# NODE 2: dispatch_tools
# ---------------------------------------------------------------------------

def dispatch_tools(state: ResearchState) -> dict:
    """
    pending_tool_calls listesindeki her tool'u çalıştırır.

    analyze_news_sentiment sonuçlarını NewsAnalysis nesnelerine dönüştürür
    ve news alanına yazar — parent FinanceState.news'e çıkar.

    Tool sonuçlarını OpenAI Responses API formatında loop_messages'e ekler;
    bir sonraki call_model çağrısında LLM bunları görecek.
    """
    registry = _make_registry()
    pending: list[dict] = state.get("pending_tool_calls") or []
    tool_result_msgs: list[dict] = []
    extracted_news: list = []

    for tc in pending:
        try:
            result = registry.dispatch(tc["name"], tc["input"])
            result_str = json.dumps(result, default=str)
            if tc["name"] == "analyze_news_sentiment" and isinstance(result, list):
                for item in result:
                    if "error" not in item:
                        try:
                            extracted_news.append(NewsAnalysis(**item))
                        except Exception:
                            pass
        except Exception as exc:
            result_str = json.dumps({"error": str(exc), "tool": tc["name"]})

        # OpenAI Responses API: function_call_output formatı
        tool_result_msgs.append({
            "type": "function_call_output",
            "call_id": tc["id"],
            "output": result_str,
        })

    return {
        "loop_messages": tool_result_msgs,
        "pending_tool_calls": [],
        "news": extracted_news,
    }


# ---------------------------------------------------------------------------
# CONDITION
# ---------------------------------------------------------------------------

def should_continue(state: ResearchState) -> Literal["dispatch_tools", "__end__"]:
    """
    Tool call var mı?
        Evet → dispatch_tools (döngüye devam)
        Hayır → END (subgraph tamamlandı)
    """
    return "dispatch_tools" if state.get("pending_tool_calls") else "__end__"


# ---------------------------------------------------------------------------
# SUBGRAPH DERLEME
# ---------------------------------------------------------------------------

def build_research_subgraph():
    """
    ResearchState tabanlı tool-loop subgraph'ını derler ve döndürür.

    finance_graph.py'da kullanım:
        builder.add_node("collect_news", build_research_subgraph())

    State key eşleşmeleri (LangGraph otomatik):
        Girdi : FinanceState.ticker → ResearchState.ticker
        Çıktı : ResearchState.news  → FinanceState.news
    """
    builder = StateGraph(ResearchState)

    builder.add_node("call_model", call_model)
    builder.add_node("dispatch_tools", dispatch_tools)

    builder.add_edge(START, "call_model")
    builder.add_conditional_edges(
        "call_model",
        should_continue,
        {"dispatch_tools": "dispatch_tools", "__end__": END},
    )
    builder.add_edge("dispatch_tools", "call_model")

    return builder.compile()
