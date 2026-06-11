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
    # parent'tan gelir — FinanceState.ticker ile aynı isim, LangGraph otomatik eşleştirir
    ticker: str

    # OpenAI Responses API "flat list" formatı:
    #   {"role": "user", "content": "..."}              ← kullanıcı sorusu
    #   {"type": "function_call", "call_id": ..., ...}  ← LLM'in tool talebi
    #   {"type": "function_call_output", "call_id":...} ← tool sonucu
    #
    # Neden `loop_messages` ismi?
    #   FinanceState'in `messages` alanı BaseMessage listesi tutar (LangChain formatı).
    #   Aynı ismi kullansaydık LangGraph parent'ın BaseMessage listesini bu subgraph'a
    #   enjekte ederdi — format uyumsuzluğu nedeniyle LLM çağrısı çökerdi.
    #   Farklı isim = tamamen izole, çakışma yok.
    #
    # Neden `operator.add`?
    #   Her node sadece o turda ürettiği YENİ mesajları döndürür.
    #   operator.add bunları mevcut listeye ekler (biriktirici davranış).
    #   Böylece provider.generate() her çağrıda tam geçmişi görür.
    loop_messages: Annotated[list[Any], operator.add]

    # call_model → dispatch_tools arasında köprü görevi görür.
    # call_model LLM'den gelen tool taleplerini buraya yazar;
    # dispatch_tools bu listeyi okuyup her tool'u çalıştırır, sonra listeyi temizler.
    # should_continue bu alanın doluluk durumuna bakarak yönlendirme yapar.
    # Format: [{"id": "call_abc", "name": "analyze_news_sentiment", "input": {...}}, ...]
    pending_tool_calls: list[dict]

    # subgraph'ın tek çıktısı — parent FinanceState.news ile eşleşir.
    # dispatch_tools tool sonuçlarını NewsAnalysis nesnelerine çevirip buraya yazar.
    # LangGraph subgraph bitince bu değeri parent state'e otomatik taşır.
    news: list   # gerçekte list[NewsAnalysis]


# ---------------------------------------------------------------------------
# HELPER
# ---------------------------------------------------------------------------

def _make_registry() -> ToolRegistry:
    # Subgraph yalnızca analyze_news_sentiment tool'unu kullanır.
    # Yeni bir tool eklemek istersen registry.register(...) ile buraya ekle;
    # LLM description'a bakarak onu da çağırmaya başlar.
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

    # state'teki mevcut mesaj geçmişini kopyala (provider.generate'e verilecek).
    # `or []` guard'ı: state'te `loop_messages` key hiç yoksa (ilk çağrı) patlamaz.
    loop_messages: list = list(state.get("loop_messages") or [])

    # Bu node'un state'e EKLEYECEĞİ yeni mesajları buraya toplayacağız.
    # operator.add reducer bunları mevcut loop_messages'e birleştirir.
    new_msgs: list = []

    # --- İLK ÇAĞRI DURUMU ---
    # loop_messages boşsa subgraph henüz başlıyor demektir.
    # LLM'e "ne yapmasını istiyoruz" diye bir kullanıcı mesajı oluşturuyoruz.
    # Bu mesaj hem new_msgs'e hem de bu turda provider'a verilecek loop_messages'e giriyor.
    if not loop_messages:
        init_msg: dict = {
            "role": "user",
            "content": (
                f"Fetch and analyze recent news sentiment for the stock ticker {ticker}. "
                "Use the analyze_news_sentiment tool to get structured results."
                # NOT: top_n burada belirtilmiyor → LLM Pydantic default'u (5) kullanır.
                # İstersen buraya "Fetch exactly 5 articles." ekleyerek yönlendirebilirsin.
            ),
        }
        new_msgs.append(init_msg)
        loop_messages = [init_msg]   # provider bu turda sadece bu mesajı görecek

    # LLM'i kayıtlı tool listesiyle çağır.
    # provider.generate() OpenAI Responses API'ye istek atar ve:
    #   - Tool call istiyorsa: response.next_action.type == "tool_calls"
    #   - Metin yanıt veriyorsa: response.next_action.type == "text"
    provider = get_provider("openai", "gpt-4o-mini")
    registry = _make_registry()
    response = provider.generate(loop_messages, tools=registry.all_tools())

    # --- ASISTAN TURN'ÜNÜ KAYDET ---
    # raw_assistant_content, OpenAI Responses API'nin döndürdüğü function_call item'larıdır.
    # Örnek:
    #   [{"type": "function_call", "id": "item_1", "call_id": "call_abc",
    #     "name": "analyze_news_sentiment", "arguments": '{"ticker":"NVDA","top_n":5}'}]
    #
    # Bu item'ları new_msgs'e ekliyoruz ki bir sonraki call_model çağrısında
    # provider, LLM'in ne talep ettiğini bilsin (konuşma geçmişi korunsun).
    if response.raw_assistant_content:
        for item in response.raw_assistant_content:
            # item bazen dict, bazen Pydantic nesnesi olabiliyor — her iki forma hazırız.
            new_msgs.append(item if isinstance(item, dict) else item.model_dump())

    # --- PENDING TOOL CALLS ---
    # LLM tool call talep ettiyse bu listeyi dispatch_tools node'u okuyacak.
    # LLM metin yanıt verdiyse bu liste boş olur → should_continue → END.
    pending = [
        {"id": tc.id, "name": tc.name, "input": tc.input}
        for tc in response.next_action.tool_calls
    ]

    return {
        # operator.add: sadece yeni mesajlar döndürülür, state'e birikimli eklenir
        "loop_messages": new_msgs,
        # boşsa → should_continue END'e gider; doluysa → dispatch_tools'a gider
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

    # LLM'e geri gönderilecek tool sonuç mesajları (OpenAI Responses API formatı)
    tool_result_msgs: list[dict] = []

    # Parent graph'a aktarılacak NewsAnalysis nesneleri
    extracted_news: list = []

    for tc in pending:
        try:
            # registry.dispatch: tool adına bakarak doğru handler'ı çağırır,
            # tc["input"] dict'ini Pydantic schema ile validate eder.
            result = registry.dispatch(tc["name"], tc["input"])
            result_str = json.dumps(result, default=str)

            # --- NEWS ÇIKARMA ---
            # Sadece analyze_news_sentiment tool'u news üretiyor.
            # Sonuç başarılıysa (list of dict) her item'ı NewsAnalysis'e çevir.
            # "error" key'i varsa o item bozuk demektir, atla.
            if tc["name"] == "analyze_news_sentiment" and isinstance(result, list):
                for item in result:
                    if "error" not in item:
                        try:
                            # NewsAnalysis(ticker=..., sentiment=..., urgency=..., ...)
                            extracted_news.append(NewsAnalysis(**item))
                        except Exception:
                            # item eksik/bozuk field içeriyorsa sessizce atla;
                            # tek bir bozuk haber tüm subgraph'ı durdurmasın.
                            pass

        except Exception as exc:
            # Tool'un kendisi exception fırlatırsa (API down, timeout vb.):
            # hatayı JSON'a çevir ve LLM'e gönder.
            # LLM "tool başarısız oldu" bilgisini görünce farklı bir strateji deneyebilir
            # ya da özür metni üretebilir — subgraph patlamaz.
            result_str = json.dumps({"error": str(exc), "tool": tc["name"]})

        # --- TOOL SONUÇ MESAJI ---
        # OpenAI Responses API'de tool sonuçları şu formatta eklenir:
        #   {"type": "function_call_output", "call_id": "<LLM'in talep ettiği id>", "output": "..."}
        # call_id eşleşmesi kritik: LLM hangi talebine yanıt geldiğini bununla anlıyor.
        tool_result_msgs.append({
            "type": "function_call_output",
            "call_id": tc["id"],
            "output": result_str,
        })

    return {
        # Tool sonuç mesajları loop_messages'e birikir; bir sonraki call_model görecek
        "loop_messages": tool_result_msgs,
        # Listeyi temizle: should_continue bir sonraki call_model'den sonra tekrar bakacak
        "pending_tool_calls": [],
        # Parent FinanceState.news'e aktarılacak veri — anahtar isim eşleşmesi sayesinde otomatik
        "news": extracted_news,
    }


# ---------------------------------------------------------------------------
# CONDITION
# ---------------------------------------------------------------------------

def should_continue(state: ResearchState) -> Literal["dispatch_tools", "__end__"]:
    """
    call_model'dan sonra çalışır; iki yoldan birini seçer:

        pending_tool_calls DOLU  →  "dispatch_tools"
            LLM bir veya daha fazla tool çağrısı talep etti.
            dispatch_tools node'u bunları çalıştırır, sonra call_model'a geri döner.
            Bu, agent/loop.py'deki `if response.next_action.type == "tool_calls":` bloğuna karşılık gelir.

        pending_tool_calls BOŞ   →  "__end__"
            LLM düz metin yanıt verdi; artık tool çalıştırmak gerekmiyor.
            Subgraph tamamlandı, kontrol parent graph'a (analyze_sentiment node'una) geçer.
            Bu, agent/loop.py'deki `if response.next_action.type == "text": return` satırına karşılık gelir.

    LangGraph "__end__" string'ini özel olarak tanır ve END node'una yönlendirir.
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

    # İki node'u kaydet
    builder.add_node("call_model", call_model)
    builder.add_node("dispatch_tools", dispatch_tools)

    # Graf her zaman call_model'dan başlar
    builder.add_edge(START, "call_model")

    # call_model'dan sonra should_continue karar verir:
    #   tool_calls var → dispatch_tools (döngü devam)
    #   tool_calls yok → END (subgraph bitti)
    builder.add_conditional_edges(
        "call_model",
        should_continue,
        {"dispatch_tools": "dispatch_tools", "__end__": END},
    )

    # dispatch_tools bitince her zaman call_model'a geri dön.
    # LLM tool sonuçlarını değerlendirir; yeni tool talep edebilir ya da metin döner.
    builder.add_edge("dispatch_tools", "call_model")

    # compile(): grafı çalıştırılabilir hale getirir.
    # Checkpointer yok — subgraph kendi state'ini tutmaz; parent graph yönetir.
    return builder.compile()


# ---------------------------------------------------------------------------
# STANDALONE ÇALIŞTIRMA (debug / manuel test)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    ticker = sys.argv[1] if len(sys.argv) > 1 else "NVDA"
    print(f"=== research_subgraph standalone: {ticker} ===\n")

    # Mermaid diagram — subgraph iç yapısını görmek için
    sg = build_research_subgraph()
    print("--- Mermaid ---")
    print(sg.get_graph().draw_mermaid())

    # Subgraph'ı tek başına invoke et (parent graph olmadan)
    # Breakpoint koymak için call_model ve dispatch_tools fonksiyonlarını kullan.
    print(f"--- Invoke: {ticker} ---")
    result = sg.invoke({"ticker": ticker})

    news = result.get("news", [])
    print(f"\nnews ({len(news)} items):")
    for n in news:
        print(f"  [{n.sentiment:8s}] {n.summary[:80]}")

    print(f"\nloop_messages ({len(result.get('loop_messages', []))} items):")
    for msg in result.get("loop_messages", []):
        msg_type = msg.get("type") or msg.get("role", "?")
        print(f"  {msg_type}")
