"""Day 17 + Day 20: Multi-turn tool-use agent loop.

Day 20 additions:
- structlog JSON logging (tool_name, args, duration_ms, tokens, status per call)
- Error recovery: tool exceptions returned to LLM as structured JSON so it can retry
- Token budget guard at 100 000 tokens (configurable)
- Per-run trace written to traces/run_<id>.json
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import structlog
from langsmith import traceable as ls_traceable, set_run_metadata as ls_set_metadata

from providers.base import LLMProvider
from agent.registry import ToolRegistry
from agent.tracing import write_trace

logger = structlog.get_logger(__name__)

# Varsayılan token bütçesi: tek bir agent çalışmasında harcanabilecek maksimum token.
# Aşılırsa loop durur ve o ana kadarki yanıtı döndürür.
TOKEN_BUDGET_DEFAULT = 100_000


@dataclass
class ToolCallLog:
    """Bir tool çağrısının sonucunu tutar — trace dosyasına yazılır."""
    name: str
    args: dict[str, Any]
    duration_ms: float
    status: str        # "success" | "error"
    error: str | None = None


@dataclass
class IterationLog:
    """Tek bir LLM iterasyonundaki tüm tool çağrılarını ve token kullanımını saklar."""
    iteration: int
    tool_calls: list[ToolCallLog]
    tokens_used: int


@dataclass
class AgentResult:
    """run_agent() fonksiyonunun dönüş değeri."""
    answer: str
    iterations: int
    total_tokens: int
    logs: list[IterationLog] = field(default_factory=list)
    trace_path: str | None = None   # traces/run_<id>.json dosyasının yolu


@ls_traceable(run_type="tool")
def _dispatch_tool(name: str, registry: ToolRegistry, args: dict) -> Any:
    """Tool çağrısını LangSmith'te 'tool' run olarak izler."""
    return registry.dispatch(name, args)


@ls_traceable(run_type="chain", name="finance_agent_run")
def run_agent(
    query: str,
    provider: LLMProvider,
    registry: ToolRegistry,
    max_iterations: int = 10,
    token_budget: int = TOKEN_BUDGET_DEFAULT,
    write_trace_file: bool = True,
    metadata: dict | None = None,   # {"ticker": "NVDA", "provider": "openai", "model": "gpt-4o-mini"}
) -> AgentResult:
    """Tool-use döngüsü: LLM yanıt üretene ya da bütçe/iterasyon sınırına gelene kadar çalışır.

    Akış:
      1. LLM'e mesajları gönder.
      2. LLM tool çağırmak istiyorsa → tool'ları çalıştır, sonuçları mesaj geçmişine ekle.
      3. LLM düz metin yanıt verdiyse → döngüden çık, AgentResult döndür.
      4. Token bütçesi aşıldıysa → erken çık.
      5. max_iterations dolduysа → uyarıyla çık.
    """
    run_id = uuid.uuid4().hex[:12]   # her çalışmayı trace dosyasında ayırt etmek için
    run_start = time.perf_counter()
    log = logger.bind(run_id=run_id)  # tüm log satırlarına run_id otomatik eklenir

    # LangSmith custom metadata: ticker, provider, model gibi bilgileri dashboard'da görünür kılar.
    # LANGCHAIN_TRACING_V2=false iken set_run_metadata no-op olarak davranır.
    _meta = {"provider": type(provider).__name__, "run_id": run_id, **(metadata or {})}
    try:
        ls_set_metadata(_meta)
    except Exception:
        pass

    messages: list[dict] = [{"role": "user", "content": query}]
    tools = registry.all_tools()
    total_tokens = 0
    logs: list[IterationLog] = []

    log.info("agent_start", query=query[:120], tools=[t.name for t in tools], token_budget=token_budget)

    for iteration in range(1, max_iterations + 1):
        t_llm = time.perf_counter()
        log.info("llm_call", iteration=iteration, messages=len(messages))

        response = provider.generate(messages, tools=tools)
        total_tokens += response.usage.total_tokens
        llm_ms = round((time.perf_counter() - t_llm) * 1000, 1)

        log.info(
            "llm_response",
            iteration=iteration,
            action=response.next_action.type,
            tokens=response.usage.total_tokens,
            total_tokens=total_tokens,
            duration_ms=llm_ms,
        )

        # ── Token bütçesi kontrolü ─────────────────────────────────────────────
        # Bu kontrol LLM yanıtından SONRA yapılır; çünkü token sayısını ancak
        # yanıt geldikten sonra öğrenebiliriz.
        if total_tokens > token_budget:
            log.warning("token_budget_exceeded", total_tokens=total_tokens, budget=token_budget)
            logs.append(IterationLog(iteration=iteration, tool_calls=[], tokens_used=response.usage.total_tokens))
            result = AgentResult(
                answer=response.text or "[token budget exceeded]",
                iterations=iteration,
                total_tokens=total_tokens,
                logs=logs,
            )
            if write_trace_file:
                result.trace_path = _flush_trace(run_id, run_start, query, result)
            return result

        # ── LLM düz metin yanıt verdi → döngüyü bitir ─────────────────────────
        if response.next_action.type == "text":
            logs.append(IterationLog(iteration=iteration, tool_calls=[], tokens_used=response.usage.total_tokens))
            result = AgentResult(
                answer=response.text,
                iterations=iteration,
                total_tokens=total_tokens,
                logs=logs,
            )
            if write_trace_file:
                result.trace_path = _flush_trace(run_id, run_start, query, result)
            log.info("agent_done", iterations=iteration, total_tokens=total_tokens, answer_len=len(result.answer))
            return result

        # ── Tool çağrısı dalı ──────────────────────────────────────────────────
        # LLM'in asistan mesajını (tool call talepleriyle birlikte) geçmişe ekle;
        # provider bunu kendi formatına (OpenAI Responses API vs Anthropic) dönüştürür.
        provider.extend_messages_with_assistant_turn(messages, response)

        iteration_calls: list[ToolCallLog] = []
        tool_results: list[dict] = []

        for tc in response.next_action.tool_calls:
            t_tool = time.perf_counter()
            log.info("tool_call", iteration=iteration, tool=tc.name, args=tc.input)

            try:
                raw = _dispatch_tool(tc.name, registry=registry, args=tc.input)
                result_str = json.dumps(raw, default=str)
                duration_ms = round((time.perf_counter() - t_tool) * 1000, 1)
                log.info("tool_result", tool=tc.name, status="success", duration_ms=duration_ms, result=result_str[:200])
                iteration_calls.append(ToolCallLog(
                    name=tc.name, args=tc.input, duration_ms=duration_ms, status="success"
                ))
            except Exception as exc:
                duration_ms = round((time.perf_counter() - t_tool) * 1000, 1)
                error_msg = f"{type(exc).__name__}: {exc}"
                # Tam traceback'i log'a yaz (sadece mesajı değil)
                log.error("tool_error", tool=tc.name, error=error_msg, duration_ms=duration_ms, exc_info=True)
                # Hatayı structured JSON olarak LLM'e gönder; LLM farklı bir input
                # deneyebilir ya da bu tool'u atlayıp devam edebilir.
                result_str = json.dumps({"error": error_msg, "tool": tc.name, "hint": "try a different input or skip this tool"})
                iteration_calls.append(ToolCallLog(
                    name=tc.name, args=tc.input, duration_ms=duration_ms, status="error", error=error_msg
                ))

            tool_results.append({"tool_call_id": tc.id, "result": result_str})

        logs.append(IterationLog(
            iteration=iteration,
            tool_calls=iteration_calls,
            tokens_used=response.usage.total_tokens,
        ))

        # Tool sonuçlarını mesaj geçmişine ekle; bir sonraki iterasyonda LLM bunları görecek
        provider.extend_messages_with_tool_results(messages, tool_results)

    # ── max_iterations doldu ───────────────────────────────────────────────────
    # Buraya düşmek genellikle prompt'un çok geniş ya da tool'ların sonsuz döngüye
    # girdiğine işaret eder; max_iterations değerini artırmak yerine önce nedeni araştır.
    log.warning("max_iterations_reached", max_iterations=max_iterations)
    result = AgentResult(
        answer="[max iterations reached]",
        iterations=max_iterations,
        total_tokens=total_tokens,
        logs=logs,
    )
    if write_trace_file:
        result.trace_path = _flush_trace(run_id, run_start, query, result)
    return result


def _flush_trace(run_id: str, run_start: float, query: str, result: AgentResult) -> str:
    """Çalışmanın özetini traces/run_<id>.json dosyasına yazar ve dosya yolunu döndürür."""
    total_duration_ms = round((time.perf_counter() - run_start) * 1000, 1)
    trace = {
        "run_id": run_id,
        "query": query,
        "total_tokens": result.total_tokens,
        "iterations": result.iterations,
        "total_duration_ms": total_duration_ms,
        "iterations_detail": [
            {
                "iteration": it.iteration,
                "tokens_used": it.tokens_used,
                "tool_calls": [
                    {
                        "name": tc.name,
                        "args": tc.args,
                        "duration_ms": tc.duration_ms,
                        "status": tc.status,
                        **({"error": tc.error} if tc.error else {}),
                    }
                    for tc in it.tool_calls
                ],
            }
            for it in result.logs
        ],
        # Yanıtın tamamı yerine ilk 500 karakter — trace dosyalarını küçük tutar
        "answer_snippet": result.answer[:500],
    }
    return write_trace(run_id, trace)
