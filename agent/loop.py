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

from providers.base import LLMProvider
from agent.registry import ToolRegistry
from agent.tracing import write_trace

logger = structlog.get_logger(__name__)

TOKEN_BUDGET_DEFAULT = 100_000


@dataclass
class ToolCallLog:
    name: str
    args: dict[str, Any]
    duration_ms: float
    status: str        # "success" | "error"
    error: str | None = None


@dataclass
class IterationLog:
    iteration: int
    tool_calls: list[ToolCallLog]
    tokens_used: int


@dataclass
class AgentResult:
    answer: str
    iterations: int
    total_tokens: int
    logs: list[IterationLog] = field(default_factory=list)
    trace_path: str | None = None


def run_agent(
    query: str,
    provider: LLMProvider,
    registry: ToolRegistry,
    max_iterations: int = 10,
    token_budget: int = TOKEN_BUDGET_DEFAULT,
    write_trace_file: bool = True,
) -> AgentResult:
    run_id = uuid.uuid4().hex[:12]
    run_start = time.perf_counter()
    log = logger.bind(run_id=run_id)

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

        # ── Token budget guard ─────────────────────────────────────────────────
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

        # ── Final text answer ──────────────────────────────────────────────────
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

        # ── Tool call branch ───────────────────────────────────────────────────
        provider.extend_messages_with_assistant_turn(messages, response)

        iteration_calls: list[ToolCallLog] = []
        tool_results: list[dict] = []

        for tc in response.next_action.tool_calls:
            t_tool = time.perf_counter()
            log.info("tool_call", iteration=iteration, tool=tc.name, args=tc.input)

            try:
                raw = registry.dispatch(tc.name, tc.input)
                result_str = json.dumps(raw, default=str)
                duration_ms = round((time.perf_counter() - t_tool) * 1000, 1)
                log.info("tool_result", tool=tc.name, status="success", duration_ms=duration_ms, result=result_str[:200])
                iteration_calls.append(ToolCallLog(
                    name=tc.name, args=tc.input, duration_ms=duration_ms, status="success"
                ))
            except Exception as exc:
                duration_ms = round((time.perf_counter() - t_tool) * 1000, 1)
                error_msg = f"{type(exc).__name__}: {exc}"
                # Log with full traceback so the error is visible in the trace
                log.error("tool_error", tool=tc.name, error=error_msg, duration_ms=duration_ms, exc_info=True)
                # Return structured error to LLM so it can retry or gracefully handle
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

        provider.extend_messages_with_tool_results(messages, tool_results)

    # ── max_iterations exhausted ───────────────────────────────────────────────
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
        "answer_snippet": result.answer[:500],
    }
    return write_trace(run_id, trace)
