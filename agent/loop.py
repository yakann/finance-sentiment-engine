"""Day 17: Multi-turn tool-use agent loop.

think → act → observe → repeat
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from providers.base import LLMProvider
from agent.registry import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class IterationLog:
    iteration: int
    tool_calls: list[dict[str, Any]]
    tokens_used: int


@dataclass
class AgentResult:
    answer: str
    iterations: int
    total_tokens: int
    logs: list[IterationLog] = field(default_factory=list)


def run_agent(
    query: str,
    provider: LLMProvider,
    registry: ToolRegistry,
    max_iterations: int = 10,
    token_budget: int = 50_000,
) -> AgentResult:
    """Run a multi-turn tool-use loop until a final text answer is produced.

    Each iteration:
      1. Call the provider with current messages + available tools.
      2. If next_action == "text": return the final answer.
      3. If next_action == "tool_calls": dispatch each tool, append results, repeat.

    Guards: max_iterations, total token_budget.
    """
    messages: list[dict] = [{"role": "user", "content": query}]
    tools = registry.all_tools()
    total_tokens = 0
    logs: list[IterationLog] = []

    for iteration in range(1, max_iterations + 1):
        logger.info("[iter %d] → provider  messages=%d", iteration, len(messages))

        response = provider.generate(messages, tools=tools)
        total_tokens += response.usage.total_tokens

        logger.info(
            "[iter %d] ← provider  action=%s  tokens=%d  total=%d",
            iteration,
            response.next_action.type,
            response.usage.total_tokens,
            total_tokens,
        )

        if total_tokens > token_budget:
            logger.warning("[iter %d] token budget exceeded (%d > %d)", iteration, total_tokens, token_budget)
            logs.append(IterationLog(iteration=iteration, tool_calls=[], tokens_used=response.usage.total_tokens))
            return AgentResult(
                answer=response.text or "[token budget exceeded]",
                iterations=iteration,
                total_tokens=total_tokens,
                logs=logs,
            )

        if response.next_action.type == "text":
            logs.append(IterationLog(iteration=iteration, tool_calls=[], tokens_used=response.usage.total_tokens))
            return AgentResult(
                answer=response.text,
                iterations=iteration,
                total_tokens=total_tokens,
                logs=logs,
            )

        # ── tool_calls branch ──────────────────────────────────────────────
        provider.extend_messages_with_assistant_turn(messages, response)

        iteration_calls: list[dict] = []
        tool_results: list[dict] = []

        for tc in response.next_action.tool_calls:
            logger.info("[iter %d] tool=%s  input=%s", iteration, tc.name, tc.input)
            try:
                raw = registry.dispatch(tc.name, tc.input)
                result_str = json.dumps(raw, default=str)
            except Exception as exc:
                result_str = f"Error: {exc}"

            snippet = result_str[:300]
            logger.info("[iter %d] tool=%s  result=%s", iteration, tc.name, snippet)

            iteration_calls.append({"name": tc.name, "input": tc.input, "result": snippet})
            tool_results.append({"tool_call_id": tc.id, "result": result_str})

        logs.append(IterationLog(
            iteration=iteration,
            tool_calls=iteration_calls,
            tokens_used=response.usage.total_tokens,
        ))

        provider.extend_messages_with_tool_results(messages, tool_results)

    logger.warning("max_iterations=%d reached", max_iterations)
    return AgentResult(
        answer="[max iterations reached]",
        iterations=max_iterations,
        total_tokens=total_tokens,
        logs=logs,
    )
