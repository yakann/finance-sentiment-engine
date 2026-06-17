"""Day 31 — LLM-as-Judge Evaluators + 6-Model Baseline

3 evaluators registered with LangSmith:
  1. sentiment_accuracy — exact match, deterministic (0.0 or 1.0)
  2. reasoning_quality  — LLM judge: "Does this output give the correct sentiment
                         with correct reasoning?" (0–5 → normalized to 0–1)
  3. brief_quality      — pairwise LLM judge: "Which summary is better, A or B?"
                         (used via evaluate_comparative)

6 provider × model combos run against 'finance-sentiment-v1' dataset (Day 30):
  openai    / gpt-4.1-mini
  openai    / gpt-4.1-nano
  groq      / llama-3.3-70b-versatile
  groq      / llama-3.1-8b-instant
  anthropic / claude-haiku-4-5-20251001
  anthropic / claude-3-5-haiku-20241022

Usage:
    python scripts/run_day31.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=True)

from openai import OpenAI as OpenAIClient
from langsmith import evaluate
from langsmith.evaluation import evaluate_comparative

from analyzer.runner import SYSTEM, _build_messages
from providers.factory import get_provider
from schemas import NewsAnalysis


DATASET_NAME = "finance-sentiment-v1"
JUDGE_MODEL = "gpt-4o-mini"

EVAL_COMBOS: list[tuple[str, str]] = [
    ("openai",    "gpt-4.1-mini"),
    ("openai",    "gpt-4.1-nano"),
    ("groq",      "llama-3.3-70b-versatile"),
    ("groq",      "llama-3.1-8b-instant"),
    ("anthropic", "claude-haiku-4-5-20251001"),
    ("anthropic", "claude-3-5-haiku-20241022"),
]

# Groq free-tier is TPM-limited; keep concurrency low to avoid 429s.
_CONCURRENCY: dict[str, int] = {
    "openai":    5,
    "groq":      2,
    "anthropic": 3,
}

_judge_client: OpenAIClient | None = None


def _judge() -> OpenAIClient:
    global _judge_client
    if _judge_client is None:
        _judge_client = OpenAIClient()
    return _judge_client


# ── Evaluator 1: Sentiment Accuracy (exact match, deterministic) ─────────────

def sentiment_accuracy(outputs: dict, reference_outputs: dict) -> dict:
    """Exact-match comparison between predicted and expected sentiment."""
    predicted = (outputs.get("sentiment") or "").lower().strip()
    expected  = (reference_outputs.get("expected_sentiment") or "").lower().strip()
    return {
        "key":   "sentiment_accuracy",
        "score": 1.0 if predicted == expected else 0.0,
    }


# ── Evaluator 2: Reasoning Quality (LLM judge, 0–5 → normalized) ─────────────

_REASONING_PROMPT = """\
You are a financial analysis evaluator. Rate the quality of the model output below.

News item:
  Ticker : {ticker}
  Title  : {title}
  Summary: {news_summary}

Model output:
  Sentiment : {sentiment}
  Brief     : {brief}

Expected sentiment: {expected}

Scoring rubric (0–5):
  5 = Correct sentiment AND specific reasoning grounded in the news text
  4 = Correct sentiment with adequate reasoning
  3 = Correct sentiment but reasoning is vague or generic
  2 = Wrong sentiment, but the analysis shows partial understanding
  1 = Wrong sentiment with poor reasoning
  0 = Incoherent output or sentiment field missing

Respond with JSON only — no markdown, no extra keys:
{{"score": <integer 0-5>, "comment": "<one sentence>"}}"""


def reasoning_quality(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """LLM judge scores whether the sentiment is correct and well-reasoned (0–5)."""
    prompt = _REASONING_PROMPT.format(
        ticker=inputs.get("ticker", ""),
        title=inputs.get("title", ""),
        news_summary=inputs.get("summary", ""),
        sentiment=outputs.get("sentiment", ""),
        brief=outputs.get("summary_text", ""),
        expected=reference_outputs.get("expected_sentiment", ""),
    )
    resp = _judge().chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    result  = json.loads(resp.choices[0].message.content)
    raw     = max(0, min(5, int(result.get("score", 0))))
    return {
        "key":     "reasoning_quality",
        "score":   raw / 5.0,
        "comment": result.get("comment", ""),
    }


# ── Evaluator 3: Brief Quality (pairwise, used with evaluate_comparative) ─────

_PAIRWISE_PROMPT = """\
You are a financial analyst evaluating two AI-generated summaries of the same news.

News:
  Ticker : {ticker}
  Title  : {title}

Summary A: {brief_a}
Summary B: {brief_b}

Which summary is better for a financial analyst?
Criteria: accuracy · clarity · conciseness · actionability

Respond with JSON only — no markdown, no extra keys:
{{"winner": "A" | "B" | "tie", "reason": "<one sentence>"}}"""


def brief_quality(inputs: dict, outputs_a: dict, outputs_b: dict) -> dict:
    """Pairwise LLM judge: which brief is better?
    score = 0.0 → A wins, 0.5 → tie, 1.0 → B wins
    """
    prompt = _PAIRWISE_PROMPT.format(
        ticker=inputs.get("ticker", ""),
        title=inputs.get("title", ""),
        brief_a=outputs_a.get("summary_text") or "(no summary)",
        brief_b=outputs_b.get("summary_text") or "(no summary)",
    )
    resp = _judge().chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    result = json.loads(resp.choices[0].message.content)
    winner = result.get("winner", "tie")
    score  = 0.0 if winner == "A" else (1.0 if winner == "B" else 0.5)
    return {
        "key":     "brief_quality",
        "score":   score,
        "comment": result.get("reason", ""),
    }


# ── Target function factory ───────────────────────────────────────────────────

def make_target(provider_name: str, model: str):
    """Returns an async target function for the given provider/model combo."""
    provider = get_provider(provider_name, model)

    async def target(inputs: dict) -> dict:
        messages = _build_messages(
            title=inputs["title"],
            summary=inputs["summary"],
            ticker=inputs["ticker"],
        )
        analysis, _ = await provider.agenerate_structured(
            messages=messages,
            schema=NewsAnalysis,
            system=SYSTEM,
        )
        return {
            "sentiment":    analysis.sentiment,
            "urgency":      analysis.urgency,
            "key_event":    analysis.key_event,
            "summary_text": analysis.summary,
        }

    target.__name__ = f"target_{provider_name}_{model.replace('-', '_').replace('.', '_')}"
    return target


# ── Aggregate scores from ExperimentResults ───────────────────────────────────

def _aggregate(results) -> dict[str, float]:
    """Extract mean scores for each evaluator key from an ExperimentResults object."""
    buckets: dict[str, list[float]] = {}
    for row in results:
        for ev in (row.evaluation_results or {}).get("results", []):
            key = ev.key
            if ev.score is not None:
                buckets.setdefault(key, []).append(ev.score)
    return {k: sum(v) / len(v) for k, v in buckets.items() if v}


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 65)
    print("Day 31 — LLM-as-Judge Evaluators + 6-Model Baseline")
    print(f"  Dataset : {DATASET_NAME}")
    print(f"  Combos  : {len(EVAL_COMBOS)}")
    print(f"  Judge   : {JUDGE_MODEL}")
    print("=" * 65)

    experiment_names: list[str] = []   # for pairwise step
    summary_rows: list[dict]    = []

    for provider_name, model in EVAL_COMBOS:
        combo_key   = f"{provider_name}/{model}"
        prefix      = f"day31-{provider_name}-{model.replace('/', '-')}"
        concurrency = _CONCURRENCY.get(provider_name, 3)
        print(f"\n▶  {combo_key}  (concurrency={concurrency})")

        target = make_target(provider_name, model)

        try:
            t0      = time.perf_counter()
            results = evaluate(
                target,
                data=DATASET_NAME,
                evaluators=[sentiment_accuracy, reasoning_quality],
                experiment_prefix=prefix,
                max_concurrency=concurrency,
            )
            elapsed = time.perf_counter() - t0

            exp_name = results.experiment_name
            experiment_names.append(exp_name)

            scores = _aggregate(results)
            row = {
                "combo":             combo_key,
                "experiment":        exp_name,
                "sentiment_accuracy": scores.get("sentiment_accuracy", float("nan")),
                "reasoning_quality":  scores.get("reasoning_quality",  float("nan")),
                "elapsed_s":          elapsed,
                "error":              None,
            }
            summary_rows.append(row)
            print(
                f"  ✓ {elapsed:.0f}s  "
                f"sentiment_acc={row['sentiment_accuracy']:.0%}  "
                f"reasoning_quality={row['reasoning_quality']:.2f}  "
                f"→ {exp_name}"
            )

        except Exception as exc:
            summary_rows.append({
                "combo": combo_key, "experiment": None,
                "sentiment_accuracy": float("nan"),
                "reasoning_quality":  float("nan"),
                "elapsed_s":          0.0, "error": str(exc),
            })
            print(f"  ✗ failed: {exc}")

    # ── Pairwise comparison (brief_quality) ───────────────────────────────────
    # Compare the first two successful experiments (gpt-4.1-mini vs llama-3.3-70b)
    pairwise_exp_names = [
        r["experiment"] for r in summary_rows
        if r["experiment"] and not r["error"]
    ]

    if len(pairwise_exp_names) >= 2:
        exp_a, exp_b = pairwise_exp_names[0], pairwise_exp_names[1]
        label_a = summary_rows[0]["combo"]
        label_b = summary_rows[1]["combo"]
        print(f"\n── Pairwise brief_quality ──────────────────────────────────")
        print(f"  A: {label_a}  ({exp_a})")
        print(f"  B: {label_b}  ({exp_b})")
        try:
            comp_results = evaluate_comparative(
                [exp_a, exp_b],
                evaluators=[brief_quality],
            )
            # score meaning: 0.0 = A wins, 0.5 = tie, 1.0 = B wins
            pairwise_scores = []
            for row in comp_results:
                for ev in (row.evaluation_results or {}).get("results", []):
                    if ev.key == "brief_quality" and ev.score is not None:
                        pairwise_scores.append(ev.score)

            if pairwise_scores:
                avg = sum(pairwise_scores) / len(pairwise_scores)
                a_wins = sum(1 for s in pairwise_scores if s < 0.5)
                b_wins = sum(1 for s in pairwise_scores if s > 0.5)
                ties   = len(pairwise_scores) - a_wins - b_wins
                print(
                    f"  Results ({len(pairwise_scores)} examples): "
                    f"A wins={a_wins}  ties={ties}  B wins={b_wins}  "
                    f"avg_score={avg:.2f}"
                )
                winner = label_a if avg < 0.45 else (label_b if avg > 0.55 else "tie")
                print(f"  → Overall winner: {winner}")
        except Exception as exc:
            print(f"  ✗ pairwise failed: {exc}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("BASELINE SCORES — 6 provider × model")
    print("=" * 65)
    header = f"{'Combo':<38} {'SentAcc':>7} {'ReasonQ':>7} {'Time(s)':>7}"
    print(header)
    print("-" * len(header))
    for r in summary_rows:
        if r["error"]:
            print(f"{r['combo']:<38} {'ERROR':>7} {'ERROR':>7} {'—':>7}")
        else:
            sa = r["sentiment_accuracy"]
            rq = r["reasoning_quality"]
            sa_str = f"{sa:.0%}" if sa == sa else "N/A"
            rq_str = f"{rq:.2f}" if rq == rq else "N/A"
            print(f"{r['combo']:<38} {sa_str:>7} {rq_str:>7} {r['elapsed_s']:>7.0f}")
    print("=" * 65)
    print("\nAll experiments visible in LangSmith under project 'finance-agent'.")
    print("Done.")


if __name__ == "__main__":
    main()
