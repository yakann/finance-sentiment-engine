"""
Day 34 — CI evaluation script

Usage:
  python scripts/run_eval_ci.py                      # fast: 10 examples, gpt-4.1-mini
  python scripts/run_eval_ci.py --full               # full: all examples, all combos
  python scripts/run_eval_ci.py --subset 5           # custom subset size
  python scripts/run_eval_ci.py --threshold 0.80     # override pass threshold

Exit code 1 when any combo falls below --threshold (default 0.85).
Writes eval_ci_results.json and eval_summary.md to project root.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from dotenv import load_dotenv

load_dotenv()

from analyzer.runner import analyze_batch

# ── Defaults ──────────────────────────────────────────────────────────────────

THRESHOLD = 0.85

FAST_COMBOS = [
    ("openai", "gpt-4.1-mini"),
]

FULL_COMBOS = [
    ("openai", "gpt-4.1-mini"),
    ("openai", "gpt-4.1-nano"),
    ("groq",   "llama-3.3-70b-versatile"),
    ("groq",   "llama-3.1-8b-instant"),
]

PROVIDER_CONCURRENCY = {"openai": 10, "groq": 2}

LABELS_PATH = PROJECT_ROOT / "eval" / "labels.jsonl"
NEWS_CACHE  = PROJECT_ROOT / "cache" / "news.jsonl"


# ── Data loading ──────────────────────────────────────────────────────────────

def _norm(link: str) -> str:
    return link.split("?.tsrc=")[0].split("&.tsrc=")[0].rstrip("/")


def load_eval_set(subset: int | None) -> tuple[list[dict], list[dict]]:
    """Return (labels, news_items) matched and optionally truncated."""
    labels   = [json.loads(l) for l in LABELS_PATH.read_text().splitlines() if l.strip()]
    all_news = [json.loads(l) for l in NEWS_CACHE.read_text().splitlines()  if l.strip()]
    news_map = {_norm(n["link"]): n for n in all_news}

    matched_labels = []
    matched_news   = []
    for lbl in labels:
        news = news_map.get(_norm(lbl["link"]))
        if news:
            matched_labels.append(lbl)
            matched_news.append(news)

    if subset:
        matched_labels = matched_labels[:subset]
        matched_news   = matched_news[:subset]

    return matched_labels, matched_news


# ── Accuracy ──────────────────────────────────────────────────────────────────

def compute_accuracy(labels: list[dict], results) -> dict:
    n = len(labels)
    if n == 0:
        return {"sentiment_acc": 0.0, "urgency_acc": 0.0, "key_event_acc": 0.0, "n": 0}
    sent = sum(r.analysis.sentiment == l["expected_sentiment"] for l, r in zip(labels, results))
    urg  = sum(r.analysis.urgency   == l["expected_urgency"]   for l, r in zip(labels, results))
    key  = sum(r.analysis.key_event == l["expected_key_event"] for l, r in zip(labels, results))
    return {
        "sentiment_acc": sent / n,
        "urgency_acc":   urg  / n,
        "key_event_acc": key  / n,
        "n": n,
    }


# ── Eval runner ───────────────────────────────────────────────────────────────

async def run_combos(
    labels: list[dict],
    news_items: list[dict],
    combos: list[tuple[str, str]],
) -> list[dict]:
    rows: list[dict] = []
    for provider, model in combos:
        print(f"  ▶ {provider}/{model} ({len(news_items)} examples)...", flush=True)
        t0 = time.perf_counter()
        try:
            stats = await analyze_batch(
                news_items=news_items,
                provider_name=provider,
                model=model,
                concurrency=PROVIDER_CONCURRENCY.get(provider, 5),
            )
            elapsed = time.perf_counter() - t0
            acc = compute_accuracy(labels, stats.results)
            rows.append({
                "provider":       provider,
                "model":          model,
                "sentiment_acc":  acc["sentiment_acc"],
                "urgency_acc":    acc["urgency_acc"],
                "key_event_acc":  acc["key_event_acc"],
                "avg_latency_ms": stats.avg_latency_ms,
                "cost_usd":       stats.cost_usd,
                "n":              acc["n"],
                "elapsed_s":      round(elapsed, 1),
                "error":          None,
            })
            print(
                f"    sentiment={acc['sentiment_acc']:.0%}  "
                f"urgency={acc['urgency_acc']:.0%}  "
                f"key_event={acc['key_event_acc']:.0%}  "
                f"latency={stats.avg_latency_ms:.0f}ms  "
                f"cost=${stats.cost_usd:.5f}",
                flush=True,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            print(f"    ✗ {exc}", flush=True)
            rows.append({
                "provider":       provider,
                "model":          model,
                "sentiment_acc":  None,
                "urgency_acc":    None,
                "key_event_acc":  None,
                "avg_latency_ms": None,
                "cost_usd":       None,
                "n":              0,
                "elapsed_s":      round(elapsed, 1),
                "error":          str(exc),
            })
    return rows


# ── Output writers ────────────────────────────────────────────────────────────

def write_json(rows: list[dict], n_labels: int, threshold: float, passed: bool, path: Path) -> None:
    path.write_text(
        json.dumps(
            {"threshold": threshold, "passed": passed, "n_labels": n_labels, "results": rows},
            indent=2,
        )
    )


def write_markdown(rows: list[dict], n_labels: int, threshold: float, passed: bool, path: Path) -> None:
    status = "✅ PASSED" if passed else "❌ FAILED"
    lines = [
        f"## Eval Results — {status}",
        "",
        f"**Threshold:** sentiment_accuracy ≥ {threshold:.0%}  ",
        f"**Examples:** {n_labels}  ",
        "",
        "| Provider | Model | Sentiment ↑ | Urgency | Key Event | Latency | Cost |",
        "|----------|-------|:-----------:|:-------:|:---------:|:-------:|:----:|",
    ]
    for r in rows:
        if r["error"]:
            lines.append(f"| {r['provider']} | {r['model']} | ERROR | — | — | — | — |")
        else:
            flag = " ⚠️" if r["sentiment_acc"] is not None and r["sentiment_acc"] < threshold else ""
            lines.append(
                f"| {r['provider']} | {r['model']} "
                f"| {r['sentiment_acc']:.0%}{flag} "
                f"| {r['urgency_acc']:.0%} "
                f"| {r['key_event_acc']:.0%} "
                f"| {r['avg_latency_ms']:.0f} ms "
                f"| ${r['cost_usd']:.5f} |"
            )
    lines += [
        "",
        f"*Failing combos (sentiment < {threshold:.0%}):* "
        + (
            ", ".join(
                f"{r['provider']}/{r['model']}"
                for r in rows
                if r["sentiment_acc"] is not None and r["sentiment_acc"] < threshold
            ) or "none"
        ),
    ]
    path.write_text("\n".join(lines) + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CI eval — sentiment accuracy gate")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--full",   action="store_true", help="Run all examples + all combos")
    mode.add_argument("--subset", type=int, default=None, help="Number of examples (default 10)")
    p.add_argument("--threshold", type=float, default=THRESHOLD, help="Min sentiment accuracy")
    p.add_argument("--output-json", default="eval_ci_results.json")
    p.add_argument("--output-md",   default="eval_summary.md")
    return p.parse_args()


async def main() -> int:
    args = parse_args()

    subset = None if args.full else (args.subset or 10)
    combos = FULL_COMBOS if args.full else FAST_COMBOS

    print("=" * 60)
    print(f"CI EVAL — {'FULL' if args.full else f'FAST (subset={subset})'}")
    print(f"  combos    : {[f'{p}/{m}' for p,m in combos]}")
    print(f"  threshold : sentiment_acc ≥ {args.threshold:.0%}")
    print("=" * 60)

    labels, news_items = load_eval_set(subset)
    if not labels:
        print("ERROR: no matched examples found in cache/news.jsonl")
        return 1

    print(f"\n→ {len(labels)} examples loaded\n")

    rows = await run_combos(labels, news_items, combos)

    # ── Threshold gate ────────────────────────────────────────────────────────
    failures = [
        r for r in rows
        if r["sentiment_acc"] is not None and r["sentiment_acc"] < args.threshold
    ]
    errors = [r for r in rows if r["error"]]
    passed = len(failures) == 0 and len(errors) == 0

    # ── Write outputs ─────────────────────────────────────────────────────────
    json_path = PROJECT_ROOT / args.output_json
    md_path   = PROJECT_ROOT / args.output_md
    write_json(rows, len(labels), args.threshold, passed, json_path)
    write_markdown(rows, len(labels), args.threshold, passed, md_path)

    print(f"\n→ JSON : {json_path}")
    print(f"→ MD   : {md_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if passed:
        print("RESULT: ✅ PASSED — all combos above threshold")
    else:
        if failures:
            for r in failures:
                print(f"RESULT: ❌ FAILED — {r['provider']}/{r['model']} "
                      f"sentiment_acc={r['sentiment_acc']:.0%} < {args.threshold:.0%}")
        if errors:
            for r in errors:
                print(f"RESULT: ❌ ERROR  — {r['provider']}/{r['model']}: {r['error'][:80]}")
    print("=" * 60)

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
