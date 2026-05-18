"""
Eval harness: runs 4 provider×model combos against labeled examples,
computes field-by-field accuracy, and writes eval/results.md.
"""
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from analyzer.runner import analyze_batch

EVAL_COMBOS = [
    ("openai", "gpt-4.1-mini"),
    ("openai", "gpt-4.1-nano"),
    ("groq", "llama-3.3-70b-versatile"),
    ("groq", "llama-3.1-8b-instant"),
]

LABELS_PATH = Path(__file__).parent / "labels.jsonl"
RESULTS_PATH = Path(__file__).parent / "results.md"


def load_labels() -> list[dict]:
    return [json.loads(l) for l in LABELS_PATH.read_text().splitlines() if l.strip()]


def _normalize_link(link: str) -> str:
    """Strip RSS tracking params for stable matching."""
    return link.split("?.tsrc=")[0].split("&.tsrc=")[0].rstrip("/")


def load_news_for_labels(labels: list[dict]) -> list[dict]:
    """Match news items to labels via normalized link."""
    news_path = Path(__file__).parent.parent / "cache" / "news.jsonl"
    all_news = [json.loads(l) for l in news_path.read_text().splitlines() if l.strip()]
    link_to_news = {_normalize_link(n["link"]): n for n in all_news}
    matched = []
    for lbl in labels:
        key = _normalize_link(lbl["link"])
        news_item = link_to_news.get(key)
        if news_item is None:
            print(f"  [warn] no news match for: {lbl['link'][:70]}...")
        else:
            matched.append(news_item)
    return matched


def compute_accuracy(labels: list[dict], results) -> dict:
    sentiment_hits = 0
    urgency_hits = 0
    key_event_hits = 0
    n = 0

    for lbl, item_result in zip(labels, results):
        a = item_result.analysis
        sentiment_hits += int(a.sentiment == lbl["expected_sentiment"])
        urgency_hits += int(a.urgency == lbl["expected_urgency"])
        # key_event: exact match
        key_event_hits += int(a.key_event == lbl["expected_key_event"])
        n += 1

    if n == 0:
        return {"sentiment_acc": 0.0, "urgency_acc": 0.0, "key_event_acc": 0.0, "n": 0}

    return {
        "sentiment_acc": sentiment_hits / n,
        "urgency_acc": urgency_hits / n,
        "key_event_acc": key_event_hits / n,
        "n": n,
    }


async def run_eval() -> None:
    labels = load_labels()
    news_items = load_news_for_labels(labels)

    # Keep only labels that have a matching news item
    news_path = Path(__file__).parent.parent / "cache" / "news.jsonl"
    all_news = [json.loads(l) for l in news_path.read_text().splitlines() if l.strip()]
    link_to_news = {_normalize_link(n["link"]): n for n in all_news}
    matched_labels = [lbl for lbl in labels if _normalize_link(lbl["link"]) in link_to_news]

    print(f"Eval set: {len(matched_labels)}/{len(labels)} labeled examples matched\n")

    rows = []
    for provider, model in EVAL_COMBOS:
        print(f"▶ {provider}/{model} ...")
        t0 = time.perf_counter()
        try:
            stats = await analyze_batch(
                news_items=news_items,
                provider_name=provider,
                model=model,
                concurrency=5,
            )
            elapsed = time.perf_counter() - t0
            acc = compute_accuracy(matched_labels, stats.results)
            rows.append({
                "provider": provider,
                "model": model,
                "sentiment_acc": acc["sentiment_acc"],
                "urgency_acc": acc["urgency_acc"],
                "key_event_acc": acc["key_event_acc"],
                "avg_latency_ms": stats.avg_latency_ms,
                "cost_usd": stats.cost_usd,
                "n": acc["n"],
                "error": None,
            })
            print(
                f"  ✓ sentiment={acc['sentiment_acc']:.0%}  "
                f"urgency={acc['urgency_acc']:.0%}  "
                f"key_event={acc['key_event_acc']:.0%}  "
                f"latency={stats.avg_latency_ms:.0f}ms  "
                f"cost=${stats.cost_usd:.5f}"
            )
        except Exception as e:
            rows.append({
                "provider": provider, "model": model,
                "sentiment_acc": None, "urgency_acc": None, "key_event_acc": None,
                "avg_latency_ms": None, "cost_usd": None, "n": 0, "error": str(e),
            })
            print(f"  ✗ failed: {e}")

    write_results_md(rows, len(matched_labels))


def write_results_md(rows: list[dict], n_labels: int) -> None:
    lines = [
        "# Eval Results\n",
        f"**Eval set:** {n_labels} labeled examples  ",
        "**Ground truth:** human-labeled (Turkish notes in `labels.jsonl`)\n",
        "| Provider | Model | sentiment_acc | urgency_acc | key_event_acc | avg_latency_ms | cost_per_run |",
        "|----------|-------|:-------------:|:-----------:|:-------------:|:--------------:|:------------:|",
    ]

    for r in rows:
        if r["error"]:
            lines.append(
                f"| {r['provider']} | {r['model']} | ERROR | ERROR | ERROR | — | — |"
            )
        else:
            lines.append(
                f"| {r['provider']} | {r['model']} "
                f"| {r['sentiment_acc']:.0%} "
                f"| {r['urgency_acc']:.0%} "
                f"| {r['key_event_acc']:.0%} "
                f"| {r['avg_latency_ms']:.0f} ms "
                f"| ${r['cost_usd']:.5f} |"
            )

    RESULTS_PATH.write_text("\n".join(lines) + "\n")
    print(f"\nResults written → {RESULTS_PATH}")


if __name__ == "__main__":
    asyncio.run(run_eval())
