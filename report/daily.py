"""Daily Markdown brief generator.

Reads cached LLM analyses and produces a prioritised brief of the day's
most urgent market-moving news.

Usage:
    python -m report.daily --date 2026-05-19
    python -m report.daily --date 2026-05-19 --top-n 5 --provider openai --model gpt-4.1-mini
"""

import argparse
import json
from datetime import date as _date
from pathlib import Path

URGENCY_ORDER = {"high": 0, "medium": 1, "low": 2}
SENTIMENT_ORDER = {"bullish": 0, "bearish": 1, "neutral": 2}
SENTIMENT_EMOJI = {"bullish": "🟢", "bearish": "🔴", "neutral": "⚪"}


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _load_analysis(cache_dir: Path, provider: str, model: str) -> list[dict]:
    slug = model.replace("/", "-")
    path = cache_dir / f"analysis_{provider}_{slug}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Analysis cache not found: {path}")
    return _load_jsonl(path)


def _load_news(cache_dir: Path) -> list[dict]:
    path = cache_dir / "news.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"News cache not found: {path}")
    return _load_jsonl(path)


def generate_report(
    date: str,
    top_n: int = 5,
    provider: str = "openai",
    model: str = "gpt-4.1-mini",
    cache_dir: Path = Path("cache"),
) -> str:
    analyses = _load_analysis(cache_dir, provider, model)
    news = _load_news(cache_dir)

    # Join by index — analyses are written in the same order as news.jsonl
    items: list[dict] = []
    for i, analysis in enumerate(analyses):
        news_item = news[i] if i < len(news) else {}
        items.append({**analysis, "link": news_item.get("link", ""), "title": news_item.get("title", "")})

    items.sort(
        key=lambda x: (
            URGENCY_ORDER.get(x["urgency"], 99),
            SENTIMENT_ORDER.get(x["sentiment"], 99),
        )
    )

    top = items[:top_n]

    lines = [
        f"# Daily Finance Brief — {date}",
        f"## Top {top_n} Urgent Movements",
        "",
    ]
    for item in top:
        emoji = SENTIMENT_EMOJI.get(item["sentiment"], "")
        lines.append(f"### {item['ticker']} · {emoji} {item['sentiment']} · {item['urgency']}")
        lines.append(item["summary"])
        lines.append("")
        if item["link"]:
            lines.append(f"[Full article]({item['link']})")
        lines.append("")

    lines += [
        "---",
        f"*Generated with `{provider}/{model}` · {len(analyses)} items analyzed*",
    ]

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate daily finance brief from cached analyses")
    parser.add_argument("--date", default=str(_date.today()), help="Report date (YYYY-MM-DD, default: today)")
    parser.add_argument("--top-n", type=int, default=5, help="Number of top items (default: 5)")
    parser.add_argument("--provider", default="openai", help="Provider used for analysis (default: openai)")
    parser.add_argument("--model", default="gpt-4.1-mini", help="Model slug (default: gpt-4.1-mini)")
    parser.add_argument("--cache-dir", default="cache", help="Cache directory (default: cache)")
    parser.add_argument("--output-dir", default="reports", help="Output directory (default: reports)")
    args = parser.parse_args()

    report = generate_report(
        date=args.date,
        top_n=args.top_n,
        provider=args.provider,
        model=args.model,
        cache_dir=Path(args.cache_dir),
    )

    print(report)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.date}.md"
    output_path.write_text(report)
    print(f"\n✓ Saved → {output_path}")


if __name__ == "__main__":
    main()
