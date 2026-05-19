import argparse
import asyncio
import json
from pathlib import Path

from analyzer.runner import analyze_batch, save_results, print_performance_table

BENCHMARK_COMBOS = [
    ("openai", "gpt-4.1-mini"),
    ("openai", "gpt-4.1-nano"),
    ("groq",   "llama-3.3-70b-versatile"),
    ("groq",   "llama-3.1-8b-instant"),
]


def load_news(limit: int) -> list[dict]:
    path = Path("cache/news.jsonl")
    items = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return items[:limit] if limit > 0 else items


async def run_single(args) -> None:
    news = load_news(args.limit)
    concurrency_str = str(args.concurrency) if args.concurrency else "provider default"
    print(f"Running {args.provider}/{args.model} on {len(news)} items (concurrency={concurrency_str})...")
    stats = await analyze_batch(
        news_items=news,
        provider_name=args.provider,
        model=args.model,
        concurrency=args.concurrency,
    )
    path = save_results(stats)
    print(f"Saved {stats.count} results → {path}")
    print_performance_table([stats])


async def run_benchmark(args) -> None:
    news = load_news(args.limit)
    concurrency_str = str(args.concurrency) if args.concurrency else "provider defaults"
    print(f"Benchmark: {len(BENCHMARK_COMBOS)} combinations × {len(news)} items (concurrency={concurrency_str})\n")
    all_stats = []
    for provider, model in BENCHMARK_COMBOS:
        print(f"▶ {provider}/{model} ...")
        try:
            stats = await analyze_batch(
                news_items=news,
                provider_name=provider,
                model=model,
                concurrency=args.concurrency,
            )
            path = save_results(stats)
            print(f"  ✓ {stats.count} items in {stats.total_time_s:.2f}s → {path}")
            all_stats.append(stats)
        except Exception as e:
            print(f"  ✗ failed: {e}")
    print_performance_table(all_stats)


def main() -> None:
    parser = argparse.ArgumentParser(description="Finance Sentiment Engine")
    parser.add_argument("--provider", default="openai", choices=["openai", "groq", "anthropic"])
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--concurrency", type=int, default=None,
                        help="Max concurrent requests (default: provider-specific safe value)")
    parser.add_argument("--limit", type=int, default=0, help="Max news items (0 = all)")
    parser.add_argument("--benchmark", action="store_true", help="Run all 4 provider×model combos")
    args = parser.parse_args()

    if args.benchmark:
        asyncio.run(run_benchmark(args))
    else:
        asyncio.run(run_single(args))


if __name__ == "__main__":
    main()
