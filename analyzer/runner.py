import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from providers.base import LLMUsage
from providers.factory import get_provider
from schemas import NewsAnalysis

SYSTEM = (
    "You are a financial analyst. Analyze the news and return structured JSON.\n\n"
    "Urgency — use price-impact framing, not event type:\n"
    "- high: news that could move the stock >3% in a single trading day "
    "(e.g., earnings surprise, major M&A, regulatory action, product recall)\n"
    "- medium: material information that plays out over days but is unlikely to "
    "cause a single-day shock (e.g., analyst target change, product launch, "
    "exec appointment, earnings preview)\n"
    "- low: informational — unlikely to move the price on its own "
    "(e.g., market commentary, speculative op-ed, sector overview)\n\n"
    "key_event — pick exactly one of these 10 buckets:\n"
    "- earnings: earnings previews, results, guidance, expectations\n"
    "- analyst_action: analyst target changes, predictions, buy/sell ratings\n"
    "- company_communication: CEO/exec statements, internal narratives\n"
    "- product_event: launches, safety incidents, recalls, concrete product news\n"
    "- competitive_dynamics: competitor moves, market share shifts, customer churn\n"
    "- policy_geopolitical: regulation, trade policy, government action, public sentiment\n"
    "- business_action: M&A, partnerships, pricing changes, supply chain\n"
    "- market_dynamics: market cap milestones, sector moves, trading patterns\n"
    "- insider_activity: insider/institutional buys, sells, speculation\n"
    "- other: comparative analysis, mixed signals, off-topic\n"
)

FEW_SHOT = [
    {
        "role": "user",
        "content": (
            "Ticker: NVDA\n"
            "Title: Nvidia Reports Record Revenue, Stock Jumps 8%\n"
            "Summary: Nvidia posted Q3 earnings of $18.1B, beating analyst expectations "
            "by 12%. Data center revenue surged 200% YoY."
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "ticker": "NVDA",
            "sentiment": "bullish",
            "urgency": "high",
            "key_event": "earnings",
            "summary": "Nvidia Q3 revenue $18.1B, beat by 12%. Data center up 200% YoY. Stock +8%.",
        }),
    },
    {
        "role": "user",
        "content": (
            "Ticker: TSLA\n"
            "Title: Tesla Slips After China Trip Disappoints\n"
            "Summary: Tesla investors wanted a breakthrough in China, but the "
            "Trump-Xi meeting ended without major trade agreements."
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "ticker": "TSLA",
            "sentiment": "bearish",
            "urgency": "medium",
            "key_event": "policy_geopolitical",
            "summary": "Trump-Xi talks ended without trade deal; China market breakthrough hopes unmet.",
        }),
    },
    {
        "role": "user",
        "content": (
            "Ticker: AAPL\n"
            "Title: Apple Analyst Raises Price Target to $220\n"
            "Summary: Wedbush analyst Dan Ives raised his Apple price target from $200 "
            "to $220, citing strong iPhone 15 demand."
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "ticker": "AAPL",
            "sentiment": "bullish",
            "urgency": "medium",
            "key_event": "analyst_action",
            "summary": "Wedbush raises AAPL target to $220 on strong iPhone 15 demand.",
        }),
    },
    {
        "role": "user",
        "content": (
            "Ticker: MSFT\n"
            "Title: Microsoft: AI Will Change Everything, Says CEO\n"
            "Summary: Satya Nadella reiterated Microsoft's long-term AI strategy "
            "at a conference, saying AI is the platform shift of the decade."
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "ticker": "MSFT",
            "sentiment": "neutral",
            "urgency": "low",
            "key_event": "company_communication",
            "summary": "Nadella reiterates AI strategy at conference; no new product or financial data.",
        }),
    },
]

PRICE_PER_1M: dict[tuple[str, str], dict[str, float]] = {
    ("openai", "gpt-4.1-mini"):           {"input": 0.40, "output": 1.60},
    ("openai", "gpt-4.1-nano"):            {"input": 0.10, "output": 0.40},
    ("openai", "gpt-4o-mini"):             {"input": 0.15, "output": 0.60},
    ("groq",   "llama-3.3-70b-versatile"): {"input": 0.59, "output": 0.79},
    ("groq",   "llama-3.1-8b-instant"):    {"input": 0.05, "output": 0.08},
}


def _build_messages(title: str, summary: str, ticker: str) -> list[dict]:
    return FEW_SHOT + [
        {"role": "user", "content": f"Ticker: {ticker}\nTitle: {title}\nSummary: {summary}"}
    ]


def calc_cost(provider_name: str, model: str, input_tokens: int, output_tokens: int) -> float:
    prices = PRICE_PER_1M.get((provider_name, model), {"input": 0.0, "output": 0.0})
    return (input_tokens * prices["input"] + output_tokens * prices["output"]) / 1_000_000


@dataclass
class ItemResult:
    analysis: NewsAnalysis
    latency_ms: float
    usage: LLMUsage


@dataclass
class BatchStats:
    provider: str
    model: str
    count: int
    total_time_s: float
    avg_latency_ms: float
    sum_latency_ms: float          # sum of per-item latencies (incl. retry backoff)
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    cost_usd: float
    num_failures: int = 0
    results: list[ItemResult] = field(default_factory=list)


async def analyze_batch(
    news_items: list[dict],
    provider_name: str,
    model: str,
    concurrency: int | None = None,
) -> BatchStats:
    provider = get_provider(provider_name, model)
    effective_concurrency = concurrency if concurrency is not None else provider.default_concurrency
    sem = asyncio.Semaphore(effective_concurrency)
    item_results: list[ItemResult | BaseException] = []

    async def analyze_one(item: dict) -> ItemResult:
        messages = _build_messages(item["title"], item["summary"], item["ticker"])
        async with sem:
            t0 = time.perf_counter()
            analysis, usage = await provider.agenerate_structured(
                messages=messages,
                schema=NewsAnalysis,
                system=SYSTEM,
            )
            latency_ms = (time.perf_counter() - t0) * 1000
        return ItemResult(analysis=analysis, latency_ms=latency_ms, usage=usage)

    t_start = time.perf_counter()
    item_results = await asyncio.gather(
        *[analyze_one(item) for item in news_items],
        return_exceptions=True,
    )
    total_time_s = time.perf_counter() - t_start

    successes = [r for r in item_results if isinstance(r, ItemResult)]
    failures = [r for r in item_results if isinstance(r, BaseException)]
    if failures:
        for err in failures:
            print(f"  [warn] item failed: {err}")

    latencies = [r.latency_ms for r in successes]
    total_input = sum(r.usage.input_tokens for r in successes)
    total_output = sum(r.usage.output_tokens for r in successes)
    total_tokens = sum(r.usage.total_tokens for r in successes)
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    sum_latency = sum(latencies)

    return BatchStats(
        provider=provider_name,
        model=model,
        count=len(successes),
        total_time_s=total_time_s,
        avg_latency_ms=avg_latency,
        sum_latency_ms=sum_latency,
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        total_tokens=total_tokens,
        cost_usd=calc_cost(provider_name, model, total_input, total_output),
        num_failures=len(failures),
        results=successes,
    )


def save_results(stats: BatchStats, output_dir: Path = Path("cache")) -> Path:
    model_slug = stats.model.replace("/", "-")
    path = output_dir / f"analysis_{stats.provider}_{model_slug}.jsonl"
    with path.open("w") as f:
        for item in stats.results:
            f.write(item.analysis.model_dump_json() + "\n")
    return path


def print_performance_table(all_stats: list[BatchStats]) -> None:
    print("\n## Performance Comparison\n")
    header = (
        f"{'Provider':<8} {'Model':<28} {'n':>3} {'Wall(s)':>8} "
        f"{'SumItem(s)':>11} {'Parallel?':>10} {'Avg(ms)':>8} "
        f"{'Fails':>5} {'Tokens':>8} {'Cost($)':>9}"
    )
    print(header)
    print("-" * len(header))
    for s in all_stats:
        # If wall_time << sum_latency → parallelism is working
        # If wall_time ≈ sum_latency  → requests are sequential
        sum_s = s.sum_latency_ms / 1000
        parallel = "YES" if sum_s > s.total_time_s * 1.5 else "NO ⚠️"
        print(
            f"{s.provider:<8} {s.model:<28} {s.count:>3} "
            f"{s.total_time_s:>8.2f} {sum_s:>11.2f} {parallel:>10} "
            f"{s.avg_latency_ms:>8.0f} {s.num_failures:>5} "
            f"{s.total_tokens:>8} {s.cost_usd:>9.5f}"
        )
    print()
