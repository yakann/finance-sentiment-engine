"""Yahoo Finance RSS scraper with append-only JSONL cache and dedup."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from datetime import datetime, timezone

import feedparser
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

RSS_URL = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={tickers}&region=US&lang=en-US"
CACHE_PATH = Path(__file__).parent.parent / "cache" / "news.jsonl"

# Known name variants per ticker for relevance filtering.
_TICKER_NAMES: dict[str, list[str]] = {
    "NVDA": ["nvidia", "nvda"],
    "TSLA": ["tesla", "tsla"],
    "MSFT": ["microsoft", "msft"],
    "AAPL": ["apple", "aapl"],
    "GOOGL": ["google", "googl", "alphabet"],
    "AMZN": ["amazon", "amzn"],
    "META": ["meta", "facebook"],
    "NFLX": ["netflix", "nflx"],
    # Turkish (BIST)
    "THYAO.IS": ["turkish airlines", "thy", "thyao"],
    "EREGL.IS": ["eregli", "erdemir", "eregl"],
    "GARAN.IS": ["garanti", "garan"],
    "AKBNK.IS": ["akbank", "akbnk"],
    "ISCTR.IS": ["is bankasi", "isbank", "isctr"],
    "KCHOL.IS": ["koc holding", "kchol"],
    "SAHOL.IS": ["sabanci", "sahol"],
    "SISE.IS": ["sise cam", "sisecam", "sise"],
    "VESTL.IS": ["vestel", "vestl"],
    "TOASO.IS": ["tofas", "toaso"],
    "BIMAS.IS": ["bim", "bimas"],
    "PGSUS.IS": ["pegasus", "pgsus"],
}


def _mentions_ticker(text: str, ticker: str) -> bool:
    """Return True if text contains the ticker symbol or a known company-name variant.

    For exchange-suffixed tickers (e.g. THYAO.IS), always also search for the
    base symbol (THYAO) so that news headlines that omit the suffix still match.
    """
    text_lower = text.lower()
    ticker_upper = ticker.upper()

    if ticker_upper in _TICKER_NAMES:
        names = _TICKER_NAMES[ticker_upper]
    else:
        # Base ticker before exchange suffix (e.g. "BMW" from "BMW.DE")
        base = ticker_upper.split(".")[0].lower()
        names = [base, ticker_upper.lower()]

    return any(name in text_lower for name in names)


class NewsItem(BaseModel):
    title: str
    link: str
    published: str
    summary: str
    ticker: str


def _link_hash(link: str) -> str:
    return hashlib.sha256(link.encode()).hexdigest()


def _load_seen_hashes() -> set[str]:
    if not CACHE_PATH.exists():
        return set()
    seen = set()
    with CACHE_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                seen.add(_link_hash(item["link"]))
            except (json.JSONDecodeError, KeyError):
                pass
    return seen


def _append_to_cache(items: list[NewsItem]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CACHE_PATH.open("a", encoding="utf-8") as f:
        for item in items:
            f.write(item.model_dump_json() + "\n")


@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def _fetch_feed(url: str) -> feedparser.FeedParserDict:
    feed = feedparser.parse(url)
    if feed.get("bozo") and not feed.entries:
        raise ConnectionError(f"Feed returned bozo error with no entries: {url}")
    return feed


def fetch_news(tickers: list[str], limit: int = 50) -> list[NewsItem]:
    tickers_str = ",".join(tickers)
    url = RSS_URL.format(tickers=tickers_str)
    logger.info("Fetching RSS feed for tickers: %s", tickers_str)

    seen = _load_seen_hashes()
    new_items: list[NewsItem] = []

    for ticker in tickers:
        ticker_url = RSS_URL.format(tickers=ticker)
        try:
            feed = _fetch_feed(ticker_url)
        except Exception as e:
            logger.warning("Failed to fetch feed for %s: %s", ticker, e)
            continue

        for entry in feed.entries:
            if len(new_items) >= limit:
                break

            link = entry.get("link", "")
            if not link:
                continue

            h = _link_hash(link)
            if h in seen:
                continue

            combined = f"{entry.get('title', '')} {entry.get('summary', '')}"
            if not _mentions_ticker(combined, ticker):
                logger.debug("Skipping off-topic article for %s: %s", ticker, entry.get("title", "")[:60])
                continue

            published = entry.get("published", datetime.now(timezone.utc).isoformat())
            item = NewsItem(
                title=entry.get("title", ""),
                link=link,
                published=published,
                summary=entry.get("summary", ""),
                ticker=ticker,
            )
            new_items.append(item)
            seen.add(h)

        if len(new_items) >= limit:
            break

    _append_to_cache(new_items)
    logger.info("Fetched %d new items, saved to %s", len(new_items), CACHE_PATH)
    return new_items


def _main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Yahoo Finance RSS news")
    parser.add_argument("--tickers", nargs="+", default=["NVDA", "TSLA", "MSFT"],
                        help="Ticker symbols to fetch")
    parser.add_argument("--limit", type=int, default=50,
                        help="Max number of new items to fetch")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    items = fetch_news(tickers=args.tickers, limit=args.limit)
    print(f"\nFetched {len(items)} new items.")
    for item in items[:5]:
        print(f"  [{item.ticker}] {item.title[:80]}")
    if len(items) > 5:
        print(f"  ... and {len(items) - 5} more.")


if __name__ == "__main__":
    _main()
