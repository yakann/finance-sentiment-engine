from concurrent.futures import ThreadPoolExecutor, as_completed

import yfinance as yf
from pydantic import BaseModel

from agent.tools.base import Tool

# fmt: off
_PEERS: dict[str, list[str]] = {
    # Semiconductors
    "NVDA": ["AMD", "INTC", "QCOM", "AVGO"],
    "AMD":  ["NVDA", "INTC", "QCOM", "AVGO"],
    "INTC": ["NVDA", "AMD", "QCOM", "AVGO"],
    "QCOM": ["NVDA", "AMD", "INTC", "AVGO"],
    "AVGO": ["NVDA", "AMD", "QCOM", "INTC"],
    # Big Tech
    "AAPL": ["MSFT", "GOOGL", "META", "AMZN"],
    "MSFT": ["AAPL", "GOOGL", "META", "AMZN"],
    "GOOGL":["META", "MSFT", "SNAP", "AMZN"],
    "GOOG": ["META", "MSFT", "SNAP", "AMZN"],
    "META": ["GOOGL", "SNAP", "MSFT", "PINS"],
    "AMZN": ["MSFT", "GOOGL", "WMT", "SHOP"],
    # EV / Auto
    "TSLA": ["GM", "F", "RIVN", "NIO"],
    "RIVN": ["TSLA", "GM", "F", "NIO"],
    "NIO":  ["TSLA", "GM", "F", "RIVN"],
    # Payments / Fintech
    "V":    ["MA", "AXP", "PYPL", "SQ"],
    "MA":   ["V", "AXP", "PYPL", "SQ"],
    "PYPL": ["V", "MA", "SQ", "AFRM"],
    "SQ":   ["PYPL", "V", "MA", "AFRM"],
    # Banks
    "JPM":  ["BAC", "WFC", "GS", "MS"],
    "BAC":  ["JPM", "WFC", "GS", "MS"],
    "GS":   ["JPM", "MS", "BAC", "WFC"],
    "MS":   ["JPM", "GS", "BAC", "WFC"],
    # Healthcare / Pharma
    "JNJ":  ["PFE", "MRK", "ABT", "BMY"],
    "PFE":  ["JNJ", "MRK", "ABT", "BMY"],
    "MRK":  ["JNJ", "PFE", "ABT", "BMY"],
    # Retail / E-commerce
    "WMT":  ["AMZN", "TGT", "COST", "HD"],
    "COST": ["WMT", "TGT", "AMZN", "HD"],
    "TGT":  ["WMT", "COST", "AMZN", "HD"],
    # Cloud / Enterprise
    "CRM":  ["MSFT", "ORCL", "SAP", "NOW"],
    "ORCL": ["MSFT", "CRM", "SAP", "NOW"],
    "NOW":  ["CRM", "MSFT", "ORCL", "WDAY"],
}
# fmt: on


class _GetCompetitorAnalysisInput(BaseModel):
    ticker: str


def _fetch_peer_snapshot(symbol: str) -> dict:
    info = yf.Ticker(symbol).info
    if not info or info.get("symbol") is None:
        return {
            "ticker": symbol,
            "price": None,
            "pe_ratio": None,
            "revenue_growth_yoy_pct": None,
        }

    _price = info.get("currentPrice")
    price = _price if _price is not None else info.get("regularMarketPrice")
    _pe = info.get("trailingPE")
    pe = _pe if _pe is not None else info.get("forwardPE")
    rev_growth = info.get("revenueGrowth")

    return {
        "ticker": symbol,
        "price": round(price, 2) if price is not None else None,
        "pe_ratio": round(pe, 2) if pe is not None else None,
        "revenue_growth_yoy_pct": round(rev_growth * 100, 2) if rev_growth is not None else None,
    }


def _get_competitor_analysis(inputs: _GetCompetitorAnalysisInput) -> dict:
    ticker = inputs.ticker.upper()
    peers = _PEERS.get(ticker)

    if peers is None:
        return {
            "error": (
                f"No peer mapping for '{ticker}'. "
                f"Supported tickers: {sorted(_PEERS.keys())}"
            )
        }

    all_symbols = [ticker] + peers

    # Fetch all snapshots concurrently — yfinance calls are I/O bound
    snapshots: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=min(len(all_symbols), 8)) as pool:
        futures = {pool.submit(_fetch_peer_snapshot, s): s for s in all_symbols}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                snapshots[symbol] = future.result()
            except Exception as e:
                snapshots[symbol] = {
                    "ticker": symbol,
                    "price": None,
                    "pe_ratio": None,
                    "revenue_growth_yoy_pct": None,
                    "fetch_error": str(e),
                }

    comparison = [snapshots[s] for s in all_symbols]

    return {
        "ticker": ticker,
        "peers": peers,
        "comparison": comparison,
    }


get_competitor_analysis = Tool(
    name="get_competitor_analysis",
    description=(
        "Compares a stock against its sector peers using a built-in peer mapping. "
        "Returns current price, P/E ratio, and revenue growth YoY (%) for the ticker "
        "and each competitor side-by-side. "
        "Supported sectors: semiconductors, big tech, EV/auto, payments, banks, "
        "healthcare, retail, cloud/enterprise."
    ),
    input_schema=_GetCompetitorAnalysisInput,
    handler=_get_competitor_analysis,
)
