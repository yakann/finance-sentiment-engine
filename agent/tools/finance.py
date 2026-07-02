import yfinance as yf
from pydantic import BaseModel
from agent.tools.base import Tool
from agent.tools.utils import format_large_number


class _GetStockDataInput(BaseModel):
    ticker: str
    period: str = "1mo"


def _get_stock_data(inputs: _GetStockDataInput) -> dict:
    t = yf.Ticker(inputs.ticker.upper())
    hist = t.history(period=inputs.period)
    if hist.empty:
        return {"error": f"No data found for ticker '{inputs.ticker}'"}

    # Drop rows where Close is NaN; take Volume from the same aligned row
    valid = hist.dropna(subset=["Close"])
    if valid.empty:
        return {"error": f"No valid price data for ticker '{inputs.ticker}'"}

    price = round(float(valid["Close"].iloc[-1]), 4)
    last_vol = valid["Volume"].iloc[-1]
    volume = int(last_vol) if last_vol is not None and last_vol == last_vol else 0  # NaN check
    first_close = float(valid["Close"].iloc[0])
    pct_change = round((price - first_close) / first_close * 100, 4) if first_close != 0 else None

    fast = t.fast_info
    market_cap = getattr(fast, "market_cap", None)
    mc_raw = float(market_cap) if market_cap is not None and market_cap > 0 else None
    mc_formatted = format_large_number(mc_raw)

    # Currency — fast_info.currency is the most reliable source for exchange-listed stocks
    currency = getattr(fast, "currency", None) or "USD"

    return {
        "ticker": inputs.ticker.upper(),
        "period": inputs.period,
        "currency": currency,
        "price": price,
        "volume": volume,
        "pct_change": pct_change,
        "market_cap": int(mc_raw) if mc_raw is not None else None,
        "market_cap_formatted": mc_formatted,
    }


get_stock_data = Tool(
    name="get_stock_data",
    description=(
        "Fetches real-time stock data for a given ticker symbol using Yahoo Finance. "
        "Returns current price, trading volume, percentage change over the period, and market cap."
    ),
    input_schema=_GetStockDataInput,
    handler=_get_stock_data,
)
