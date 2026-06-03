import yfinance as yf
from pydantic import BaseModel
from agent.tools.base import Tool


class _GetStockDataInput(BaseModel):
    ticker: str
    period: str = "1mo"


def _get_stock_data(inputs: _GetStockDataInput) -> dict:
    t = yf.Ticker(inputs.ticker.upper())
    hist = t.history(period=inputs.period)
    if hist.empty:
        return {"error": f"No data found for ticker '{inputs.ticker}'"}

    price = round(float(hist["Close"].iloc[-1]), 4)
    volume = int(hist["Volume"].iloc[-1])
    first_close = float(hist["Close"].iloc[0])
    pct_change = round((price - first_close) / first_close * 100, 4) if first_close else None

    info = t.fast_info
    market_cap = getattr(info, "market_cap", None)

    return {
        "ticker": inputs.ticker.upper(),
        "period": inputs.period,
        "price": price,
        "volume": volume,
        "pct_change": pct_change,
        "market_cap": int(market_cap) if market_cap else None,
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
