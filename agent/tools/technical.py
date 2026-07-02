import pandas as pd
import yfinance as yf
from pydantic import BaseModel

from agent.tools.base import Tool


class _GetTechnicalAnalysisInput(BaseModel):
    ticker: str
    period: str = "1y"  # needs ≥200 trading days for EMA200


def _rsi(close: pd.Series, window: int = 14) -> float | None:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Wilder's smoothing: com = window - 1  →  alpha = 1/window
    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()
    rs = avg_gain / avg_loss
    val = (100 - 100 / (1 + rs)).iloc[-1]
    return round(float(val), 2) if pd.notna(val) else None


def _ema(close: pd.Series, span: int) -> float | None:
    if len(close) < span:
        return None
    val = close.ewm(span=span, adjust=False).mean().iloc[-1]
    return round(float(val), 4) if pd.notna(val) else None


def _macd(close: pd.Series) -> dict:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    line = ema12 - ema26
    signal = line.ewm(span=9, adjust=False).mean()
    histogram = line - signal

    def _last(s: pd.Series) -> float | None:
        v = s.iloc[-1]
        return round(float(v), 4) if pd.notna(v) else None

    return {
        "macd_line": _last(line),
        "signal_line": _last(signal),
        "histogram": _last(histogram),
    }


def _trend(price: float, ema20: float | None, ema50: float | None, ema200: float | None) -> str:
    if ema20 is not None and ema50 is not None and ema200 is not None:
        if price > ema20 > ema50 > ema200:
            return "bullish"
        if price < ema20 < ema50 < ema200:
            return "bearish"
    return "neutral"


def _get_technical_analysis(inputs: _GetTechnicalAnalysisInput) -> dict:
    t = yf.Ticker(inputs.ticker.upper())
    hist = t.history(period=inputs.period)

    if hist.empty:
        return {"error": f"No data found for ticker '{inputs.ticker}'"}

    close = hist["Close"].dropna()
    if len(close) < 26:
        return {"error": f"Insufficient history for '{inputs.ticker}' (need ≥26 trading days)"}

    price = round(float(close.iloc[-1]), 4)
    ema20 = _ema(close, 20)
    ema50 = _ema(close, 50)
    ema200 = _ema(close, 200)

    return {
        "ticker": inputs.ticker.upper(),
        "price": price,
        "rsi_14": _rsi(close),
        "ema_20": ema20,
        "ema_50": ema50,
        "ema_200": ema200,
        "macd": _macd(close),
        "trend_direction": _trend(price, ema20, ema50, ema200),
    }


get_technical_analysis = Tool(
    name="get_technical_analysis",
    description=(
        "Computes technical indicators for a given ticker from Yahoo Finance price history. "
        "Returns RSI-14, EMA 20/50/200, MACD (line, signal, histogram), "
        "and overall trend direction (bullish/bearish/neutral)."
    ),
    input_schema=_GetTechnicalAnalysisInput,
    handler=_get_technical_analysis,
)
