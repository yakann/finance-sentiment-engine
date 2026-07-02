import yfinance as yf
from pydantic import BaseModel
from agent.tools.base import Tool


class _GetValuationInput(BaseModel):
    ticker: str


def _get_valuation(inputs: _GetValuationInput) -> dict:
    t = yf.Ticker(inputs.ticker.upper())
    info = t.info

    if not info or info.get("symbol") is None:
        return {"error": f"No data found for ticker '{inputs.ticker}'"}

    def _r(val: float | None, n: int = 2) -> float | None:
        return round(val, n) if val is not None else None

    currency = info.get("currency", "USD")

    ev_to_ebitda = _r(info.get("enterpriseToEbitda"))
    price_to_book = _r(info.get("priceToBook"))

    # Flag ratios that are implausible — usually caused by currency unit mixing in yfinance
    # for non-USD stocks (Enterprise Value in local currency, EBITDA normalised differently).
    # USD stocks use a lenient threshold (200) because growth/tech companies can legitimately
    # have high EV/EBITDA (e.g. Tesla ~80–120x). Non-USD stocks use 100 because yfinance
    # frequently mixes TRY/USD units, producing garbage values like THYAO 159x (real: ~3.7x).
    warnings: list[str] = []
    ev_ebitda_threshold = 200 if currency == "USD" else 100
    if ev_to_ebitda is not None and (ev_to_ebitda > ev_ebitda_threshold or ev_to_ebitda < 0):
        warnings.append(
            f"EV/EBITDA={ev_to_ebitda} is outside the plausible range (0–{ev_ebitda_threshold}) "
            f"for {currency}-denominated stocks. "
            "This likely reflects a currency mismatch in Yahoo Finance data. Treat with caution."
        )
    pb_threshold = 50 if currency == "USD" else 30
    if price_to_book is not None and price_to_book > pb_threshold:
        warnings.append(
            f"Price/Book={price_to_book} is unusually high for a {currency} stock. "
            "Verify that book value is reported in the same currency as the share price."
        )

    result = {
        "ticker": inputs.ticker.upper(),
        "currency": currency,
        "forward_pe": _r(info.get("forwardPE")),
        "peg_ratio": _r(info.get("pegRatio")),
        "ev_to_ebitda": ev_to_ebitda,
        "ev_to_sales": _r(info.get("enterpriseToRevenue")),
        "price_to_book": price_to_book,
    }
    if warnings:
        result["warnings"] = warnings
    return result


get_valuation = Tool(
    name="get_valuation",
    description=(
        "Fetches valuation multiples for a given ticker using Yahoo Finance. "
        "Returns forward P/E, PEG ratio, EV/EBITDA, EV/Sales, and price-to-book."
    ),
    input_schema=_GetValuationInput,
    handler=_get_valuation,
)
