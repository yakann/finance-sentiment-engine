import yfinance as yf
from pydantic import BaseModel
from agent.tools.base import Tool


class _GetFinancialMetricsInput(BaseModel):
    ticker: str


def _get_financial_metrics(inputs: _GetFinancialMetricsInput) -> dict:
    t = yf.Ticker(inputs.ticker.upper())
    info = t.info

    if not info or info.get("symbol") is None:
        return {"error": f"No data found for ticker '{inputs.ticker}'"}

    def _pct(val: float | None) -> float | None:
        return round(val * 100, 2) if val is not None else None

    eps_actual = info.get("trailingEps")
    _cy = info.get("epsCurrentYear")
    eps_estimate = _cy if _cy is not None else info.get("forwardEps")
    free_cash_flow = info.get("freeCashflow")
    debt_to_equity = info.get("debtToEquity")

    return {
        "ticker": inputs.ticker.upper(),
        "currency": info.get("currency", "USD"),
        "revenue_growth_yoy_pct": _pct(info.get("revenueGrowth")),
        "eps_actual_ttm": round(eps_actual, 4) if eps_actual is not None else None,
        "eps_estimate_current_year": round(eps_estimate, 4) if eps_estimate is not None else None,
        "free_cash_flow": int(free_cash_flow) if free_cash_flow is not None else None,
        "gross_margin_pct": _pct(info.get("grossMargins")),
        "debt_to_equity": round(debt_to_equity, 2) if debt_to_equity is not None else None,
    }


get_financial_metrics = Tool(
    name="get_financial_metrics",
    description=(
        "Fetches fundamental financial metrics for a given ticker using Yahoo Finance. "
        "Returns revenue growth YoY (%), EPS actual (TTM) vs current-year estimate, "
        "free cash flow (USD), gross margin (%), and debt-to-equity ratio."
    ),
    input_schema=_GetFinancialMetricsInput,
    handler=_get_financial_metrics,
)
