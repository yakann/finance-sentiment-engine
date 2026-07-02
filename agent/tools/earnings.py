from datetime import datetime, timezone

import pandas as pd
import yfinance as yf
from pydantic import BaseModel

from agent.tools.base import Tool


class _GetEarningsInput(BaseModel):
    ticker: str


def _get_earnings(inputs: _GetEarningsInput) -> dict:
    t = yf.Ticker(inputs.ticker.upper())

    # ── Next earnings date ────────────────────────────────────────────────────
    next_earnings_date = None
    warnings: list[str] = []
    try:
        cal = t.calendar
        if isinstance(cal, dict):
            dates = cal.get("Earnings Date", [])
            if dates:
                next_earnings_date = str(dates[0])
    except Exception as e:
        warnings.append(f"calendar fetch failed: {e}")

    # ── Last 4 quarters beat/miss history ────────────────────────────────────
    history: list[dict] = []
    try:
        df = t.earnings_dates
        if df is not None and not df.empty:
            past = df[df["Reported EPS"].notna()].head(4)
            for dt_idx, row in past.iterrows():
                surprise = row.get("Surprise(%)")
                surprise_f = float(surprise) if pd.notna(surprise) else None

                if surprise_f is None:
                    beat_miss = None
                elif surprise_f > 0:
                    beat_miss = "beat"
                elif surprise_f < 0:
                    beat_miss = "miss"
                else:
                    beat_miss = "in-line"

                eps_est = row.get("EPS Estimate")
                eps_act = row.get("Reported EPS")

                history.append({
                    "date": str(dt_idx.date()),
                    "eps_estimate": round(float(eps_est), 4) if pd.notna(eps_est) else None,
                    "eps_actual": round(float(eps_act), 4) if pd.notna(eps_act) else None,
                    "surprise_pct": round(surprise_f, 2) if surprise_f is not None else None,
                    "beat_miss": beat_miss,
                })
    except Exception as e:
        warnings.append(f"earnings history fetch failed: {e}")

    if not history and next_earnings_date is None:
        return {"error": f"No earnings data found for ticker '{inputs.ticker}'"}

    # Staleness check: if newest reported quarter is > 2 years old, flag it
    if history:
        try:
            newest_date = datetime.strptime(history[0]["date"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
            age_days = (datetime.now(timezone.utc) - newest_date).days
            if age_days > 730:
                warnings.append(
                    f"Earnings data is stale — the most recent reported quarter is from "
                    f"{history[0]['date']} ({age_days // 365} years ago). "
                    "Yahoo Finance may not carry full EPS history for this ticker. "
                    "Do NOT use these figures to assess recent performance."
                )
        except (ValueError, KeyError):
            pass

    result: dict = {
        "ticker": inputs.ticker.upper(),
        "next_earnings_date": next_earnings_date,
        "last_4_quarters": history,
    }
    if warnings:
        result["warnings"] = warnings
    return result


get_earnings = Tool(
    name="get_earnings",
    description=(
        "Fetches earnings data for a given ticker using Yahoo Finance. "
        "Returns the next scheduled earnings date and the last 4 quarters of "
        "EPS beat/miss history (estimate, actual, surprise %, and beat/miss verdict)."
    ),
    input_schema=_GetEarningsInput,
    handler=_get_earnings,
)
