"""Day 21 tests: brief CLI — structured Markdown output for NVDA, TSLA, MSFT."""

import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_brief(ticker: str) -> str:
    path = Path("briefs") / f"{ticker.upper()}.md"
    assert path.exists(), f"Brief file not found: {path}. Run: python brief.py {ticker}"
    return path.read_text(encoding="utf-8")


def _assert_sections(content: str, ticker: str) -> None:
    required = [
        f"# {ticker.upper()}",
        "Investment Brief",
        "## 1. Company Snapshot",
        "## 2. Recent News & Sentiment",
        "## 3. Key Risk Factors",
        "## 4. Analyst Verdict",
        "## 5. Sources",
        "Analyst Engine",
    ]
    for section in required:
        assert section in content, f"[{ticker}] Missing section: {section!r}"


def _assert_stock_data(content: str, ticker: str) -> None:
    assert re.search(r"Current Price.*\$[\d,\.]+", content), \
        f"[{ticker}] Missing 'Current Price $X' row"
    assert re.search(r"Market Cap.*\$[\d,\.]+", content), \
        f"[{ticker}] Missing 'Market Cap $X' row"


def _assert_sentiment_section(content: str, ticker: str) -> None:
    # At least one sentiment emoji bullet
    has_sentiment_bullet = any(
        emoji in content for emoji in ("🟢", "🔴", "⚪")
    )
    assert has_sentiment_bullet, f"[{ticker}] No sentiment emoji bullets found"
    assert "Overall Sentiment" in content, f"[{ticker}] Missing 'Overall Sentiment' line"


def _assert_risks_section(content: str, ticker: str) -> None:
    # Section 3 must have at least 3 bullet points
    section_match = re.search(
        r"## 3\. Key Risk Factors(.*?)(?=## 4\.|$)", content, re.DOTALL
    )
    assert section_match, f"[{ticker}] Key Risk Factors section not found"
    bullets = re.findall(r"^-\s+", section_match.group(1), re.MULTILINE)
    assert len(bullets) >= 3, f"[{ticker}] Expected ≥3 risk bullets, found {len(bullets)}"


def _assert_verdict(content: str, ticker: str) -> None:
    assert "Recommendation" in content, f"[{ticker}] Missing Recommendation"
    has_verdict = any(word in content for word in ("Buy", "Hold", "Watch"))
    assert has_verdict, f"[{ticker}] Verdict must contain Buy/Hold/Watch"


def _assert_sources(content: str, ticker: str) -> None:
    assert "Sources" in content, f"[{ticker}] Missing Sources section"
    assert "Yahoo Finance" in content or "http" in content.lower(), \
        f"[{ticker}] Sources section appears empty"


def _assert_file_size(ticker: str) -> None:
    path = Path("briefs") / f"{ticker.upper()}.md"
    size = path.stat().st_size
    assert size >= 800, f"[{ticker}] Brief file too small ({size} bytes) — likely truncated"


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_nvda_brief():
    print("\n" + "=" * 70)
    print("TEST 1: NVDA brief — all sections, stock data, 10-K risks")
    print("=" * 70)

    ticker = "NVDA"
    content = _read_brief(ticker)

    _assert_sections(content, ticker)
    _assert_stock_data(content, ticker)
    _assert_sentiment_section(content, ticker)
    _assert_risks_section(content, ticker)
    _assert_verdict(content, ticker)
    _assert_sources(content, ticker)
    _assert_file_size(ticker)

    print(f"  File size: {Path('briefs/NVDA.md').stat().st_size:,} bytes")
    print(f"  Sections: OK")
    print("  PASSED")


def test_tsla_brief():
    print("\n" + "=" * 70)
    print("TEST 2: TSLA brief — all sections, graceful 10-K fallback")
    print("=" * 70)

    ticker = "TSLA"
    content = _read_brief(ticker)

    _assert_sections(content, ticker)
    _assert_stock_data(content, ticker)
    _assert_sentiment_section(content, ticker)
    _assert_risks_section(content, ticker)
    _assert_verdict(content, ticker)
    _assert_sources(content, ticker)
    _assert_file_size(ticker)

    print(f"  File size: {Path('briefs/TSLA.md').stat().st_size:,} bytes")
    print(f"  Sections: OK")
    print("  PASSED")


def test_msft_brief():
    print("\n" + "=" * 70)
    print("TEST 3: MSFT brief — all sections, graceful 10-K fallback")
    print("=" * 70)

    ticker = "MSFT"
    content = _read_brief(ticker)

    _assert_sections(content, ticker)
    _assert_stock_data(content, ticker)
    _assert_sentiment_section(content, ticker)
    _assert_risks_section(content, ticker)
    _assert_verdict(content, ticker)
    _assert_sources(content, ticker)
    _assert_file_size(ticker)

    print(f"  File size: {Path('briefs/MSFT.md').stat().st_size:,} bytes")
    print(f"  Sections: OK")
    print("  PASSED")


def test_generate_fresh_nvda():
    """End-to-end: generate + save + assert for NVDA (live API call)."""
    print("\n" + "=" * 70)
    print("TEST 4: End-to-end generation — NVDA (live agent run)")
    print("=" * 70)

    from brief import generate_brief, save_brief

    content = generate_brief("NVDA")
    path = save_brief("NVDA", content)

    assert path.exists(), "brief file not written"
    _assert_sections(content, "NVDA")
    _assert_stock_data(content, "NVDA")
    _assert_sentiment_section(content, "NVDA")
    _assert_risks_section(content, "NVDA")
    _assert_verdict(content, "NVDA")
    _assert_sources(content, "NVDA")

    print(f"  Generated and saved → {path}")
    print(f"  File size: {path.stat().st_size:,} bytes")
    print("  PASSED")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=" * 70)
    print("DAY 21 TESTS — Brief CLI: NVDA · TSLA · MSFT")
    print("=" * 70)
    print("NOTE: Tests 1–3 require briefs to already exist (run brief.py first).")
    print("      Test 4 is a live end-to-end generation (requires API keys).")
    print()

    run_live = "--live" in sys.argv

    if run_live:
        test_generate_fresh_nvda()
    else:
        test_nvda_brief()
        test_tsla_brief()
        test_msft_brief()

    print("\n" + "=" * 70)
    print("ALL DAY 21 TESTS PASSED")
    print("brief.py: generate_brief() + save_brief() → briefs/<TICKER>.md")
    print("=" * 70)
