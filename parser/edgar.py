"""
SEC EDGAR 10-K parser.

Downloads a 10-K filing HTML from SEC EDGAR, strips noisy markup, and
splits the document into named sections (Item 1, Item 1A, Item 7, etc.).

Usage:
    python -m parser.edgar --cik 0001045810 --output data/nvda_10k_2025.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

import warnings

import requests
from bs4 import BeautifulSoup, Comment, Tag, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# EDGAR constants
# ---------------------------------------------------------------------------

EDGAR_BASE = "https://www.sec.gov"
EDGAR_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"
# SEC requires a non-empty User-Agent that includes company/email info.
USER_AGENT = "finance-sentiment-engine research@example.com"

HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
    "Accept": "application/json, text/html, */*",
}

# ---------------------------------------------------------------------------
# Section regex patterns  (matches EDGAR's known Item heading patterns)
# ---------------------------------------------------------------------------
# Matches: "Item 1.", "ITEM 1A.", "Item 1A:", "ITEM 1 —", "Item 7 MD&A" etc.
SECTION_PATTERN = re.compile(
    r"(?:^|\n)\s*"
    r"(?:ITEM|Item)\s+"
    r"(?P<num>\d{1,2}[A-C]?)"
    r"[.\s:—\-]+"
    r"(?P<title>[^\n]{3,80})",
    re.MULTILINE,
)

# Canonical section names we care about (used to label captured sections).
KNOWN_ITEMS: dict[str, str] = {
    "1":  "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "2":  "Properties",
    "3":  "Legal Proceedings",
    "4":  "Mine Safety Disclosures",
    "5":  "Market for Registrant Common Equity",
    "6":  "Reserved",
    "7":  "MD&A",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8":  "Financial Statements",
    "9":  "Changes in and Disagreements with Accountants",
    "9A": "Controls and Procedures",
    "9B": "Other Information",
    "10": "Directors Executive Officers and Corporate Governance",
    "11": "Executive Compensation",
    "12": "Security Ownership",
    "13": "Certain Relationships",
    "14": "Principal Accountant Fees",
    "15": "Exhibits",
}


# ---------------------------------------------------------------------------
# EDGAR API helpers
# ---------------------------------------------------------------------------

def _get(url: str, retries: int = 3, delay: float = 1.0) -> requests.Response:
    """GET with simple retry and SEC-mandated rate limiting."""
    for attempt in range(retries):
        resp = requests.get(url, headers=HEADERS, timeout=30)
        if resp.status_code == 200:
            return resp
        if resp.status_code == 429:
            wait = delay * (2 ** attempt)
            logger.warning("Rate limited — waiting %.1fs", wait)
            time.sleep(wait)
        else:
            resp.raise_for_status()
    raise RuntimeError(f"Failed to GET {url} after {retries} attempts")


def find_10k_filing(cik: str) -> tuple[str, str]:
    """
    Return (accession_number, primary_document) for the most recent 10-K.

    Queries the EDGAR submissions endpoint and walks recent filings.
    """
    padded = cik.lstrip("0").zfill(10)
    url = EDGAR_SUBMISSIONS.format(cik=padded)
    logger.info("Fetching submissions: %s", url)
    data = _get(url).json()

    filings = data.get("filings", {}).get("recent", {})
    forms = filings.get("form", [])
    accessions = filings.get("accessionNumber", [])
    primary_docs = filings.get("primaryDocument", [])

    for i, form in enumerate(forms):
        if form in ("10-K", "10-K/A"):
            acc = accessions[i].replace("-", "")
            doc = primary_docs[i]
            logger.info("Found %s: accession=%s  doc=%s", form, acc, doc)
            return acc, doc

    raise ValueError(f"No 10-K found for CIK {cik}")


def build_filing_url(cik: str, accession: str, document: str) -> str:
    padded = cik.lstrip("0").zfill(10)
    return f"{EDGAR_BASE}/Archives/edgar/data/{int(padded)}/{accession}/{document}"


def download_filing_html(url: str) -> str:
    logger.info("Downloading filing: %s", url)
    resp = _get(url)
    resp.encoding = resp.apparent_encoding or "utf-8"
    return resp.text


# ---------------------------------------------------------------------------
# HTML cleaning
# ---------------------------------------------------------------------------

_STRIP_TAGS = {"table", "font", "style", "script", "meta", "head", "noscript"}
_INLINE_STYLE_RE = re.compile(r'\s*style\s*=\s*["\'][^"\']*["\']', re.IGNORECASE)
_MULTI_BLANK_RE = re.compile(r"\n{3,}")
_NBSP_RE = re.compile(r"\xa0+|\u00a0+")


def clean_html(raw_html: str) -> str:
    """
    Parse HTML with BeautifulSoup, remove noisy elements, return plain text.

    Strips: <table>, <font>, inline styles, HTML comments, <script>/<style>.
    Preserves paragraph and section structure via newline separators.
    """
    soup = BeautifulSoup(raw_html, "lxml")

    # Remove comments
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        comment.extract()

    # Remove noisy structural tags (but keep their text content)
    for tag_name in _STRIP_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # Strip inline style attributes from remaining tags
    for tag in soup.find_all(True):
        if isinstance(tag, Tag):
            attrs_to_keep = {k: v for k, v in tag.attrs.items() if k not in ("style", "class", "id")}
            tag.attrs = attrs_to_keep

    text = soup.get_text(separator="\n")

    # Normalise whitespace
    text = _NBSP_RE.sub(" ", text)
    text = _MULTI_BLANK_RE.sub("\n\n", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    return text.strip()


# ---------------------------------------------------------------------------
# Section splitting
# ---------------------------------------------------------------------------

def _canonical_label(num: str, raw_title: str) -> str:
    """Build a clean section key like 'Item 1A - Risk Factors'."""
    canonical = KNOWN_ITEMS.get(num.upper(), raw_title.strip())
    return f"Item {num} - {canonical}"


def split_sections(text: str) -> dict[str, str]:
    """
    Split plain text into sections keyed by canonical Item label.

    Uses SECTION_PATTERN to find headings; text between consecutive headings
    becomes the section body.  The 'full_text' key always holds the raw text.
    """
    matches = list(SECTION_PATTERN.finditer(text))
    if not matches:
        logger.warning("No section headings found — returning as single block")
        return {"full_text": text}

    sections: dict[str, str] = {}
    for idx, match in enumerate(matches):
        num = match.group("num").upper()
        raw_title = match.group("title")
        label = _canonical_label(num, raw_title)

        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[start:end].strip()

        # Skip near-empty sections (likely TOC references)
        if len(body) < 100:
            continue

        # If the same item appears twice (TOC + body), keep the longer copy
        if label in sections and len(sections[label]) >= len(body):
            continue

        sections[label] = body

    sections["full_text"] = text
    logger.info("Extracted %d sections", len(sections) - 1)
    return sections


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def parse_10k(cik: str, output_path: Optional[str] = None) -> dict[str, str]:
    """
    Full pipeline: find → download → clean → split → save.

    Returns the sections dict.
    """
    accession, document = find_10k_filing(cik)
    url = build_filing_url(cik, accession, document)
    raw_html = download_filing_html(url)

    logger.info("Cleaning HTML (%d chars)", len(raw_html))
    plain_text = clean_html(raw_html)
    logger.info("Plain text length: %d chars", len(plain_text))

    sections = split_sections(plain_text)

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Metadata block
        result = {
            "_meta": {
                "cik": cik,
                "accession": accession,
                "source_document": document,
                "source_url": url,
                "char_count": len(plain_text),
                "section_count": len(sections) - 1,
            },
            **sections,
        }
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Saved to %s", output_path)

    return sections


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    parser = argparse.ArgumentParser(description="Download and parse a 10-K from SEC EDGAR")
    parser.add_argument("--cik", default="0001045810", help="SEC CIK number (default: NVDA)")
    parser.add_argument("--output", default="data/nvda_10k_2025.json", help="Output JSON path")
    args = parser.parse_args()

    sections = parse_10k(cik=args.cik, output_path=args.output)
    print(f"\n✅  Done — {len(sections) - 1} sections extracted → {args.output}")
    for key in sections:
        if key != "full_text":
            snippet = sections[key][:80].replace("\n", " ")
            print(f"  {key}: {snippet}…")


if __name__ == "__main__":
    _cli()
