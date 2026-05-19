#!/usr/bin/env bash
# demo.sh — end-to-end Finance Sentiment Engine demo on 5 headlines
#
# Usage:
#   bash scripts/demo.sh
#
# Requires:
#   - .env with OPENAI_API_KEY set (or export it before running)
#   - uv / pip dependencies installed

set -euo pipefail

PROVIDER="${PROVIDER:-openai}"
MODEL="${MODEL:-gpt-4.1-nano}"
LIMIT=5

# Resolve project root regardless of where the script is called from
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

# Load .env if present and not already set
if [ -f ".env" ]; then
  # shellcheck disable=SC2046
  export $(grep -v '^\s*#' .env | grep -v '^\s*$' | xargs)
fi

echo "╔══════════════════════════════════════════════════╗"
echo "║        Finance Sentiment Engine — Demo           ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""
echo "Provider : $PROVIDER"
echo "Model    : $MODEL"
echo "Items    : $LIMIT headlines"
echo ""

echo "▶ Step 1/3  Scraping latest financial news…"
python -c "
import json
from pathlib import Path
from scraper.yahoo import fetch_news

TICKERS = ['NVDA', 'TSLA', 'AAPL', 'MSFT', 'AMZN']
items = fetch_news(tickers=TICKERS, limit=20)
print(f'  Fetched {len(items)} items from RSS feeds.')
cache = Path('cache')
cache.mkdir(exist_ok=True)
path = cache / 'news.jsonl'
path.write_text('\n'.join(json.dumps(i.model_dump()) for i in items))
print(f'  Saved → {path}')
"
echo ""

echo "▶ Step 2/3  Analyzing top $LIMIT headlines with $PROVIDER/$MODEL…"
python main.py --provider "$PROVIDER" --model "$MODEL" --limit "$LIMIT"
echo ""

echo "▶ Step 3/3  Generating today's brief…"
TODAY="$(python -c 'from datetime import date; print(date.today())')"
python -m report.daily --date "$TODAY" --top-n "$LIMIT" --provider "$PROVIDER" --model "$MODEL"
echo ""
echo "✅  Done. Report saved to reports/${TODAY}.md"
