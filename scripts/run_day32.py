"""Day 32 — Braintrust Evaluation Platform Comparison

LangSmith (Day 31) ile aynı 6 model kombinasyonunu Braintrust'ta koşturur.
UX farklarını gözlemlemek için tasarlandı.

LangSmith farkları:
  - Dataset inline geçiliyor (cloud'a upload gerekmez)
  - Scorer imzası: (input, output, expected) → LangSmith'teki (outputs, reference_outputs) yerine
  - Pairwise evaluator SDK'da yok → Braintrust UI'dan karşılaştırma yapılır
  - Sonuçlar: https://www.braintrust.dev/app

Evaluators:
  1. sentiment_accuracy  — exact match (0.0 veya 1.0)
  2. reasoning_quality   — LLM judge gpt-4o-mini (0–5 → normalize 0–1)

Kullanım:
    python scripts/run_day32.py
    python scripts/run_day32.py --quick   # sadece ilk 2 combo

Gereklilik:
    BRAINTRUST_API_KEY .env içinde dolu olmalı
    (braintrust.dev → Settings → API Keys)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))  # run_day30 import'u için

from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=True)

from openai import OpenAI as OpenAIClient

# Braintrust ana eval fonksiyonu ve Score tipi
from braintrust import Eval, Score

# Proje bileşenleri — Day 31 ile aynı
from analyzer.runner import SYSTEM, _build_messages
from providers.factory import get_provider
from schemas import NewsAnalysis

# 50 örneklik golden dataset — Day 30'dan import
# (LangSmith'te bu cloud dataset'ten çekiliyordu; Braintrust'ta inline geçiyoruz)
from run_day30 import GOLDEN_EXAMPLES


# ── Sabitler ──────────────────────────────────────────────────────────────────

BRAINTRUST_PROJECT = "finance-sentiment"   # Braintrust dashboard'daki proje adı
JUDGE_MODEL        = "gpt-4o-mini"         # Day 31 ile aynı jüri model

EVAL_COMBOS: list[tuple[str, str]] = [
    ("openai",    "gpt-4.1-mini"),
    ("openai",    "gpt-4.1-nano"),
    ("groq",      "llama-3.3-70b-versatile"),
    ("groq",      "llama-3.1-8b-instant"),
    ("anthropic", "claude-haiku-4-5-20251001"),
    ("anthropic", "claude-3-5-haiku-20241022"),
]

# LangSmith ile aynı concurrency değerleri
_CONCURRENCY: dict[str, int] = {
    "openai":    5,
    "groq":      2,
    "anthropic": 3,
}

_judge_client: OpenAIClient | None = None


def _judge() -> OpenAIClient:
    global _judge_client
    if _judge_client is None:
        _judge_client = OpenAIClient()
    return _judge_client


# ── Braintrust Dataset ────────────────────────────────────────────────────────
#
# LangSmith: dataset_name string → cloud'dan çekilir
#   evaluate(target, data="finance-sentiment-v1", ...)
#
# Braintrust: data callable → list[{input, expected}] inline
#   Eval(..., data=get_dataset, ...)
#
# Avantaj: network bağımlılığı yok, hızlı, versiyonlanabilir
# Dezavantaj: dataset UI'da görünmüyor (yalnızca experiment results)

def get_dataset() -> list[dict]:
    """50 golden örneği Braintrust formatına çevirir.

    LangSmith formatı: inputs={...}, outputs={expected_sentiment: ...}
    Braintrust formatı: input={...}, expected={expected_sentiment: ...}
    Fark: 's' yok (input/expected tekil).
    """
    return [
        {
            "input":    ex["inputs"],    # ticker, title, summary
            "expected": ex["outputs"],   # expected_sentiment, expected_urgency, expected_key_event
        }
        for ex in GOLDEN_EXAMPLES
    ]


# ── Scorer 1: Sentiment Accuracy ──────────────────────────────────────────────
#
# LangSmith imzası:
#   def sentiment_accuracy(outputs: dict, reference_outputs: dict) → dict
#   return {"key": "sentiment_accuracy", "score": 1.0}
#
# Braintrust imzası:
#   def sentiment_accuracy(input, output, expected) → float | Score | dict
#   return 1.0  # fonksiyon adı metrik adı olur
#
# Kritik fark: LangSmith "key" parametresi ister; Braintrust fonksiyon adını kullanır.

def sentiment_accuracy(input, output, expected) -> float:
    """Tahmin edilen sentiment ile beklenen etiket tam eşleşmesi.
    Braintrust: sadece float döndür — fonksiyon adı metrik adı olur."""
    predicted = (output.get("sentiment") or "").lower().strip()
    exp_sent  = (expected.get("expected_sentiment") or "").lower().strip()
    return 1.0 if predicted == exp_sent else 0.0


# ── Scorer 2: Reasoning Quality ───────────────────────────────────────────────
#
# LangSmith imzası:
#   def reasoning_quality(inputs, outputs, reference_outputs) → dict
#   return {"key": "reasoning_quality", "score": raw/5.0, "comment": "..."}
#
# Braintrust imzası:
#   def reasoning_quality(input, output, expected) → Score
#   Score(name=..., score=..., metadata={...})
#
# Score nesnesi metadata taşıyabilir → Braintrust UI'da "comment" görünür.

_REASONING_PROMPT = """\
You are a financial analysis evaluator. Rate the quality of the model output below.

News item:
  Ticker : {ticker}
  Title  : {title}
  Summary: {news_summary}

Model output:
  Sentiment : {sentiment}
  Brief     : {brief}

Expected sentiment: {expected}

Scoring rubric (0–5):
  5 = Correct sentiment AND specific reasoning grounded in the news text
  4 = Correct sentiment with adequate reasoning
  3 = Correct sentiment but reasoning is vague or generic
  2 = Wrong sentiment, but the analysis shows partial understanding
  1 = Wrong sentiment with poor reasoning
  0 = Incoherent output or sentiment field missing

Respond with JSON only — no markdown, no extra keys:
{{"score": <integer 0-5>, "comment": "<one sentence>"}}"""


def reasoning_quality(input, output, expected) -> Score:
    """LLM jüri (gpt-4o-mini) reasoning kalitesini 0–5 puanlar → normalize 0–1.
    Score.metadata içindeki comment Braintrust trace UI'da görülebilir."""
    prompt = _REASONING_PROMPT.format(
        ticker=input.get("ticker", ""),
        title=input.get("title", ""),
        news_summary=input.get("summary", ""),
        sentiment=output.get("sentiment", ""),
        brief=output.get("summary_text", ""),
        expected=expected.get("expected_sentiment", ""),
    )

    resp = _judge().chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )

    result = json.loads(resp.choices[0].message.content)
    raw    = max(0, min(5, int(result.get("score", 0))))

    return Score(
        name="reasoning_quality",
        score=raw / 5.0,
        metadata={"comment": result.get("comment", "")},
    )


# ── Task Factory ──────────────────────────────────────────────────────────────
#
# LangSmith: async def target(inputs) → dict
#   LangSmith aevaluate() async-native, max_concurrency ile paralel çalışır
#
# Braintrust: async def task(input) → dict
#   Braintrust Eval() async task destekler; kendi event loop'unu yönetir
#
# İkisi de aynı provider.agenerate_structured() çağrısı yapar.
# Fark: parametre adı "inputs" → "input" (tekil)

def make_task(provider_name: str, model: str):
    """Verilen provider/model için Braintrust-uyumlu async task döndürür."""
    provider = get_provider(provider_name, model)

    async def task(input_dict: dict) -> dict:
        messages = _build_messages(
            title=input_dict["title"],
            summary=input_dict["summary"],
            ticker=input_dict["ticker"],
        )
        analysis, _ = await provider.agenerate_structured(
            messages=messages,
            schema=NewsAnalysis,
            system=SYSTEM,
        )
        return {
            "sentiment":    analysis.sentiment,
            "urgency":      analysis.urgency,
            "key_event":    analysis.key_event,
            "summary_text": analysis.summary,
        }

    task.__name__ = f"task_{provider_name}_{model.replace('-', '_').replace('.', '_')}"
    return task


# ── Skor Özeti Yardımcısı ─────────────────────────────────────────────────────

def _extract_scores(result) -> dict[str, float]:
    """EvalResultWithSummary'den metrik ortalamaları çıkarır.

    Braintrust: result.summary.scores → Dict[str, SummaryScore]
    SummaryScore.score → float (mean across all examples)
    """
    try:
        scores_map = result.summary.scores  # {metric_name: SummaryScore}
        return {
            name: s.score
            for name, s in scores_map.items()
            if s is not None and s.score is not None
        }
    except AttributeError:
        return {}


# ── Main ──────────────────────────────────────────────────────────────────────

def main(quick: bool = False) -> None:
    # BRAINTRUST_API_KEY kontrolü
    api_key = os.getenv("BRAINTRUST_API_KEY", "").strip()
    if not api_key:
        print(
            "[ERROR] BRAINTRUST_API_KEY bulunamadı.\n"
            "\n"
            "  Adımlar:\n"
            "  1. https://www.braintrust.dev adresinde hesap aç\n"
            "  2. Settings → API Keys → 'Create API Key'\n"
            "  3. .env dosyasına ekle: BRAINTRUST_API_KEY=bt_...\n"
        )
        sys.exit(1)

    combos  = EVAL_COMBOS[:2] if quick else EVAL_COMBOS
    dataset = get_dataset()

    print("=" * 65)
    print("Day 32 — Braintrust Evaluation")
    print(f"  Project  : {BRAINTRUST_PROJECT}")
    print(f"  Combos   : {len(combos)}{' (quick)' if quick else ''}")
    print(f"  Examples : {len(dataset)}")
    print(f"  Judge    : {JUDGE_MODEL}")
    print("=" * 65)

    summary_rows: list[dict] = []

    for provider_name, model in combos:
        combo_key   = f"{provider_name}/{model}"
        exp_name    = f"day32-{provider_name}-{model.replace('/', '-')}"
        concurrency = _CONCURRENCY.get(provider_name, 3)

        print(f"\n▶  {combo_key}  (concurrency={concurrency})")

        task = make_task(provider_name, model)

        try:
            t0 = time.perf_counter()

            # Braintrust Eval — LangSmith aevaluate() karşılığı
            #
            # LangSmith:
            #   await aevaluate(target, data="dataset-name", evaluators=[...])
            #
            # Braintrust:
            #   Eval(project_name, data=callable, task=fn, scores=[...])
            #
            # İki önemli fark:
            #   1. data = inline list (cloud dataset adı değil)
            #   2. Senkron çağrı (await gerektirmez — Eval kendi loop'unu yönetir)
            result = Eval(
                BRAINTRUST_PROJECT,
                data=lambda: dataset,
                task=task,
                scores=[sentiment_accuracy, reasoning_quality],
                experiment_name=exp_name,
                max_concurrency=concurrency,
            )

            elapsed = time.perf_counter() - t0
            scores  = _extract_scores(result)
            sa      = scores.get("sentiment_accuracy", float("nan"))
            rq      = scores.get("reasoning_quality",  float("nan"))

            # Braintrust experiment URL
            exp_url = ""
            try:
                exp_url = result.summary.experiment_url or ""
            except AttributeError:
                pass

            row = {
                "combo":              combo_key,
                "experiment":         exp_name,
                "sentiment_accuracy": sa,
                "reasoning_quality":  rq,
                "elapsed_s":          elapsed,
                "url":                exp_url,
                "error":              None,
            }
            summary_rows.append(row)

            sa_str = f"{sa:.0%}" if sa == sa else "N/A"
            rq_str = f"{rq:.2f}" if rq == rq else "N/A"
            print(f"  ✓ {elapsed:.0f}s  sentiment_acc={sa_str}  reasoning_quality={rq_str}")
            if exp_url:
                print(f"  → {exp_url}")

        except Exception as exc:
            summary_rows.append({
                "combo": combo_key, "experiment": exp_name,
                "sentiment_accuracy": float("nan"),
                "reasoning_quality":  float("nan"),
                "elapsed_s": 0.0, "url": "", "error": str(exc),
            })
            print(f"  ✗ failed: {exc}")

    # ── Özet tablo ────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("BRAINTRUST BASELINE SCORES")
    print("=" * 65)
    header = f"{'Combo':<38} {'SentAcc':>7} {'ReasonQ':>7} {'Time(s)':>7}"
    print(header)
    print("-" * len(header))
    for r in summary_rows:
        if r["error"]:
            print(f"{r['combo']:<38} {'ERROR':>7} {'ERROR':>7} {'—':>7}")
        else:
            sa     = r["sentiment_accuracy"]
            rq     = r["reasoning_quality"]
            sa_str = f"{sa:.0%}" if sa == sa else "N/A"
            rq_str = f"{rq:.2f}" if rq == rq else "N/A"
            print(f"{r['combo']:<38} {sa_str:>7} {rq_str:>7} {r['elapsed_s']:>7.0f}")

    print("=" * 65)
    print(f"\nTüm experimentler Braintrust'ta '{BRAINTRUST_PROJECT}' projesi altında görünür.")
    print(
        "\nNot: Pairwise (brief_quality) Braintrust SDK'da aevaluate_comparative() karşılığı yok.\n"
        "     Braintrust UI'da iki experiment'ı 'Compare' ile yan yana incele."
    )
    print("\nDone.")


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    main(quick=quick)
