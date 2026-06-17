"""Day 31 — LLM-as-Judge Evaluators + 6-Model Baseline

3 evaluators registered with LangSmith:
  1. sentiment_accuracy — exact match, deterministic (0.0 or 1.0)
  2. reasoning_quality  — LLM judge: "Does this output give the correct sentiment
                         with correct reasoning?" (0–5 → normalized to 0–1)
  3. brief_quality      — pairwise LLM judge: "Which summary is better, A or B?"
                         (used via evaluate_comparative)

6 provider × model combos run against 'finance-sentiment-v1' dataset (Day 30):
  openai    / gpt-4.1-mini
  openai    / gpt-4.1-nano
  groq      / llama-3.3-70b-versatile
  groq      / llama-3.1-8b-instant
  anthropic / claude-haiku-4-5-20251001
  anthropic / claude-3-5-haiku-20241022

Usage:
    python scripts/run_day31.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

# Proje kökünü Python path'ine ekliyoruz ki
# "from analyzer.runner import ..." gibi import'lar çalışsın.
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
# .env dosyasından LANGSMITH_API_KEY, OPENAI_API_KEY, GROQ_API_KEY,
# ANTHROPIC_API_KEY gibi değişkenleri okuyoruz.
load_dotenv(ROOT / ".env", override=True)

# Judge LLM için OpenAI client — reasoning_quality ve brief_quality
# evaluator'larında "jüri" olarak gpt-4o-mini kullanacağız.
from openai import OpenAI as OpenAIClient

# LangSmith'in iki ana değerlendirme fonksiyonu:
#   aevaluate()            → async target'ları destekleyen evaluate versiyonu
#   evaluate_comparative() → iki farklı experiment'ı örnek-bazlı karşılaştırır (pairwise)
from langsmith import aevaluate
from langsmith.evaluation import evaluate_comparative

# Day 28/29'dan gelen altyapı:
#   SYSTEM        → LLM'e verilen sistem prompt'u (finans analist rolü + urgency kılavuzu)
#   _build_messages → few-shot örnekleri + kullanıcı haberi birleştiren yardımcı fonksiyon
from analyzer.runner import SYSTEM, _build_messages

# Provider factory: "openai"/"groq"/"anthropic" string'inden doğru client döndürür.
from providers.factory import get_provider

# Pydantic şema: modelin üretmesi gereken yapıyı tanımlar
# (ticker, sentiment, urgency, key_event, summary)
from schemas import NewsAnalysis


# ── Sabitler ──────────────────────────────────────────────────────────────────

# Day 30'da LangSmith'e yüklenen 50 örneklik golden dataset'in adı.
# evaluate() bu ismi kullanarak LangSmith'ten örnekleri çeker.
DATASET_NAME = "finance-sentiment-v1"

# Hem reasoning_quality hem de brief_quality evaluator'larında
# "jüri" olarak kullanacağımız model.
# gpt-4o-mini seçildi: hızlı, ucuz, JSON mode destekliyor.
JUDGE_MODEL = "gpt-4o-mini"

# Karşılaştıracağımız 6 provider × model kombinasyonu.
# Her biri ayrı bir LangSmith "experiment" olarak kaydedilecek.
EVAL_COMBOS: list[tuple[str, str]] = [
    ("openai",    "gpt-4.1-mini"),           # OpenAI'ın küçük ama güçlü modeli
    ("openai",    "gpt-4.1-nano"),            # OpenAI'ın en ucuz/hızlı modeli
    ("groq",      "llama-3.3-70b-versatile"), # Groq'ta çalışan Meta'nın 70B Llama'sı
    ("groq",      "llama-3.1-8b-instant"),    # Groq'ta çalışan küçük 8B Llama
    ("anthropic", "claude-haiku-4-5-20251001"), # Anthropic'in yeni Haiku modeli
    ("anthropic", "claude-3-5-haiku-20241022"), # Anthropic'in önceki Haiku modeli
]

# Groq ücretsiz katmanında dakika başına token limiti (TPM) çok dar:
#   llama-3.3-70b → ~12K TPM, llama-3.1-8b → ~6K TPM
# Bu yüzden Groq için eşzamanlı istek sayısını 2'ye düşürüyoruz.
# Daha yüksek değer → 429 RateLimitError fırtınası.
_CONCURRENCY: dict[str, int] = {
    "openai":    5,  # OpenAI limitleri rahat, 5 paralel istek sorunsuz
    "groq":      2,  # TPM limiti nedeniyle düşük tutuldu
    "anthropic": 3,  # Anthropic orta seviye limit
}

# Judge client singleton: her evaluator çağrısında yeniden OpenAI() oluşturmamak için
# modül düzeyinde bir değişkende saklıyoruz. İlk kullanımda _judge() içinde atanır.
_judge_client: OpenAIClient | None = None


def _judge() -> OpenAIClient:
    """Lazy singleton: ilk çağrıda OpenAI client oluşturur, sonra aynısını döndürür."""
    global _judge_client
    if _judge_client is None:
        _judge_client = OpenAIClient()
    return _judge_client


# ── Evaluator 1: Sentiment Accuracy ──────────────────────────────────────────
#
# En basit evaluator: LLM çağrısı yok, tamamen deterministik.
#
# LangSmith evaluate() fonksiyonu bu fonksiyonu her örnek için şöyle çağırır:
#   sentiment_accuracy(
#       outputs         = {"sentiment": "bullish", ...},   # modelin ürettiği
#       reference_outputs = {"expected_sentiment": "bullish", ...}  # golden label
#   )
#
# Dönüş değeri:
#   {"key": "sentiment_accuracy", "score": 1.0}  → doğru tahmin
#   {"key": "sentiment_accuracy", "score": 0.0}  → yanlış tahmin
#
# LangSmith bu score'ları toplayıp experiment sayfasında ortalama gösterir.

def sentiment_accuracy(outputs: dict, reference_outputs: dict) -> dict:
    """Modelin tahmin ettiği sentiment ile golden label'ı karşılaştırır.
    Tam eşleşme → 1.0, farklı → 0.0. LLM çağrısı yok, saf Python."""
    predicted = (outputs.get("sentiment") or "").lower().strip()
    expected  = (reference_outputs.get("expected_sentiment") or "").lower().strip()
    return {
        "key":   "sentiment_accuracy",
        "score": 1.0 if predicted == expected else 0.0,
    }


# ── Evaluator 2: Reasoning Quality ───────────────────────────────────────────
#
# LLM-as-judge pattern: jüri LLM (gpt-4o-mini), değerlendirilen LLM'nin
# çıktısını görüp 0–5 arası puan verir.
#
# Neden ayrı bir "jüri" LLM kullanıyoruz?
#   → Exact match, sadece doğru/yanlış söyler.
#   → Ama model yanlış tahminde de iyi gerekçe yazıyor olabilir (partial credit).
#   → Ya da doğru tahmin etmiş ama gerekçe tamamen boş/alakasız olabilir.
#   → Jüri LLM bu nüansları görebilir.
#
# LangSmith bu fonksiyonu şöyle çağırır (3 argüman → inputs de geliyor):
#   reasoning_quality(
#       inputs            = {"ticker": "NVDA", "title": "...", "summary": "..."},
#       outputs           = {"sentiment": "bullish", "summary_text": "..."},
#       reference_outputs = {"expected_sentiment": "bullish", ...}
#   )
#
# Score 0–5 → biz bunu 5'e bölerek 0.0–1.0 aralığına normalize ediyoruz.
# LangSmith'te tüm metriklerin 0–1 arasında olması grafikleri karşılaştırmayı kolaylaştırır.

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


def reasoning_quality(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Jüri LLM (gpt-4o-mini) modelin gerekçesini 0–5 ile puanlar.
    Prompt'a haber + model çıktısı + beklenen sentiment gönderilir.
    Dönen ham skor 5'e bölünerek 0–1 aralığına normalize edilir."""

    # Prompt şablonuna dataset'ten gelen haber bilgilerini ve
    # değerlendirilen modelin çıktısını yerleştiriyoruz.
    prompt = _REASONING_PROMPT.format(
        ticker=inputs.get("ticker", ""),
        title=inputs.get("title", ""),
        news_summary=inputs.get("summary", ""),
        sentiment=outputs.get("sentiment", ""),
        brief=outputs.get("summary_text", ""),
        expected=reference_outputs.get("expected_sentiment", ""),
    )

    # temperature=0 → deterministik çıktı. Jüri her seferinde aynı soruya
    # aynı puanı versin istiyoruz (tekrarlanabilirlik).
    resp = _judge().chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},  # OpenAI garantili JSON döndürür
        temperature=0,
    )

    result = json.loads(resp.choices[0].message.content)

    # Güvenli skor alma: jüri 0–5 dışı sayı yazarsa max/min ile kırpıyoruz.
    raw = max(0, min(5, int(result.get("score", 0))))

    return {
        "key":     "reasoning_quality",
        "score":   raw / 5.0,           # 0.0–1.0 normalize
        "comment": result.get("comment", ""),  # LangSmith trace'inde görünür
    }


# ── Evaluator 3: Brief Quality (Pairwise) ────────────────────────────────────
#
# Pairwise (ikili karşılaştırma) evaluator:
#   → İki farklı modelin AYNI haber için ürettiği summary'i karşılaştırır.
#   → "A mı daha iyi, B mi, yoksa berabere mi?" sorusunu jüri LLM'e sorar.
#
# Bu evaluator evaluate() ile DEĞİL, evaluate_comparative() ile kullanılır.
# evaluate_comparative() iki experiment'ın (iki ayrı evaluate() koşusu) çıktılarını
# eşleştirerek her örnek için (outputs_a, outputs_b) çiftini bu fonksiyona gönderir.
#
# LangSmith bu fonksiyonu şöyle çağırır:
#   brief_quality(
#       inputs     = {"ticker": "NVDA", "title": "..."},   # aynı haber
#       outputs_a  = {"summary_text": "..."},               # model A'nın çıktısı
#       outputs_b  = {"summary_text": "..."},               # model B'nin çıktısı
#   )
#
# Score anlamı:
#   0.0 → A kazandı
#   0.5 → berabere
#   1.0 → B kazandı

_PAIRWISE_PROMPT = """\
You are a financial analyst evaluating two AI-generated summaries of the same news.

News:
  Ticker : {ticker}
  Title  : {title}

Summary A: {brief_a}
Summary B: {brief_b}

Which summary is better for a financial analyst?
Criteria: accuracy · clarity · conciseness · actionability

Respond with JSON only — no markdown, no extra keys:
{{"winner": "A" | "B" | "tie", "reason": "<one sentence>"}}"""


def brief_quality(inputs: dict, outputs_a: dict, outputs_b: dict) -> dict:
    """Pairwise jüri: iki modelin summary'sini karşılaştırır.
    evaluate_comparative() tarafından çağrılır; evaluate() değil.
    score: 0.0=A kazandı, 0.5=berabere, 1.0=B kazandı"""

    prompt = _PAIRWISE_PROMPT.format(
        ticker=inputs.get("ticker", ""),
        title=inputs.get("title", ""),
        brief_a=outputs_a.get("summary_text") or "(no summary)",
        brief_b=outputs_b.get("summary_text") or "(no summary)",
    )

    resp = _judge().chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )

    result = json.loads(resp.choices[0].message.content)
    winner = result.get("winner", "tie")

    # "A" → 0.0, "B" → 1.0, "tie" → 0.5
    # Bu encoding LangSmith'in pairwise görselleştirmesiyle uyumlu:
    # düşük skor = A daha iyi, yüksek skor = B daha iyi.
    score = 0.0 if winner == "A" else (1.0 if winner == "B" else 0.5)

    return {
        "key":     "brief_quality",
        "score":   score,
        "comment": result.get("reason", ""),
    }


# ── Target Function Factory ───────────────────────────────────────────────────
#
# evaluate() fonksiyonu bir "target" fonksiyon ister:
#   target(inputs: dict) → dict
#
# inputs → dataset'teki bir örneğin inputs alanı: {"ticker": "NVDA", "title": "...", "summary": "..."}
# dönen dict → modelin ürettiği çıktı: {"sentiment": "bullish", "summary_text": "..."}
#
# Biz 6 farklı combo için 6 ayrı target fonksiyon üretiyoruz.
# make_target() bir closure döndürür: içinde provider sabittir, her çağrıda
# sadece inputs değişir.
#
# async def target → LangSmith'in evaluate() fonksiyonu async target'ları destekler.
# Async kullanıyoruz çünkü tüm provider'larımızda agenerate_structured() var
# ve bu fonksiyon max_concurrency kadar örneği paralel işleyebilir.

def make_target(provider_name: str, model: str):
    """Verilen provider/model için LangSmith-uyumlu async target fonksiyon üretir.

    Dönen target(inputs) fonksiyonu:
      1. Dataset örneğinden few-shot mesaj listesi oluşturur
      2. Provider'dan structured output (NewsAnalysis) ister
      3. LangSmith'in beklediği dict formatında döndürür
    """
    # Provider nesnesi closure içinde sabitlenir.
    # OpenAIProvider / GroqProvider / AnthropicProvider döner.
    provider = get_provider(provider_name, model)

    async def target(inputs: dict) -> dict:
        # _build_messages: analyzer/runner.py'den gelen yardımcı.
        # Few-shot örnekler + "Ticker: X\nTitle: Y\nSummary: Z" user mesajı
        # tek bir messages listesi olarak döner.
        messages = _build_messages(
            title=inputs["title"],
            summary=inputs["summary"],
            ticker=inputs["ticker"],
        )

        # agenerate_structured: provider'a mesajları gönderir ve
        # yanıtı doğrudan NewsAnalysis Pydantic modeline parse eder.
        # system=SYSTEM → her provider bunu kendi API formatına çevirir
        #   (OpenAI: messages[0] role=system, Anthropic: top-level system param)
        analysis, _ = await provider.agenerate_structured(
            messages=messages,
            schema=NewsAnalysis,
            system=SYSTEM,
        )

        # LangSmith evaluator'ları bu dict'e erişir.
        # "summary_text" adını kullandık çünkü "summary" hem inputs'ta hem
        # outputs'ta olursa karışıklık çıkabilir.
        return {
            "sentiment":    analysis.sentiment,   # "bullish" | "bearish" | "neutral"
            "urgency":      analysis.urgency,      # "high" | "medium" | "low"
            "key_event":    analysis.key_event,    # earnings, policy_geopolitical, vs.
            "summary_text": analysis.summary,      # ≤200 karakter özet
        }

    # LangSmith experiment adını oluştururken fonksiyon adını kullanır.
    # Anlamlı bir isim veriyoruz ki LangSmith UI'da hangi model olduğu belli olsun.
    target.__name__ = f"target_{provider_name}_{model.replace('-', '_').replace('.', '_')}"
    return target


# ── Skor Toplama Yardımcısı ───────────────────────────────────────────────────
#
# evaluate() bir ExperimentResults nesnesi döndürür.
# Bu nesne üzerinde iterate ettiğimizde her örnek için bir "row" alırız.
# Her row'da evaluation_results → {"results": [EvaluationResult, ...]} var.
# EvaluationResult'ta .key (evaluator adı) ve .score (0–1 arası float) var.
#
# _aggregate() tüm örnekleri gezerek her evaluator için ortalama score hesaplar.

def _aggregate(results) -> dict[str, float]:
    """ExperimentResults'tan her evaluator key'i için ortalama score döndürür.

    Örnek dönüş:
      {"sentiment_accuracy": 0.72, "reasoning_quality": 0.68}
    """
    buckets: dict[str, list[float]] = {}
    for row in results:
        # row.evaluation_results → {"results": [EvaluationResult, ...]}
        for ev in (row.evaluation_results or {}).get("results", []):
            if ev.score is not None:
                buckets.setdefault(ev.key, []).append(ev.score)
    # Her key için liste ortalaması
    return {k: sum(v) / len(v) for k, v in buckets.items() if v}


# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    print("=" * 65)
    print("Day 31 — LLM-as-Judge Evaluators + 6-Model Baseline")
    print(f"  Dataset : {DATASET_NAME}")
    print(f"  Combos  : {len(EVAL_COMBOS)}")
    print(f"  Judge   : {JUDGE_MODEL}")
    print("=" * 65)

    # Her combo için experiment adını saklıyoruz.
    # Döngü bittikten sonra pairwise karşılaştırma için kullanacağız.
    experiment_names: list[str] = []
    summary_rows: list[dict]    = []

    # ── Adım 1: Her provider/model için evaluate() koştur ────────────────────
    for provider_name, model in EVAL_COMBOS:
        combo_key   = f"{provider_name}/{model}"

        # LangSmith'te experiment adının öneki. Gerçek ad şu formatta olur:
        # "day31-openai-gpt-4.1-mini-abc123"
        prefix      = f"day31-{provider_name}-{model.replace('/', '-')}"

        concurrency = _CONCURRENCY.get(provider_name, 3)
        print(f"\n▶  {combo_key}  (concurrency={concurrency})")

        # Bu combo için async target fonksiyon üret
        target = make_target(provider_name, model)

        try:
            t0 = time.perf_counter()

            # evaluate() şu sırada çalışır:
            #   1. LangSmith'ten DATASET_NAME'deki tüm örnekleri çeker (50 adet)
            #   2. Her örnek için target(inputs) çağrısını async olarak koşturur
            #      (max_concurrency kadar paralel)
            #   3. Her target tamamlandıkça evaluator'ları senkron çalıştırır:
            #      → sentiment_accuracy(outputs, reference_outputs)
            #      → reasoning_quality(inputs, outputs, reference_outputs)
            #   4. Tüm sonuçları LangSmith'e yükler ve ExperimentResults döner
            results = await aevaluate(
                target,
                data=DATASET_NAME,
                evaluators=[sentiment_accuracy, reasoning_quality],
                experiment_prefix=prefix,
                max_concurrency=concurrency,
            )

            elapsed = time.perf_counter() - t0

            # results.experiment_name → LangSmith'teki benzersiz experiment adı
            # Pairwise karşılaştırma için bu ismi saklıyoruz.
            exp_name = results.experiment_name
            experiment_names.append(exp_name)

            # Tüm örneklerin score'larını toplayıp ortalama hesapla
            scores = _aggregate(results)

            row = {
                "combo":              combo_key,
                "experiment":         exp_name,
                "sentiment_accuracy": scores.get("sentiment_accuracy", float("nan")),
                "reasoning_quality":  scores.get("reasoning_quality",  float("nan")),
                "elapsed_s":          elapsed,
                "error":              None,
            }
            summary_rows.append(row)

            print(
                f"  ✓ {elapsed:.0f}s  "
                f"sentiment_acc={row['sentiment_accuracy']:.0%}  "
                f"reasoning_quality={row['reasoning_quality']:.2f}  "
                f"→ {exp_name}"
            )

        except Exception as exc:
            # Bir combo başarısız olsa bile diğerleri devam etsin
            summary_rows.append({
                "combo": combo_key, "experiment": None,
                "sentiment_accuracy": float("nan"),
                "reasoning_quality":  float("nan"),
                "elapsed_s": 0.0, "error": str(exc),
            })
            print(f"  ✗ failed: {exc}")

    # ── Adım 2: Pairwise karşılaştırma (brief_quality) ───────────────────────
    #
    # evaluate_comparative() iki experiment'ı alır ve her örnek için
    # her iki modelin çıktısını yan yana brief_quality'e gönderir.
    #
    # Burada ilk iki başarılı experiment'ı karşılaştırıyoruz:
    # genellikle openai/gpt-4.1-mini vs openai/gpt-4.1-nano olacak.

    # Sadece başarılı experiment isimlerini al
    pairwise_exp_names = [
        r["experiment"] for r in summary_rows
        if r["experiment"] and not r["error"]
    ]

    if len(pairwise_exp_names) >= 2:
        exp_a   = pairwise_exp_names[0]
        exp_b   = pairwise_exp_names[1]
        label_a = summary_rows[0]["combo"]
        label_b = summary_rows[1]["combo"]

        print(f"\n── Pairwise brief_quality ──────────────────────────────────")
        print(f"  A: {label_a}  ({exp_a})")
        print(f"  B: {label_b}  ({exp_b})")

        try:
            # evaluate_comparative() şu sırada çalışır:
            #   1. exp_a ve exp_b'deki tüm run'ları eşleştirir (aynı example_id üzerinden)
            #   2. Her eşleşen çift için brief_quality(inputs, outputs_a, outputs_b) çağırır
            #   3. Sonuçları LangSmith'e kaydeder
            comp_results = evaluate_comparative(
                [exp_a, exp_b],
                evaluators=[brief_quality],
            )

            # Pairwise score'ları topla:
            # 0.0 = A kazandı, 0.5 = berabere, 1.0 = B kazandı
            pairwise_scores = []
            for row in comp_results:
                for ev in (row.evaluation_results or {}).get("results", []):
                    if ev.key == "brief_quality" and ev.score is not None:
                        pairwise_scores.append(ev.score)

            if pairwise_scores:
                avg    = sum(pairwise_scores) / len(pairwise_scores)
                a_wins = sum(1 for s in pairwise_scores if s < 0.5)  # A kazandı
                b_wins = sum(1 for s in pairwise_scores if s > 0.5)  # B kazandı
                ties   = len(pairwise_scores) - a_wins - b_wins

                print(
                    f"  Results ({len(pairwise_scores)} examples): "
                    f"A wins={a_wins}  ties={ties}  B wins={b_wins}  "
                    f"avg_score={avg:.2f}"
                )

                # Ortalama skor 0.45'in altı → A genel olarak daha iyi
                # Ortalama skor 0.55'in üstü → B genel olarak daha iyi
                # Arada → jüri kararsız, berabere
                winner = label_a if avg < 0.45 else (label_b if avg > 0.55 else "tie")
                print(f"  → Overall winner: {winner}")

        except Exception as exc:
            print(f"  ✗ pairwise failed: {exc}")

    # ── Adım 3: Terminal özet tablosu ────────────────────────────────────────
    print("\n" + "=" * 65)
    print("BASELINE SCORES — 6 provider × model")
    print("=" * 65)

    # NaN kontrolü: float("nan") == float("nan") → False olduğu için
    # "sa == sa" ifadesi NaN olmayan durumlarda True döner.
    header = f"{'Combo':<38} {'SentAcc':>7} {'ReasonQ':>7} {'Time(s)':>7}"
    print(header)
    print("-" * len(header))
    for r in summary_rows:
        if r["error"]:
            print(f"{r['combo']:<38} {'ERROR':>7} {'ERROR':>7} {'—':>7}")
        else:
            sa     = r["sentiment_accuracy"]
            rq     = r["reasoning_quality"]
            sa_str = f"{sa:.0%}" if sa == sa else "N/A"  # NaN kontrolü
            rq_str = f"{rq:.2f}" if rq == rq else "N/A"
            print(f"{r['combo']:<38} {sa_str:>7} {rq_str:>7} {r['elapsed_s']:>7.0f}")

    print("=" * 65)
    print("\nAll experiments visible in LangSmith under project 'finance-agent'.")
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
