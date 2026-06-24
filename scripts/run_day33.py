"""
Day 33 — Ragas-based RAG quality evaluation

Metrics (ragas 0.4.x):
  faithfulness      : answer claims are supported by retrieved context
  answer_relevancy  : answer addresses the question (cosine similarity of reverse-generated Qs)
  context_precision : retrieved chunks ranked by relevance (uses reference answer)
  context_recall    : reference answer statements attributable to retrieved context

Golden set : 20 NVIDIA 10-K questions with expected answers
RAG impls  : numpy (cosine scratch), qdrant (HNSW), langchain (Qdrant + Cohere rerank)
Output     : evals/rag_ragas_results.md
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv()


# ─── Golden Set — 20 NVIDIA 10-K questions ────────────────────────────────────

GOLDEN_SET: list[dict] = [
    {
        "id": 1,
        "question": "What are NVIDIA's main revenue streams and business segments?",
        "expected_answer": (
            "NVIDIA operates two reporting segments: Compute & Networking and Graphics. "
            "Revenue streams include Data Center (the largest, driven by H100/H200/Blackwell GPUs), "
            "Gaming (GeForce RTX), Professional Visualization, Automotive, and OEM & Other."
        ),
        "relevant_sections": ["Item 1", "Item 9"],
    },
    {
        "id": 2,
        "question": "What were NVIDIA's total revenues in fiscal year 2025?",
        "expected_answer": (
            "NVIDIA's total revenue in fiscal year 2025 was approximately $130.5 billion, "
            "representing over 100% year-over-year growth, primarily driven by Data Center "
            "demand for AI training and inference hardware."
        ),
        "relevant_sections": ["Item 9", "Item 7"],
    },
    {
        "id": 3,
        "question": "What AI-related risks does NVIDIA identify in its 10-K?",
        "expected_answer": (
            "NVIDIA identifies AI-related risks including: evolving AI regulations and liability, "
            "customer concentration in hyperscalers, geopolitical export restrictions on AI chips "
            "to China, competition from custom silicon (Google TPU, Amazon Trainium), "
            "and rapid technology obsolescence."
        ),
        "relevant_sections": ["Item 1A"],
    },
    {
        "id": 4,
        "question": "Who are NVIDIA's main competitors in the GPU and AI chip market?",
        "expected_answer": (
            "NVIDIA's main competitors include AMD (Instinct GPU series), Intel (Gaudi accelerators), "
            "and hyperscalers building custom AI chips: Google (TPU), Amazon (Trainium/Inferentia), "
            "Microsoft, and Meta. Qualcomm and MediaTek compete in mobile/edge AI."
        ),
        "relevant_sections": ["Item 1", "Item 1A"],
    },
    {
        "id": 5,
        "question": "What is NVIDIA's data center segment revenue and growth?",
        "expected_answer": (
            "NVIDIA's Data Center segment revenue was approximately $115 billion in fiscal year 2025, "
            "growing over 140% year-over-year. Growth was driven by demand for H100/H200 GPUs for "
            "AI model training and inference, plus networking products."
        ),
        "relevant_sections": ["Item 9", "Item 7"],
    },
    {
        "id": 6,
        "question": "What supply chain and manufacturing risks does NVIDIA face?",
        "expected_answer": (
            "NVIDIA depends on TSMC as its primary GPU manufacturer, creating concentration risk. "
            "Additional risks include long semiconductor lead times, geographic concentration "
            "of manufacturing in Taiwan, component shortages, and export restrictions."
        ),
        "relevant_sections": ["Item 1A"],
    },
    {
        "id": 7,
        "question": "What is NVIDIA's research and development expenditure?",
        "expected_answer": (
            "NVIDIA's R&D expenses were approximately $8 billion in fiscal year 2025. "
            "Investment areas include GPU architecture, CUDA software platform, AI inference, "
            "networking technologies, and automotive autonomous driving platforms."
        ),
        "relevant_sections": ["Item 9", "Item 7"],
    },
    {
        "id": 8,
        "question": "What export control and regulatory restrictions affect NVIDIA?",
        "expected_answer": (
            "US export regulations restrict NVIDIA from selling advanced AI chips (H100, A100 "
            "and derivatives) to China, Russia, and other restricted countries without licenses. "
            "These controls have significantly reduced China revenue. NVIDIA develops "
            "export-compliant alternative products."
        ),
        "relevant_sections": ["Item 1A", "Item 1"],
    },
    {
        "id": 9,
        "question": "How does NVIDIA describe its gaming segment performance?",
        "expected_answer": (
            "NVIDIA's Gaming segment revenue was approximately $11 billion in fiscal year 2025. "
            "The GeForce RTX 40 series drove growth. Gaming is the second-largest segment but is "
            "dwarfed by Data Center. RTX AI laptops and desktops expand the installed base."
        ),
        "relevant_sections": ["Item 9", "Item 1"],
    },
    {
        "id": 10,
        "question": "What are NVIDIA's cybersecurity risk management policies?",
        "expected_answer": (
            "NVIDIA has a cybersecurity risk management program overseen by the board's audit "
            "committee, including regular risk assessments, penetration testing, incident response "
            "procedures, employee security training, and third-party security reviews."
        ),
        "relevant_sections": ["Item 1C"],
    },
    {
        "id": 11,
        "question": "How does NVIDIA protect its intellectual property and patents?",
        "expected_answer": (
            "NVIDIA protects IP through patents, trade secrets, copyrights, and trademarks. "
            "The company holds thousands of patents covering GPU architectures, AI computing, "
            "and networking. The CUDA software stack is protected as a trade secret and competitive moat."
        ),
        "relevant_sections": ["Item 1", "Item 1A"],
    },
    {
        "id": 12,
        "question": "What is NVIDIA's capital return policy including dividends and buybacks?",
        "expected_answer": (
            "NVIDIA pays quarterly cash dividends and executes share buyback programs. "
            "The company authorizes billions in stock repurchases while maintaining modest "
            "dividends, prioritizing R&D investment and growth over large dividend distributions."
        ),
        "relevant_sections": ["Item 5", "Item 6"],
    },
    {
        "id": 13,
        "question": "What is NVIDIA's gross margin and profitability trend?",
        "expected_answer": (
            "NVIDIA's gross margin expanded significantly to approximately 74-75% in fiscal "
            "year 2025, up from around 66% in fiscal year 2024. The improvement reflects the "
            "high-margin Data Center GPU mix shift and operating leverage on large revenue growth."
        ),
        "relevant_sections": ["Item 9", "Item 7"],
    },
    {
        "id": 14,
        "question": "How many employees does NVIDIA have and how does it describe its culture?",
        "expected_answer": (
            "NVIDIA had approximately 36,000 employees as of fiscal year 2025. The company "
            "emphasizes an engineering-driven culture of innovation, intellectual curiosity, "
            "and speed, competing aggressively for AI and semiconductor design talent."
        ),
        "relevant_sections": ["Item 1"],
    },
    {
        "id": 15,
        "question": "What are NVIDIA's main product lines including H100 and Blackwell?",
        "expected_answer": (
            "NVIDIA's key products include the H100/H200 (Hopper architecture) data center GPUs, "
            "Blackwell architecture (B100, B200, GB200 NVL72 rack-scale systems), GeForce RTX 40/50 "
            "series for gaming, RTX Pro for professional visualization, Jetson for edge AI, "
            "and DRIVE for autonomous vehicles."
        ),
        "relevant_sections": ["Item 1", "Item 1A"],
    },
    {
        "id": 16,
        "question": "What is NVIDIA's automotive segment strategy and revenue?",
        "expected_answer": (
            "NVIDIA's Automotive segment generated approximately $1.7 billion in fiscal year 2025. "
            "The DRIVE platform provides autonomous vehicle computing. Key partnerships include "
            "Mercedes-Benz, BYD, and other automakers. NVIDIA sees automotive as a long-term "
            "growth opportunity in autonomous driving and in-vehicle AI."
        ),
        "relevant_sections": ["Item 1", "Item 9"],
    },
    {
        "id": 17,
        "question": "How does NVIDIA describe the competitive significance of its CUDA platform?",
        "expected_answer": (
            "CUDA is NVIDIA's parallel computing platform and key competitive moat. It has millions "
            "of developers and is deeply integrated into AI frameworks like PyTorch and TensorFlow. "
            "The large installed CUDA developer base creates switching costs and ecosystem lock-in."
        ),
        "relevant_sections": ["Item 1", "Item 1A"],
    },
    {
        "id": 18,
        "question": "What are NVIDIA's customer concentration risks?",
        "expected_answer": (
            "A small number of large customers — hyperscalers like Microsoft, Amazon, Google, "
            "and Meta — represent a significant portion of Data Center revenue. Loss or reduction "
            "of spending from any major customer could materially impact NVIDIA's financial results."
        ),
        "relevant_sections": ["Item 1A"],
    },
    {
        "id": 19,
        "question": "What is NVIDIA's NVLink and networking strategy?",
        "expected_answer": (
            "NVIDIA uses NVLink for high-bandwidth GPU-to-GPU connections within a server node. "
            "Its networking strategy includes InfiniBand and Ethernet (Spectrum-X) for connecting "
            "large GPU clusters. Mellanox acquisition (2020) underpins the networking portfolio, "
            "integral to NVIDIA's full-stack AI infrastructure offering."
        ),
        "relevant_sections": ["Item 1"],
    },
    {
        "id": 20,
        "question": "What geographic risks does NVIDIA face related to China and Taiwan?",
        "expected_answer": (
            "NVIDIA faces significant geographic risks: Taiwan houses TSMC (primary GPU foundry), "
            "creating geopolitical risk from potential China-Taiwan conflict. US export restrictions "
            "have substantially reduced China revenue. China previously represented over 20% of sales "
            "before export controls, now significantly lower due to regulatory restrictions."
        ),
        "relevant_sections": ["Item 1A"],
    },
]


# ─── RAG Wrappers ─────────────────────────────────────────────────────────────

class _NumPyRAG:
    name = "numpy"

    def __init__(self) -> None:
        self._embeddings = None
        self._chunks: list[dict] = []
        self._chunk_map: dict[str, dict] = {}

    def setup(self) -> None:
        from rag.rag_numpy import DATA_PATH, chunk_sections, embed_chunks, load_10k

        print("  [numpy] Loading & chunking 10-K...")
        sections = load_10k(DATA_PATH)
        self._chunks = chunk_sections(sections)
        print(f"  [numpy] {len(self._chunks)} chunks — embedding (~30 s)...")
        self._embeddings = embed_chunks(self._chunks)
        self._chunk_map = {c["chunk_id"]: c for c in self._chunks}
        print(f"  [numpy] Ready — shape {self._embeddings.shape}")

    def query(self, q: str) -> dict:
        from rag.rag_numpy import rag_answer

        result = rag_answer(q, self._embeddings, self._chunks)
        contexts = [
            self._chunk_map[cid]["text"]
            for cid in result["chunks_used"]
            if cid in self._chunk_map
        ]
        return {"answer": result["answer"], "contexts": contexts}


class _QdrantRAG:
    name = "qdrant"

    def setup(self) -> None:
        from rag.rag_qdrant import DATA_PATH, build_qdrant_index, chunk_sections, load_10k

        print("  [qdrant] Loading & chunking 10-K...")
        sections = load_10k(DATA_PATH)
        chunks = chunk_sections(sections)
        print(f"  [qdrant] {len(chunks)} chunks — building index (skips if cached)...")
        build_qdrant_index(chunks)
        print("  [qdrant] Ready.")

    def query(self, q: str) -> dict:
        from rag.rag_qdrant import qdrant_search, rag_answer

        top_chunks = qdrant_search(q, top_k=5)
        answer = rag_answer(q, top_chunks)
        return {"answer": answer, "contexts": [c["text"] for c in top_chunks]}


class _LangChainRAG:
    name = "langchain"

    def __init__(self) -> None:
        self._vectorstore = None
        self._cohere_key = ""

    def setup(self) -> None:
        from rag.rag_langchain import get_vectorstore

        cohere_key = os.getenv("COHERE_API_KEY", "")
        if not cohere_key:
            raise EnvironmentError("COHERE_API_KEY not set — LangChain RAG requires Cohere.")
        self._cohere_key = cohere_key
        self._vectorstore = get_vectorstore()
        print("  [langchain] Ready.")

    def query(self, q: str) -> dict:
        from langchain_classic.retrievers.contextual_compression import (
            ContextualCompressionRetriever,
        )
        from langchain_cohere import CohereRerank
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI

        base_retriever = self._vectorstore.as_retriever(search_kwargs={"k": 10})
        compressor = CohereRerank(
            model="rerank-v3.5", top_n=5, cohere_api_key=self._cohere_key
        )
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=base_retriever
        )
        docs = retriever.invoke(q)
        contexts = [d.page_content for d in docs]
        context_str = "\n\n---\n\n".join(
            f"[Chunk {i} | section={d.metadata.get('section', '?')}]\n{d.page_content}"
            for i, d in enumerate(docs, 1)
        )
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a financial analyst. "
                "Answer using ONLY the provided context from an NVIDIA 10-K filing.",
            ),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ])
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        answer = (prompt | llm | StrOutputParser()).invoke(
            {"context": context_str, "question": q}
        )
        return {"answer": answer, "contexts": contexts}


# ─── Ragas Evaluation ─────────────────────────────────────────────────────────

def evaluate_with_ragas(records: list[dict]) -> dict[str, list[float | None]]:
    """
    records: list of {question, answer, contexts, reference}
    Returns metric_name → per-sample score list (None on parse failure)
    """
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["reference"],
        )
        for r in records
    ]
    dataset = EvaluationDataset(samples=samples)

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
        embeddings=OpenAIEmbeddings(model="text-embedding-3-small"),
        raise_exceptions=False,
        show_progress=True,
    )

    df = result.to_pandas()
    return {
        "faithfulness": df["faithfulness"].tolist(),
        "answer_relevancy": df["answer_relevancy"].tolist(),
        "context_precision": df["context_precision"].tolist(),
        "context_recall": df["context_recall"].tolist(),
    }


# ─── Markdown Report ──────────────────────────────────────────────────────────

_METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
_METRIC_LABELS = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Ans Relevancy",
    "context_precision": "Ctx Precision",
    "context_recall": "Ctx Recall",
}


def _fmt(v: float | None) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "–"
    return f"{v:.3f}"


def _avg(vals: list[float | None]) -> float:
    ok = [v for v in vals if v is not None and isinstance(v, float) and v == v]
    return sum(ok) / len(ok) if ok else float("nan")


def save_markdown(all_results: dict, path: Path) -> None:
    impl_names = list(all_results.keys())

    lines = [
        "# RAG Evaluation — Day 33 (Ragas Metrics)",
        "",
        "**Framework:** ragas 0.4.x  ",
        "**Document:** NVIDIA 10-K 2025  ",
        "**Golden set:** 20 questions with ground-truth expected answers  ",
        f"**Implementations:** {', '.join(impl_names)}  ",
        "",
        "## Metric Definitions",
        "",
        "| Metric | What it measures | Requires |",
        "|--------|-----------------|----------|",
        "| **Faithfulness** | Fraction of answer claims supported by retrieved context | answer + contexts |",
        "| **Answer Relevancy** | How well the answer addresses the question (reverse-QG cosine) | answer + question |",
        "| **Context Precision** | Proportion of retrieved chunks relevant to the reference answer | contexts + reference |",
        "| **Context Recall** | Fraction of reference answer statements attributable to context | contexts + reference |",
        "",
        "## Summary — Average Scores",
        "",
        "| Implementation | Faithfulness ↑ | Ans Relevancy ↑ | Ctx Precision ↑ | Ctx Recall ↑ |",
        "|----------------|:--------------:|:---------------:|:---------------:|:------------:|",
    ]

    for name in impl_names:
        if "error" in all_results[name] and not all_results[name].get("scores"):
            lines.append(f"| {name} | ERROR | ERROR | ERROR | ERROR |")
            continue
        s = all_results[name].get("scores", {})
        lines.append(
            f"| {name} "
            f"| {_fmt(_avg(s.get('faithfulness', [])))} "
            f"| {_fmt(_avg(s.get('answer_relevancy', [])))} "
            f"| {_fmt(_avg(s.get('context_precision', [])))} "
            f"| {_fmt(_avg(s.get('context_recall', [])))} |"
        )

    lines += ["", "## Per-Question Results", ""]

    for name in impl_names:
        data = all_results[name]
        records = data.get("records", [])
        scores = data.get("scores", {})
        if not records:
            continue

        lines += [
            f"### {name.upper()}",
            "",
            "| # | Question (truncated) | Faith | AnsRel | CtxPrec | CtxRec | Latency |",
            "|:-:|----------------------|:-----:|:------:|:-------:|:------:|:-------:|",
        ]

        faith = scores.get("faithfulness", [None] * len(records))
        rel = scores.get("answer_relevancy", [None] * len(records))
        prec = scores.get("context_precision", [None] * len(records))
        rec = scores.get("context_recall", [None] * len(records))

        for i, r in enumerate(records):
            q = r["question"]
            q_short = (q[:50] + "…") if len(q) > 50 else q
            lat = r.get("latency_s", 0.0)
            lines.append(
                f"| {r['id']:2d} | {q_short} "
                f"| {_fmt(faith[i] if i < len(faith) else None)} "
                f"| {_fmt(rel[i] if i < len(rel) else None)} "
                f"| {_fmt(prec[i] if i < len(prec) else None)} "
                f"| {_fmt(rec[i] if i < len(rec) else None)} "
                f"| {lat:.1f}s |"
            )
        lines.append("")

    lines += [
        "## Metric Notes",
        "",
        "- **Faithfulness = 0** for short, direct answers is common — ragas NLI parser needs multiple "
        "  verifiable claims. Use as a relative comparison across impls, not absolute.",
        "- **Answer Relevancy** uses embedding cosine similarity of reverse-generated questions — "
        "  scores above 0.85 indicate strong topical alignment.",
        "- **Context Precision/Recall** both require the `reference` (ground truth) field — "
        "  they measure retrieval quality relative to expected answer coverage.",
        "",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved → {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("DAY 33 — Ragas-based RAG quality evaluation")
    print("  20 questions × up to 3 implementations × 4 metrics")
    print("=" * 70)

    # ── 1. Setup implementations ──────────────────────────────────────────────
    impls: list = []
    for cls in (_NumPyRAG, _QdrantRAG, _LangChainRAG):
        impl = cls()
        try:
            print(f"\n→ Setting up {impl.name}...")
            impl.setup()
            impls.append(impl)
        except Exception as exc:
            print(f"  ⚠️  {impl.name} setup failed — skipping: {exc}")

    if not impls:
        print("ERROR: No RAG implementation could be initialised.")
        sys.exit(1)

    print(f"\n→ Active: {[i.name for i in impls]}")

    # ── 2. Run RAG queries (cached to JSON) ───────────────────────────────────
    cache_path = Path("evals/rag_ragas_cache.json")
    cache_path.parent.mkdir(exist_ok=True)
    raw_cache: dict = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    print("\n[Phase 1/2] Running RAG queries...")
    for impl in impls:
        if impl.name in raw_cache:
            print(f"  [{impl.name}] loaded {len(raw_cache[impl.name])} cached answers")
            continue

        raw_cache[impl.name] = []
        for item in GOLDEN_SET:
            qid, q = item["id"], item["question"]
            print(f"  [{impl.name}] Q{qid:02d} {q[:50]}…", end=" ", flush=True)
            t0 = time.time()
            try:
                out = impl.query(q)
                elapsed = time.time() - t0
                raw_cache[impl.name].append({
                    "id": qid,
                    "question": q,
                    "answer": out["answer"],
                    "contexts": out["contexts"],
                    "reference": item["expected_answer"],
                    "relevant_sections": item["relevant_sections"],
                    "latency_s": round(elapsed, 2),
                })
                print(f"✓ {elapsed:.1f}s")
            except Exception as exc:
                elapsed = time.time() - t0
                print(f"✗ {str(exc)[:60]}")
                raw_cache[impl.name].append({
                    "id": qid,
                    "question": q,
                    "answer": "",
                    "contexts": [],
                    "reference": item["expected_answer"],
                    "relevant_sections": item["relevant_sections"],
                    "latency_s": round(elapsed, 2),
                    "error": str(exc),
                })
            time.sleep(0.2)

        cache_path.write_text(json.dumps(raw_cache, indent=2, ensure_ascii=False))
        print(f"  [{impl.name}] → cached to {cache_path}")

    # ── 3. Ragas evaluation ───────────────────────────────────────────────────
    print("\n[Phase 2/2] Running ragas metrics (LLM judge calls)...")
    all_results: dict = {}

    for impl_name, records in raw_cache.items():
        valid = [r for r in records if r.get("answer") and r.get("contexts")]
        skipped = len(records) - len(valid)
        print(f"\n  [{impl_name}] {len(valid)} valid records"
              + (f" ({skipped} skipped)" if skipped else ""))

        if not valid:
            all_results[impl_name] = {"records": [], "scores": {}, "error": "no valid records"}
            continue

        try:
            scores = evaluate_with_ragas(valid)
            all_results[impl_name] = {"records": valid, "scores": scores}
            for metric, vals in scores.items():
                ok = [v for v in vals if v is not None]
                avg = sum(ok) / len(ok) if ok else float("nan")
                print(f"    {metric:20s}: {avg:.3f}")
        except Exception as exc:
            print(f"  ✗ ragas failed for {impl_name}: {exc}")
            all_results[impl_name] = {"records": valid, "scores": {}, "error": str(exc)}

    # ── 4. Save outputs ───────────────────────────────────────────────────────
    md_path = Path("evals/rag_ragas_results.md")
    json_path = Path("evals/rag_ragas_results.json")

    save_markdown(all_results, md_path)
    json_path.write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False, default=str)
    )
    print(f"  Saved JSON → {json_path}")

    print("\n" + "=" * 70)
    print("DAY 33 COMPLETE")
    print(f"  Report  : {md_path}")
    print(f"  Raw JSON: {json_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
