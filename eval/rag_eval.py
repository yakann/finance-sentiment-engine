"""
RAG Evaluation Harness — Day 14

Metrics
-------
- Recall@5        : fraction of top-5 retrieved chunks from the expected 10-K section(s)
- Faithfulness    : LLM-as-judge — are answer claims supported by retrieved chunks? (0–1)
- Answer Relevance: LLM-as-judge — does the answer address the question? (0–1)

Coverage: 3 implementations × 15 queries = 45 evaluations
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

# Ensure CWD is the project root so relative imports in rag_*.py resolve correctly.
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_judge_client = OpenAI()


# ─── 15 Evaluation Queries ────────────────────────────────────────────────────

QUERIES: list[dict] = [
    {
        "id": 1,
        "query": "What are NVIDIA's main revenue streams and business segments?",
        "expected_sections": ["Item 1", "Item 9"],
    },
    {
        "id": 2,
        "query": "What were NVIDIA's total revenues in fiscal year 2025?",
        "expected_sections": ["Item 9", "Item 7"],
    },
    {
        "id": 3,
        "query": "What AI-related risks does NVIDIA identify?",
        "expected_sections": ["Item 1A"],
    },
    {
        "id": 4,
        "query": "Who are NVIDIA's main competitors in the GPU and AI chip market?",
        "expected_sections": ["Item 1", "Item 1A"],
    },
    {
        "id": 5,
        "query": "What is NVIDIA's data center segment revenue and growth?",
        "expected_sections": ["Item 9", "Item 7"],
    },
    {
        "id": 6,
        "query": "What supply chain and manufacturing risks does NVIDIA face?",
        "expected_sections": ["Item 1A"],
    },
    {
        "id": 7,
        "query": "What is NVIDIA's research and development expenditure?",
        "expected_sections": ["Item 9", "Item 7"],
    },
    {
        "id": 8,
        "query": "What export control and regulatory restrictions affect NVIDIA?",
        "expected_sections": ["Item 1A", "Item 1"],
    },
    {
        "id": 9,
        "query": "How does NVIDIA describe its gaming segment performance?",
        "expected_sections": ["Item 9", "Item 1"],
    },
    {
        "id": 10,
        "query": "What are NVIDIA's cybersecurity policies and risk management practices?",
        "expected_sections": ["Item 1C"],
    },
    {
        "id": 11,
        "query": "How does NVIDIA protect its intellectual property and patents?",
        "expected_sections": ["Item 1", "Item 1A"],
    },
    {
        "id": 12,
        "query": "What is NVIDIA's dividend and capital return policy?",
        "expected_sections": ["Item 6", "Item 5"],
    },
    {
        "id": 13,
        "query": "What is NVIDIA's gross margin and profitability trend?",
        "expected_sections": ["Item 9", "Item 7"],
    },
    {
        "id": 14,
        "query": "How many employees does NVIDIA have and what is its culture?",
        "expected_sections": ["Item 1"],
    },
    {
        "id": 15,
        "query": "What are NVIDIA's main product lines including H100 and Blackwell?",
        "expected_sections": ["Item 1", "Item 1A"],
    },
]


# ─── LLM-as-Judge ─────────────────────────────────────────────────────────────

_FAITHFULNESS_PROMPT = """\
You are evaluating whether an AI answer is faithful to provided source chunks.

Question: {question}

Source chunks:
{chunks}

Answer: {answer}

Rate faithfulness 0.0–1.0:
- 1.0: Every claim is directly supported by source chunks
- 0.5: Most claims supported; minor inference beyond sources
- 0.0: Claims not found in or contradicted by source chunks

Respond ONLY with valid JSON: {{"score": <float 0-1>, "reason": "<one sentence>"}}"""

_RELEVANCE_PROMPT = """\
You are evaluating whether an AI answer addresses the question asked.

Question: {question}

Answer: {answer}

Rate answer relevance 0.0–1.0:
- 1.0: Directly and completely addresses the question
- 0.5: Partially addresses it or includes off-topic content
- 0.0: Does not address the question at all

Respond ONLY with valid JSON: {{"score": <float 0-1>, "reason": "<one sentence>"}}"""


def _llm_judge(prompt: str) -> tuple[float, str]:
    """Call gpt-4o-mini as judge; return (score, reason)."""
    try:
        resp = _judge_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=150,
        )
        content = resp.choices[0].message.content.strip()
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return float(data["score"]), str(data.get("reason", ""))
        return 0.0, "parse_error"
    except Exception as exc:
        return 0.0, f"error: {str(exc)[:80]}"


def recall_at_k(retrieved: list[dict], expected_sections: list[str], k: int = 5) -> float:
    """Fraction of top-k retrieved chunks whose section matches any expected section."""
    top_k = retrieved[:k]
    hits = sum(
        1 for c in top_k
        if any(exp in c.get("section", "") for exp in expected_sections)
    )
    return hits / k if k > 0 else 0.0


# ─── RAG Implementation Wrappers ──────────────────────────────────────────────

class NumPyRAG:
    """Day 9 — brute-force cosine similarity over an in-memory NumPy matrix."""

    name = "numpy"

    def __init__(self) -> None:
        self._embeddings: np.ndarray | None = None
        self._chunks: list[dict] = []
        self._chunk_map: dict[str, dict] = {}

    def setup(self) -> None:
        from rag.rag_numpy import load_10k, chunk_sections, embed_chunks, DATA_PATH

        print("  [numpy] Loading & chunking 10-K...")
        sections = load_10k(DATA_PATH)
        self._chunks = chunk_sections(sections)
        print(f"  [numpy] {len(self._chunks)} chunks. Embedding (this takes ~30 s)...")
        self._embeddings = embed_chunks(self._chunks)
        self._chunk_map = {c["chunk_id"]: c for c in self._chunks}
        print(f"  [numpy] Ready — matrix {self._embeddings.shape}")

    def query(self, q: str) -> dict:
        from rag.rag_numpy import rag_answer

        result = rag_answer(q, self._embeddings, self._chunks)
        # rag_numpy.rag_answer returns chunk_ids; look up full dicts for section info.
        retrieved = [
            {"section": self._chunk_map[cid]["section"], "text": self._chunk_map[cid]["text"]}
            for cid in result["chunks_used"]
            if cid in self._chunk_map
        ]
        return {"answer": result["answer"], "chunks": retrieved}


class QdrantRAG:
    """Day 10 — HNSW vector index via Qdrant with metadata filtering."""

    name = "qdrant"

    def setup(self) -> None:
        from rag.rag_qdrant import load_10k, chunk_sections, build_qdrant_index, DATA_PATH

        print("  [qdrant] Loading & chunking 10-K...")
        sections = load_10k(DATA_PATH)
        chunks = chunk_sections(sections)
        print(f"  [qdrant] {len(chunks)} chunks. Building index (skips if exists)...")
        build_qdrant_index(chunks)
        print("  [qdrant] Ready.")

    def query(self, q: str) -> dict:
        from rag.rag_qdrant import qdrant_search, rag_answer

        top_chunks = qdrant_search(q, top_k=5)
        answer = rag_answer(q, top_chunks)
        return {
            "answer": answer,
            "chunks": [{"section": c["section"], "text": c["text"]} for c in top_chunks],
        }


class LangChainRAG:
    """Day 13 — LangChain LCEL pipeline with Qdrant retrieval + Cohere reranking."""

    name = "langchain"

    def __init__(self) -> None:
        self._vectorstore = None
        self._openai_key: str = ""
        self._cohere_key: str = ""

    def setup(self) -> None:
        from rag.rag_langchain import get_vectorstore

        print("  [langchain] Initialising vectorstore (skips embedding if collection exists)...")
        self._vectorstore = get_vectorstore()
        self._openai_key = os.getenv("OPENAI_API_KEY", "")
        self._cohere_key = os.getenv("COHERE_API_KEY", "")
        if not self._cohere_key:
            raise EnvironmentError("COHERE_API_KEY not set — needed for LangChain reranking.")
        print("  [langchain] Ready.")

    def query(self, q: str) -> dict:
        from langchain_classic.retrievers.contextual_compression import (
            ContextualCompressionRetriever,
        )
        from langchain_cohere import CohereRerank
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI

        # Two-stage retrieval: Qdrant (top-10) → Cohere cross-encoder (top-5).
        # We invoke the retriever explicitly so we can capture the chunks for Recall@5,
        # then generate the answer from those same chunks (no double Cohere call).
        base_retriever = self._vectorstore.as_retriever(search_kwargs={"k": 10})
        compressor = CohereRerank(
            model="rerank-v3.5",
            top_n=5,
            cohere_api_key=self._cohere_key,
        )
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever,
        )
        docs = retriever.invoke(q)
        chunks = [
            {"section": d.metadata.get("section", ""), "text": d.page_content}
            for d in docs
        ]

        # Build answer from the reranked chunks directly (mirrors the LCEL chain logic).
        context = "\n\n---\n\n".join(
            f"[Chunk {i} | section={c['section']}]\n{c['text']}"
            for i, c in enumerate(chunks, 1)
        )
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a financial analyst assistant. "
                "Answer using ONLY the provided context excerpts from an NVIDIA 10-K filing. "
                "Be specific and cite which chunks support your answer.",
            ),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ])
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=self._openai_key)
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": q})

        return {"answer": answer, "chunks": chunks}


# ─── Evaluation Loop ──────────────────────────────────────────────────────────

def run_evaluation(impls: list) -> list[dict]:
    results: list[dict] = []
    impl_names = [i.name for i in impls]

    for q_item in QUERIES:
        qid = q_item["id"]
        query = q_item["query"]
        expected = q_item["expected_sections"]

        print(f"\n{'─' * 62}")
        print(f"Q{qid:02d}: {query}")

        for impl in impls:
            print(f"  [{impl.name}] ...", end=" ", flush=True)
            t0 = time.time()
            try:
                out = impl.query(query)
                latency = round(time.time() - t0, 2)
                answer: str = out["answer"]
                chunks: list[dict] = out["chunks"]

                r5 = recall_at_k(chunks, expected, k=5)

                chunks_txt = "\n\n".join(
                    f"[{i + 1}] ({c['section']})\n{c['text'][:400]}"
                    for i, c in enumerate(chunks[:5])
                )
                faith_score, faith_reason = _llm_judge(
                    _FAITHFULNESS_PROMPT.format(
                        question=query,
                        chunks=chunks_txt,
                        answer=answer[:1000],
                    )
                )
                rel_score, rel_reason = _llm_judge(
                    _RELEVANCE_PROMPT.format(question=query, answer=answer[:1000])
                )

                results.append({
                    "query_id":         qid,
                    "query":            query,
                    "impl":             impl.name,
                    "recall_at_5":      round(r5, 3),
                    "faithfulness":     round(faith_score, 3),
                    "answer_relevance": round(rel_score, 3),
                    "latency_s":        latency,
                    "answer_preview":   answer[:300].replace("\n", " "),
                    "faith_reason":     faith_reason,
                    "rel_reason":       rel_reason,
                })
                print(
                    f"R@5={r5:.2f} | Faith={faith_score:.2f} | "
                    f"Rel={rel_score:.2f} | {latency:.1f}s"
                )

            except Exception as exc:
                latency = round(time.time() - t0, 2)
                print(f"ERROR: {exc}")
                results.append({
                    "query_id":         qid,
                    "query":            query,
                    "impl":             impl.name,
                    "recall_at_5":      0.0,
                    "faithfulness":     0.0,
                    "answer_relevance": 0.0,
                    "latency_s":        latency,
                    "answer_preview":   f"ERROR: {str(exc)[:200]}",
                    "faith_reason":     "",
                    "rel_reason":       "",
                })

    return results


# ─── Markdown Report ──────────────────────────────────────────────────────────

_IMPL_ORDER = ["numpy", "qdrant", "langchain"]


def save_results_md(results: list[dict], path: Path) -> None:
    by_impl: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_impl[r["impl"]].append(r)

    # Compute per-implementation averages
    averages: dict[str, dict[str, float]] = {}
    for name in _IMPL_ORDER:
        rows = by_impl.get(name, [])
        if not rows:
            continue
        averages[name] = {
            "recall_at_5":      sum(r["recall_at_5"]      for r in rows) / len(rows),
            "faithfulness":     sum(r["faithfulness"]      for r in rows) / len(rows),
            "answer_relevance": sum(r["answer_relevance"]  for r in rows) / len(rows),
            "latency_s":        sum(r["latency_s"]         for r in rows) / len(rows),
        }

    best: dict[str, float] = {
        m: max(averages[n][m] for n in averages)
        for m in ("recall_at_5", "faithfulness", "answer_relevance")
    }

    def _bold_if_best(v: float, metric: str) -> str:
        s = f"{v:.3f}"
        return f"**{s}**" if abs(v - best[metric]) < 1e-9 else s

    lines: list[str] = [
        "# RAG Evaluation Results — Day 14",
        "",
        "**Setup:** 3 RAG implementations × 15 queries = 45 total evaluations  ",
        "**Document:** NVIDIA 10-K 2025 (EDGAR) — 19 SEC sections, ~1 000 tiktoken chunks",
        "",
        "## Metric Definitions",
        "",
        "| Metric | Description | Range |",
        "|--------|-------------|-------|",
        "| **Recall@5** | Fraction of top-5 retrieved chunks that come from the expected 10-K section(s) | 0–1 |",
        "| **Faithfulness** | LLM judge (`gpt-4o-mini`): are answer claims supported by retrieved chunks? | 0–1 |",
        "| **Answer Relevance** | LLM judge (`gpt-4o-mini`): does the answer address the question? | 0–1 |",
        "",
        "## Summary",
        "",
        "| Implementation | Recall@5 ↑ | Faithfulness ↑ | Answer Relevance ↑ | Avg Latency |",
        "|----------------|:----------:|:--------------:|:------------------:|:-----------:|",
    ]

    for name in _IMPL_ORDER:
        if name not in averages:
            continue
        a = averages[name]
        lines.append(
            f"| {name} "
            f"| {_bold_if_best(a['recall_at_5'], 'recall_at_5')} "
            f"| {_bold_if_best(a['faithfulness'], 'faithfulness')} "
            f"| {_bold_if_best(a['answer_relevance'], 'answer_relevance')} "
            f"| {a['latency_s']:.1f}s |"
        )

    lines += [
        "",
        "## Per-Query Results",
        "",
        "| Q# | Query (truncated to 45 chars) | Impl | R@5 | Faith | Rel | Latency |",
        "|:--:|-------------------------------|------|:---:|:-----:|:---:|:-------:|",
    ]

    def _impl_rank(r: dict) -> int:
        try:
            return _IMPL_ORDER.index(r["impl"])
        except ValueError:
            return 99

    for r in sorted(results, key=lambda x: (x["query_id"], _impl_rank(x))):
        q_short = (r["query"][:45] + "…") if len(r["query"]) > 45 else r["query"]
        lines.append(
            f"| {r['query_id']:2d} | {q_short} | {r['impl']} "
            f"| {r['recall_at_5']:.2f} | {r['faithfulness']:.2f} "
            f"| {r['answer_relevance']:.2f} | {r['latency_s']:.1f}s |"
        )

    lines += ["", "## Detailed Answers", ""]

    for qid in range(1, 16):
        q_rows = [r for r in results if r["query_id"] == qid]
        if not q_rows:
            continue
        lines += [f"### Q{qid}: {q_rows[0]['query']}", ""]
        for r in sorted(q_rows, key=_impl_rank):
            lines += [
                f"**{r['impl'].upper()}**  ",
                f"Recall@5: `{r['recall_at_5']:.2f}` | "
                f"Faithfulness: `{r['faithfulness']:.2f}` | "
                f"Relevance: `{r['answer_relevance']:.2f}` | "
                f"Latency: `{r['latency_s']:.1f}s`  ",
                f"> {r['answer_preview']}",
                "",
            ]
            if r["faith_reason"]:
                lines += [f"*Faith reason:* {r['faith_reason']}  ", ""]
            if r["rel_reason"]:
                lines += [f"*Relevance reason:* {r['rel_reason']}", ""]
            lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n✅ Results saved → {path}")


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 62)
    print("RAG EVALUATION HARNESS — Day 14")
    print("3 implementations × 15 queries = 45 evaluations")
    print("=" * 62)

    impls: list = [NumPyRAG(), QdrantRAG(), LangChainRAG()]

    print("\n[Phase 1/2] Setting up implementations...")
    for impl in impls:
        print(f"\n→ {impl.name}")
        impl.setup()

    print("\n\n[Phase 2/2] Running evaluation (45 rows + 90 LLM judge calls)...")
    results = run_evaluation(impls)

    out_dir = Path(__file__).parent
    save_results_md(results, out_dir / "rag_results.md")

    json_path = out_dir / "rag_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✅ JSON saved → {json_path}")

    # Print summary
    print("\n" + "=" * 62)
    print("SUMMARY")
    print("=" * 62)
    by_impl: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_impl[r["impl"]].append(r)
    for name in _IMPL_ORDER:
        rows = by_impl.get(name, [])
        if not rows:
            continue
        r5    = sum(r["recall_at_5"]      for r in rows) / len(rows)
        faith = sum(r["faithfulness"]      for r in rows) / len(rows)
        rel   = sum(r["answer_relevance"]  for r in rows) / len(rows)
        lat   = sum(r["latency_s"]         for r in rows) / len(rows)
        print(f"  {name:10s}  R@5={r5:.3f}  Faith={faith:.3f}  Rel={rel:.3f}  Lat={lat:.1f}s")


if __name__ == "__main__":
    main()
