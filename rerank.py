"""
RAG Reranking — Day 12

Two-stage retrieval: broad vector search (top-10) → rerank → top-5.

Why reranking works
  • Vector search uses bi-encoders: query and passage embedded separately → fast but coarse
  • Rerankers are cross-encoders: query + passage concatenated as one input → fine-grained
  • Typical gain: +5–15 Recall@5 points over vector-only retrieval

Implementations
  1. Cohere Rerank API           — cloud, single HTTP call, SOTA accuracy
  2. BAAI/bge-reranker-large     — open-source CrossEncoder, runs on local CPU

Recall@5 evaluation methodology
  • Oracle / gold set   = Cohere reranked top-5 (industry-grade cross-encoder as reference)
  • Recall@5(method)    = |method_top5 ∩ cohere_top5| / 5
  • Shows how much plain vector search already captures, and how close BGE gets to Cohere

Prerequisites
  • OPENAI_API_KEY  — for embeddings
  • COHERE_API_KEY  — https://dashboard.cohere.com/api-keys (free tier: 1000 req/month)
  • Qdrant running  — docker run -p 6333:6333 qdrant/qdrant
  • BGE model auto-downloads from HuggingFace (~1.1 GB first run)
"""

from __future__ import annotations

import os
import time
import json
from pathlib import Path
from typing import Callable

import numpy as np
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

load_dotenv()

# ── Config ───────────────────────────────────────────────────────────────────
DATA_PATH      = Path("data/nvda_10k_2025.json")
EMBED_MODEL    = "text-embedding-3-small"
EMBED_DIM      = 1536
CHUNK_SIZE     = 500
CHUNK_OVERLAP  = 50
BATCH_SIZE     = 100
TOP_K_RETRIEVE = 10   # retrieve wider pool for reranking
TOP_K_FINAL    = 5    # final top-K after reranking
COLLECTION     = "nvda_10k"

# Cohere rerank model — use the latest v3 multilingual model
COHERE_RERANK_MODEL = "rerank-v3.5"

# BGE cross-encoder model (downloaded from HuggingFace on first run)
BGE_MODEL = "BAAI/bge-reranker-large"

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
qdrant        = QdrantClient(host="localhost", port=6333)

EVAL_QUERIES = [
    "What are the main AI-related risks NVIDIA faces?",
    "What was the revenue from the data center segment?",
    "Who are NVIDIA's main competitors?",
    "What are NVIDIA's supply chain risks?",
    "How does NVIDIA plan to address regulatory compliance?",
]


# ── 1. Data loading & chunking (same as Day 9/10) ────────────────────────────
def load_10k(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    sections = []
    for key, value in data.items():
        if key in ("_meta", "full_text") or not isinstance(value, str) or not value.strip():
            continue
        sections.append({"section": key, "text": value.strip()})
    return sections


def chunk_sections(sections: list[dict]) -> list[dict]:
    enc    = tiktoken.get_encoding("cl100k_base")
    chunks = []
    for sec in sections:
        tokens    = enc.encode(sec["text"])
        start     = 0
        chunk_idx = 0
        while start < len(tokens):
            end         = min(start + CHUNK_SIZE, len(tokens))
            chunk_text  = enc.decode(tokens[start:end])
            chunks.append({
                "chunk_id":    f"{sec['section']}::chunk_{chunk_idx}",
                "section":     sec["section"],
                "chunk_idx":   chunk_idx,
                "text":        chunk_text,
                "token_count": end - start,
            })
            chunk_idx += 1
            if end == len(tokens):
                break
            start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


# ── 2. Embedding helpers ─────────────────────────────────────────────────────
def embed_texts(texts: list[str]) -> list[list[float]]:
    all_vecs = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch    = texts[i : i + BATCH_SIZE]
        response = openai_client.embeddings.create(model=EMBED_MODEL, input=batch)
        vecs     = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
        all_vecs.extend(vecs)
        print(f"  Embedded {min(i + BATCH_SIZE, len(texts))}/{len(texts)} chunks...", end="\r")
        time.sleep(0.1)
    print()
    return all_vecs


def embed_query(query: str) -> list[float]:
    resp = openai_client.embeddings.create(model=EMBED_MODEL, input=[query])
    return resp.data[0].embedding


# ── 3. Qdrant index (reuse Day-10 collection) ────────────────────────────────
def ensure_qdrant_index(chunks: list[dict]) -> None:
    existing = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION in existing:
        print(f"  Collection '{COLLECTION}' already exists — skipping upsert.")
        return

    qdrant.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
    )
    texts  = [c["text"] for c in chunks]
    vecs   = embed_texts(texts)
    points = [
        PointStruct(
            id=idx,
            vector=vecs[idx],
            payload={k: chunks[idx][k] for k in ("section", "chunk_idx", "text", "chunk_id")},
        )
        for idx in range(len(chunks))
    ]
    for i in range(0, len(points), 256):
        qdrant.upsert(collection_name=COLLECTION, points=points[i : i + 256])
    print(f"  ✅ {len(points)} points stored in Qdrant")


# ── 4. Base retrieval: top-10 via vector search ──────────────────────────────
def vector_search(query: str, top_k: int = TOP_K_RETRIEVE) -> list[dict]:
    q_vec = embed_query(query)
    hits  = qdrant.search(
        collection_name=COLLECTION,
        query_vector=q_vec,
        limit=top_k,
        with_payload=True,
    )
    return [
        {
            "chunk_id": h.payload["chunk_id"],
            "section":  h.payload["section"],
            "text":     h.payload["text"],
            "score":    h.score,
        }
        for h in hits
    ]


# ── 5a. Cohere Reranker ───────────────────────────────────────────────────────
def rerank_cohere(query: str, candidates: list[dict], top_n: int = TOP_K_FINAL) -> list[dict]:
    """
    Sends (query, documents) to Cohere Rerank API.
    Returns top-N candidates sorted by relevance_score descending.

    Cohere cross-encoder sees: [CLS] query [SEP] passage [SEP]
    and outputs a relevance score in [0, 1].
    """
    import cohere  # lazy import — only needed if COHERE_API_KEY is set

    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "COHERE_API_KEY not set.\n"
            "  1. Sign up at https://dashboard.cohere.com/api-keys (free tier: 1000 req/month)\n"
            "  2. Add  COHERE_API_KEY=<key>  to your .env file"
        )

    co      = cohere.ClientV2(api_key=api_key)
    docs    = [c["text"] for c in candidates]
    response = co.rerank(
        model=COHERE_RERANK_MODEL,
        query=query,
        documents=docs,
        top_n=top_n,
    )

    reranked = []
    for result in response.results:
        candidate = candidates[result.index].copy()
        candidate["rerank_score"] = result.relevance_score
        candidate["original_rank"] = result.index + 1  # 1-based
        reranked.append(candidate)

    return reranked   # already sorted by relevance_score desc


# ── 5b. BGE Reranker (local CrossEncoder) ────────────────────────────────────
_bge_model = None   # module-level cache so model loads only once


def _get_bge_model():
    global _bge_model
    if _bge_model is None:
        from sentence_transformers import CrossEncoder
        print(f"  Loading {BGE_MODEL} (first run downloads ~1.1 GB)...")
        _bge_model = CrossEncoder(BGE_MODEL, max_length=512)
        print("  ✅ BGE model loaded")
    return _bge_model


def rerank_bge(query: str, candidates: list[dict], top_n: int = TOP_K_FINAL) -> list[dict]:
    """
    Local CPU reranking with BAAI/bge-reranker-large.
    CrossEncoder encodes (query, passage) pairs together → relevance logit.
    Score range: raw logit (no fixed bound), higher = more relevant.
    """
    model  = _get_bge_model()
    pairs  = [(query, c["text"]) for c in candidates]
    scores = model.predict(pairs)   # numpy array, shape (N,)

    indexed = sorted(
        enumerate(scores),
        key=lambda x: x[1],
        reverse=True,
    )[:top_n]

    reranked = []
    for new_rank, (orig_idx, score) in enumerate(indexed, 1):
        candidate = candidates[orig_idx].copy()
        candidate["rerank_score"]  = float(score)
        candidate["original_rank"] = orig_idx + 1  # 1-based
        reranked.append(candidate)

    return reranked


# ── 6. Recall@5 metric ───────────────────────────────────────────────────────
def recall_at_k(retrieved: list[dict], gold: list[dict], k: int = TOP_K_FINAL) -> float:
    """
    Recall@K = |retrieved_top_k ∩ gold_top_k| / K

    We use Cohere top-5 as the gold/oracle set because it's the state-of-the-art
    cross-encoder.  This lets us measure:
      • How much plain vector search already captures (no-rerank baseline)
      • How closely BGE aligns with the Cohere oracle
    """
    gold_ids     = {c["chunk_id"] for c in gold[:k]}
    retrieved_ids = {c["chunk_id"] for c in retrieved[:k]}
    return len(retrieved_ids & gold_ids) / k


# ── 7. Pretty print helpers ──────────────────────────────────────────────────
def print_results_table(
    query: str,
    no_rerank:    list[dict],
    cohere_ranks: list[dict],
    bge_ranks:    list[dict],
) -> None:
    print(f"\n{'=' * 78}")
    print(f"🔍 QUERY: {query}")
    print(f"{'=' * 78}")

    headers = ["Rank", "chunk_id", "score", "rerank_score", "orig_rank"]

    def _row(i: int, c: dict, show_orig: bool = False) -> str:
        chunk_id   = c["chunk_id"][:42]
        score      = f"{c.get('score', 0.0):.4f}"
        rerank_scr = f"{c.get('rerank_score', c.get('score', 0.0)):.4f}"
        orig       = str(c.get("original_rank", i + 1)) if show_orig else "-"
        return f"  {i:<5} {chunk_id:<43} {score:>8}  {rerank_scr:>12}  {orig:>9}"

    hdr = f"  {'Rank':<5} {'chunk_id':<43} {'score':>8}  {'rerank_score':>12}  {'orig_rank':>9}"
    sep = "  " + "-" * 75

    print("\n📌 NO RERANK  (top-5 of top-10 by cosine)")
    print(hdr)
    print(sep)
    for i, c in enumerate(no_rerank, 1):
        print(_row(i, c))

    print("\n🌐 COHERE RERANK  (Cohere rerank-v3.5)")
    print(hdr)
    print(sep)
    for i, c in enumerate(cohere_ranks, 1):
        print(_row(i, c, show_orig=True))

    print("\n🤗 BGE RERANK  (BAAI/bge-reranker-large, local CPU)")
    print(hdr)
    print(sep)
    for i, c in enumerate(bge_ranks, 1):
        print(_row(i, c, show_orig=True))


def print_recall_summary(recall_table: list[dict]) -> None:
    print(f"\n{'=' * 78}")
    print("📊 RECALL@5 SUMMARY  (oracle = Cohere top-5)")
    print(f"{'=' * 78}")
    print(f"  {'Query':<55} {'No-rerank':>10}  {'BGE':>7}")
    print("  " + "-" * 75)
    total_no_rerank = 0.0
    total_bge       = 0.0
    for row in recall_table:
        q  = row["query"][:54]
        nr = row["no_rerank"]
        bg = row["bge"]
        total_no_rerank += nr
        total_bge       += bg
        print(f"  {q:<55} {nr:>10.2f}  {bg:>7.2f}")
    n = len(recall_table)
    print("  " + "-" * 75)
    print(f"  {'MEAN':<55} {total_no_rerank / n:>10.2f}  {total_bge / n:>7.2f}")
    print(f"\n  Cohere Recall@5 = 1.00 by definition (it IS the oracle)")
    print(
        f"\n  Interpretation:\n"
        f"    • No-rerank mean = {total_no_rerank / n:.2f} → "
        f"{total_no_rerank / n * 100:.0f}% of Cohere top-5 already in vector top-5\n"
        f"    • BGE mean       = {total_bge / n:.2f} → "
        f"{total_bge / n * 100:.0f}% agreement with Cohere oracle (open-source parity)"
    )


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 78)
    print("RAG RERANKING — Day 12  (top-10 → rerank → top-5)")
    print("  Two methods: Cohere API  vs  BAAI/bge-reranker-large (local)")
    print("=" * 78)

    # --- Build index ---
    print("\n[1/4] Loading & chunking 10-K...")
    sections = load_10k(DATA_PATH)
    chunks   = chunk_sections(sections)
    print(f"  → {len(sections)} sections → {len(chunks)} chunks")

    print(f"\n[2/4] Ensuring Qdrant index (collection='{COLLECTION}')...")
    ensure_qdrant_index(chunks)

    # Warm up BGE model before the query loop
    print(f"\n[3/4] Loading BGE model...")
    _get_bge_model()

    # --- Evaluate ---
    print(f"\n[4/4] Running {len(EVAL_QUERIES)} queries...")
    recall_table: list[dict] = []

    for query in EVAL_QUERIES:
        # Stage 1: broad vector retrieval (top-10)
        candidates = vector_search(query, top_k=TOP_K_RETRIEVE)

        # Stage 2a: Cohere rerank
        try:
            cohere_top5 = rerank_cohere(query, candidates)
        except EnvironmentError as e:
            print(f"\n⚠️  Cohere skipped: {e}")
            cohere_top5 = None

        # Stage 2b: BGE rerank
        bge_top5 = rerank_bge(query, candidates)

        # No-rerank baseline: just the first 5 of the cosine-sorted top-10
        no_rerank_top5 = candidates[:TOP_K_FINAL]

        # Recall@5 (only computable when Cohere ran)
        if cohere_top5 is not None:
            r_no_rerank = recall_at_k(no_rerank_top5, cohere_top5)
            r_bge       = recall_at_k(bge_top5, cohere_top5)
            recall_table.append({
                "query":     query,
                "no_rerank": r_no_rerank,
                "bge":       r_bge,
            })
            print_results_table(query, no_rerank_top5, cohere_top5, bge_top5)
        else:
            # Fallback: show BGE vs no-rerank without Cohere
            print(f"\n{'=' * 78}")
            print(f"🔍 QUERY: {query}")
            print(f"\n  ⚠️  Cohere unavailable — showing BGE vs vector baseline only")
            print(f"\n  No-rerank top-5 chunk IDs:")
            for i, c in enumerate(no_rerank_top5, 1):
                print(f"    [{i}] {c['chunk_id']}  score={c['score']:.4f}")
            print(f"\n  BGE top-5 chunk IDs (rerank_score = cross-encoder logit):")
            for i, c in enumerate(bge_top5, 1):
                print(
                    f"    [{i}] {c['chunk_id']}  rerank={c['rerank_score']:.4f}"
                    f"  (was #{c['original_rank']})"
                )

    # --- Summary ---
    if recall_table:
        print_recall_summary(recall_table)
    else:
        print(
            "\n📊 Recall@5 skipped (Cohere API key not set).\n"
            "  → Set COHERE_API_KEY in .env to enable full evaluation."
        )

    print("\n✅ Day 12 complete!")
    print("   Key takeaways:")
    print("   • Two-stage retrieval: broad vector search → cross-encoder rerank")
    print("   • Cohere (cloud): zero setup, state-of-the-art, 1000 free req/month")
    print("   • BGE (local): free, private, ~90-95% parity with Cohere on English text")
    print("   • Recall@5 gap shows how much precision is gained by reranking")


if __name__ == "__main__":
    main()
