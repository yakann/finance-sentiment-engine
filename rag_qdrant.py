"""
RAG with Qdrant Vector DB — Day 10

What you gain over pure NumPy (Day 9):
  ✅ Persistence  — vectors survive process restarts; no re-embedding
  ✅ Filtering    — metadata filters (section, date, ticker …) at query time
  ✅ Scale        — HNSW index, sub-linear search, horizontal sharding

Steps:
  1. Connect to local Qdrant (docker run -p 6333:6333 qdrant/qdrant)
  2. Create collection  (vector_size=1536, distance=Cosine)
  3. Upsert chunks      payload: {section, chunk_idx, text}
  4. Search             query_vector + optional section filter
  5. Compare results    with Day-9 NumPy scores on the same 3 queries
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

load_dotenv()

# ── Config ──────────────────────────────────────────────────────────────────
DATA_PATH      = Path("data/nvda_10k_2025.json")
EMBED_MODEL    = "text-embedding-3-small"
EMBED_DIM      = 1536
CHUNK_SIZE     = 500
CHUNK_OVERLAP  = 50
BATCH_SIZE     = 100
TOP_K          = 5
COLLECTION     = "nvda_10k"

openai_client  = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
qdrant         = QdrantClient(host="localhost", port=6333)


# ── 1. Load & Flatten 10-K ──────────────────────────────────────────────────
def load_10k(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    sections = []
    for key, value in data.items():
        if key in ("_meta", "full_text") or not isinstance(value, str) or not value.strip():
            continue
        sections.append({"section": key, "text": value.strip()})
    return sections


# ── 2. Token-Accurate Chunking ───────────────────────────────────────────────
def chunk_sections(sections: list[dict]) -> list[dict]:
    enc    = tiktoken.get_encoding("cl100k_base")
    chunks = []

    for sec in sections:
        tokens    = enc.encode(sec["text"])
        start     = 0
        chunk_idx = 0

        while start < len(tokens):
            end         = min(start + CHUNK_SIZE, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text  = enc.decode(chunk_tokens)

            chunks.append({
                "chunk_id":    f"{sec['section']}::chunk_{chunk_idx}",
                "section":     sec["section"],
                "chunk_idx":   chunk_idx,
                "text":        chunk_text,
                "token_count": len(chunk_tokens),
            })

            chunk_idx += 1
            if end == len(tokens):
                break
            start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


# ── 3. Batch Embedding ───────────────────────────────────────────────────────
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


# ── 4. Qdrant: Create Collection & Upsert ────────────────────────────────────
def build_qdrant_index(chunks: list[dict]) -> None:
    """Create (or recreate) the collection and upsert all chunk vectors."""
    existing = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION in existing:
        print(f"  Collection '{COLLECTION}' already exists — skipping upsert.")
        print("  (Delete with: qdrant.delete_collection(COLLECTION) to force re-index)")
        return

    qdrant.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
    )
    print(f"  Created collection '{COLLECTION}'")

    texts = [c["text"] for c in chunks]
    vecs  = embed_texts(texts)

    points = [
        PointStruct(
            id=idx,
            vector=vecs[idx],
            payload={
                "section":   chunks[idx]["section"],
                "chunk_idx": chunks[idx]["chunk_idx"],
                "text":      chunks[idx]["text"],
                "chunk_id":  chunks[idx]["chunk_id"],
            },
        )
        for idx in range(len(chunks))
    ]

    # Upsert in batches of 256
    for i in range(0, len(points), 256):
        qdrant.upsert(collection_name=COLLECTION, points=points[i : i + 256])
        print(f"  Upserted {min(i + 256, len(points))}/{len(points)} points...", end="\r")
    print(f"\n  ✅ {len(points)} points stored in Qdrant")


# ── 5. Qdrant Search ─────────────────────────────────────────────────────────
def qdrant_search(
    query: str,
    top_k: int = TOP_K,
    section_filter: str | None = None,
) -> list[dict]:
    """
    Search Qdrant.  Optional `section_filter` restricts results to one section.

    Filter example:
        must=[FieldCondition(key="section", match=MatchValue(value="Item 1A"))]
    """
    q_vec = embed_query(query)

    search_filter = None
    if section_filter:
        search_filter = Filter(
            must=[
                FieldCondition(
                    key="section",
                    match=MatchValue(value=section_filter),
                )
            ]
        )

    hits = qdrant.search(
        collection_name=COLLECTION,
        query_vector=q_vec,
        limit=top_k,
        query_filter=search_filter,
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


# ── 6. NumPy Cosine Search (Day 9 baseline) ───────────────────────────────────
def cosine_search_numpy(
    query: str,
    embeddings: np.ndarray,
    chunks: list[dict],
    top_k: int = TOP_K,
) -> list[dict]:
    q_vec        = np.array(embed_query(query), dtype=np.float32)
    dot_products = embeddings @ q_vec
    emb_norms    = np.linalg.norm(embeddings, axis=1)
    q_norm       = np.linalg.norm(q_vec)
    cosine_scores = dot_products / (emb_norms * q_norm + 1e-10)
    top_indices  = np.argsort(cosine_scores)[::-1][:top_k]
    return [{"chunk_id": chunks[i]["chunk_id"], "score": float(cosine_scores[i])} for i in top_indices]


# ── 7. RAG Answer ─────────────────────────────────────────────────────────────
def rag_answer(query: str, top_chunks: list[dict]) -> str:
    context_parts = [
        f"[Chunk {i} | {c['chunk_id']} | score={c['score']:.4f}]\n{c['text']}"
        for i, c in enumerate(top_chunks, 1)
    ]
    context = "\n\n---\n\n".join(context_parts)

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a financial analyst assistant. "
                    "Answer using ONLY the provided context excerpts from an NVIDIA 10-K filing. "
                    "Be specific and cite which chunks support your answer."
                ),
            },
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
        temperature=0.2,
        max_tokens=600,
    )
    return response.choices[0].message.content


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("RAG WITH QDRANT — Day 10  (vs. NumPy Day 9 baseline)")
    print("=" * 70)

    # --- Index build ---
    print("\n[1/4] Loading & chunking 10-K...")
    sections = load_10k(DATA_PATH)
    chunks   = chunk_sections(sections)
    print(f"  → {len(sections)} sections → {len(chunks)} chunks")

    print(f"\n[2/4] Building Qdrant index (collection='{COLLECTION}')...")
    build_qdrant_index(chunks)

    # --- Queries ---
    queries = [
        "What are the AI-related risks?",
        "Revenue from data center segment?",
        "Main competitors?",
    ]

    # Build NumPy baseline (re-embed only if collection was fresh above)
    print("\n[3/4] Building NumPy baseline for comparison...")
    texts      = [c["text"] for c in chunks]
    numpy_vecs = np.array(embed_texts(texts), dtype=np.float32)
    print(f"  → NumPy matrix: {numpy_vecs.shape}")

    # --- Run & compare ---
    print("\n[4/4] Running queries — Qdrant vs NumPy")
    print("=" * 70)

    for query in queries:
        print(f"\n🔍 QUERY: {query}")
        print("-" * 70)

        qdrant_results = qdrant_search(query)
        numpy_results  = cosine_search_numpy(query, numpy_vecs, chunks)

        answer = rag_answer(query, qdrant_results)
        print(f"💬 ANSWER (Qdrant context):\n{answer}")

        print(f"\n📊 TOP-{TOP_K} CHUNK COMPARISON")
        print(f"  {'Rank':<5} {'Qdrant chunk_id':<45} {'Qdrant':>8}  {'NumPy':>8}")
        print(f"  {'-'*5} {'-'*45} {'-'*8}  {'-'*8}")
        for rank, (q_r, n_r) in enumerate(zip(qdrant_results, numpy_results), 1):
            match = "✅" if q_r["chunk_id"] == n_r["chunk_id"] else "⚠️ "
            print(f"  {rank:<5} {q_r['chunk_id'][:44]:<45} {q_r['score']:>8.4f}  {n_r['score']:>8.4f} {match}")

        print("=" * 70)

    # --- Filter demo ---
    print("\n🔎 FILTER DEMO — restrict search to 'Item 1A' (Risk Factors) section")
    print("-" * 70)
    risk_query   = "What are the AI-related risks?"
    risk_results = qdrant_search(risk_query, section_filter="Item 1A")

    if risk_results:
        for i, r in enumerate(risk_results, 1):
            print(f"  [{i}] score={r['score']:.4f}  section={r['section']}  {r['chunk_id']}")
    else:
        # Fallback: list available sections and try first one
        info = qdrant.get_collection(COLLECTION)
        sample = qdrant.scroll(COLLECTION, limit=5, with_payload=True)[0]
        available = sorted({p.payload["section"] for p in sample})
        print(f"  ⚠️  No results for 'Item 1A'. Available sections (sample): {available}")
        if available:
            fallback_section = available[0]
            print(f"\n  Retrying filter with section='{fallback_section}'...")
            risk_results = qdrant_search(risk_query, section_filter=fallback_section)
            for i, r in enumerate(risk_results, 1):
                print(f"  [{i}] score={r['score']:.4f}  section={r['section']}  {r['chunk_id']}")

    print("\n✅ Day 10 complete!")
    print("   Qdrant gains vs NumPy:")
    print("   • Persistence : vectors survive process restarts (no re-embedding)")
    print("   • Filtering   : section/ticker/date filters at query time, zero extra cost")
    print("   • Scale       : HNSW index → sub-linear O(log N) search vs O(N) brute-force")


if __name__ == "__main__":
    main()
