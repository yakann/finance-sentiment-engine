"""
RAG from scratch with pure NumPy — Day 9

Demonstrates what a Vector DB does under the hood:
  1. Chunk text with tiktoken (token-accurate splitting)
  2. Embed chunks via OpenAI text-embedding-3-small
  3. Store embeddings as numpy.ndarray (N, 1536)
  4. At query time: cosine similarity = (A @ B) / (||A|| * ||B||)  → top-5
  5. Feed top-5 chunks as context to an LLM → answer
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────
DATA_PATH = Path("data/nvda_10k_2025.json")
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
CHUNK_SIZE = 500        # tokens per chunk
CHUNK_OVERLAP = 50      # token overlap between consecutive chunks
BATCH_SIZE = 100        # embeddings per API call
TOP_K = 5

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ── 1. Load & Flatten 10-K Text ─────────────────────────────────────────────
def load_10k(path: Path) -> list[dict]:
    """Return list of {section, text} dicts from the 10-K JSON."""
    data = json.loads(path.read_text())
    sections = []
    for key, value in data.items():
        if key in ("_meta", "full_text") or not isinstance(value, str) or not value.strip():
            continue
        sections.append({"section": key, "text": value.strip()})
    return sections


# ── 2. Token-Accurate Chunking ───────────────────────────────────────────────
def chunk_sections(sections: list[dict], chunk_size: int = CHUNK_SIZE,
                   overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Split each section into chunks of `chunk_size` tokens with `overlap` token
    overlap between adjacent chunks.  Uses tiktoken so we never exceed model
    token limits regardless of character variance.
    """
    enc = tiktoken.get_encoding("cl100k_base")
    chunks = []

    for sec in sections:
        tokens = enc.encode(sec["text"])
        start = 0
        chunk_idx = 0

        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = enc.decode(chunk_tokens)

            chunks.append({
                "chunk_id": f"{sec['section']}::chunk_{chunk_idx}",
                "section": sec["section"],
                "text": chunk_text,
                "token_count": len(chunk_tokens),
            })

            chunk_idx += 1
            if end == len(tokens):
                break
            start += chunk_size - overlap   # slide with overlap

    return chunks


# ── 3. Batch Embedding ───────────────────────────────────────────────────────
def embed_chunks(chunks: list[dict], batch_size: int = BATCH_SIZE) -> np.ndarray:
    """
    Embed all chunks in batches.  Returns ndarray of shape (N, EMBED_DIM).
    The numpy array IS the "vector database" — each row is one chunk.
    """
    all_embeddings = []
    total = len(chunks)

    for i in range(0, total, batch_size):
        batch = chunks[i: i + batch_size]
        texts = [c["text"] for c in batch]

        response = client.embeddings.create(model=EMBED_MODEL, input=texts)
        batch_vecs = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
        all_embeddings.extend(batch_vecs)

        print(f"  Embedded {min(i + batch_size, total)}/{total} chunks...", end="\r")
        time.sleep(0.1)   # light rate-limit courtesy

    print()
    return np.array(all_embeddings, dtype=np.float32)   # (N, 1536)


# ── 4. Cosine Similarity Search ──────────────────────────────────────────────
def cosine_search(query: str, embeddings: np.ndarray,
                  chunks: list[dict], top_k: int = TOP_K) -> list[dict]:
    """
    Embed query → compute cosine similarity against all stored vectors.

    cosine(A, B) = (A · B) / (||A|| * ||B||)

    With L2-normalised vectors this reduces to a simple dot product (A @ B.T),
    but we compute it explicitly here to show the full formula.
    """
    # Embed query
    response = client.embeddings.create(model=EMBED_MODEL, input=[query])
    q_vec = np.array(response.data[0].embedding, dtype=np.float32)  # (1536,)

    # Dot products:  embeddings @ q_vec  →  shape (N,)
    dot_products = embeddings @ q_vec

    # L2 norms
    emb_norms = np.linalg.norm(embeddings, axis=1)   # (N,)
    q_norm = np.linalg.norm(q_vec)                   # scalar

    # Cosine similarities
    cosine_scores = dot_products / (emb_norms * q_norm + 1e-10)

    # Top-K indices (descending)
    top_indices = np.argsort(cosine_scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            **chunks[idx],
            "score": float(cosine_scores[idx]),
        })
    return results


# ── 5. RAG Answer Generation ─────────────────────────────────────────────────
def rag_answer(query: str, embeddings: np.ndarray, chunks: list[dict]) -> dict:
    """Retrieve top-5 chunks, build context, ask LLM for an answer."""
    top_chunks = cosine_search(query, embeddings, chunks)

    context_parts = []
    for i, c in enumerate(top_chunks, 1):
        context_parts.append(
            f"[Chunk {i} | {c['chunk_id']} | score={c['score']:.4f}]\n{c['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    system_prompt = (
        "You are a financial analyst assistant. "
        "Answer the user's question using ONLY the provided context excerpts from an NVIDIA 10-K filing. "
        "Be specific and cite which chunks support your answer. "
        "If the context doesn't contain enough information, say so."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {query}"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=600,
    )

    return {
        "query": query,
        "answer": response.choices[0].message.content,
        "chunks_used": [c["chunk_id"] for c in top_chunks],
        "scores": [c["score"] for c in top_chunks],
    }


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("RAG FROM SCRATCH — Pure NumPy Vector Store")
    print("=" * 70)

    # Build index
    print("\n[1/3] Loading & chunking 10-K...")
    sections = load_10k(DATA_PATH)
    chunks = chunk_sections(sections)
    print(f"  → {len(sections)} sections → {len(chunks)} chunks "
          f"(~{sum(c['token_count'] for c in chunks):,} tokens total)")

    print(f"\n[2/3] Embedding {len(chunks)} chunks with {EMBED_MODEL}...")
    embeddings = embed_chunks(chunks)
    print(f"  → Embedding matrix shape: {embeddings.shape}  "
          f"({embeddings.nbytes / 1024:.1f} KB in memory)")

    # Queries
    queries = [
        "What are the AI-related risks?",
        "Revenue from data center segment?",
        "Main competitors?",
    ]

    print("\n[3/3] Running RAG queries...")
    print("=" * 70)

    for query in queries:
        print(f"\n🔍 QUERY: {query}")
        print("-" * 70)

        result = rag_answer(query, embeddings, chunks)

        print(f"💬 ANSWER:\n{result['answer']}")
        print(f"\n📎 Chunks used ({TOP_K} total):")
        for chunk_id, score in zip(result["chunks_used"], result["scores"]):
            print(f"   • {chunk_id}  (cosine={score:.4f})")
        print("=" * 70)

    # Explain what a Vector DB does
    print("\n📚 WHAT DOES A VECTOR DB DO? (demonstrated above)")
    print("-" * 70)
    print(f"  1. TEXT → TOKENS  : tiktoken splits text into tokens (BPE encoding)")
    print(f"  2. TOKENS → VECTOR: OpenAI maps each chunk to a {EMBED_DIM}-dim float vector")
    print(f"  3. STORE VECTORS  : numpy ndarray {embeddings.shape} — this IS the vector DB")
    print(f"  4. QUERY SEARCH   : embed query → cosine similarity → sort → top-{TOP_K}")
    print(f"  5. GENERATION     : top-{TOP_K} chunks as context → LLM → grounded answer")
    print()
    print("  A production Vector DB (Pinecone, Weaviate, FAISS) does the same thing")
    print("  but adds: persistent storage, HNSW/IVF indexing for sub-linear search,")
    print("  metadata filters, and horizontal scaling. The math is identical.")


if __name__ == "__main__":
    main()
