#!/usr/bin/env python3
"""
Day 11 — Chunking Strategy Comparison

Compares 4 chunking strategies on 15 queries using Recall@5.

Strategies:
  1. Fixed-size (500 tokens, no overlap)
  2. Fixed-size with overlap (500 tok / 100 tok overlap)
  3. Recursive  (paragraph → sentence → word fallback, à la LangChain)
  4. Semantic   (paragraph-level embedding + similarity-drop splitting)

Ground truth: correct answer is in the expected section(s).
Recall@5 = 1 if any top-5 chunk belongs to an expected section.
"""

import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH    = Path("data/nvda_10k_2025.json")
CACHE_DIR    = Path("experiments/cache")
OUTPUT_PATH  = Path("chunking_comparison.md")
EMBED_MODEL  = "text-embedding-3-small"
EMBED_DIM    = 1536
TOP_K        = 5
BATCH_SIZE   = 100
CHUNK_SIZE   = 500   # tokens
OVERLAP      = 100   # tokens (strategy 2)
MAX_SEMANTIC = 600   # max tokens for a semantic chunk

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
enc    = tiktoken.get_encoding("cl100k_base")


# ── Ground Truth: 15 queries ──────────────────────────────────────────────────
QUERIES = [
    {
        "id": "q01",
        "query": "What was NVIDIA's total revenue in fiscal year 2025?",
        "expected_sections": ["Item 7 - MD&A"],
    },
    {
        "id": "q02",
        "query": "What are the main AI-related risks NVIDIA faces?",
        "expected_sections": ["Item 1A - Risk Factors"],
    },
    {
        "id": "q03",
        "query": "Who are NVIDIA's main competitors in the GPU market?",
        "expected_sections": ["Item 1 - Business", "Item 1A - Risk Factors"],
    },
    {
        "id": "q04",
        "query": "What is NVIDIA's data center segment revenue?",
        "expected_sections": ["Item 7 - MD&A"],
    },
    {
        "id": "q05",
        "query": "What are NVIDIA's main product lines and platforms?",
        "expected_sections": ["Item 1 - Business"],
    },
    {
        "id": "q06",
        "query": "What cybersecurity risks and incidents does NVIDIA disclose?",
        "expected_sections": ["Item 1C - Cybersecurity"],
    },
    {
        "id": "q07",
        "query": "What is NVIDIA's gross profit and operating income?",
        "expected_sections": ["Item 7 - MD&A"],
    },
    {
        "id": "q08",
        "query": "What are the export control and China-related regulatory risks?",
        "expected_sections": ["Item 1A - Risk Factors"],
    },
    {
        "id": "q09",
        "query": "Where are NVIDIA's main office and facility properties located?",
        "expected_sections": ["Item 2 - Properties"],
    },
    {
        "id": "q10",
        "query": "What is NVIDIA's gaming segment revenue and market performance?",
        "expected_sections": ["Item 7 - MD&A"],
    },
    {
        "id": "q11",
        "query": "What legal proceedings and litigation is NVIDIA involved in?",
        "expected_sections": ["Item 3 - Legal Proceedings"],
    },
    {
        "id": "q12",
        "query": "What is NVIDIA's research and development expense?",
        "expected_sections": ["Item 7 - MD&A"],
    },
    {
        "id": "q13",
        "query": "What are the supply chain and manufacturing concentration risks?",
        "expected_sections": ["Item 1A - Risk Factors"],
    },
    {
        "id": "q14",
        "query": "What is NVIDIA's cash position and liquidity situation?",
        "expected_sections": ["Item 7 - MD&A"],
    },
    {
        "id": "q15",
        "query": "What are the market risks related to interest rates and foreign exchange?",
        "expected_sections": [
            "Item 7A - Quantitative and Qualitative Disclosures About Market Risk"
        ],
    },
]


# ── Data Loading ──────────────────────────────────────────────────────────────
def load_10k(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    sections = []
    for key, value in data.items():
        if key in ("_meta", "full_text") or not isinstance(value, str) or not value.strip():
            continue
        sections.append({"section": key, "text": value.strip()})
    return sections


# ── Strategy 1: Fixed-size, no overlap ───────────────────────────────────────
def chunk_fixed(sections: list[dict]) -> list[dict]:
    chunks = []
    for sec in sections:
        tokens = enc.encode(sec["text"])
        for i, start in enumerate(range(0, len(tokens), CHUNK_SIZE)):
            end = min(start + CHUNK_SIZE, len(tokens))
            chunks.append({
                "chunk_id": f"{sec['section']}::fixed__{i}",
                "section":  sec["section"],
                "text":     enc.decode(tokens[start:end]),
            })
    return chunks


# ── Strategy 2: Fixed-size with overlap ──────────────────────────────────────
def chunk_fixed_overlap(sections: list[dict]) -> list[dict]:
    chunks = []
    for sec in sections:
        tokens = enc.encode(sec["text"])
        i, start = 0, 0
        while start < len(tokens):
            end = min(start + CHUNK_SIZE, len(tokens))
            chunks.append({
                "chunk_id": f"{sec['section']}::overlap__{i}",
                "section":  sec["section"],
                "text":     enc.decode(tokens[start:end]),
            })
            i += 1
            if end == len(tokens):
                break
            start += CHUNK_SIZE - OVERLAP
    return chunks


# ── Strategy 3: Recursive ─────────────────────────────────────────────────────
def chunk_recursive(sections: list[dict]) -> list[dict]:
    """
    Mimics LangChain RecursiveCharacterTextSplitter:
    Try to split by double-newline → newline → sentence → word.
    Merge small pieces until CHUNK_SIZE tokens, respecting natural boundaries.
    """
    separators = ["\n\n", "\n", ". ", " ", ""]

    def _split(text: str, seps: list[str]) -> list[str]:
        if not seps or len(enc.encode(text)) <= CHUNK_SIZE:
            return [text]
        sep = seps[0]
        if sep == "":
            # last resort: split into individual characters (won't happen in practice)
            return [text[i:i+1] for i in range(len(text))]
        raw_parts = text.split(sep)
        result = []
        for part in raw_parts:
            if not part.strip():
                continue
            rejoined = part if sep in (" ", "") else part + sep
            if len(enc.encode(rejoined)) > CHUNK_SIZE:
                result.extend(_split(rejoined, seps[1:]))
            else:
                result.append(rejoined)
        return result

    def _merge(splits: list[str]) -> list[str]:
        merged, current, current_len = [], "", 0
        for piece in splits:
            plen = len(enc.encode(piece))
            if current_len + plen > CHUNK_SIZE and current:
                merged.append(current.strip())
                current, current_len = piece, plen
            else:
                current += piece
                current_len += plen
        if current.strip():
            merged.append(current.strip())
        return merged

    chunks = []
    for sec in sections:
        splits = _split(sec["text"], separators)
        merged = _merge(splits)
        for i, text in enumerate(merged):
            if text.strip():
                chunks.append({
                    "chunk_id": f"{sec['section']}::recursive__{i}",
                    "section":  sec["section"],
                    "text":     text,
                })
    return chunks


# ── Strategy 4: Semantic ──────────────────────────────────────────────────────
def chunk_semantic(sections: list[dict]) -> list[dict]:
    """
    1. Split each section into paragraphs (double-newline boundaries).
    2. Embed all paragraphs via OpenAI (cached).
    3. Compute cosine similarity between adjacent paragraph embeddings.
    4. Split where similarity < 25th-percentile threshold.
    5. Merge resulting segments to stay within MAX_SEMANTIC tokens.
    """
    # Collect all paragraphs with their section index
    all_paras: list[str] = []
    sec_para_map: list[tuple[dict, list[int]]] = []  # (section, [para_indices])

    for sec in sections:
        paras = [p.strip() for p in re.split(r"\n\n+", sec["text"]) if p.strip()]
        if not paras:
            paras = [sec["text"].strip()]
        start_idx = len(all_paras)
        all_paras.extend(paras)
        indices = list(range(start_idx, start_idx + len(paras)))
        sec_para_map.append((sec, indices, paras))

    # Embed all paragraphs (cached)
    cache_path = CACHE_DIR / "embeddings_semantic_paras.npy"
    if cache_path.exists():
        print(f"  [semantic] Loading paragraph embeddings from cache...")
        para_vecs = np.load(cache_path)
    else:
        print(f"  [semantic] Embedding {len(all_paras)} paragraphs...")
        para_vecs = _batch_embed(all_paras)
        np.save(cache_path, para_vecs)
        print(f"  [semantic] Saved paragraph cache.")

    # Build semantic chunks per section
    chunks = []
    for sec, indices, paras in sec_para_map:
        n = len(paras)
        vecs = para_vecs[indices[0]: indices[0] + n]

        if n <= 1:
            text = paras[0] if paras else ""
            if text:
                chunks.append({
                    "chunk_id": f"{sec['section']}::semantic__0",
                    "section":  sec["section"],
                    "text":     text,
                })
            continue

        # Cosine similarities between adjacent paragraphs
        normed = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10)
        sims   = np.array([float(normed[i] @ normed[i + 1]) for i in range(n - 1)])

        threshold    = float(np.percentile(sims, 25))
        split_points = {i + 1 for i, s in enumerate(sims) if s < threshold}

        # Build segments by merging paragraphs between split points
        segments, seg_start = [], 0
        for sp in sorted(split_points):
            segments.append("\n\n".join(paras[seg_start:sp]))
            seg_start = sp
        segments.append("\n\n".join(paras[seg_start:]))

        # Enforce MAX_SEMANTIC token limit per chunk
        chunk_idx = 0
        for seg in segments:
            if not seg.strip():
                continue
            seg_tokens = enc.encode(seg)
            if len(seg_tokens) <= MAX_SEMANTIC:
                chunks.append({
                    "chunk_id": f"{sec['section']}::semantic__{chunk_idx}",
                    "section":  sec["section"],
                    "text":     seg,
                })
                chunk_idx += 1
            else:
                for j in range(0, len(seg_tokens), MAX_SEMANTIC):
                    sub = enc.decode(seg_tokens[j: j + MAX_SEMANTIC])
                    chunks.append({
                        "chunk_id": f"{sec['section']}::semantic__{chunk_idx}",
                        "section":  sec["section"],
                        "text":     sub,
                    })
                    chunk_idx += 1
    return chunks


# ── Embedding helpers ─────────────────────────────────────────────────────────
MAX_EMBED_TOKENS = 8000  # leave a small buffer below the 8192 model limit


def _truncate_for_embed(text: str) -> str:
    """Truncate text to MAX_EMBED_TOKENS tokens to satisfy embedding API limits."""
    tokens = enc.encode(text)
    if len(tokens) <= MAX_EMBED_TOKENS:
        return text
    return enc.decode(tokens[:MAX_EMBED_TOKENS])


def _batch_embed(texts: list[str]) -> np.ndarray:
    vecs = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = [_truncate_for_embed(t) for t in texts[i: i + BATCH_SIZE]]
        resp  = client.embeddings.create(model=EMBED_MODEL, input=batch)
        batch_vecs = [item.embedding for item in sorted(resp.data, key=lambda x: x.index)]
        vecs.extend(batch_vecs)
        time.sleep(0.05)
    return np.array(vecs, dtype=np.float32)


def embed_chunks(chunks: list[dict], cache_path: Path) -> np.ndarray:
    if cache_path.exists():
        print(f"  → Cache hit: {cache_path.name}  ({len(chunks)} chunks)")
        return np.load(cache_path)
    texts = [c["text"] for c in chunks]
    print(f"  → Embedding {len(texts)} chunks...")
    vecs = _batch_embed(texts)
    np.save(cache_path, vecs)
    print(f"  → Saved: {cache_path.name}")
    return vecs


# ── Retrieval ─────────────────────────────────────────────────────────────────
def retrieve_top_k(query: str, embeddings: np.ndarray, chunks: list[dict]) -> list[dict]:
    resp  = client.embeddings.create(model=EMBED_MODEL, input=[query])
    q_vec = np.array(resp.data[0].embedding, dtype=np.float32)
    norms  = np.linalg.norm(embeddings, axis=1)
    q_norm = np.linalg.norm(q_vec)
    scores = (embeddings @ q_vec) / (norms * q_norm + 1e-10)
    top_idx = np.argsort(scores)[::-1][:TOP_K]
    return [{**chunks[idx], "rank": i + 1, "score": float(scores[idx])}
            for i, idx in enumerate(top_idx)]


def recall_at_k(retrieved: list[dict], expected_sections: list[str]) -> bool:
    return any(r["section"] in expected_sections for r in retrieved)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("DAY 11 — Chunking Strategy Comparison  (Recall@5)")
    print("=" * 70)

    sections = load_10k(DATA_PATH)
    print(f"\nLoaded {len(sections)} sections from 10-K\n")

    # ── Build all chunk sets ──────────────────────────────────────────────────
    print("Building chunk sets...")
    strategy_chunks = {
        "fixed_no_overlap": chunk_fixed(sections),
        "fixed_overlap":    chunk_fixed_overlap(sections),
        "recursive":        chunk_recursive(sections),
        "semantic":         chunk_semantic(sections),
    }
    for name, chunks in strategy_chunks.items():
        token_counts = [len(enc.encode(c["text"])) for c in chunks]
        avg_tok = sum(token_counts) / len(token_counts) if token_counts else 0
        print(f"  {name:<22}: {len(chunks):>4} chunks  (avg {avg_tok:.0f} tok)")

    # ── Embed all chunk sets ──────────────────────────────────────────────────
    print("\nEmbedding chunk sets...")
    all_embeddings = {}
    for name, chunks in strategy_chunks.items():
        cache_path = CACHE_DIR / f"embeddings_{name}.npy"
        all_embeddings[name] = embed_chunks(chunks, cache_path)

    # ── Run 15 queries × 4 strategies ────────────────────────────────────────
    print(f"\nRunning {len(QUERIES)} queries × {len(strategy_chunks)} strategies...\n")

    # Cache query vectors to avoid 4× the API calls (one embed per query suffices)
    query_vec_cache: dict[str, np.ndarray] = {}

    results: dict[str, list[dict]] = {name: [] for name in strategy_chunks}

    for q in QUERIES:
        # Embed query once, reuse across strategies
        if q["query"] not in query_vec_cache:
            resp = client.embeddings.create(model=EMBED_MODEL, input=[q["query"]])
            query_vec_cache[q["query"]] = np.array(
                resp.data[0].embedding, dtype=np.float32
            )
        q_vec = query_vec_cache[q["query"]]

        row_hits = {}
        for name, chunks in strategy_chunks.items():
            emb    = all_embeddings[name]
            norms  = np.linalg.norm(emb, axis=1)
            q_norm = np.linalg.norm(q_vec)
            scores = (emb @ q_vec) / (norms * q_norm + 1e-10)
            top_idx = np.argsort(scores)[::-1][:TOP_K]
            retrieved = [chunks[i] for i in top_idx]
            hit = recall_at_k(retrieved, q["expected_sections"])
            results[name].append({"query_id": q["id"], "hit": hit, "retrieved": retrieved})
            row_hits[name] = "✅" if hit else "❌"

        icons = "  ".join(f"{v}" for v in row_hits.values())
        print(f"  {q['id']}  {icons}  {q['query'][:55]}")

    # ── Compute Recall@5 ─────────────────────────────────────────────────────
    recall: dict[str, float] = {
        name: sum(r["hit"] for r in res) / len(res)
        for name, res in results.items()
    }

    best = max(recall, key=recall.get)

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("RECALL@5 RESULTS")
    print("=" * 55)
    labels = {
        "fixed_no_overlap": "Fixed (500, no overlap)",
        "fixed_overlap":    "Fixed (500/100 overlap)",
        "recursive":        "Recursive",
        "semantic":         "Semantic",
    }
    for name, score in sorted(recall.items(), key=lambda x: -x[1]):
        bar  = "█" * round(score * 20)
        star = " ⭐" if name == best else ""
        print(f"  {labels[name]:<26} {score:.1%}  {bar}{star}")
    print()

    # ── Write chunking_comparison.md ─────────────────────────────────────────
    n_fixed   = len(strategy_chunks["fixed_no_overlap"])
    n_overlap = len(strategy_chunks["fixed_overlap"])
    n_rec     = len(strategy_chunks["recursive"])
    n_sem     = len(strategy_chunks["semantic"])

    per_query_rows = ""
    for i, q in enumerate(QUERIES):
        cells = " | ".join(
            ("✅" if results[name][i]["hit"] else "❌")
            for name in ["fixed_no_overlap", "fixed_overlap", "recursive", "semantic"]
        )
        per_query_rows += f"| {q['id']} | {q['query']} | {cells} |\n"

    overlap_delta = (n_overlap / n_fixed - 1) * 100

    md = f"""# Day 11 — Chunking Strategy Comparison

**Dataset:** NVIDIA 10-K 2025 (`nvda_10k_2025.json`)
**Metric:** Recall@5 — does the correct section appear in the top-5 retrieved chunks?
**Embedding model:** `{EMBED_MODEL}`
**Queries:** {len(QUERIES)} ground-truth Q&A pairs with section-level labels

---

## Summary: Recall@5

| Strategy | # Chunks | Recall@5 | Notes |
|---|---:|:---:|---|
| Fixed (500 tok, no overlap) | {n_fixed} | {recall['fixed_no_overlap']:.1%} | Baseline — fast, simple, misses boundaries |
| Fixed (500 tok, 100 overlap) | {n_overlap} | {recall['fixed_overlap']:.1%} | +{overlap_delta:.0f}% more chunks, better boundary coverage |
| Recursive | {n_rec} | {recall['recursive']:.1%} | Respects paragraph / sentence structure |
| Semantic | {n_sem} | {recall['semantic']:.1%} | Splits at embedding similarity drops |

**🏆 Best strategy:** `{best}` ({recall[best]:.1%} Recall@5)

---

## Per-Query Results

| Query ID | Question | Fixed | Fixed+Overlap | Recursive | Semantic |
|---|---|:---:|:---:|:---:|:---:|
{per_query_rows}
---

## Key Takeaways

1. **Fixed (no overlap)** is the fastest baseline but suffers at chunk boundaries — answers
   that straddle two chunks are missed entirely.

2. **Fixed + overlap** recovers boundary answers at the cost of ~{overlap_delta:.0f}% more chunks
   and some redundancy in the index.

3. **Recursive** respects the document's natural structure (paragraphs → sentences),
   which is a significant advantage for structured filings like 10-Ks where sections
   and paragraphs already encode topical boundaries.

4. **Semantic** splits where embedding similarity drops, capturing genuine topic transitions
   regardless of formatting — effective when a section discusses multiple distinct topics
   without explicit separators.

> 💡 **For 10-K / long structured documents:** Recursive or semantic chunking generally
> wins because the document's own structure (sections, paragraphs) already marks topic
> boundaries. Fixed-size chunking performs best when text is uniformly dense with
> little structure (e.g., raw log files, flat prose).

---

*Generated by `experiments/chunking_comparison.py` — Day 11 RAG Chunking Experiment*
"""

    OUTPUT_PATH.write_text(md)
    print(f"✅ Results written → {OUTPUT_PATH}")

    return recall


if __name__ == "__main__":
    main()
