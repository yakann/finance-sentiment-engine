"""
RAG with LangChain — Day 13

LangChain port of the Day-12 two-stage pipeline (rerank.py, 425 lines → ~150 lines).

Abstractions used
  RecursiveCharacterTextSplitter  → replaces manual tiktoken chunking
  OpenAIEmbeddings + QdrantVectorStore.from_documents → replaces embed + upsert loop
  ContextualCompressionRetriever + CohereRerank → replaces manual rerank_cohere()
  LCEL chain (retriever | prompt | llm | StrOutputParser) → replaces rag_answer()

Trade-offs vs the raw Day-12 implementation
  ✅ ~65 % fewer lines
  ✅ Composable, swappable components via LCEL |
  ✅ Easy to swap embedding/retriever/reranker/LLM
  ❌ Character-based chunking — not token-accurate (RCTS doesn't count tokens)
  ❌ No BGE local reranker (needs a custom BaseDocumentCompressor)
  ❌ No Recall@5 evaluation harness
  ❌ Less visibility into per-chunk scores & batching control

Prerequisites
  OPENAI_API_KEY  — embeddings + chat
  COHERE_API_KEY  — reranking  (free tier: 1000 req/month)
  Qdrant running  — docker run -p 6333:6333 qdrant/qdrant
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_cohere import CohereRerank
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH   = Path("data/nvda_10k_2025.json")
COLLECTION  = "nvda_10k_lc"         # separate collection from Day-10/12 runs
CHUNK_SIZE  = 2000                   # characters ≈ 500 tokens (4 chars/token avg)
CHUNK_OVERLAP = 200                  # ≈ 50-token overlap
TOP_K_RETRIEVE = 10
TOP_K_FINAL    = 5

QUERIES = [
    "What are the AI-related risks?",
    "Revenue from data center segment?",
    "Main competitors?",
]

# ── 1. Load 10-K sections as LangChain Documents ──────────────────────────────
def load_documents(path: Path) -> list[Document]:
    data = json.loads(path.read_text())
    return [
        Document(page_content=v.strip(), metadata={"section": k})
        for k, v in data.items()
        if k not in ("_meta", "full_text") and isinstance(v, str) and v.strip()
    ]


# ── 2. Build or reuse Qdrant vectorstore ─────────────────────────────────────
def get_vectorstore() -> QdrantVectorStore:
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    client = QdrantClient(url="http://localhost:6333")
    existing = [c.name for c in client.get_collections().collections]

    if COLLECTION in existing:
        print(f"  Collection '{COLLECTION}' already exists — reusing.")
        return QdrantVectorStore(
            client=client,
            collection_name=COLLECTION,
            embedding=embeddings,
        )

    print("  Building vectorstore (embedding all chunks)...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
    docs   = load_documents(DATA_PATH)
    chunks = splitter.split_documents(docs)
    print(f"  → {len(docs)} sections → {len(chunks)} chunks")

    vectorstore = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        url="http://localhost:6333",
        collection_name=COLLECTION,
    )
    print(f"  ✅ {len(chunks)} chunks stored in '{COLLECTION}'")
    return vectorstore


# ── 3. Build LCEL chain ───────────────────────────────────────────────────────
def build_chain(vectorstore: QdrantVectorStore) -> RunnablePassthrough:
    """Build a two-stage RAG chain using LCEL.

    Pipeline (left to right):
        query (str)
          │
          ├─▶ [base_retriever]  QdrantVectorStore cosine search → top-10 Documents
          │         │
          │   [CohereRerank]    Cross-encoder rerank → top-5 Documents
          │         │
          │   [format_docs]     list[Document] → formatted context string
          │
          ├─▶ [RunnablePassthrough]  query passes through unchanged as "question"
          │
          ▼
        [ChatPromptTemplate]   merges {context} + {question} into a chat prompt
          │
        [ChatOpenAI]           gpt-4o-mini generates the answer
          │
        [StrOutputParser]      AIMessage → plain string
          │
          ▼
        answer (str)

    Two-stage retrieval detail:
        Stage 1 — bi-encoder (fast, coarse):
            OpenAI text-embedding-3-small embeds the query and does ANN search
            in Qdrant (HNSW). Returns TOP_K_RETRIEVE (10) candidates with
            cosine similarity scores.

        Stage 2 — cross-encoder (slow, fine-grained):
            Cohere rerank-v3.5 concatenates query + each passage as a single
            input and scores relevance. Returns the top TOP_K_FINAL (5)
            documents, stored in doc.metadata["relevance_score"].

    Args:
        vectorstore: An already-populated QdrantVectorStore instance.

    Returns:
        A compiled LCEL Runnable. Call ``chain.invoke(query: str) -> str``.

    Raises:
        EnvironmentError: If COHERE_API_KEY is not set in the environment.

    Example::

        vectorstore = get_vectorstore()
        chain = build_chain(vectorstore)
        answer = chain.invoke("What are NVIDIA's main AI risks?")
    """
    cohere_api_key = os.getenv("COHERE_API_KEY")
    if not cohere_api_key:
        raise EnvironmentError(
            "COHERE_API_KEY not set.\n"
            "  1. Sign up at https://dashboard.cohere.com/api-keys (free tier: 1000 req/month)\n"
            "  2. Add  COHERE_API_KEY=<your-key>  to your .env file"
        )

    base_retriever = vectorstore.as_retriever(
        search_kwargs={"k": TOP_K_RETRIEVE}
    )

    # Cohere cross-encoder reranks top-10 → top-5
    compressor = CohereRerank(
        model="rerank-v3.5",
        top_n=TOP_K_FINAL,
        cohere_api_key=cohere_api_key,
    )
    retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )

    def format_docs(docs: list[Document]) -> str:
        return "\n\n---\n\n".join(
            f"[Chunk {i} | section={doc.metadata.get('section', '?')}]\n{doc.page_content}"
            for i, doc in enumerate(docs, 1)
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

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    # LCEL: retriever → format → prompt → llm → string
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 70)
    print("RAG WITH LANGCHAIN — Day 13  (two-stage: Qdrant → Cohere rerank → GPT)")
    print("=" * 70)

    print("\n[1/3] Initialising vectorstore...")
    vectorstore = get_vectorstore()

    print("\n[2/3] Building LCEL chain (Qdrant → CohereRerank → gpt-4o-mini)...")
    chain = build_chain(vectorstore)

    print("\n[3/3] Running queries...")
    print("=" * 70)
    for query in QUERIES:
        print(f"\n🔍 QUERY: {query}")
        print("-" * 70)
        answer = chain.invoke(query)
        print(f"💬 ANSWER:\n{answer}")
        print("=" * 70)

    print("\n✅ Day 13 complete!")
    print("   LangChain gains vs Day-12 raw code:")
    print("   • ~65 % fewer lines (~150 vs 425)")
    print("   • LCEL makes it trivial to swap any component")
    print("   • Trade-off: less control over chunking accuracy & evaluation harness")


if __name__ == "__main__":
    main()
