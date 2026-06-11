"""Day 19: 10-K RAG tool — wraps Week-2 LangChain RAG as an agent-callable tool."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel

from agent.tools.base import Tool

load_dotenv()

# Ticker → Qdrant koleksiyon adı (Week-2 rag_langchain.py ile önceden indexlendi)
_TICKER_COLLECTIONS: dict[str, str] = {
    "NVDA": "nvda_10k_lc",
}

# Ticker → yerel JSON veri dosyası (koleksiyon yoksa buradan inşa edilir)
_TICKER_DATA: dict[str, Path] = {
    "NVDA": Path(__file__).parent.parent.parent / "data" / "nvda_10k_2025.json",
}

# Chunk boyutu token değil karakter cinsindendir — RecursiveCharacterTextSplitter karakter sayar.
# 2000 karakter ≈ tipik İngilizce metinde 500 token.
_CHUNK_SIZE = 2000
_CHUNK_OVERLAP = 200   # %10 örtüşme: chunk sınırındaki cümleler bölünmesin
_TOP_K_RETRIEVE = 10   # geniş ilk geçiş; Cohere reranker bunu daraltacak
_TOP_K_FINAL = 5       # rerank sonrası LLM'e gönderilen son chunk sayısı
_QDRANT_URL = "http://localhost:6333"


def _get_or_build_vectorstore(ticker: str):
    """Return QdrantVectorStore for ticker; index from data file if collection missing."""
    from langchain_core.documents import Document
    from langchain_openai import OpenAIEmbeddings
    from langchain_qdrant import QdrantVectorStore
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from qdrant_client import QdrantClient

    collection = _TICKER_COLLECTIONS[ticker]
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    client = QdrantClient(url=_QDRANT_URL)
    existing = [c.name for c in client.get_collections().collections]

    if collection in existing:
        # Koleksiyon zaten indexli — yeniden embed etmeden bağlan
        return QdrantVectorStore(
            client=client,
            collection_name=collection,
            embedding=embeddings,
        )

    # Koleksiyon yok — veri dosyasından inşa et
    data_path = _TICKER_DATA[ticker]
    data = json.loads(data_path.read_text())
    # JSON'daki her anahtar bir 10-K bölümü adıdır; metadata ve full_text atla
    docs = [
        Document(
            page_content=v.strip(),
            metadata={"section": k, "ticker": ticker},
        )
        for k, v in data.items()
        if k not in ("_meta", "full_text") and isinstance(v, str) and v.strip()
    ]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_CHUNK_SIZE,
        chunk_overlap=_CHUNK_OVERLAP,
        add_start_index=True,  # karakter offsetini metadata'ya yazar, debug için faydalı
    )
    chunks = splitter.split_documents(docs)
    # from_documents tüm chunk'ları embed edip tek seferde Qdrant'a yükler
    return QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        url=_QDRANT_URL,
        collection_name=collection,
    )


class _Query10KInput(BaseModel):
    ticker: str
    question: str


def _query_10k(inputs: _Query10KInput) -> dict[str, Any]:
    ticker = inputs.ticker.upper()

    # Desteklenmeyen ticker için structured hata döndür; agent loop bunu LLM'e iletir,
    # LLM de web_search ile fallback yapabilir.
    if ticker not in _TICKER_COLLECTIONS:
        return {
            "error": f"No 10-K index for '{ticker}'. Supported tickers: {list(_TICKER_COLLECTIONS.keys())}"
        }

    from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
    from langchain_cohere import CohereRerank
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_openai import ChatOpenAI

    vectorstore = _get_or_build_vectorstore(ticker)
    # Reranker'ın eleme yapabilmesi için gerekenin fazlasını getir
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": _TOP_K_RETRIEVE})

    cohere_api_key = os.getenv("COHERE_API_KEY")
    if cohere_api_key:
        # İki aşamalı: vektör arama (recall) → Cohere cross-encoder (precision)
        compressor = CohereRerank(
            model="rerank-v3.5",
            top_n=_TOP_K_FINAL,
            cohere_api_key=cohere_api_key,
        )
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever,
        )
    else:
        # Cohere key yok — reranking atla, vektör aramasının top-K'sını kullan
        retriever = vectorstore.as_retriever(search_kwargs={"k": _TOP_K_FINAL})

    docs = retriever.invoke(inputs.question)

    # LLM kaynağını gösterebilsin diye bölüm adını context'e göm
    context = "\n\n---\n\n".join(
        f"[Chunk {i} | section={doc.metadata.get('section', '?')}]\n{doc.page_content}"
        for i, doc in enumerate(docs, 1)
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            f"You are a financial analyst assistant. "
            f"Answer using ONLY the provided context excerpts from {ticker}'s 10-K annual filing. "
            f"Be specific and cite which sections support your answer.",
        ),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ])

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,   # düşük sıcaklık → olgusal yanıt, hallucination azalır
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    # LCEL zinciri: context sabitlenir, soru olduğu gibi geçirilir
    chain = (
        {"context": lambda _: context, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke(inputs.question)

    # Her chunk'ın ilk 200 karakterini okunabilir kaynak referansı olarak döndür
    sources = [
        {
            "section": doc.metadata.get("section", "unknown"),
            "snippet": doc.page_content[:200].strip(),
        }
        for doc in docs
    ]

    return {"answer": answer, "sources": sources}


query_10k = Tool(
    name="query_10k",
    description=(
        "Queries a company's 10-K annual SEC filing using RAG (Retrieval-Augmented Generation). "
        "Pass a stock ticker and a specific question about the company's annual report. "
        "Returns a detailed answer grounded in the actual 10-K text, plus the source sections cited. "
        "Best for: business strategy, risk factors, revenue breakdown, competition, R&D, and outlook. "
        "Supported tickers: NVDA."
    ),
    input_schema=_Query10KInput,
    handler=_query_10k,
)
