# Finance Sentiment Engine

> A multi-provider LLM pipeline that scrapes financial news, classifies each story by sentiment, urgency, and event type, then renders a priority-ranked daily brief in Markdown.
> Built in 7 days as a hands-on AI engineering project — swap OpenAI, Anthropic, or Groq with a single flag and compare accuracy vs. cost in seconds.

![Demo](docs/demo.gif)

## Architecture

```mermaid
flowchart LR
    subgraph Ingest
        A[scraper/yahoo.py\nRSS feeds] -->|news.jsonl| B[(cache/)]
    end
    subgraph Analyze
        B -->|news.jsonl| C[analyzer/runner.py\nasync batch]
        C -->|LLM structured output| D[(cache/analysis_*.jsonl)]
    end
    subgraph Report
        D --> E[report/daily.py\nMarkdown generator]
        E --> F[reports/YYYY-MM-DD.md]
    end
    subgraph Providers
        G[OpenAI\ngpt-4.1-mini / nano]
        H[Anthropic\nClaude]
        I[Groq\nLlama 3.x]
    end
    C <-->|LLMProvider interface| G & H & I
```

## Quickstart

```bash
# 1. Clone & install
git clone https://github.com/<your-handle>/finance-sentiment-engine
cd finance-sentiment-engine
uv sync          # or: pip install -r requirements.txt

# 2. Set API keys
cp .env.example .env
# Edit .env and fill in your keys

# 3. Run end-to-end demo (5 headlines → brief)
bash scripts/demo.sh
```

## Usage

### Fetch & Analyze News

```bash
# Analyze with default model (gpt-4.1-nano — best cost/accuracy balance)
python main.py --provider openai --model gpt-4.1-nano

# Benchmark all providers at once
python main.py --benchmark

# Limit to first 5 items (quick smoke-test)
python main.py --limit 5
```

Results are saved to `cache/analysis_{provider}_{model}.jsonl`.

### Generate Daily Brief

```bash
# Today's brief (reads from cache, writes to reports/)
python -m report.daily

# Specific date
python -m report.daily --date 2026-05-18

# Custom top-n and model
python -m report.daily --date 2026-05-18 --top-n 10 --model gpt-4.1-mini
```

Output is printed to stdout **and** saved as `reports/{date}.md`.

### Sample Output

```markdown
# Daily Finance Brief — 2026-05-18
## Top 5 Urgent Movements

### TSLA · ⚪ neutral · medium
Tesla raised U.S. Model Y prices for the first time in two years; stock formed a lower buy point.

[Full article](https://finance.yahoo.com/...)
```

## Eval Results

Evaluated on **36 human-labeled examples** with a 10-bucket `key_event` enum and price-impact urgency rubric.

| Provider | Model | sentiment_acc | urgency_acc | key_event_acc | avg_latency | cost/run |
|----------|-------|:-------------:|:-----------:|:-------------:|:-----------:|:--------:|
| openai | gpt-4.1-mini | 83% | 67% | 78% | 1 261 ms | $0.016 |
| openai | **gpt-4.1-nano** ✅ | 69% | **75%** | 58% | 912 ms | **$0.004** |
| groq | llama-3.3-70b-versatile | 56% | 50% | 44% | 11 496 ms† | $0.022 |
| groq | llama-3.1-8b-instant | 52% | 39% | 39% | 27 531 ms† | $0.002 |

† Groq latency inflated by free-tier TPM rate-limit backoff — true inference is ~200–400 ms.

→ Full analysis: [`eval/results.md`](eval/results.md)

## Provider Matrix

| Provider | `generate` | `generate_structured` | Structured output mechanism |
|----------|------------|----------------------|-----------------------------|
| OpenAI | Responses API | `beta.chat.completions.parse` | Native JSON schema / parsed Pydantic |
| Anthropic | Messages API | Tool use + `tool_choice` | Forces tool call, extracts `input` field |
| Groq | Chat Completions API | `response_format={"type":"json_object"}` | JSON mode + manual Pydantic parse |

## Low-level Usage

```python
from providers.factory import get_provider

provider = get_provider("openai", "gpt-4.1-nano")
response = provider.generate([{"role": "user", "content": "Hello!"}])
print(response.text)
print(response.usage)  # input / output / total tokens
```

## What I Learned

Seven days of building this pipeline taught me more than any tutorial about the gap between "LLM works in the playground" and "LLM is reliable in production."

**GPT-4.1-nano is the sleeper hit.** It beats mini on urgency accuracy (75 % vs 67 %) at a quarter of the cost. This makes sense in hindsight: urgency is a rubric-driven signal — once you give the model a clear price-impact definition, a smaller model can follow it perfectly. Sentiment and complex event taxonomy are where the parameter count starts to matter.

**Open-weight models struggle with long constrained prompts.** Groq's Llama models plateau around 50 % sentiment accuracy and fall to 39–44 % on `key_event`. The pattern is consistent: when you hand them a 10-item Literal enum inside a multi-thousand-token few-shot prompt, instruction-following degrades noticeably. GPT-class models treat the enum as a hard constraint; Llama models treat it as a suggestion.

**Free-tier rate limits are a real engineering problem.** Groq's 12 K / 6 K TPM limits turned a 400 ms model into a 27-second one in batch mode. The fix — exponential backoff with `tenacity`, per-provider concurrency caps — added more code than the actual LLM calls.

**Structured output is not free.** Each provider implements it differently: OpenAI parses directly into a Pydantic model, Anthropic forces a tool call you then unwrap, Groq uses JSON mode requiring a manual `model_validate` call. Hiding this behind a common `LLMProvider` interface was the right call — it kept the analyzer layer clean and made the provider swap genuinely one-line.

## Day 13 — LangChain Port: Line Count & Control Trade-off

`rag_langchain.py` ports the Day-12 two-stage pipeline (vector search → Cohere rerank → GPT answer) to pure LangChain / LCEL.

### Line Count Comparison

| File | Lines | Description |
|------|------:|-------------|
| `rag_numpy.py` | 239 | Day 9 — NumPy brute-force cosine search |
| `rag_qdrant.py` | 320 | Day 10 — Qdrant vector DB + filter demo |
| `rerank.py` | 424 | Day 12 — Two-stage rerank (Cohere + BGE) |
| **`rag_langchain.py`** | **185** | **Day 13 — LangChain / LCEL port** |

**56 % fewer lines** than the equivalent raw implementation (`rerank.py`).

### What LangChain Hides (and What You Lose)

| Raw Day-12 code | LangChain equivalent | Control lost |
|-----------------|---------------------|-------------|
| `tiktoken` + manual sliding window | `RecursiveCharacterTextSplitter` | Character-based, not token-accurate — chunk sizes are approximate |
| `openai.embeddings.create` in batches of 100 | `OpenAIEmbeddings` | No visibility into batch size or rate-limit retry logic |
| Manual `qdrant.upsert` loop | `QdrantVectorStore.from_documents` | Can't tune upsert batch size or inspect intermediate state |
| `rerank_cohere()` → raw `relevance_score` per chunk | `ContextualCompressionRetriever + CohereRerank` | Rerank scores are stripped from final `Document` objects |
| `rerank_bge()` local CrossEncoder | ❌ not ported | No BGE local reranker (needs a custom `BaseDocumentCompressor`) |
| `recall_at_k()` evaluation harness | ❌ not ported | No automated Recall@5 metric — you need `langchain_benchmarks` or manual eval |
| Per-chunk score visible at every step | Hidden inside chain | Harder to debug when retrieval quality degrades |

### The Key Insight

> LangChain is excellent for rapid prototyping — swapping the LLM, retriever, or reranker is one line.  
> But because you built Days 9–12 from scratch, you now know exactly what each abstraction is doing underneath. When something breaks in production, that mental model is what lets you debug it.

```bash
# Run the LangChain port (requires Qdrant running on localhost:6333)
python rag_langchain.py
```
