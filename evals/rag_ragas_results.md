# RAG Evaluation — Day 33 (Ragas Metrics)

**Framework:** ragas 0.4.x  
**Document:** NVIDIA 10-K 2025  
**Golden set:** 20 questions with ground-truth expected answers  
**Implementations:** numpy, qdrant, langchain  

## Metric Definitions

| Metric | What it measures | Requires |
|--------|-----------------|----------|
| **Faithfulness** | Fraction of answer claims supported by retrieved context | answer + contexts |
| **Answer Relevancy** | How well the answer addresses the question (reverse-QG cosine) | answer + question |
| **Context Precision** | Proportion of retrieved chunks relevant to the reference answer | contexts + reference |
| **Context Recall** | Fraction of reference answer statements attributable to context | contexts + reference |

## Summary — Average Scores

| Implementation | Faithfulness ↑ | Ans Relevancy ↑ | Ctx Precision ↑ | Ctx Recall ↑ |
|----------------|:--------------:|:---------------:|:---------------:|:------------:|
| numpy | 0.918 | 0.737 | 0.671 | 0.423 |
| qdrant | 0.883 | 0.791 | 0.668 | 0.413 |
| langchain | 0.968 | 0.803 | 0.599 | 0.478 |

## Per-Question Results

### NUMPY

| # | Question (truncated) | Faith | AnsRel | CtxPrec | CtxRec | Latency |
|:-:|----------------------|:-----:|:------:|:-------:|:------:|:-------:|
|  1 | What are NVIDIA's main revenue streams and busines… | 1.000 | 0.967 | 0.950 | 0.500 | 3.8s |
|  2 | What were NVIDIA's total revenues in fiscal year 2… | 1.000 | 0.000 | 0.000 | 0.000 | 1.2s |
|  3 | What AI-related risks does NVIDIA identify in its … | 1.000 | 0.971 | 0.500 | 0.800 | 4.0s |
|  4 | Who are NVIDIA's main competitors in the GPU and A… | 0.786 | 1.000 | 1.000 | 0.500 | 3.5s |
|  5 | What is NVIDIA's data center segment revenue and g… | 1.000 | 0.864 | 0.200 | 0.000 | 4.2s |
|  6 | What supply chain and manufacturing risks does NVI… | 1.000 | 0.985 | 0.917 | 0.000 | 5.7s |
|  7 | What is NVIDIA's research and development expendit… | 0.500 | 0.000 | 0.000 | 0.000 | 5.2s |
|  8 | What export control and regulatory restrictions af… | 1.000 | 0.875 | 1.000 | 0.667 | 10.3s |
|  9 | How does NVIDIA describe its gaming segment perfor… | 1.000 | 0.802 | 0.333 | 0.000 | 3.2s |
| 10 | What are NVIDIA's cybersecurity risk management po… | 1.000 | 0.936 | 1.000 | 0.500 | 5.0s |
| 11 | How does NVIDIA protect its intellectual property … | 1.000 | 0.971 | 0.500 | 0.667 | 2.5s |
| 12 | What is NVIDIA's capital return policy including d… | 1.000 | 0.916 | 0.500 | 0.667 | 4.1s |
| 13 | What is NVIDIA's gross margin and profitability tr… | 0.667 | 0.000 | 0.000 | 0.000 | 1.1s |
| 14 | How many employees does NVIDIA have and how does i… | 0.917 | 0.864 | 1.000 | 0.000 | 4.3s |
| 15 | What are NVIDIA's main product lines including H10… | 0.789 | 0.765 | 1.000 | 0.333 | 7.0s |
| 16 | What is NVIDIA's automotive segment strategy and r… | 1.000 | 0.000 | 0.833 | 0.500 | 4.0s |
| 17 | How does NVIDIA describe the competitive significa… | 0.750 | 0.955 | 0.887 | 0.667 | 2.8s |
| 18 | What are NVIDIA's customer concentration risks? | 1.000 | 0.947 | 1.000 | 1.000 | 3.6s |
| 19 | What is NVIDIA's NVLink and networking strategy? | 0.950 | 0.965 | 1.000 | 0.667 | 4.0s |
| 20 | What geographic risks does NVIDIA face related to … | 1.000 | 0.966 | 0.804 | 1.000 | 7.0s |

### QDRANT

| # | Question (truncated) | Faith | AnsRel | CtxPrec | CtxRec | Latency |
|:-:|----------------------|:-----:|:------:|:-------:|:------:|:-------:|
|  1 | What are NVIDIA's main revenue streams and busines… | 1.000 | 0.967 | 0.950 | 0.500 | 4.0s |
|  2 | What were NVIDIA's total revenues in fiscal year 2… | 0.500 | 0.000 | 0.000 | 0.000 | 1.2s |
|  3 | What AI-related risks does NVIDIA identify in its … | 1.000 | 0.988 | 0.500 | 0.600 | 9.8s |
|  4 | Who are NVIDIA's main competitors in the GPU and A… | 0.769 | 1.000 | 1.000 | 0.500 | 3.9s |
|  5 | What is NVIDIA's data center segment revenue and g… | 1.000 | 0.865 | 0.325 | 0.000 | 3.1s |
|  6 | What supply chain and manufacturing risks does NVI… | 1.000 | 0.987 | 0.917 | 0.000 | 6.5s |
|  7 | What is NVIDIA's research and development expendit… | 0.500 | 0.000 | 0.000 | 0.000 | 1.2s |
|  8 | What export control and regulatory restrictions af… | 1.000 | 0.875 | 1.000 | 0.667 | 6.6s |
|  9 | How does NVIDIA describe its gaming segment perfor… | 0.947 | 0.825 | 0.333 | 0.000 | 5.0s |
| 10 | What are NVIDIA's cybersecurity risk management po… | 1.000 | 0.938 | 1.000 | 0.500 | 6.4s |
| 11 | How does NVIDIA protect its intellectual property … | 0.800 | 0.971 | 0.500 | 0.667 | 3.5s |
| 12 | What is NVIDIA's capital return policy including d… | 1.000 | 0.916 | 0.500 | 0.667 | 3.3s |
| 13 | What is NVIDIA's gross margin and profitability tr… | 0.900 | 0.000 | 0.000 | 0.000 | 4.4s |
| 14 | How many employees does NVIDIA have and how does i… | 0.923 | 0.864 | 1.000 | 0.000 | 2.7s |
| 15 | What are NVIDIA's main product lines including H10… | 0.700 | 0.765 | 1.000 | 0.667 | 7.1s |
| 16 | What is NVIDIA's automotive segment strategy and r… | 1.000 | 0.938 | 0.833 | 0.500 | 6.4s |
| 17 | How does NVIDIA describe the competitive significa… | 0.625 | 1.000 | 0.887 | 0.667 | 3.4s |
| 18 | What are NVIDIA's customer concentration risks? | 1.000 | 0.950 | 1.000 | 1.000 | 3.3s |
| 19 | What is NVIDIA's NVLink and networking strategy? | 1.000 | 0.974 | 0.806 | 0.667 | 3.7s |
| 20 | What geographic risks does NVIDIA face related to … | 1.000 | 1.000 | 0.804 | 0.667 | 7.5s |

### LANGCHAIN

| # | Question (truncated) | Faith | AnsRel | CtxPrec | CtxRec | Latency |
|:-:|----------------------|:-----:|:------:|:-------:|:------:|:-------:|
|  1 | What are NVIDIA's main revenue streams and busines… | 1.000 | 1.000 | 0.950 | 0.500 | 4.2s |
|  2 | What were NVIDIA's total revenues in fiscal year 2… | 1.000 | 0.000 | 0.000 | 0.000 | 2.4s |
|  3 | What AI-related risks does NVIDIA identify in its … | 1.000 | 0.971 | 0.000 | 1.000 | 3.8s |
|  4 | Who are NVIDIA's main competitors in the GPU and A… | 1.000 | 1.000 | 1.000 | 0.500 | 3.0s |
|  5 | What is NVIDIA's data center segment revenue and g… | 1.000 | 0.848 | 0.000 | 0.000 | 2.2s |
|  6 | What supply chain and manufacturing risks does NVI… | 1.000 | 0.985 | 0.833 | 0.000 | 4.7s |
|  7 | What is NVIDIA's research and development expendit… | 1.000 | 0.845 | 0.000 | 0.000 | 1.4s |
|  8 | What export control and regulatory restrictions af… | 1.000 | 0.892 | 1.000 | 0.667 | 6.2s |
|  9 | How does NVIDIA describe its gaming segment perfor… | 0.818 | 0.817 | 0.000 | 0.000 | 3.1s |
| 10 | What are NVIDIA's cybersecurity risk management po… | 1.000 | 0.849 | 0.333 | 1.000 | 4.3s |
| 16 | What is NVIDIA's automotive segment strategy and r… | 1.000 | 0.000 | 1.000 | 0.500 | 4.8s |
| 17 | How does NVIDIA describe the competitive significa… | 0.909 | 0.913 | 1.000 | 0.333 | 2.9s |
| 18 | What are NVIDIA's customer concentration risks? | 0.875 | 0.952 | 0.917 | 1.000 | 3.8s |
| 19 | What is NVIDIA's NVLink and networking strategy? | 0.923 | 0.974 | 0.950 | 0.667 | 3.1s |
| 20 | What geographic risks does NVIDIA face related to … | 1.000 | 1.000 | 1.000 | 1.000 | 4.9s |

## Metric Notes

- **Faithfulness = 0** for short, direct answers is common — ragas NLI parser needs multiple   verifiable claims. Use as a relative comparison across impls, not absolute.
- **Answer Relevancy** uses embedding cosine similarity of reverse-generated questions —   scores above 0.85 indicate strong topical alignment.
- **Context Precision/Recall** both require the `reference` (ground truth) field —   they measure retrieval quality relative to expected answer coverage.
