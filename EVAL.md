# Retrieval evaluation

Setup: all 8 synthetic contracts in `data/contracts/`, `chunk_text(..., max_chars=500, overlap=50)`, `HashingEmbedder()`, and all 12 queries from `data/queries.jsonl`. A hit is relevant when the returned span overlaps a labeled span in the same document.

| Retriever | recall@3 | Queries missed |
| --- | ---: | --- |
| BM25 only | 11/12 = 0.917 | q12 |
| Dense only (HashingEmbedder) | 11/12 = 0.917 | q12 |
| Hybrid (RRF) | 11/12 = 0.917 | q12 |

All scores were computed locally using the provided offline embedder; they are public-query results, not hidden grader scores.
