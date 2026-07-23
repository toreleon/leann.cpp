# leann.cpp v0.2 — BEIR SciFact result

The native C++/ggml spike met the selected storage, recall, and latency point
on one public local-RAG-scale corpus:

| Metric | leann.cpp PQ64×4 | Dense HNSW |
|---|---:|---:|
| Index bytes | 385,975 | 16,713,632 |
| Index / raw text | 4.959% | 214.755% |
| Recall@3 vs exact dense embeddings | 0.924444 | 0.996667 |
| Mean search latency | 1,273.827 ms | 0.395 ms |
| P95 search latency | 1,750.226 ms | 0.474 ms |

The compact index is 43.30× smaller than the dense HNSW baseline. It stores no
dense document vectors and recomputes exactly 64 candidates per query through
the same Nomic GGUF model used at build time.

## Scope

- Corpus: BEIR SciFact, 5,183 documents and all 300 test queries.
- Model: `nomic-embed-text-v1.5.Q4_K_M.gguf`, 768 dimensions.
- Backend: llama.cpp/ggml Metal on Apple M4 Pro.
- Index: trained PQ with 64 subquantizers × 4 bits, pruned base degree 3,
  compact upper HNSW layers.
- Search: upper-layer routing, flat ADC because `N < 100,000`, exact top-64
  recomputation in batches of 16.

Recall is overlap with exact dense top-3 under the same normalized Nomic
embeddings. It is not SciFact qrels recall. Both latency figures exclude query
embedding; compact latency includes document reads and GGUF recomputation,
while dense HNSW latency is resident-vector graph search.

Full provenance and machine-readable metrics are in
`scifact-v0.2-results.json`. The corpus came from the
[official BEIR dataset distribution](https://github.com/beir-cellar/beir/wiki/Datasets-available).
