# leann.cpp v0.2 vs Official LEANN

## Executive conclusion

The zero-Python, native C++/ggml gap is real, but official LEANN should not be
described simply as a Python vector index. Its control plane is Python, while
its HNSW data plane is a customized FAISS C++ fork using compact CSR and ZMQ
embedding recomputation.

On BEIR SciFact, at nearly identical recall, leann.cpp produced a 39.25%
smaller vector index, required 4.81 times fewer embedding recomputations, and
was 5.13 times faster in the measured real-GGUF comparison on the same
machine. Official LEANN achieved 0.33 percentage points higher Recall@3 and is
substantially more mature as a product.

The appropriate positioning is therefore:

> **leann.cpp is a zero-Python, single-process, selective-recomputation vector
> index with in-process llama.cpp/ggml for directly embedded local RAG.**

It should not be positioned as the first C++ implementation of LEANN because
official LEANN already has a C++ graph engine.

## Head-to-head results

| Metric | leann.cpp v0.2 | Official LEANN | Interpretation |
|---|---:|---:|---|
| Corpus / queries | 5,183 / 300 | 5,183 / 300 | Same BEIR SciFact fixture |
| Vector index | 385,975 B | 635,358 B | leann.cpp is 39.25% smaller |
| Vector index / raw text | 4.959% | 8.164% | Raw payload is 7,782,636 B |
| Vector-serving artifacts | 385,975 B | 721,992 B | Official includes ID map, passage offsets, and metadata |
| Text store | 7,824,128 B | 8,066,226 B | Binary concatenation vs JSONL |
| Total durable storage | 8,210,103 B | 8,788,218 B | leann.cpp is 6.58% smaller |
| Recall@3 | 0.924444 | 0.927778 | Exact dense top-3 overlap over all 300 queries |
| Exact candidate embeddings/query | 64.00 | 307.81 | Official requires 4.81× more |
| Mean latency, real GGUF | 1,273.827 ms | 6,534.278 ms | Official is 5.13× slower in the measured runs |
| P50 latency | 1,196.477 ms | 6,372.138 ms |  |
| P95 latency | 1,750.226 ms | 8,642.614 ms |  |

The matched-recall official point uses HNSW `complexity=8` and
`batch_size=16`. At the higher official point, `complexity=16` reached
Recall@3 0.968889 while requiring an average of 431.78 candidate embeddings
per query.

## Measurement scope and methodology

- Sources:
  [official StarTrail-org/LEANN](https://github.com/StarTrail-org/LEANN),
  [pinned commit](https://github.com/StarTrail-org/LEANN/tree/7a34d8856b7aa92da47097af02e8f26b341a90a3),
  and the [LEANN paper](https://arxiv.org/abs/2506.08276).
- Official LEANN was pinned at commit
  `7a34d8856b7aa92da47097af02e8f26b341a90a3`. The core came from that
  commit, and the official HNSW binary package was version 0.3.7.
- Official HNSW used `M=32`, `efConstruction=200`, compact CSR,
  recomputation enabled, and cosine distance.
- leann.cpp used PQ64×4, graph-build `M=16`, pruned base degree 3, and exact
  reranking of 64 candidates.
- Both systems used corpus vectors produced by the same
  `nomic-embed-text-v1.5.Q4_K_M.gguf` model and llama.cpp revision `c588c4f`.
  A parity check over 11 evenly distributed documents found a minimum cosine
  similarity of 0.99999982 between the benchmark cache and llama-server.
- Official recall and candidate counts were measured over all 300 queries
  using the exact cached corpus vectors. This removes model inference from
  that run without changing graph traversal or result IDs.
- Official real-GGUF latency used the supported integration path:
  ZMQ embedding worker → OpenAI-compatible provider → llama-server. It was
  measured over 20 evenly distributed queries at indices
  `0,15,31,47,62,78,94,110,125,141,157,173,188,204,220,236,251,267,283,299`.
- leann.cpp latency was measured over all 300 queries using in-process
  llama.cpp/ggml Metal.
- Both latency measurements exclude query embedding and cold start.
- Because official latency uses a 20-query sample while leann.cpp uses all 300
  queries, the latency ratios are directional experimental results rather
  than formal confidence bounds.
- With candidate embeddings served from an in-memory cache, the official
  graph/IPC path at the matched-recall point had a mean latency of 407.323 ms
  and P95 latency of 523.203 ms. This isolates part of the integration
  overhead and is not an end-to-end RAG latency result.

## Architecture comparison

| Component | leann.cpp | Official LEANN |
|---|---|---|
| Control plane | C++20 | Python |
| Graph engine | Native compact graph + PQ | Custom FAISS C++ HNSW + compact CSR |
| Recomputation | In-process llama.cpp | ZMQ worker + provider abstraction |
| Local ggml route | Native GGUF | llama-server/OpenAI-compatible endpoint |
| Runtime | One process and one binary/library | Python packages + native extension + worker |
| Measured search strategy | PQ scan/routing → exact top-64 | Selective graph traversal → exact recomputation |
| Product surface | Research spike CLI/library | CLI, ingestion, MCP, filtering, hybrid/BM25, HNSW/DiskANN |

## Interpretation

1. **Storage:** leann.cpp wins on this small corpus because of compact PQ codes
   and aggressive graph pruning. Official LEANN retains a denser HNSW graph to
   support higher recall.
2. **Recall and compute:** the matched-recall points are nearly identical, but
   official LEANN recomputes approximately 308 vectors per query while
   leann.cpp recomputes exactly 64.
3. **Latency:** the difference comes from both the candidate count and the
   Python/ZMQ/HTTP integration path. This is where native in-process
   llama.cpp integration provides concrete value.
4. **Maturity:** official LEANN wins decisively. leann.cpp does not yet provide
   its ingestion ecosystem, updates and deletes, metadata filters, hybrid
   retrieval, production concurrency, or million-chunk validation.
5. **Absorption risk:** this should be revised from low to **medium**. Official
   LEANN already has C++ HNSW and a local provider route, so it could absorb a
   llama.cpp provider. A single-process C++ ABI and zero-Python deployment
   remain meaningful differentiators.

## Recommended next steps

1. Keep this benchmark as a regression gate for storage, recall, candidate
   count, and matched-recall latency.
2. Run 100K- and then 1M-chunk corpora. SciFact is too small to validate
   LEANN's scaling claims.
3. Replace the flat ADC path with graph-guided PQ traversal at larger scales
   while retaining the flat path as an exact-ish fast path for small corpora.
4. Add concurrent-query safety, reusable embedding batches, and an mmap
   document store.
5. Define a stable on-disk format, C ABI, and a direct integration example for
   the llama.cpp server.
