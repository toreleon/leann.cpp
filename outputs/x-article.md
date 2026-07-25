# leann.cpp: a small native vector index for local RAG

Most local RAG prototypes feel small until you look at the vector index on
disk. The source documents may fit comfortably on a laptop, but the dense
embeddings and graph can turn into the largest part of the system.

LEANN takes a useful approach to that problem: keep a compact search
structure, then recompute the document embeddings that are actually needed
during a query instead of storing every full-precision vector forever.

I started leann.cpp to see how that idea would look in the deployment model I
want for local RAG: one native process, GGUF embeddings in-process through
llama.cpp/ggml, and no Python service or ZMQ worker to operate.

The code is now public:

https://github.com/toreleon/leann.cpp

## What is in the index

leann.cpp is a C++20 research implementation. It stores trained, bit-packed
PQ64x4 codes, a pruned CSR graph, and compact upper HNSW layers. At search
time it uses the compact representation to choose a small candidate set,
recomputes those document embeddings with the local GGUF model, and reranks
them exactly.

It does not store the full FP32 document embedding matrix.

That last point is the reason the project exists. Storage saved in the index
is traded for model work at query time. The practical question is whether the
trade still gives useful recall and acceptable latency.

## Comparing it with official LEANN

I compared leann.cpp with StarTrail-org/LEANN on BEIR SciFact. The test used
the same 5,183 documents, all 300 test queries for recall, and the same
normalized Nomic Q4_K_M GGUF embedding signal. Official LEANN was pinned at
commit 7a34d88.

I tuned the comparison around nearly identical Recall@3:

leann.cpp reached 0.924444 Recall@3 with a 385,975-byte vector index and
recomputed 64 candidate embeddings per query.

Official LEANN reached 0.927778 Recall@3 with a 635,358-byte vector index and
recomputed 307.81 candidate embeddings per query on average.

At that operating point, the leann.cpp vector index was 39.25% smaller and it
used 4.81 times fewer embedding recomputations. Official LEANN retained a
0.003334 absolute recall advantage.

There is an important storage detail here. Once document text is included,
the complete durable artifacts were 8,210,103 bytes for leann.cpp and
8,788,218 bytes for official LEANN, a 6.58% reduction. SciFact is small and
its text dominates the total, so the index-only result should not be confused
with an end-to-end storage claim at large scale.

## Real GGUF latency

I also ran both systems on the same Apple M4 Pro with
nomic-embed-text-v1.5.Q4_K_M.gguf.

leann.cpp averaged 1.274 seconds across all 300 queries. Its p50 was 1.196
seconds and its p95 was 1.750 seconds.

Official LEANN averaged 6.534 seconds across 20 evenly spaced queries. Its p50
was 6.372 seconds and its p95 was 8.643 seconds.

Both timers exclude query embedding and cold start. The leann.cpp path runs
llama.cpp in-process. The official path uses its supported ZMQ worker,
provider layer, and llama-server integration.

The measured mean is 5.13 times lower for leann.cpp, but I treat that ratio as
directional. The query counts differ, and this is not a formal confidence
bound. The candidate count and process boundaries both contribute to the
latency difference.

## Why a native implementation is still useful

Official LEANN is not simply a Python vector index. Its control plane is
Python, while the graph data plane is a custom FAISS C++ HNSW implementation.
It is also far more mature than leann.cpp. It already includes ingestion,
filtering, hybrid and BM25 retrieval, MCP support, multiple backends, and a
broader provider system.

So I am not claiming that leann.cpp is the first C++ implementation of LEANN.
That would be inaccurate.

The narrower goal is useful on its own: make selective recomputation directly
embeddable in a C++ local RAG application, with a single process and native
GGUF inference.

## Current limits

This is still a research spike, not a production vector database.

Dense embeddings remain in memory while the index is built. There are no
incremental updates or deletes yet. Metadata filtering and production
concurrency are not implemented. Most importantly, SciFact is only about
7.78 MB, so it cannot validate the scaling behavior that makes low-storage
indexing interesting.

The next useful tests are 100,000 and 1 million chunks. After that I want to
add graph-guided PQ traversal, mmap-friendly storage, concurrent query safety,
and a stable C ABI that can be embedded by llama.cpp applications.

If you work on local RAG, llama.cpp, compact vector search, or retrieval
benchmarks, I would value a second set of eyes. Try it on another machine,
bring a larger corpus, challenge the methodology, or open an issue.

Repository:

https://github.com/toreleon/leann.cpp

Full benchmark methodology and raw comparison:

https://github.com/toreleon/leann.cpp/blob/main/outputs/official-leann-comparison.md

Official LEANN:

https://github.com/StarTrail-org/LEANN

LEANN paper:

https://arxiv.org/abs/2506.08276
