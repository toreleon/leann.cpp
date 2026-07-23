# leann.cpp

`leann.cpp` is a native C++20 spike for low-storage local RAG. It builds an
HNSW graph with hnswlib, discards the dense vectors, stores a pruned CSR graph
plus trained product-quantization (PQ) codes, and recomputes promising document
embeddings in batches with a GGUF model through llama.cpp/ggml.

This repository is an independent LEANN-style implementation, not an
accuracy-compatible port of official LEANN. Official LEANN has a Python
control plane and a custom FAISS C++ data plane; this project targets a
zero-Python, single-process C++/ggml deployment. The current `v0.2-spike`
answers the first implementation and measurement questions:

> Can a llama.cpp embedding model traverse a compact graph by selectively
> recomputing document embeddings, and what recall/latency/storage trade-off
> does that produce?

## What works

- Native C++ runtime with no Python service.
- GGUF embedding through the current llama.cpp C API.
- HNSW construction through hnswlib.
- High-degree-preserving base-graph pruning plus compact upper HNSW layers.
- Trained, bit-packed PQ with asymmetric distance computation (ADC); SimHash
  remains available as an ablation.
- Greedy upper-layer routing, graph beam search for large indexes, and flat ADC
  for small local indexes.
- Exact llama.cpp recomputation and ranking of only the selected candidates.
- Batched on-demand recomputation.
- Model fingerprint validation.
- `build`, `search`, `stats`, and exact-ground-truth `bench` commands, including
  a same-embedding dense HNSW baseline.
- Reproducible BEIR SciFact preparation and parameter-sweep scripts.
- Deterministic hash embedder for fast development and CI.

## Build

The default build fetches a pinned hnswlib revision:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Enable the native llama.cpp backend with an existing checkout:

```bash
cmake -S . -B build-llama \
  -DCMAKE_BUILD_TYPE=Release \
  -DLEANN_ENABLE_LLAMA=ON \
  -DLEANN_LLAMA_CPP_SOURCE_DIR=/path/to/llama.cpp
cmake --build build-llama -j
```

If `LEANN_LLAMA_CPP_SOURCE_DIR` is omitted, CMake fetches a pinned llama.cpp
revision. On macOS, the llama.cpp defaults provide the ggml Metal backend; on
other platforms, use the usual llama.cpp CMake backend flags.

For environments without CMake, the developer Makefile can build the
non-llama path:

```bash
make HNSWLIB_DIR=/path/to/hnswlib
make test HNSWLIB_DIR=/path/to/hnswlib
```

## Quick start

Input is UTF-8 text with one already-chunked document per non-empty line.
Build and query with one GGUF embedding model:

```bash
./build-llama/leann build \
  --docs samples/documents.txt \
  --index out/demo \
  --embedder llama \
  --model /path/to/embedding-model.gguf \
  --approx pq \
  --pq-subquantizers 64 \
  --pq-bits 4

./build-llama/leann search \
  --index out/demo \
  --query "How can compact vector search save disk space?" \
  --embedder llama \
  --model /path/to/embedding-model.gguf \
  --top-k 3 \
  --ef-search 64 \
  --rerank-ratio 0.25
```

The same model, pooling metadata, and `--gpu-layers` setting must be used for
build and search. This avoids silently mixing approximate codes produced on
different numerical backends.
`--ctx` is the token capacity per document sequence; `--parallel` sequences
share a llama.cpp context sized as their product.
`--pq-subquantizers` must evenly divide the embedding dimension. The default
64 works with common 384-, 512-, 768-, 1024-, and 1536-dimensional models.

Use the hash backend to exercise the complete pipeline without a model:

```bash
./build/leann build \
  --docs samples/documents.txt \
  --index out/demo-hash \
  --embedder hash \
  --hash-dim 256

./build/leann search \
  --index out/demo-hash \
  --query "local vector index storage" \
  --embedder hash \
  --hash-dim 256
```

## Benchmark

`bench` computes dense exact ground truth in memory, then reports recall,
latency, and recomputation counts for the compact index. It can also build a
dense HNSW from the identical normalized embeddings:

```bash
./build-llama/leann bench \
  --index out/demo \
  --queries samples/queries.txt \
  --embedder llama \
  --model /path/to/embedding-model.gguf \
  --top-k 3 \
  --ef-search 64 \
  --recompute-batch 16 \
  --rerank-ratio 0.25 \
  --dense-baseline 1 \
  --ground-truth-cache out/demo.bench.f32
```

The optional cache is a benchmark-only artifact; it is never loaded by
`search` and is not part of the low-storage index.

Prepare the public BEIR SciFact corpus and run a repeatable sweep:

```bash
python3 scripts/prepare_beir_scifact.py --output work/datasets/scifact

python3 scripts/run_sweep.py \
  --binary build-llama/leann \
  --index pq64=out/scifact \
  --queries work/datasets/scifact/queries-test.txt \
  --embedder llama \
  --model /path/to/embedding-model.gguf \
  --ground-truth-cache work/datasets/scifact/model.bench.f32 \
  --ef-search 32,64,96 \
  --dense-baseline \
  --output out/sweep
```

Sweep at least these knobs:

| Knob | Storage | Recall | Latency |
|---|---:|---:|---:|
| `--low-degree` down | lower | usually lower | can fall or rise |
| `--pq-subquantizers` up | higher code bytes | usually higher | nearly neutral |
| `--pq-bits` up | higher codebook/codes | usually higher | nearly neutral |
| `--ef-search` up | unchanged | higher | higher |
| `--rerank-ratio` down | unchanged | wider approximate beam | slightly higher |
| `--recompute-batch` up | unchanged | neutral/slightly changed | model-dependent |
| `--scan-limit` up | unchanged | can improve small-index recall | more ADC work |

`search_ms` excludes the one query-embedding call but includes document reads,
document embedding recomputation, graph traversal, and exact ranking.

### Official LEANN baseline

The reproducible SciFact comparison pins
[StarTrail-org/LEANN](https://github.com/StarTrail-org/LEANN) at commit
`7a34d88`, uses the same Nomic GGUF corpus vectors, and matches recall before
comparing candidate counts and latency.

| Metric | leann.cpp v0.2 | Official LEANN |
|---|---:|---:|
| Vector index | 385,975 B | 635,358 B |
| Vector-serving artifacts | 385,975 B | 721,992 B |
| Total storage including text | 8,210,103 B | 8,788,218 B |
| Recall@3, all 300 queries | 0.924444 | 0.927778 |
| Candidate embeddings/query | 64.00 | 307.81 |
| Mean real-GGUF latency | 1,273.827 ms | 6,534.278 ms |
| P50 real-GGUF latency | 1,196.477 ms | 6,372.138 ms |
| P95 real-GGUF latency | 1,750.226 ms | 8,642.614 ms |

At nearly identical recall, leann.cpp's vector index is 39.25% smaller and it
uses 4.81 times fewer embedding recomputations. Its measured real-GGUF mean
latency is 5.13 times lower. Official LEANN is nevertheless much more mature:
its Python control plane and custom FAISS C++ data plane support ingestion,
MCP, filtering, hybrid/BM25 retrieval, and multiple graph backends.

Official recall and candidate counts use all 300 queries with the exact shared
corpus vectors. Official real-GGUF latency uses 20 evenly spaced queries
through its supported ZMQ/OpenAI-compatible path to llama-server; leann.cpp
latency uses all 300 queries with llama.cpp in-process. Both timers exclude
query embedding and cold start, so the latency ratio is a directional
experimental result rather than a formal confidence bound.

The result supports positioning leann.cpp as a **zero-Python, single-process
selective-recomputation index with in-process llama.cpp/ggml**, not as the
first C++ implementation of LEANN. See
[the complete comparison](outputs/official-leann-comparison.md) and its
[machine-readable metrics](outputs/official-leann-comparison.json) for timing
scope and caveats.

## Runtime artifacts

For prefix `out/demo`, the builder writes:

- `out/demo.leann`: header, embedder fingerprint, pruned CSR graph, compact
  upper layers, and either PQ metadata/codes/codebook or a SimHash table.
- `out/demo.docs`: raw chunk store with offsets.

No FP32 document embedding is written. Run `leann stats --index out/demo` to
compare index bytes with raw document bytes and the omitted dense-vector size.

## Honest spike boundaries

The implementation now has trained PQ and retained upper routing layers, but
it is still a research spike rather than the paper's complete system:

- all dense embeddings remain in RAM during build;
- PQ training is ordinary per-subspace Lloyd k-means, without OPQ/residual PQ;
- graph pruning selects from existing HNSW neighbors rather than performing
  reconstruction search;
- flat ADC is intentionally used below `--scan-limit` (100k nodes by default);
- a line-oriented document store with no deletes or incremental updates;
- single-process synchronous serving.

Those differences matter. Do not claim the paper's “under 5%” or recall/latency
numbers from this code without measuring them on the target corpus and model.
See [DESIGN.md](DESIGN.md), [VALIDATION.md](VALIDATION.md), and
[ROADMAP.md](ROADMAP.md).

## References

- Y. Wang et al., [LEANN: A Low-Storage Vector
  Index](https://arxiv.org/abs/2506.08276), MLSys 2026.
- [StarTrail-org/LEANN](https://github.com/StarTrail-org/LEANN), official
  Python control plane and custom native backends.
- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp).
- [nmslib/hnswlib](https://github.com/nmslib/hnswlib).

## License

MIT. Dependencies retain their own licenses.
