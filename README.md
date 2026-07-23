# leann.cpp

`leann.cpp` is a native C++20 spike for low-storage local RAG. It builds an
HNSW graph with hnswlib, discards the dense vectors, stores a pruned CSR graph
plus trained product-quantization (PQ) codes, and recomputes promising document
embeddings in batches with a GGUF model through a pinned llama.cpp/ggml C API.

This repository is an independent LEANN-style implementation, not an
accuracy-compatible port of official LEANN. Official LEANN has a Python
control plane and a custom FAISS C++ data plane; this project targets a
zero-Python, single-process C++/ggml deployment. The current `v0.3` development
line adds fail-closed artifact integrity to the measured `v0.2` retrieval
spike:

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
- A non-cryptographic model/config fingerprint guardrail.
- SHA-256-protected indexes, lazy per-chunk CRC32C document validation, shared
  corpus identity, and fail-closed pair publication.
- Fail-fast rejection of non-finite configuration, query, embedding, PQ, and
  cosine-distance values before they can enter integer conversions or ranking.
- Const, mutex-protected document reads so one immutable index/document pair
  can serve concurrent searches.
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
make HNSWLIB_DIR=/path/to/hnswlib \
  test persistence-test core-safety-test
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

The same model description/size, pooling metadata, and `--gpu-layers` setting
must be used for build and search. This guardrail catches common configuration
mismatches, but it does not yet cryptographically identify the GGUF weights,
llama.cpp build, or Metal/CUDA/Vulkan backend.
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

- `out/demo.leann`: corpus identity, embedder fingerprint, pruned CSR graph,
  compact upper layers, PQ/SimHash data, and a full-file SHA-256 footer.
- `out/demo.docs`: corpus identity, offsets, per-chunk CRC32C values, a
  SHA-256-protected metadata region, and raw chunk bytes.

No FP32 document embedding is written. Run `leann stats --index out/demo` to
compare index bytes with raw document bytes and the omitted dense-vector size.

### Artifact integrity and format migration

The `.leann` v3 and `.docs` v2 formats deliberately reject older spike
artifacts; rebuild a v0.2 index from its source chunks. Both new artifacts
carry the same deterministic 256-bit identity derived from ordered chunk
lengths and bytes. `search`, `bench`, and `stats` reject a mixed pair before
query embedding or corpus scanning.

Index loading verifies SHA-256 over the complete compact index. Opening a
document store verifies only its header, offset table, and checksum table, so a
100 GB raw corpus is not scanned at startup. Each fetched chunk is checked
against CRC32C before it is returned for recomputation.

Builders acquire adjacent `.lock` directories, use unique temporary and backup
names, validate both completed artifacts, publish the document store, and
rename the index last as the commit marker. An ordinary failure rolls back to
the previous pair; interruption can leave the index absent or a stale lock,
but never makes a mixed pair pass validation. C++20 has no portable fsync or
two-file atomic rename, so this is fail-closed publication rather than a
power-loss atomicity claim. After a machine-level interruption, stop all
builders and inspect any `.bak.*` artifacts before removing a stale lock.
If the new pair commits but backup/lock cleanup fails, `build` reports that
distinct committed state and retains the recoverable path instead of silently
claiming cleanup success.

### Concurrency and numeric safety

`Index` is immutable after loading, and `DocumentStore::read` /
`DocumentStore::read_many` are const and serialize access to their shared file
stream. Multiple search workers can therefore share one loaded index/document
pair. Moving or destroying that pair while reads are active is unsupported.
Each worker must use its own `Embedder`, or an embedder implementation that
explicitly guarantees concurrent calls; the generic embedder interface does
not add hidden serialization.

Library validation rejects NaN and infinity in build/search ratios, raw query
vectors, backend embeddings, PQ calculations, and exact cosine ranking.
Positive but extremely small rerank ratios saturate the approximate beam at the
index size instead of converting an infinite intermediate to an integer.

## Honest spike boundaries

The implementation now has trained PQ and retained upper routing layers, but
it is still a research spike rather than the paper's complete system:

- all dense embeddings remain in RAM during build;
- PQ training is ordinary per-subspace Lloyd k-means, without OPQ/residual PQ;
- graph pruning selects from existing HNSW neighbors rather than performing
  reconstruction search;
- flat ADC is intentionally used below `--scan-limit` (100k nodes by default);
- a line-oriented document store with no deletes or incremental updates;
- no automatic recovery command yet for stale locks/backups after a
  machine-level interruption;
- no bundled service mode, cancellation, or request scheduler.

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
