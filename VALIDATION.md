# Validation record

This file records validation of the measured `v0.2-spike` and the subsequent
`v0.3` artifact-integrity hardening. Results are implementation measurements
on one public corpus and one machine, not a reproduction of every LEANN paper
result.

## v0.3 artifact integrity gate

The machine-runnable clean-build command is:

```bash
make HNSWLIB_DIR=/path/to/hnswlib \
  BUILD_DIR=/new/empty/build-directory \
  test persistence-test core-safety-test
```

Validated on Apple arm64 with AppleClang 21:

- Existing PQ, SimHash, compact upper-layer, search, fingerprint-mismatch, and
  truncation tests: pass.
- SHA-256 known-answer vectors for empty input and `abc`: pass.
- CRC32C `123456789` known-answer vector: pass.
- Valid `.leann` v3 / `.docs` v2 round trip and nonzero shared identity: pass.
- Both directions of a same-count, same-byte-length cross-pair swap are
  rejected before the counting embedder is invoked: pass.
- Index payload/footer corruption, truncation, and appended bytes: rejected.
- Document fixed-header and metadata corruption: rejected during open before
  trusting the affected count/table.
- Lazy document payload corruption: rejected by CRC32C during chunk read.
- Document truncation and appended bytes: rejected.
- Hostile `UINT64_MAX` metadata count: rejected without allocation.
- An injected embedding failure during replacement preserves both previous
  artifacts byte-for-byte and removes owned temporary files/locks: pass.
- Fault injection at index backup, document backup, document activation, and
  final index activation restores the previous pair byte-for-byte: pass.
- Publication from both-artifact, index-only, document-only, and empty prior
  states installs one complete new pair and removes owned backups: pass.
- Injected backup/lock cleanup failures report a distinct “pair committed;
  cleanup required” state while retaining the recoverable path: pass.
- A pre-existing build lock blocks a second build without deleting that lock:
  pass.
- Two simultaneous same-prefix builders are serialized by the lock; the
  losing builder fails before embedding and the winning pair validates: pass.
- NaN and both infinities in build/search ratios, text-query embeddings, raw
  query vectors, build embeddings, and rerank embeddings: rejected before the
  unsafe operation.
- Twelve synchronized workers sharing one const document store completed 300
  mixed `read` / `read_many` rounds each with exact byte equality: pass.
- Concurrent-read locking preserves lazy CRC32C rejection and move-only
  document-store use: pass.
- The core and persistence suites pass under combined ASan + UBSan.

The validation used a new temporary build directory, compiled with
`-Wall -Wextra -Wpedantic -Werror`, and exited 0. It does not prove
fsync/power-loss recovery or that an arbitrary caller-provided embedder is
thread-safe.

## Build and correctness

Validated on Apple arm64 with AppleClang 21:

- C++20 Makefile build and unit test: pass.
- CMake Release build: pass.
- CMake install to a temporary prefix: pass.
- ASan + UBSan test with `halt_on_error=1`: pass.
- Truncated index rejection: pass.
- Wrong embedder fingerprint rejection: pass.
- Pinned llama.cpp + ggml CPU static build and final executable link: pass.
- Pinned llama.cpp + ggml Metal static build and runtime search: pass.
- Real GGUF build/search/benchmark path: pass.
- Trained PQ including non-byte-aligned code packing: pass.
- SimHash compatibility path: pass.
- Compact upper-layer round trip and routing: pass.
- Dense HNSW baseline from the same normalized embeddings: pass.
- Benchmark embedding-cache round trip: pass.

The sanitizer run initially exposed hnswlib's unaligned `size_t` label slot for
even FP32 dimensions. The builder now gives only the temporary HNSW vectors one
zero-valued padding coordinate. The repeat sanitizer run passed, and stored
embedding dimensions remain unchanged.

## BEIR SciFact / Nomic GGUF run

Environment and provenance:

- Machine: Apple M4 Pro, 14 cores, 48 GB RAM; macOS 26.5.1.
- Backend: llama.cpp/ggml Metal, `gpu-layers=99`.
- Model: `nomic-ai/nomic-embed-text-v1.5-GGUF`,
  `nomic-embed-text-v1.5.Q4_K_M.gguf`.
- Model SHA-256:
  `d4e388894e09cf3816e8b0896d81d265b55e7a9fff9ab03fe8bf4ef5e11295ac`.
- llama.cpp revision:
  `c588c4f47683e73ad2d69f50480bec6cc85fd0f7`.
- hnswlib revision:
  `d9b3608c83d83b46c96e25088cb1d729b29dcfe9`.
- Corpus: BEIR SciFact, 5,183 documents and all 300 test queries.
- SciFact archive SHA-256:
  `536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165`.
- Prepared raw text: 7,782,636 bytes, excluding line separators.

Index configuration:

```text
graph_degree=16, ef_construction=100
low_degree=3, hub_ratio=0.02
PQ subquantizers=64, bits=4, iterations=10, training samples=4096
embedding dimension=768
```

Search configuration:

```text
top_k=3, ef_search=64, recompute_batch=16
rerank_ratio=0.25, approximate_scan_limit=100000
```

Since this corpus has fewer than 100k nodes, search used flat ADC over compact
PQ codes after upper-layer routing, then recomputed exactly 64 candidates.
Recall below is agreement with exact dense top-3 under the same Nomic
embeddings; it is not qrels-based BEIR task recall.

Observed over all 300 test queries:

```text
Recall@3                         0.924444
Mean compact-index latency       1273.827 ms
P50 compact-index latency        1196.477 ms
P95 compact-index latency        1750.226 ms
Mean exact recomputations        64.000
Mean approximate distances       5238.940
Mean upper-layer hops            3.290

Dense HNSW Recall@3              0.996667
Dense HNSW mean search latency   0.395 ms
Dense HNSW P95 search latency    0.474 ms
Dense HNSW serialized bytes      16,713,632
```

Both latency measurements exclude query embedding. The compact measurement
includes document-store reads, GGUF recomputation, ADC, and exact reranking;
the dense HNSW measurement is graph search over already resident vectors.

Storage and build:

```text
leann.cpp index bytes             385,975
raw prepared document bytes     7,782,636
index / raw                         4.959%
dense HNSW / leann.cpp              43.302x
leann.cpp / dense HNSW               2.309%
dense FP32 vector bytes omitted 15,922,176
base directed edges                27,667
upper directed edges                3,575
build wall time                    115.001 s
build maximum RSS                1,607,237,632 bytes
```

The `.leann` SHA-256 is
`5d2f7f5081f723f75464e6381ba9492055f54c29bbf12eead1d9435bd534c5fd`.

## Official LEANN comparison

Official LEANN was pinned at
`7a34d8856b7aa92da47097af02e8f26b341a90a3`. Its HNSW backend used `M=32`,
`efConstruction=200`, compact CSR, recomputation, and cosine distance.
The corpus vectors were the exact benchmark cache above. Eleven vectors
recomputed through the same GGUF and llama-server had minimum cosine
0.99999982 against the cache.

The matched-recall full-set point was:

```text
                                leann.cpp        Official LEANN
Recall@3                         0.924444          0.927778
Vector index bytes                385,975           635,358
Vector-serving bytes              385,975           721,992
Candidate embeddings/query         64.000           307.810
```

Official recall and candidate counts use all 300 queries. Official real-GGUF
latency was separately measured on 20 evenly spaced queries:

```text
                                leann.cpp        Official LEANN
Mean latency                    1273.827 ms       6534.278 ms
P50 latency                     1196.477 ms       6372.138 ms
P95 latency                     1750.226 ms       8642.614 ms
Queries measured                       300                20
```

Both timers exclude query embedding and cold start. Official uses its supported
ZMQ/OpenAI-compatible provider route to the same llama.cpp server; leann.cpp
uses llama.cpp in-process. Full methodology and caveats are in
`outputs/official-leann-comparison.md`.

## Historical v0.1 real GGUF smoke run

Configuration:

- Model: `nomic-ai/nomic-embed-text-v1.5-GGUF`,
  `nomic-embed-text-v1.5.Q4_K_M.gguf`
- Model SHA-256:
  `d4e388894e09cf3816e8b0896d81d265b55e7a9fff9ab03fe8bf4ef5e11295ac`
- llama.cpp revision:
  `c588c4f47683e73ad2d69f50480bec6cc85fd0f7`
- hnswlib revision:
  `d9b3608c83d83b46c96e25088cb1d729b29dcfe9`
- Backend: ggml CPU + Accelerate, four threads
- Corpus: 30 synthetic documentation passages in `samples/documents.txt`
- Queries: eight queries in `samples/queries.txt`
- `top_k=3`, `ef_search=24`, `recompute_batch=8`,
  `rerank_ratio=0.5`, 128-bit sketches

Observed:

```text
Recall@3                  0.916667
Mean search latency       147.228 ms
P50 search latency        146.963 ms
P95 search latency        150.687 ms
Mean exact recomputations 24.000
Mean sketch distances     28.875
Index bytes               1,514
Raw document bytes        2,215
Dense FP32 bytes omitted  92,160
```

The corpus is far too small for index/raw overhead to be representative:
fixed headers and offsets dominate, so the compact index is 67.8% of raw text.
The useful smoke result is that the complete native path ran, returned exact
reranked results, and achieved 11/12 top-3 matches while persisting no dense
document vectors.

A separate Metal build/index/search smoke run also passed with the same model,
returning the same three document IDs in 61.1 ms. CPU and Metal embeddings were
not bit-identical, so the final fingerprint deliberately rejects changing
`--gpu-layers` between index build and search.
