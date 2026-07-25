# Roadmap from spike to repository-grade system

## Completed in v0.4 operability

- Strict per-command option validation with correction hints, replacing a
  parser that silently ignored an unrecognized option and answered with its
  defaults.
- Per-command help, a command overview, and `--version`.
- Opt-in `--format json` for `build`, `search`, `stats`, `bench`, and
  `doctor`, with UTF-8 validation and fail-closed rejection of document bytes
  that JSON cannot represent; the parsed text output is unchanged.
- Optional build progress and cooperative cancellation in the library, with
  SIGINT/SIGTERM cancellation in the CLI, exit status 130, and no leaked
  locks, temporaries, or partially published pairs.
- Advisory lock descriptors recording pid, host, and start time, without
  making acquisition depend on them.
- `leann doctor` artifact inspection with a repair mode limited to provably
  safe removals and a separate explicit `--force-unlock`.
- A `cli-cache-test` target and a `check` aggregate for the CMake-free build
  path, which previously could not run the CLI suite at all.

## Completed in v0.3 integrity hardening

- Versioned corpus identity in both artifacts with pre-embedding pair
  validation.
- Full compact-index SHA-256, bounded document-metadata SHA-256, and lazy
  per-chunk CRC32C.
- Unique temporary/backup artifacts, same-prefix build locks, index-last
  fail-closed publication, rollback, and dedicated persistence tests.
- Explicit migration and power-loss boundaries for the `.leann` v3 and
  `.docs` v2 formats.
- Fail-fast finite-value validation across build, query, PQ, and exact-ranking
  boundaries.
- Const, synchronized shared document reads with deterministic concurrency and
  sanitizer coverage.

## Completed: native read/search C API

- Versioned, dependency-free C11 ABI for opening an existing artifact pair and
  searching through a caller-owned batched embedding callback.
- Opaque search/result handles, copied result bytes, serialized callback
  calls, thread-local error detail, and exception-safe status translation.
- C11 compile proof plus C++ parity, failure, ownership, and concurrency tests.

## Completed in v0.2 spike

- Native GGUF recomputation, versioned index/document formats, fingerprint
  checks, malformed-input tests, ASan/UBSan, and macOS/Linux CI.
- Trained bit-packed PQ with ADC plus selectable SimHash ablation.
- Compact upper HNSW layers, pruned base CSR, upper-layer routing, large-index
  beam search, and small-index flat ADC.
- BEIR SciFact preparation, benchmark embedding cache, same-vector dense HNSW
  baseline, and CSV/JSON sweep tooling.
- Same-corpus, same-GGUF matched-recall comparison against official LEANN
  compact HNSW.

## Weeks 1–2: reproducible baseline

- Run the supplied sweep on a second public corpus and one target local corpus.
- Add 100K- and 1M-chunk official LEANN regression points; SciFact is too small
  to establish the scaling behavior.
- Measure Recall@3, Recall@10, p50/p95 retrieval latency, exact recomputations,
  peak RSS, build time, index/raw ratio, and index/dense-HNSW ratio.
- Record hardware, GGUF hash, llama.cpp revision, chunking, and all search
  parameters with every benchmark result.

Exit gate: repeatable numbers show a useful storage reduction at an explicitly
chosen recall target.

## Next: approximation quality

- Compare current PQ, residual/OPQ, binary sketches, and a small distilled
  embedder at equal bytes.
- Add blockwise SIMD distance kernels.
- Tune beam/recomputation semantics against the paper's Algorithm 2.

Exit gate: approximate codes reduce recomputations materially over SimHash at
the same recall and storage budget.

## Next: graph quality and format

- Implement pruning with reconstruction search, bidirectional degree
  enforcement, and connectivity checks.
- Replace 64-bit-per-node-per-offset CSR overhead with blocked or compressed
  adjacency.
- Add mmap-based immutable index loading.

Exit gate: graph pruning beats uniform degree reduction across at least two
corpora.

## Weeks 7–8: storage-bounded construction

- Add centroid sampling and soft assignment.
- Build overlapping shards under a configurable peak-storage/RSS budget.
- Merge graph layers and resolve duplicated nodes.
- Stream approximate-code training and generation.

Exit gate: peak temporary storage and RSS stay within declared budgets on a
dataset too large for dense in-memory construction.

## Weeks 9–10: local RAG integration

- Add incremental append buffer and soft deletes.
- Add a bounded hot embedding cache.
- Add a pinned llama.cpp-compatible retrieval example over the completed C
  callback ABI.
- Add service mode. Cancellation, progress, and structured JSON output shipped
  in v0.4.
- Extend recovery beyond the v0.4 `doctor` inspection: cross-prefix scanning
  and the temporaries `bench` writes next to a `--raw-latencies` target.
- Package reproducible releases for macOS arm64 and Linux x86_64.

Exit gate: end-to-end local RAG demo survives restart/update, reports its
storage and recall envelope, and has no Python runtime.

## Research questions to keep explicit

- At what corpus size does recomputation amortize better than quantized dense
  storage on CPU-only, Metal, CUDA, and Vulkan?
- Is the query workload skew high enough for a small exact-vector cache to
  dominate more elaborate approximation?
- How much hierarchy must be retained before base-graph recall stops being the
  bottleneck?
- Does the same GGUF model produce stable enough embeddings across ggml
  backends for a durable index?
