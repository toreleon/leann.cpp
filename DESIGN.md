# Design notes

## Goal

Minimize persistent vector-index storage for local RAG while retaining enough
graph guidance to recompute only a small subset of document embeddings at query
time.

The runtime dependency direction is:

```text
query text
  -> llama.cpp query embedding
  -> PQ ADC + compact upper-layer routing / base-graph beam
  -> read selected raw chunks
  -> batched llama.cpp recomputation
  -> exact cosine ranking
```

Hnswlib is used only by the offline builder. Its dense vector storage and
native serialized format are never placed in the runtime artifact.

## Build pipeline

1. Embed all chunks in batches and L2-normalize them.
2. Build an inner-product HNSW index in memory.
3. Extract compact upper layers and the base-layer adjacency lists.
4. Rank nodes by original degree; mark the top `hub_ratio` as hubs.
5. Let hubs initially keep up to `2 * graph_degree` outgoing links and let
   ordinary nodes keep `low_degree`.
6. Add reverse links, deduplicate, and cap every final adjacency list at
   `2 * graph_degree`, retaining the closest links.
7. Train one k-means codebook per PQ subspace and bit-pack the assigned
   centroid IDs for every vector. SimHash can be selected for ablation.
8. Derive a deterministic corpus identity from ordered chunk lengths and
   bytes, persist it in both artifacts, checksum the artifacts, and discard
   dense vectors.

This approximates the paper's high-degree-preserving idea. The paper rebuilds
neighbor sets with graph search during pruning; this spike selects from the
already-built base-layer neighborhood.

For an even embedding dimension, the temporary hnswlib vectors receive one
zero-valued padding coordinate. It leaves inner products unchanged and avoids
hnswlib's unaligned 64-bit label slot; the padding never enters PQ/SimHash or
the serialized format.

## Search pipeline

1. Build a per-query PQ ADC table (or a SimHash query sketch).
2. Greedily descend retained HNSW upper layers using approximate distances.
3. For `N <= approximate_scan_limit`, scan all compact codes. Otherwise run a
   bounded best-first beam on the pruned base graph from the routed entry.
4. Keep an approximate shortlist of
   `ceil(ef_search / rerank_ratio)` candidates, saturated at the index size
   before any floating-point-to-integer conversion.
5. Fetch and re-embed the best `ef_search` candidates in
   `recompute_batch_size` batches.
6. Return exact cosine top-k among those recomputed candidates.

Flat ADC is a deliberate local-RAG operating mode: at 5k nodes, scanning 64
four-bit subcodes is cheap compared with one GGUF embedding call. The graph
path remains necessary when compact-code scanning becomes material.

## Binary formats

All integer fields are little-endian. Both files are versioned and validated
when loaded.

### `.leann`

| Field | Type |
|---|---|
| magic | 8 bytes, `LEANNC04` |
| version, metric | `u32`, `u32` |
| corpus pair identity | 32-byte SHA-256 |
| dimension, approximation kind | `u32`, `u32` |
| maximum degree, entry point, maximum level | `u32` × 3 |
| sketch bits | `u32` |
| PQ subquantizers, bits, centroids, subdimension | `u32` × 4 |
| sketch seed | `u64` |
| node count, base edge count | `u64`, `u64` |
| PQ codebook value count, approximation code bytes | `u64`, `u64` |
| fingerprint size + bytes | `u32` + UTF-8 |
| model source size + bytes | `u32` + UTF-8 |
| model digest | 32-byte SHA-256, all zero when absent |
| model size | `u64`, file bytes, 0 when absent |
| pooling type, context tokens | `u32`, `u32` |
| document prefix size + bytes | `u32` + UTF-8 |
| query prefix size + bytes | `u32` + UTF-8 |
| artifact card entry count | `u32` |
| each card entry | key `u32` + UTF-8, value `u32` + UTF-8 |
| base CSR offsets, edges | `(N + 1) * u64`, `E * u32` |
| each upper-layer counts | node count + edge count, `u64` × 2 |
| each upper-layer CSR | node IDs `u32`, offsets `u64`, edges `u32` |
| PQ codebook + bit-packed codes | FP32 values + declared bytes |
| or SimHash table | `N * sketch_bits / 8` bytes |
| artifact checksum | 32-byte SHA-256 of all preceding bytes |

The format currently uses 64-bit CSR offsets to avoid a 4-billion-edge limit.
An optional blocked/varint CSR format is a future storage optimization.

### Embedder descriptor and artifact card

The descriptor section exists so an index is a self-describing artifact rather
than something that only works on the machine that built it. The fingerprint
answers "is this the right embedder?" after the fact; the descriptor answers
"which embedder do I need?" beforehand, which is the question somebody who
just downloaded the pair actually has.

`model source` is an origin string the publisher supplies (`hf:OWNER/REPO/FILE`
by convention). It is an unverified claim. `model digest` is not: it is the
SHA-256 of the model file as read at build time, so a fetched copy can be
checked against it. Neither is a trust root — a matching digest establishes
that the bytes are the ones the publisher recorded, not that the publisher is
trustworthy. `model size` is the **file** size, deliberately different from the
`llama_model_size()` tensor total already inside the fingerprint string.

Every string in the section is validated identically on write and on read:
length-capped, no C0 control characters or DEL, valid UTF-8, and card keys
unique. The read-side copy is not redundant. These values are reproduced in
three delimiter-sensitive places — the `key=value` lines of `stats`, the
tab-separated `LEANNMF1` manifest, and a shell command printed by `pull` — so
a newline forges a record rather than corrupting one, and invalid UTF-8 would
make `stats --format json` fail permanently on an otherwise valid index.

The card preserves the order it was given and rejects duplicate keys. Flag
order is therefore part of the artifact bytes: the same logical card supplied
in a different order is a different file with a different SHA-256. Nothing
time- or host-derived is ever written, so two builds of the same corpus with
the same options and the same backend produce identical bytes.

### Document and query prefixes

Instruction-prefixed embedding models (`search_document: ` / `search_query: `
for nomic-embed) only work if the prefix used at query time matches the one
used at build time. The prefixes are therefore index state, applied by `Index`
at all three embedding sites: the build pass, the query in `Index::search`, and
the rerank recomputation in `search_embedding`.

They are deliberately **not** part of the embedder fingerprint. A live embedder
cannot know an index's prefixes without first loading that index, so folding
them in would be circular. More decisively, `Index` hands the same `Embedder &`
both queries and documents with no distinguishing signal, so neither a
fingerprint nor a wrapping decorator could tell which prefix to apply.

The prefix never reaches the document store. Pair identity is derived from the
raw chunk bytes, and a search result must be the chunk, not the chunk with a
prompt glued to the front of it.

A caller that embeds its own query and calls `search_embedding` directly must
apply `query_prefix()` itself; `leann bench` does exactly this. Precomputed
embeddings cannot be reconciled with a prefix at all — a `LEANNBC2` cache
records no prefix and its vectors already exist — so those combinations are
rejected rather than silently accepted at degraded recall.

### Migration from `LEANNC03`

A v3 index is recognised by magic and rejected with a message naming the
boundary, rather than reported as "not a leann.cpp index". There is no in-place
upgrade: the descriptor and the prefixes are load-bearing for search, so a v3
pair has to be rebuilt from its source chunks.

### `LEANNMF1` manifests

A published index needs its digests to travel separately from the files they
describe, because a corrupted download cannot be trusted to report its own
corruption. `leann pull` reads a manifest and prints the exact commands that
fetch the artifacts; `leann verify` checks what landed against it.

The manifest is line-oriented and tab-separated rather than JSON, because the
repository takes no JSON parser dependency and because a small grammar is one
that can be rejected precisely. Parsing is fail-closed with no lenient mode: an
unknown key, a duplicate record, a CRLF line ending, an uppercase digest, a
name containing `..`, or a `prefix` whose `.leann` and `.docs` are not both
listed are all errors. A newer manifest is never half-understood by an older
binary.

`pull` opens no socket. That is the whole reason the binary needs no HTTP
client, no TLS surface, and no new dependency — and also the reason `pull`
alone verifies nothing.

### `.docs`

| Field | Type |
|---|---|
| magic | 8 bytes, `LEANDC02` |
| version | `u32` |
| corpus pair identity | 32-byte SHA-256 |
| document count | `u64` |
| fixed-header checksum | 32-byte SHA-256 of preceding header |
| byte offsets | `(N + 1) * u64` |
| document checksums | `N * u32` CRC32C |
| metadata checksum | 32-byte SHA-256 of all preceding metadata |
| UTF-8 document bytes | variable |

Only requested documents are read. Open verifies the bounded metadata region;
each requested document is checked against CRC32C after its bytes are read.
The entire raw corpus is not loaded or hashed during normal startup.
The opened stream is protected by an internal mutex, so const `read` and
`read_many` calls on one store are safe across concurrent workers without
reopening a pathname that may have been replaced.

## Publication protocol

An index build acquires directory locks adjacent to both target files and
uses unique same-directory `.tmp.*` and `.bak.*` paths. It closes and validates
the complete temporary pair before publication. Existing artifacts are moved
to backups, the new document store is renamed into place, and the new index is
renamed last as the commit marker. Ordinary errors trigger rollback, restoring
documents before the index so an old index is never reactivated against the
wrong chunks.

The shared identity makes racing readers fail closed. A process or machine
interruption can leave the index absent, backups, or stale locks; C++20 does
not provide portable directory fsync or a two-file atomic rename. The protocol
therefore claims transactional rollback for reported filesystem errors and
torn-pair detection, not power-loss atomic activation.

Checksum verification and parsing use the same already-open stream, so an
adjacent rename cannot make the loader validate one inode and parse another.
If the new pair commits but backup or lock removal fails, the builder reports
an explicit “committed; cleanup required” error and leaves the path available
for recovery.

### Lock descriptors

Acquisition is the `create_directory` call itself. Immediately afterwards the
builder writes an advisory `owner` file inside the lock recording `pid`,
`host`, and `started_unix`, and removes it before removing the directory. The
descriptor is metadata, never part of acquisition, so a binary that ignores it
still interoperates and a lock left by an older binary is simply one with no
descriptor. Unknown keys are ignored on read so the record can grow.

The descriptor exists so `leann doctor` can distinguish "a lock exists" from
"a lock whose owner is gone". It does not make staleness decidable: a pid is
recycled, and a shared filesystem can be mounted on a second host. Liveness is
reported as `running`, `absent`, or `unknown`, and a lock is removed only by an
explicit `--force-unlock`, which refuses when the descriptor names a process
running on this host. Removing a live lock would let two builders interleave
their publication transactions, and pair identity cannot detect that: identity
is derived from the corpus bytes alone, so two concurrent builds of the same
corpus under different build parameters produce a cross that validates.

### Cancellation boundary

`BuildConfig::should_cancel` is polled at phase boundaries. Cancellation
throws `leann::BuildCancelled`, and unwinding removes the temporary pair and
the locks through the same RAII path an error takes, so a cancelled build
leaves no `.lock`, `.tmp.*`, or `.bak.*` behind and publishes nothing.

The final poll is after the temporary pair has been validated and before
`publish_artifact_pair`. Cancellation is never observed inside the publication
transaction, so an interrupt cannot produce a mixed pair.

Cancellation is cooperative and therefore bounded rather than immediate. The
two uncancellable spans are `DocumentStore::write` and
`train_product_quantizer`, each a single opaque pass over the whole corpus;
worst-case latency is one such pass.

## Invariants

- Corpus node labels are contiguous `u32` IDs.
- The index and document store have the same nonzero corpus pair identity.
- The compact index SHA-256, document metadata SHA-256, and every fetched
  chunk CRC32C validate before affected data is used.
- Stored embeddings and query embeddings are L2-normalized.
- Configuration ratios, embeddings, PQ intermediates, and exact distances are
  finite before conversion, heap insertion, or sorting.
- Build and search embedder fingerprints must match.
- Document and query prefixes are index state, applied by `Index`, and are
  never part of the fingerprint, the document store, or the corpus identity.
- Descriptor strings are length-capped, free of control characters, and valid
  UTF-8, checked identically on write and on read; card keys are unique.
- Nothing time- or host-derived is written into an artifact.
- Every CSR edge points to an existing node.
- Every upper edge points to a node present on that layer, and the global
  entry point occurs on every retained upper layer.
- PQ subquantizers evenly divide the embedding dimension; every packed code
  references a persisted centroid.
- `exact_recomputations <= min(ef_search, N)`.
- Returned distances are exact cosine distances from recomputed embeddings,
  never approximate distances.

## Known risks

- PQ quality is sensitive to subvector width; aggressive 8-byte/vector codes
  were inadequate on the measured 768-dimensional SciFact corpus.
- Pruning to very low base degree can hurt the graph path above the flat-scan
  threshold even when local flat ADC recall is strong.
- Recomputing a chunk must be deterministic enough across repeated runs to
  preserve ranking. The fingerprint includes the requested GPU layer count to
  prevent an obvious CPU/GPU mix, but it does not yet identify Metal vs CUDA vs
  Vulkan or the exact llama.cpp build flags.
- Build memory is still `O(N * dimension)`.
- A model description/size/config fingerprint is a guardrail, not a
  cryptographic model or backend hash. The descriptor's `model digest` does
  identify the GGUF file, but it is recorded metadata rather than something
  the loader enforces: nothing checks that the model in use hashes to it.
- `context tokens` is recorded but not enforced against the live embedder, so
  a search with a different `--ctx` than the build is still accepted.
- CRC32C detects accidental chunk corruption but is not an authenticity
  mechanism.
- A `LEANNMF1` manifest is unsigned and has no trust root or revocation. A
  matching digest proves the bytes are the ones the publisher recorded, and
  nothing more.
- A recorded artifact digest pins one specific build. Embeddings are not
  bit-identical across backends or batch shapes, so the same corpus rebuilt
  elsewhere, or with a different `--batch-tokens` or `--parallel`, is a valid
  index with different bytes.
- Stale-lock/backup recovery after an unclean machine stop is currently
  manual.
- Concurrent searches may share `Index` and `DocumentStore`; concurrency of a
  caller-supplied `Embedder` remains that implementation's responsibility.
