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
8. Persist upper layers + base CSR + approximation data + embedder fingerprint;
   discard dense vectors.

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
   `ceil(ef_search / rerank_ratio)` candidates.
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
| magic | 8 bytes, `LEANNC02` |
| version, metric | `u32`, `u32` |
| dimension, approximation kind | `u32`, `u32` |
| maximum degree, entry point, maximum level | `u32` × 3 |
| sketch bits | `u32` |
| PQ subquantizers, bits, centroids, subdimension | `u32` × 4 |
| sketch seed | `u64` |
| node count, base edge count | `u64`, `u64` |
| PQ codebook value count, approximation code bytes | `u64`, `u64` |
| fingerprint size + bytes | `u32` + UTF-8 |
| base CSR offsets, edges | `(N + 1) * u64`, `E * u32` |
| each upper-layer counts | node count + edge count, `u64` × 2 |
| each upper-layer CSR | node IDs `u32`, offsets `u64`, edges `u32` |
| PQ codebook + bit-packed codes | FP32 values + declared bytes |
| or SimHash table | `N * sketch_bits / 8` bytes |

The format currently uses 64-bit CSR offsets to avoid a 4-billion-edge limit.
An optional blocked/varint CSR format is a future storage optimization.

### `.docs`

| Field | Type |
|---|---|
| magic | 8 bytes, `LEANDC01` |
| version | `u32` |
| document count | `u64` |
| byte offsets | `(N + 1) * u64` |
| UTF-8 document bytes | variable |

Only requested documents are read. The entire raw corpus is not loaded during
normal search.

## Invariants

- Corpus node labels are contiguous `u32` IDs.
- Stored embeddings and query embeddings are L2-normalized.
- Build and search embedder fingerprints must match.
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
  cryptographic model or backend hash.
