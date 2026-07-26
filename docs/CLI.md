# Command-line interface

`leann` has five commands. `leann --help` lists them, `leann <command> --help`
prints one command's options, and `leann --version` prints the version.

```text
leann build    Embed a corpus, prune the graph, and publish the artifact pair.
leann search   Retrieve the exact top-k for one query.
leann stats    Describe a published artifact pair.
leann bench    Measure recall, latency, and recomputation over a query file.
leann doctor   Report artifact health and leftovers from an interrupted build.
```

## Option handling

Every option each command reads is declared in one table in `app/main.cpp`, and
that table is the only thing the parser accepts. An option the command does not
read is an error rather than a silently ignored token:

```console
$ leann search --index corpus --query "graph pruning" --topk 5
error: unknown option for leann search: --topk; did you mean --top-k?

$ leann stats --index corpus --top-k 5
error: unknown option for leann stats: --top-k; --top-k belongs to leann search
```

This matters because the previous parser accepted anything: a mistyped
`--topk 5` left `--top-k` at its default of 3 and the command answered with
three results and exit status 0, which is indistinguishable from a correct
answer to the question that was asked. Stray positional arguments and options
missing their value are rejected the same way.

Numeric option values are parsed strictly and without locale influence. A
decimal option accepts an optional `-`, digits with an optional `.`, and an
optional `e`/`E` exponent; a leading `+`, surrounding whitespace, a hex float,
`inf`, and `nan` are all rejected with `must be a finite number`. An unsigned
option accepts digits only. Neither ever silently truncates a value.

`tests/test_cli_cache.cpp` pins the tables: every flag must be unique within a
command, must start with `--`, and must carry help text.

## Exit status

| Status | Meaning |
| --- | --- |
| 0 | success |
| 1 | error; the message is on stderr, prefixed `error: ` |
| 2 | invoked with no command; usage is printed on stderr |
| 130 | a cancellable command stopped on SIGINT or SIGTERM |

Only `build` and `bench` produce 130; see Cancellation below.

## Progress

`build` and `bench` accept `--progress auto|always|never`. `auto`, the default,
enables progress only when stderr is a terminal, so a redirected or piped run
stays quiet. Progress is written to stderr only and never to stdout, so it
cannot contaminate a parsed result stream.

```console
$ leann build --docs corpus.txt --index corpus --progress always
building graph 12288/120000 (10.2%) 6001/s eta 17s
```

## Cancellation

`build` and `bench` install handlers for SIGINT and SIGTERM; the other
commands keep the default disposition, because a command that caught a signal
without acting on it would simply be uninterruptible. The first signal
requests cancellation and restores the default disposition, so a second signal
always terminates even inside an uncancellable span.

`build` observes the request at phase boundaries and unwinds, which removes
the temporary artifacts and the build locks, then exits 130:

```console
$ leann build --docs corpus.txt --index corpus
^C
cancelled: build cancelled during building graph
$ echo $?
130
```

Two properties are deliberate and three limits are real.

- Nothing is published. Cancellation is not observed once the publication
  transaction has begun, so an interrupt can never produce a mixed pair.
- Nothing is left behind: no `.lock` directory, no `.tmp.*`, no `.bak.*`.
- Cancellation is cooperative, so it is not instantaneous. Latency is bounded
  by the longest uncancellable span, which is one full pass of
  `DocumentStore::write` or of PQ training. On a large corpus with a GGUF
  embedder that span is minutes, not milliseconds.
- `bench` observes cancellation only between queries, so a measured query is
  never truncated and reported as a result.
- `search`, `stats`, and `doctor` are not cancellable; SIGINT terminates them
  the ordinary way. They hold no locks and write no temporaries, so there is
  nothing for an orderly unwind to clean up.

## Structured output

`build`, `search`, `stats`, `bench`, and `doctor` accept `--format json`. The
default remains `text`, and JSON is strictly opt-in because the benchmark
harness parses the text form.

The text output of `search`, `stats`, and `bench` is byte-identical to v0.3;
this was checked by running both binaries over the same index and diffing.
`build` has one deliberate change: its `index:` and `documents:` lines print
the path unquoted, where v0.3 printed `index: "corpus.leann"` because
`std::filesystem::path` streams through `std::quoted`. Nothing parses those
lines.

JSON goes to stdout as a single object. For `search` this includes the metrics
that the text format writes to stderr, so a caller reads one stream instead of
two. Document text is escaped, and a document that is not valid UTF-8 is a
fail-closed error rather than a silently corrupt document:

```console
$ leann search --index corpus --query x --format json
error: cannot emit JSON for document: value is not valid UTF-8; use --format text
$ echo $?
1
```

The document is buffered and forwarded only once complete, so that failure
writes **nothing** to stdout rather than a truncated object with the error
interleaved into it. The same corpus remains fully serviceable through
`--format text`, which is byte-preserving.

### `stats`

The same seventeen keys as the text output, in the same order. The example
below is the real output of `leann stats --index corpus --format json` for a
corpus of the five short lines used by `leann build --docs corpus.txt --index
corpus --pq-subquantizers 8` with the default 256-dimension hash embedder.

```json
{
  "nodes": 5,
  "edges": 18,
  "upper_edges": 0,
  "max_level": 0,
  "dimension": 256,
  "approximation": "pq",
  "sketch_bits": 0,
  "approximation_code_bytes": 20,
  "approximation_codebook_bytes": 5120,
  "max_degree": 32,
  "entry_point": 0,
  "index_bytes": 5441,
  "raw_document_bytes": 114,
  "index_over_raw_percent": 4772.807,
  "dense_vector_bytes_avoided": 5120,
  "pair_identity": "49f2b48d8cbcde7e637cd3babbcb2570a2fe63fbf1f9058d065a3bc6ffaa75f9",
  "embedder": "leann-hash-v1:256"
}
```

The ratio is that large only because the example corpus is five short lines;
`docs/BENCHMARK_EVALUATION_PLAN.md` covers measurement at scale.

### `search`

```json
{
  "query": "cat",
  "results": [
    {
      "id": 0,
      "distance": 0.680562,
      "document": "the cat sat on the mat"
    }
  ],
  "metrics": {
    "search_ms": 0.028,
    "exact_recomputations": 5,
    "approximate_distances": 5,
    "expanded_nodes": 5,
    "upper_layer_hops": 0,
    "embedding_batches": 1
  }
}
```

`distance` is the exact cosine distance, never an approximation.

### `build`

```json
{
  "nodes": 5,
  "edges": 18,
  "build_seconds": 0.002,
  "approximation": "pq",
  "index_path": "corpus.leann",
  "index_bytes": 5441,
  "documents_path": "corpus.docs",
  "documents_bytes": 298,
  "dense_vector_bytes_avoided": 5120,
  "pair_identity": "49f2b48d…",
  "embedder": "leann-hash-v1:256"
}
```

### `bench`

The same keys as the text output, including the top-k-dependent recall key
names (`recall_at_10`, `dense_hnsw_recall_at_10`), so a consumer can switch
formats without remapping. The `dense_hnsw_*` keys appear only with
`--dense-baseline 1`.

## `leann doctor`

`doctor` reports what is on disk next to an artifact prefix: whether the pair
is present and validates, which build locks exist, and which temporaries and
backups an interrupted build left behind.

```console
$ leann doctor --index corpus
index: corpus.leann (present)
documents: corpus.docs (present)
pair: valid (identity 49f2b48d…)
lock: corpus.leann.lock held by pid 41207 on host build-01, owner absent
temporary: corpus.leann.tmp.6a17d62f59cd98dc (4440010 bytes)
hint: rerun with --repair to remove what is provably safe to remove
```

### What `--repair` will and will not do

`--repair` removes only what can be shown to be safe, and reports what it
declined and why:

- Nothing at all is removed **while a build lock is present** for the prefix.
  A lock means a build may be mid-transaction: its temporaries are still being
  written, and its backups are load-bearing, because `publish_artifact_pair`
  parks the previous pair in `.bak.*` for the whole transaction and restores
  from exactly those files on rollback.
- With no lock present, a `.bak.*` file is removed **only when the live pair
  itself loads, checksum-verifies, opens, and cross-validates**. This is not
  caution for its own sake: when a publication rolls back and the document
  restore fails, the previous index is deliberately left in its backup rather
  than reactivated against the wrong chunks, so the backup can be the only
  surviving index.
- `--repair` **never removes a build lock.**
- Every condition that can refuse the request is evaluated before anything is
  deleted, so a refusal cannot happen after a removal and discard the record
  of what was removed.

### Why a lock is never called stale

A build lock is a directory created with `mkdir`; the creation is the atomic
acquisition. A descriptor file inside records the pid, host, and start time,
which lets `doctor` say something better than "a lock exists" — but liveness
remains a probe, not a proof: a pid is recycled, a shared filesystem can be
mounted on another host, and a lock created by an older binary has no
descriptor at all. `doctor` therefore reports `running`, `absent`, or
`unknown`, and never `stale`.

Removing a lock is a separate, explicit `--force-unlock`, which refuses when a
lock names a process running on this host:

```console
$ leann doctor --index corpus --force-unlock
error: refusing --force-unlock: a lock names a process that is running on this host
```

Removing a lock that a build still holds would let two builders interleave
their publication transactions. Pair identity would not catch it: identity is
derived from the corpus bytes alone, so two concurrent builds of the *same*
corpus with different `--seed` or `--approx` produce a cross that validates.

### Scope

`doctor --index PREFIX` inspects the index and document-store paths derived
from that prefix. It does not see temporaries written next to a `bench`
`--raw-latencies` target, which normally lives in a results directory.
