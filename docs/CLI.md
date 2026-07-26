# Command-line interface

`leann` has seven commands. `leann --help` lists them, `leann <command> --help`
prints one command's options, and `leann --version` prints the version.

```text
leann build    Embed a corpus, prune the graph, and publish the artifact pair.
leann search   Retrieve the exact top-k for one query.
leann stats    Describe a published artifact pair.
leann bench    Measure recall, latency, and recomputation over a query file.
leann doctor   Report artifact health and leftovers from an interrupted build.
leann pull     Print the exact commands that fetch a published index.
leann verify   Check that an artifact pair loads and matches its manifest.
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
- `search`, `stats`, `doctor`, `pull`, and `verify` are not cancellable; SIGINT
  terminates them the ordinary way. They hold no locks and write no
  temporaries, so there is nothing for an orderly unwind to clean up.

## Structured output

Every command accepts `--format json`. The default remains `text`, and JSON is
strictly opt-in because the benchmark harness parses the text form.

The text output of `search` and `bench` is byte-identical to v0.3; this was
checked by running both binaries over the same index and diffing. `stats` is
byte-identical for its first seventeen lines and then **appends** the embedder
descriptor; the two harness parsers build a dictionary from every line
containing `=`, so appended keys are additive rather than breaking. `build` has
one deliberate change from v0.3: its `index:` and `documents:` lines print the
path unquoted, where v0.3 printed `index: "corpus.leann"` because
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

The same keys as the text output, in the same order. The example below is the
real output of `leann stats --index corpus --format json` after running
`leann build --docs corpus.txt --index corpus --pq-subquantizers 8` over the
first five lines of `samples/documents.txt` with the default 256-dimension
hash embedder.

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
  "index_bytes": 5505,
  "raw_document_bytes": 377,
  "index_over_raw_percent": 1460.212,
  "dense_vector_bytes_avoided": 5120,
  "pair_identity": "eb95bddfad9802f1c9136b2a19dda363304d0437daa993cbe49e81bc3916479c",
  "embedder": "leann-hash-v1:256",
  "model_source": "",
  "model_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
  "model_bytes": 0,
  "pooling_type": 0,
  "context_tokens": 0,
  "document_prefix": "",
  "query_prefix": "",
  "card": {}
}
```

The ratio is that large only because the example corpus is five lines;
`docs/BENCHMARK_EVALUATION_PLAN.md` covers measurement at scale.

The last eight keys are the embedder descriptor. They are empty or zero here
because the hash embedder has no model file and no descriptor options were
given. In text output the card is flattened to one `card_<key>=<value>` line
per entry; in JSON it is a nested object, so a publisher-chosen key can never
collide with a field a consumer already reads by name.

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

## Shareable indexes: `leann pull` and `leann verify`

An index and its document store are two large files that only work with the
model that built them. `build --model-source` and `--card` record where that
model came from and whatever else the publisher wants the artifact to say;
`pull` and `verify` are the download side of the same story.

### `leann pull`

`pull` prints commands. It opens no socket, resolves no DNS, and downloads
nothing:

```console
$ leann pull hf:toreleon/llama-cpp-docs.leann
repo: toreleon/llama-cpp-docs.leann (model revision main)
step 1 — fetch the manifest:
  curl -fL --retry 3 -o ./leann.manifest 'https://huggingface.co/toreleon/llama-cpp-docs.leann/resolve/main/leann.manifest'
step 2 — print the checked plan:
  leann pull hf:toreleon/llama-cpp-docs.leann --manifest ./leann.manifest
```

With the manifest in hand it prints the plan that can actually be checked —
every file, its size, its expected digest, and the command that ends the job:

```console
$ leann pull hf:toreleon/llama-cpp-docs.leann --manifest leann.manifest
repo: toreleon/llama-cpp-docs.leann (model revision main)
prefix: llama-cpp-docs
total_bytes: 1364441
model: hf:nomic-ai/nomic-embed-text-v1.5-GGUF/nomic-embed-text-v1.5.Q4_K_M.gguf (84106624 bytes, sha256:d4e388894e09cf3816e8b0896d81d265b55e7a9fff9ab03fe8bf4ef5e11295ac)
download:
  curl -fL --retry 3 -o ./llama-cpp-docs.docs 'https://huggingface.co/toreleon/llama-cpp-docs.leann/resolve/main/llama-cpp-docs.docs'
  # 1231453 bytes, sha256:c3271987fe3f4a98e1e4b5fd5d6da3b75e055d04da38ee6d975ad0886dd37319
  ...
then:
  leann verify --index ./llama-cpp-docs --manifest leann.manifest
```

`-L` is not optional in those commands: a Hugging Face `resolve` URL answers
with a redirect to a CDN. `-f` turns an HTTP error into a nonzero exit instead
of a file full of error markup that would then fail a digest check for the
wrong reason.

Printing commands rather than running them is what keeps the binary free of an
HTTP client, a TLS dependency, and a retry policy. The cost is stated plainly:
`pull` on its own verifies nothing at all. `verify` is the step that does.

`hf:datasets/OWNER/NAME` addresses a dataset repository; the bare form is a
model repository. A manifest whose `type` disagrees with the form that was
typed is refused rather than allowed to win — on the Hub `owner/name` as a
model and `owner/name` as a dataset are different repositories, so silently
switching would print downloads from one the operator did not ask for.

### `leann verify`

`verify` is a gate, not a report. `doctor` prints what it finds and exits 0
even when a pair is unusable, because it is a diagnostic. `verify` exits 1 on
any failure so a script can branch on it, and it never modifies anything:

```console
$ leann verify --index llama-cpp-docs --manifest leann.manifest
pair: valid (identity 4f62583567c94fa356eefbd093c96019252b474ad0cb8c8406f202b2dbd84e30)
model_source: hf:nomic-ai/nomic-embed-text-v1.5-GGUF/nomic-embed-text-v1.5.Q4_K_M.gguf
file: llama-cpp-docs.docs ok
file: llama-cpp-docs.leann ok
verify: ok
$ echo $?
0
```

Without `--manifest` it reports only that the pair loads and cross-validates,
and says `manifest: not checked`. With one it also compares every recorded
size and SHA-256 against the files beside `--index`, and cross-checks the
manifest's `model` record against the model the index records. If those
disagree, following the manifest would fetch a model the index refuses on the
first query; saying so during verification is more useful than a fingerprint
mismatch later.

The two halves catch different things. Index loading verifies SHA-256 over the
whole index and the document store's metadata region, but a document store's
payload is checked per chunk only when a chunk is read — so a pair with an
edited document body still loads. The manifest digest is what catches that.

A manifest whose `prefix` record disagrees with `--index` is refused rather
than checked, because comparing one pair against another pair's digests would
report a mismatch that reads exactly like corruption.

### The `LEANNMF1` manifest

Line-oriented and tab-separated, with a magic first line:

```text
LEANNMF1
repo	toreleon/llama-cpp-docs.leann
type	model
revision	main
prefix	llama-cpp-docs
file	llama-cpp-docs.docs	1231453	sha256:c327...
file	llama-cpp-docs.leann	132988	sha256:2323...
model	hf:nomic-ai/...Q4_K_M.gguf	84106624	sha256:d4e3...
```

It is not JSON because the repository takes no JSON parser dependency, and
because a grammar this small can be rejected precisely. Parsing is fail-closed
with no lenient mode: an unknown key, a duplicate singleton record, a blank
line, a CRLF line ending, an uppercase or wrong-length digest, a non-decimal
size, a name containing `..` or a leading `/`, and a `prefix` whose `.leann`
and `.docs` are not both listed are each an error. An older binary therefore
never half-understands a newer manifest.

The `repo`, `revision`, and `prefix` records and every `file` name are held to
an allowlist — letters, digits, and `.` `_` `-` `/`, and no `..` — on both the
reading and the writing side. The dot-segment rule is not cosmetic: these
values become path components of the resolve URL, and `curl` normalizes dot
segments before sending the request, so a `revision` of
`../../other/repo/resolve/main` would print a command that downloads from a
repository the operator never named while every other line still said
otherwise. That is stricter than it needs to be for a URL, and
deliberately so: `pull` writes these values into a single-quoted shell argument
that a person pastes into a terminal, and an apostrophe in a repository name
would end the quoting and turn the rest of the line from data into shell.
Escaping would also work; refusing is easier to be sure about, and no real
repository ID, git revision, or artifact name needs anything else. The `model`
record's origin string takes a weaker rule — it legitimately contains a colon,
as in `hf:owner/repo/file.gguf`, and is only ever printed as prose — but still
rejects control characters.

`--dest` is a local path rather than a URL component, so it may contain
spaces; it is quoted in the printed command and rejects only quotes and
control characters.

Like `LEANNBC2`, a manifest is not self-authenticating. It is unsigned, there
is no trust root and no revocation, and the digests are only as trustworthy as
the channel that delivered the manifest. A match establishes that the bytes are
the ones the publisher recorded — not that the publisher is trustworthy, and
not that the index is any good.

`scripts/publish_hf_index.py pack` writes the manifest, and the C++ reader and
the Python writer are deliberately strict about the same grammar so the two
cannot drift.

### Descriptor options on `build`

```text
--model-source URI      model origin recorded in the index, e.g. hf:OWNER/REPO/FILE
--document-prefix TEXT  prepended to every chunk before embedding, and on rerank
--query-prefix TEXT     prepended to the query at search time
--card KEY=VALUE        artifact card entry; repeatable
```

`--card` is the only repeatable option. Its key must be unique, non-empty, and
free of control characters, and both halves must be valid UTF-8 — a card entry
is reproduced in `stats` output, in a tab-separated manifest, and in a printed
shell command, so a newline in one would forge a record rather than corrupt it.

The prefixes travel inside the index and are applied automatically, so a
downloaded artifact is queried the way its publisher built it without the
caller having to know. Two combinations are refused rather than silently
accepted, because in both the vectors already exist and cannot be re-prefixed:

```console
$ leann build --docs corpus.txt --index corpus --embedder cache \
    --embedding-cache corpus.bc2 --document-prefix 'search_document: '
error: --document-prefix cannot be applied to --embedder cache vectors, which were already computed; bake the prefix into the cache instead

$ leann bench --index prefixed --queries q.txt --query-embedding-cache q.bc2
error: --query-embedding-cache cannot be used with an index that has a query prefix; the cached vectors do not record one
```

Both would otherwise produce no error at all — just quietly worse recall,
which is the failure mode that reads as success.

`bench --ground-truth` is deliberately **not** guarded the same way, and the
asymmetry is worth stating. The two caches carry a binding — an embedder
fingerprint and the source file's SHA-256 — so they look verified while that
binding says nothing about prefixes; that is the trap worth closing. A
`LEANN_GT1` file carries no binding at all: it is a header and a list of
neighbour IDs, with no fingerprint, no model identity, and no corpus digest.
It has always been an operator assertion that the truth was computed for this
index with this embedder, and a prefix is one more thing that assertion now
covers. Recompute ground truth when you change a prefix.
