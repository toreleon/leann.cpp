# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`leann.cpp` is a C++20 research spike for low-storage local RAG: build an HNSW
graph with hnswlib, throw away the dense document vectors, persist only a pruned
CSR graph plus trained PQ codes, then recompute the embeddings of selected
candidates at query time through in-process llama.cpp. It is an independent
LEANN-style implementation, **not** an accuracy-compatible port of official
LEANN, and its selling point is a zero-Python, single-process deployment.

Read `DESIGN.md` before touching formats or the search path — it is the
authoritative record of binary layouts and invariants.

## Build and test

Canonical build (CMake fetches a pinned hnswlib; no llama.cpp):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Single C++ test: `ctest --test-dir build -R leann_persistence_tests --output-on-failure`
(targets: `leann_tests`, `leann_persistence_tests`, `leann_core_safety_tests`,
`leann_c_api_tests`, `leann_c_header_tests`, `leann_cli_cache_tests`). Each test
is a standalone executable, so `./build/leann_persistence_tests` also works.

With the native GGUF embedder:

```bash
cmake -S . -B build-llama -DCMAKE_BUILD_TYPE=Release \
  -DLEANN_ENABLE_LLAMA=ON -DLEANN_LLAMA_CPP_SOURCE_DIR=work/reference/llama.cpp
cmake --build build-llama -j
```

Fast iteration without CMake (needs an hnswlib checkout; cannot build the llama
path):

```bash
make HNSWLIB_DIR=work/reference/hnswlib
make HNSWLIB_DIR=work/reference/hnswlib check
```

`check` runs every suite, including `cli-cache-test`. Individual targets are
`test`, `persistence-test`, `core-safety-test`, `c-api-test`, and
`cli-cache-test`.

Sanitizers (the core and persistence suites are expected to be clean). Note the
two Makefile quirks: a command-line `CXXFLAGS` **replaces** the makefile's, so
`-std=c++20` must be repeated, and the link rule uses only `LDLIBS`, so the
sanitizer flags have to be passed there as well:

```bash
make HNSWLIB_DIR=work/reference/hnswlib BUILD_DIR=build-asan \
  CXXFLAGS="-std=c++20 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer" \
  LDLIBS="-pthread -fsanitize=address,undefined" \
  core-safety-test persistence-test
```

Python harness tests (needs `numpy`; run from the repo root — the tests
`sys.path`-insert `scripts/`):

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
python3 -m unittest tests.test_collect_large_scale_results -v
python3 -m unittest tests.test_benchmark_tools.BenchmarkToolsTest.test_tie_stable_topk
```

CI (`.github/workflows/ci.yml`) runs the Python suite, then a CMake Release
build and `ctest`, on ubuntu-latest and macos-latest.

## Architecture

Four layers, each with a different dependency budget:

- **`src/` + `include/leann/` — the library (`leann::leann`).** Depends only on
  hnswlib (builder only) and optionally llama.cpp. SHA-256 and CRC32C are
  implemented in-repo (`src/checksum.cpp`); do not add crypto/hashing
  dependencies. `Index::build` (offline) uses hnswlib; nothing hnswlib-shaped
  survives into the runtime artifacts. `Index::search_embedding` is the hot path:
  PQ ADC table → greedy descent through retained upper layers → flat ADC below
  `approximate_scan_limit` (100k nodes) or a bounded base-graph beam above it →
  read the selected raw chunks → batched embedder recomputation → exact cosine
  top-k. Returned distances are always exact, never approximate.
- **`app/main.cpp` — the CLI** (`build`, `search`, `stats`, `bench`,
  `doctor`). One ~3000 line file: an `Arguments` parser, the `command_specs`
  option tables, a `JsonWriter`, the `LEANNBC2` embedding-cache and
  `LEANN_GT1` ground-truth readers, and the benchmark loop all live in an
  anonymous namespace here. `tests/test_cli_cache.cpp` does
  `#define main leann_cli_entry_for_test` then `#include "../app/main.cpp"`, so
  this file is compiled twice and its internal helpers are directly unit-tested.
  Keep new CLI logic as small free functions in that anonymous namespace rather
  than inlining it into `command_*`, and keep it warning-clean under a second
  translation unit. **Every option a command reads must have an entry in that
  command's table in `command_specs`** — validation rejects anything else, so
  an `args.get()` without a table entry becomes an unreachable option. Install
  nothing process-global (signal handlers, exit calls) outside `main`. See
  `docs/CLI.md` for the option, JSON, cancellation, and `doctor` contracts.
- **`include/leann/leann.h` + `src/c_api.cpp` — the versioned C11 ABI**
  (`LEANN_C_API_VERSION`). Read/search only: opaque `leann_searcher` /
  `leann_results` handles, a caller-owned `leann_embed_batch_fn`, owned result
  bytes, thread-local error text, `leann_status` returns with no exceptions
  escaping. The header must stay valid C11 with no C++ or llama.cpp types —
  `tests/test_c_header.c` compiles it as strict C11 as the proof. Document
  changes in `docs/C_API.md`.
- **`scripts/` + `tests/test_*.py` — the publication benchmark harness.**
  Dependency-light Python (stdlib + numpy) that prepares BEIR/NQ tiers, caches
  embeddings, computes exact ground truth, orchestrates runs against both
  leann.cpp and pinned official LEANN, attests the live embedding endpoint, and
  collects results. It is deliberately fail-closed: `collect_large_scale_results.py`
  rejects a report with a missing sweep point, artifact hash, parity check, or
  ranked result ID and never interpolates. `docs/BENCHMARK_EVALUATION_PLAN.md`
  is the operational plan; `README.md` has the command sequence.

## Invariants that tests actively defend

Breaking any of these should surface in `leann_persistence_tests` or
`leann_core_safety_tests`; if you change one, update `DESIGN.md` and
`VALIDATION.md` too.

- A `.leann` index and its `.docs` store are a **pair** sharing one deterministic
  256-bit corpus identity derived from ordered chunk lengths and bytes. A mixed
  pair must be rejected before any query embedding or corpus scan.
- Index load verifies SHA-256 over the whole compact index. Document-store open
  verifies only header/offsets/checksum table (a 100 GB corpus must not be
  scanned at startup); each fetched chunk is CRC32C-checked on read.
- Build publication is fail-closed, not power-loss atomic: adjacent `.lock`
  directories, unique `.tmp.*`/`.bak.*` names, validate the complete temporary
  pair, publish documents first, rename the index **last** as the commit marker,
  roll back documents-before-index on error. If the pair commits but cleanup
  fails, report that distinct "committed; cleanup required" state.
- Formats are versioned and reject older artifacts (`LEANNC03` / `LEANDC02`).
  A format change means bumping magic + version, updating the `DESIGN.md` tables,
  and stating the migration boundary — never silently accepting both shapes.
- NaN/infinity is rejected before it can reach an integer conversion, heap
  insert, or sort — in build/search ratios, query vectors, backend embeddings,
  PQ intermediates, and exact distances. Tiny-but-positive rerank ratios saturate
  the beam at index size instead of overflowing.
- Build and search embedder fingerprints must match. The fingerprint includes
  `--gpu-layers` (CPU and Metal embeddings are not bit-identical) but is a
  configuration guardrail, not a cryptographic model/backend hash.
- `Index` is immutable after load and `DocumentStore::read`/`read_many` are const
  and mutex-serialized, so workers may share one loaded pair — but each worker
  needs its own `Embedder`. Don't add hidden serialization to the `Embedder`
  interface.
- `--embedder cache` (streamed `LEANNBC2`) is build-only and must stay rejected
  by `search` and `bench`; queries need the real matching embedder.
- A cancelled build publishes nothing and leaves no `.lock`, `.tmp.*`, or
  `.bak.*`. The last cancellation check is before `publish_artifact_pair`, so
  cancellation is never observed inside the publication transaction.
- `doctor --repair` removes a `.tmp.*` only when no lock is present and a
  `.bak.*` only when the live pair loads and validates — a backup can be the
  only surviving index after a rolled-back publication. `--repair` never
  removes a lock; `--force-unlock` does, and refuses when the lock descriptor
  names a process running on this host. A lock is never reported as "stale":
  liveness is `running`, `absent`, or `unknown`.
- The text output of every command is what the Python harness parses, so JSON
  is strictly opt-in and the text form must stay byte-identical.

## Conventions

- C++20, 80-column wrapping, 4-space indent, `-Wall -Wextra -Wpedantic` clean
  (validation runs add `-Werror`). Public API in `namespace leann`, internals in
  `namespace leann::detail`, file-local helpers in anonymous namespaces.
- `[[nodiscard]]` on value-returning queries; errors are exceptions with specific
  messages naming the artifact/option (tests match on message substrings, so
  reword them deliberately).
- C++ tests use a local `check(condition, message)` helper and `int main()` — no
  gtest, no test framework dependency. Python tests are plain `unittest`.
- No new third-party dependencies beyond hnswlib and llama.cpp without a strong
  reason; both are pinned by commit in `CMakeLists.txt`.

## Division of labor across models

Standing preference for this repo — Opus orchestrates and codes, Sonnet absorbs
volume, Codex reviews. Delegating along these lines is pre-authorized; no need
to ask first.

- **Opus — orchestration and implementation.** Both halves: planning, design
  taste, and context assembly (decomposing the task, deciding what gets handed
  off, judging whether the result reads right) *and* the actual coding. Opus
  writes the diff it planned.
- **Sonnet — exploration and token-heavy work.** Search fan-out across many
  files, log and transcript trawls, digesting benchmark output or the 216 KB
  `collect_large_scale_results.py` — anything where the deliverable is a
  conclusion and the cost is input volume.
- **Codex `gpt-5.6-sol` — peer.** Independent review, plan critique, and a
  second opinion on output. Treat it as a peer whose verdict is worth weighing,
  not an oracle to accept unread.

Mechanics, verified against the installed plugin:

- Model override on the Agent tool: `model: "opus" | "sonnet"`. Note that
  `subagent_type: "fork"` ignores `model` — a fork always inherits the caller's
  model, so use a named agent type when the point is to switch models.
- Codex work handoff: `Agent(subagent_type: "codex:codex-rescue", prompt:
  "--model gpt-5.6-sol <task>")`, or `/codex:rescue --model gpt-5.6-sol <task>`.
  The forwarder leaves `--model` unset unless a name is given explicitly, and
  passes anything other than `spark` straight through to the Codex CLI — so an
  unrecognized model name fails there, not here.
- `/codex:review` and `/codex:adversarial-review` are marked
  `disable-model-invocation`, so they are **user-typed only**. Ask for one; do
  not try to launch a Codex review from here. `codex:codex-rescue` likewise only
  forwards `task`, never `review`.
- `/codex:status` polls a running job and `/codex:result <job-id>` prints a
  finished job's stored output; both are also user-typed.

## Evidence discipline

The docs deliberately separate "measured here" from "claimed by the paper".
`VALIDATION.md` records what was actually run, on what hardware, with which GGUF
and llama.cpp revision; `README.md`'s "Honest spike boundaries" lists what this
code does *not* do. Do not quote LEANN paper numbers, extrapolate a benchmark to
another corpus, or add a comparison table without a measurement to back it —
add the measurement to `VALIDATION.md` instead, or say it is unmeasured.

## Machine-local assets (gitignored)

`work/` holds research material that is not part of the repository and may exist
only on this machine: `work/reference/{hnswlib,llama.cpp,LEANN}` checkouts,
`work/models/nomic-embed-text-v1.5.Q4_K_M.gguf`, and prepared corpora under
`work/datasets/`. Prefer these over re-downloading, but verify they exist first
and never assume them in committed code, scripts, or CI.
