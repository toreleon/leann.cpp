# Contributing to leann.cpp

Thanks for looking. This file covers what you need to build the project, get a
change reviewed, and avoid the handful of mistakes that will get a pull request
sent back.

## What this project is, and is not

`leann.cpp` is a C++20 **research spike** for low-storage local RAG: build an
HNSW graph, discard the dense document vectors, keep only a pruned CSR graph
plus trained PQ codes, and recompute the embeddings of selected candidates at
query time through in-process llama.cpp.

It is an independent LEANN-style implementation, **not** an accuracy-compatible
port of official LEANN. Its selling point is a zero-Python, single-process
deployment.

Two consequences for contributors:

- `README.md`'s "Honest spike boundaries" section is deliberate. Changes that
  quietly widen a claim are a bigger problem here than a missing feature.
- `DESIGN.md` is the authoritative record of binary layouts and invariants.
  **Read it before touching any format or the search path.**

## Prerequisites

- A C++20 compiler. The floor is a standard library providing `std::bit_cast`,
  `std::span`, and `std::ranges`: libstdc++ 11 (GCC 11) or libc++ 14.
- CMake 3.20+ **and** network access, for the default build (it fetches a pinned
  hnswlib via `FetchContent`).
- Python 3 with `numpy`, only if you touch the `scripts/` benchmark harness.

CI runs `ubuntu-latest` and `macos-latest`. Other platforms are unmeasured —
if you build somewhere else, say so in your PR, because we cannot verify it.

## Build and test

There are two build paths on purpose. Use whichever fits what you are changing.

### CMake — the canonical path

Fetches a pinned hnswlib. This is what CI runs, so your change must pass here.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

### Makefile — fast iteration, no network

Needs an hnswlib checkout you already have. It **cannot** build the llama.cpp
path, so it is for library, CLI, and test work.

```bash
make HNSWLIB_DIR=/path/to/hnswlib check
```

`check` runs every suite. Individual targets: `test`, `persistence-test`,
`core-safety-test`, `c-api-test`, `cli-cache-test`.

### With the native GGUF embedder

```bash
cmake -S . -B build-llama -DCMAKE_BUILD_TYPE=Release \
  -DLEANN_ENABLE_LLAMA=ON -DLEANN_LLAMA_CPP_SOURCE_DIR=/path/to/llama.cpp
cmake --build build-llama -j
```

### Sanitizers

The core and persistence suites are expected to be clean. Two Makefile quirks
bite here: a command-line `CXXFLAGS` **replaces** the makefile's, so
`-std=c++20` must be repeated; and the link rule uses only `LDLIBS`, so the
sanitizer flags have to be passed there too.

```bash
make HNSWLIB_DIR=/path/to/hnswlib BUILD_DIR=build-asan \
  CXXFLAGS="-std=c++20 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer" \
  LDLIBS="-pthread -fsanitize=address,undefined" \
  core-safety-test persistence-test
```

### Python harness

Run from the repository root — the tests `sys.path`-insert `scripts/`.

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

## Source map

Four layers, each with a different dependency budget. Respect the budget.

- **`src/` + `include/leann/` — the library (`leann::leann`).** May depend on
  hnswlib (builder only) and optionally llama.cpp. `Index::build` is offline
  and uses hnswlib; nothing hnswlib-shaped survives into runtime artifacts.
- **`app/main.cpp` — the CLI.** Depends on the library. One large file: the
  `Arguments` parser, the `command_specs` option tables, a `JsonWriter`, the
  cache and ground-truth readers, and the benchmark loop.
- **`include/leann/leann.h` + `src/c_api.cpp` — the versioned C11 ABI.**
  Read/search only. Nothing C++-shaped may appear in the header.
- **`scripts/` + `tests/test_*.py` — the benchmark harness.** Python stdlib
  plus numpy, nothing heavier. Deliberately fail-closed: it rejects an
  incomplete report rather than interpolating.

SHA-256 and CRC32C are implemented in-repo (`src/checksum.cpp`). **Do not add a
crypto or hashing dependency.** More generally, no new third-party dependency
beyond hnswlib and llama.cpp without a strong reason; both are pinned by commit
in `CMakeLists.txt`.

## Code conventions

- C++20. 4-space indent, 80-column wrapping.
- Warning-clean under `-Wall -Wextra -Wpedantic`. Validation runs add `-Werror`.
- Public API in `namespace leann`, internals in `namespace leann::detail`,
  file-local helpers in anonymous namespaces.
- `[[nodiscard]]` on value-returning queries.
- Errors are exceptions whose messages name the artifact or option involved.
  **Tests match on message substrings, so reword them deliberately.**
- Comments should explain *why*, not restate the code.

An `.editorconfig` is provided; most editors will pick up the indent and
final-newline rules automatically.

## Test conventions

There is **no test framework** — no gtest, no dependency. A suite is a plain
executable:

```cpp
void check(bool condition, const std::string & message);

int main() {
    check(some_condition, "what should have been true");
    std::cout << "all leann.cpp <suite> tests passed\n";
}
```

Python tests are plain `unittest`.

**A new suite must be registered in BOTH build systems.** This is the single
most common omission:

1. `CMakeLists.txt` — an `add_executable(...)` *and* a matching
   `add_test(...)`.
2. `Makefile` — a target, plus adding it to the `check` aggregate.

Miss the Makefile and your suite silently never runs on the CMake-free path;
miss CMake and it never runs in CI.

`app/main.cpp` is compiled **twice**: once as the CLI, and once by
`tests/test_cli_cache.cpp`, which does `#define main leann_cli_entry_for_test`
and then `#include "../app/main.cpp"`. New CLI code must stay warning-clean in
both translation units. Keep new CLI logic in small free functions in the
anonymous namespace rather than inlining it into a `command_*` function.

## Invariants that will get a PR rejected

These are actively defended by `leann_persistence_tests` and
`leann_core_safety_tests`. If you deliberately change one, update `DESIGN.md`
and `VALIDATION.md` in the same PR and say so explicitly.

- **Artifact pairing.** A `.leann` index and its `.docs` store share one
  deterministic corpus identity. A mixed pair must be rejected before any query
  embedding or corpus scan.
- **Bounded startup verification.** Index load verifies SHA-256 over the whole
  compact index; document-store open verifies only header, offsets, and the
  checksum table. A 100 GB corpus must not be scanned at startup. Each fetched
  chunk is CRC32C-checked on read.
- **Fail-closed publication.** Adjacent `.lock` directories, unique
  `.tmp.*`/`.bak.*` names, validate the complete temporary pair, publish
  documents first, rename the index **last** as the commit marker, roll back
  documents-before-index on error. This is fail-closed, not power-loss atomic.
- **Versioned formats.** Formats reject older artifacts. A format change means
  bumping magic + version, updating the `DESIGN.md` tables, and stating the
  migration boundary — never silently accepting both shapes. `DESIGN.md` holds
  the current magic values; do not hardcode them elsewhere in docs.
- **No non-finite values.** NaN and infinity are rejected before reaching an
  integer conversion, heap insert, or sort — in build/search ratios, query
  vectors, backend embeddings, PQ intermediates, and exact distances.
  Tiny-but-positive rerank ratios must saturate the beam at index size rather
  than overflow, which means subnormals have to *parse*, not be refused.
- **CLI option tables.** Every option a command reads must have an entry in that
  command's table in `command_specs`. Validation rejects anything else, so an
  `args.get()` without a table entry becomes an unreachable option.
- **Stable CLI text output.** The Python harness parses the text output of every
  command. JSON is strictly opt-in via `--format json`; the text form must stay
  byte-identical.
- **C11 header purity.** `include/leann/leann.h` must stay valid C11 with no C++
  or llama.cpp types. `tests/test_c_header.c` compiles it as strict C11 as the
  proof. Document ABI changes in `docs/C_API.md`.
- **Shared-read concurrency.** `Index` is immutable after load and
  `DocumentStore::read`/`read_many` are const and mutex-serialized, so workers
  may share one loaded pair — but each worker needs its own `Embedder`. Do not
  add hidden serialization to the `Embedder` interface.
- **Nothing process-global outside `main`.** No signal handlers, no `exit()`
  calls installed from library or helper code.

## Evidence discipline

The docs deliberately separate "measured here" from "claimed by the paper".

- `VALIDATION.md` records what was actually run, on what hardware, with which
  GGUF and llama.cpp revision.
- `README.md`'s "Honest spike boundaries" lists what this code does *not* do.

So: **do not** quote LEANN paper numbers, extrapolate a benchmark to another
corpus, or add a comparison table without a measurement behind it. If you have
the measurement, add it to `VALIDATION.md`. If you do not, say it is unmeasured.
A PR that states a performance or storage claim with no backing will be asked to
remove the claim.

## Before you push

```bash
# 1. Canonical build and tests
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure

# 2. Sanitizers, if you touched the library or search path
make HNSWLIB_DIR=/path/to/hnswlib BUILD_DIR=build-asan \
  CXXFLAGS="-std=c++20 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer" \
  LDLIBS="-pthread -fsanitize=address,undefined" \
  core-safety-test persistence-test

# 3. Python harness, if you touched scripts/
python3 -m unittest discover -s tests -p 'test_*.py'

# 4. No new lines over 80 columns
awk 'length > 80 {print FILENAME":"FNR}' <files you changed>
```

## Pull requests

- Keep it focused. One reviewable change beats a broad one.
- Say what you **verified** and what you **did not**. "Builds on my Mac, not
  tested on Linux" is a useful sentence; silence is not.
- If you changed an exception message, mention it — a test probably matches on
  its substring.
- If you changed a format or an invariant above, update `DESIGN.md` and
  `VALIDATION.md` in the same PR.

## License

MIT. See `LICENSE`. By contributing you agree your contributions are licensed
under the same terms.
