# GPU-Server Benchmark Evaluation Plan

## Purpose

This is the authoritative execution plan for a fair, launch-grade comparison
between `leann.cpp` and the pinned official
[StarTrail-org/LEANN](https://github.com/StarTrail-org/LEANN) HNSW backend.

The comparison uses deterministic 100K and 1M BEIR Natural Questions tiers,
one GGUF model artifact, identical corpus and query vectors, identical exact
top-10 ground truth, and the same physical GPU server for every timed run.

Do not combine the partial macOS measurements under `work/bench-nq` with
GPU-server measurements. They are useful development evidence, but they are
not part of the final cross-system comparison.

## Required result

The final publication matrix contains all six profiles:

| Tier | Native `leann.cpp` | Official cached recompute | Official real GGUF recompute |
|---|---|---|---|
| 100K | EF 32, 64, 128, 256, 512 | complexity 32, 64, 128, 256, 512; batch 0 and 16 | complexity 32, 64, 128, 256, 512; batch 16 |
| 1M | EF 32, 64, 128, 256, 512 | complexity 32, 64, 128, 256, 512; batch 0 and 16 | complexity 32, 64, 128, 256, 512; batch 16 |

Each search sweep uses one warmup and three measured repetitions over all 300
queries. Recall is reported at 3 and 10 from one top-10 search.

The cached official profile is used only for the algorithmic
recall/candidate-recomputation Pareto curve. Its latency is not comparable with
real inference. Headline latency comes only from native real GGUF recomputation
versus official real GGUF recomputation.

## Fairness contract

The run is publishable only when all of these conditions hold:

1. Both implementations run on the same otherwise-idle GPU server.
2. Both consume the same exact-source-bound corpus and query caches.
3. Both use the same top-10 exact ground-truth artifact.
4. Native and official real recomputation use the same GGUF bytes and pinned
   llama.cpp revision.
5. Query embedding is excluded from both search timers through the shared query
   cache.
6. Candidate document recomputation, graph traversal, and exact reranking are
   included in both real search paths.
7. Official LEANN uses its pinned package-local FAISS/HNSW extension, not a
   separately installed `faiss-cpu` substitute.
8. The final matched pair has Recall@3 at least 0.90 and an absolute Recall@3
   gap no greater than 0.01.
9. Native/reference embedding cosine similarity is at least 0.9999.
10. The strict collector verifies the complete matrix, repetitions, ranked
    result IDs, runtime identity, artifact hashes, parity reports, and endpoint
    attestations.

System architecture overhead remains part of the end-to-end comparison:
`leann.cpp` recomputes in-process, while official LEANN uses its supported
ZMQ/OpenAI-compatible path. Report this distinction with the results.

## Frozen identities

Start from these revisions and artifacts:

| Component | Required identity |
|---|---|
| `leann.cpp` | freeze and record the clean GPU-run commit before preflight; the current code baseline is `be56e846d0ec7c1069c606a9823838fc67c91b32` |
| Official LEANN | `7a34d8856b7aa92da47097af02e8f26b341a90a3` |
| llama.cpp | `c588c4f47683e73ad2d69f50480bec6cc85fd0f7` |
| Model | `nomic-embed-text-v1.5.Q4_K_M.gguf` |
| Model SHA-256 | `d4e388894e09cf3816e8b0896d81d265b55e7a9fff9ab03fe8bf4ef5e11295ac` |
| BEIR NQ ZIP MD5 | `d4d3d2e48787a744b6f6e691ff534307` |
| Documents | deterministic nested 100K and 1M tiers |
| Queries | 300 deterministic positive-qrel test queries |

If `leann.cpp` changes before the run, freeze the new commit before generating
any timed evidence and record the change in the final report. Do not change
source, compiler flags, model, dataset, or benchmark scripts between profiles.

## GPU-server assumptions

The strict current attestation requires the benchmark harness, llama-server
process, and absolute GGUF path to exist on the same Linux host. A remote
OpenAI-compatible endpoint running on another machine is suitable only for a
development report because the harness cannot prove that remote process or
resolve its model path.

Recommended minimum:

- Linux x86-64 or ARM64 with an NVIDIA CUDA-capable GPU
- 48 GB system RAM; 64 GB or more preferred for the 1M official build
- 50 GB free workspace storage; 100 GB preferred for logs and retained evidence
- Python 3.10–3.12
- CMake 3.24+, a C++20 compiler, Ninja, Git, SWIG, `pkg-config`, ZeroMQ
  development headers, and CUDA toolkit
- `curl`, `jq`, `sha256sum`, `md5sum`, and `nvidia-smi`

CUDA does not accelerate the current native HNSW construction loop. It
accelerates cache generation and real GGUF candidate recomputation. Native and
official index builds remain primarily CPU-bound, so CPU model and clock must
be reported alongside the GPU.

## 1. Define paths

Run every command from the repository root in one shell or a persistent
terminal multiplexer session.

```bash
set -euo pipefail

export REPO_ROOT="$(pwd -P)"
export PYTHON_BIN="python3"
export CPU_THREADS="$(nproc)"
export LEANN_CPP_COMMIT="$(git rev-parse HEAD)"
export MODEL_PATH="$REPO_ROOT/work/models/nomic-embed-text-v1.5.Q4_K_M.gguf"
export LLAMA_CPP_DIR="$REPO_ROOT/work/reference/llama.cpp"
export LLAMA_SERVER_BUILD="$REPO_ROOT/work/llama-server-build"
export NATIVE_BUILD="$REPO_ROOT/build-gpu"
export OFFICIAL_REPO="$REPO_ROOT/work/reference/LEANN"
export OFFICIAL_VENV="$REPO_ROOT/work/official-leann-venv"
export OFFICIAL_PY="$OFFICIAL_VENV/bin/python"
export EMBEDDING_URL="http://127.0.0.1:18080"
export REAL_PROXY_URL="http://127.0.0.1:18081"
export CACHED_URL="http://127.0.0.1:18082"
export EMBED_FINGERPRINT="llama.cpp-v1:nomic-bert 137M Q4_K - Medium:83349984:768:1:gpu-layers=99"
export MODEL_IDENTITY="nomic-embed-text-v1.5.Q4_K_M.gguf sha256=d4e388894e09cf3816e8b0896d81d265b55e7a9fff9ab03fe8bf4ef5e11295ac llama.cpp=c588c4f47683e73ad2d69f50480bec6cc85fd0f7"
```

## 2. Preflight and provenance

Do not begin if another process is using the selected GPU or if the filesystem
does not have enough space.

```bash
mkdir -p work/bench-nq/gpu-server work/reference work/models

{
  date --iso-8601=seconds
  uname -a
  lscpu
  free -h
  df -h "$REPO_ROOT"
  nvidia-smi
  "$PYTHON_BIN" --version
  cmake --version
  c++ --version
  printf 'leann.cpp commit: %s\n' "$LEANN_CPP_COMMIT"
} | tee work/bench-nq/gpu-server/preflight.txt

test "$(git rev-parse HEAD)" = "$LEANN_CPP_COMMIT"

sha256sum "$MODEL_PATH"
test "$(sha256sum "$MODEL_PATH" | awk '{print $1}')" = \
  "d4e388894e09cf3816e8b0896d81d265b55e7a9fff9ab03fe8bf4ef5e11295ac"
```

The working tree may contain copied datasets or ignored benchmark outputs, but
tracked source must be clean:

```bash
test -z "$(git status --porcelain --untracked-files=no)"
```

Place the exact GGUF at `$MODEL_PATH` before preflight. The commit recorded in
`$LEANN_CPP_COMMIT` is the frozen native identity for the whole run; do not
change it between profiles.

## 3. Pin and build llama.cpp

```bash
git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA_CPP_DIR"
git -C "$LLAMA_CPP_DIR" checkout \
  c588c4f47683e73ad2d69f50480bec6cc85fd0f7

cmake -S "$LLAMA_CPP_DIR" -B "$LLAMA_SERVER_BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF

cmake --build "$LLAMA_SERVER_BUILD" \
  --target llama-server -j "$CPU_THREADS"
```

If the pinned checkout already exists, verify its commit instead of cloning:

```bash
test "$(git -C "$LLAMA_CPP_DIR" rev-parse HEAD)" = \
  "c588c4f47683e73ad2d69f50480bec6cc85fd0f7"
```

## 4. Build and test `leann.cpp` with CUDA

```bash
cmake -S "$REPO_ROOT" -B "$NATIVE_BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLEANN_ENABLE_LLAMA=ON \
  -DLEANN_LLAMA_CPP_SOURCE_DIR="$LLAMA_CPP_DIR" \
  -DGGML_CUDA=ON \
  -DLEANN_BUILD_TESTS=ON

cmake --build "$NATIVE_BUILD" -j "$CPU_THREADS"
ctest --test-dir "$NATIVE_BUILD" --output-on-failure

"$NATIVE_BUILD/leann" --help
sha256sum "$NATIVE_BUILD/leann" \
  "$LLAMA_SERVER_BUILD/bin/llama-server"
```

## 5. Install pinned official LEANN

Do not install a standalone FAISS package. Build and import the official
package-local extension.

```bash
git clone --recursive \
  https://github.com/StarTrail-org/LEANN.git "$OFFICIAL_REPO"
git -C "$OFFICIAL_REPO" checkout \
  7a34d8856b7aa92da47097af02e8f26b341a90a3
git -C "$OFFICIAL_REPO" submodule update --init --recursive

"$PYTHON_BIN" -m venv "$OFFICIAL_VENV"
"$OFFICIAL_PY" -m pip install --upgrade \
  pip setuptools wheel scikit-build-core numpy
"$OFFICIAL_PY" -m pip install -e \
  "$OFFICIAL_REPO/packages/leann-core"
"$OFFICIAL_PY" -m pip install -e \
  "$OFFICIAL_REPO/packages/leann-backend-hnsw"
```

If the checkout already exists, verify and initialize it instead of cloning:

```bash
test "$(git -C "$OFFICIAL_REPO" rev-parse HEAD)" = \
  "7a34d8856b7aa92da47097af02e8f26b341a90a3"
git -C "$OFFICIAL_REPO" submodule update --init --recursive
```

Verify that imports resolve into the pinned checkout and its package-local
extension:

```bash
"$OFFICIAL_PY" - <<'PY'
import leann
import leann_backend_hnsw
import leann_backend_hnsw.faiss
import leann_backend_hnsw._swigfaiss
import leann_backend_hnsw.hnsw_backend

for module in (
    leann,
    leann_backend_hnsw,
    leann_backend_hnsw.faiss,
    leann_backend_hnsw._swigfaiss,
    leann_backend_hnsw.hnsw_backend,
):
    print(module.__name__, module.__file__)
PY
```

## 6. Prepare deterministic NQ tiers

This command downloads the official archive when it is not already cached and
verifies the expected MD5 before extraction:

```bash
"$PYTHON_BIN" scripts/prepare_beir_nq_scale.py \
  --output work/datasets/nq-scale \
  --cache-archive work/datasets/nq.zip
```

Verify the declared tier sizes:

```bash
jq '.counts, .invariants, .qrel_coverage' \
  work/datasets/nq-scale/manifest.json
```

The gate is:

- 100,000 and 1,000,000 documents
- 300 queries
- zero retained document-text duplicates
- complete positive-qrel coverage
- `100k_is_exact_prefix_of_1m: true`

## 7. Start the CUDA embedding server

Keep the server private to localhost. Use an absolute model path because strict
provenance compares `/props.model_path` with the bound artifact.

`--cache-ram 0` is required. `llama-server` enables `--cache-idle-slots` by
default, and on every task launch it saves each idle slot's prompt state into a
server prompt cache whose default limit is 8 GiB
(`tools/server/server-context.cpp`, the `cache_idle_slots` block; unlike the
neighbouring `update_cache` path it is not restricted to completion tasks, so
embedding requests are affected too). Every document in this corpus is unique,
so that cache can never hit, but the per-task cost still grows linearly with the
number of rows the process has already embedded. Measured on the GPU server over
40,000 rows, the default configuration decays from 682 to 97 rows/s and keeps
falling, which makes the 1M corpus cache a multi-day job. With `--cache-ram 0`
throughput is flat at roughly 1,050 rows/s.

This matters beyond generation time. Official real-GGUF recomputation runs
through this same server, so with the cache enabled each timed complexity point
is served by a slower process than the one before it, and the measured latency
ordering becomes an artefact of execution order. Disabling the cache removes
overhead only from the official path — `leann.cpp` embeds in process and never
touches this server — so the change can only favour the baseline.

```bash
"$LLAMA_SERVER_BUILD/bin/llama-server" \
  --log-disable \
  --model "$MODEL_PATH" \
  --cache-ram 0 \
  --embedding \
  --pooling mean \
  --embd-normalize 2 \
  --ctx-size 32768 \
  --batch-size 32768 \
  --ubatch-size 32768 \
  --parallel 16 \
  --gpu-layers 99 \
  --threads "$CPU_THREADS" \
  --host 127.0.0.1 \
  --port 18080 \
  > work/bench-nq/gpu-server/llama-server.log 2>&1 &

export LEANN_LLAMA_SERVER_PID="$!"

until curl --fail --silent "$EMBEDDING_URL/health" >/dev/null; do
  sleep 2
done

export LLAMA_BUILD_INFO="$(
  curl --fail --silent "$EMBEDDING_URL/props" | jq -r '.build_info'
)"

curl --fail --silent "$EMBEDDING_URL/props" |
  jq '{model_path, build_info, total_slots, default_generation_settings}'
```

The expected server identity is:

- absolute model path equals `$MODEL_PATH`
- 16 total slots
- 2,048 tokens per slot, represented by a total context of 32,768
- build information from the pinned llama.cpp executable
- CUDA backend visible in the server log

## 8. Generate shared vectors, truth, and live captures

Generate 100K first. This is both a correctness gate and the prefix seed for
1M. The endpoint capture must occur while the cache checkpoint exists; the
completed cache publisher removes the checkpoint.

### 8.1 100K shared artifacts

```bash
mkdir -p work/bench-nq/100k

set -o pipefail
"$PYTHON_BIN" -u scripts/run_large_scale_benchmark.py prepare \
  --documents work/datasets/nq-scale/documents-100k.txt \
  --queries work/datasets/nq-scale/queries-test-300.txt \
  --corpus-cache work/bench-nq/100k/corpus.leannbc2 \
  --query-cache work/bench-nq/100k/queries.leannbc2 \
  --ground-truth work/bench-nq/100k/ground-truth.txt \
  --top-k 10 \
  --recall-k 3 10 \
  --embedding-url "$EMBEDDING_URL" \
  --embedding-model nomic-embed-text \
  --fingerprint "$EMBED_FINGERPRINT" \
  --model-identity "$MODEL_IDENTITY" \
  --model-artifact "$MODEL_PATH" \
  --embedding-batch-size 256 \
  --dimension 768 \
  2>&1 | tee work/bench-nq/100k/prepare.log &

export PREPARE_100K_JOB_PID="$!"

until test -s work/bench-nq/100k/corpus.leannbc2.checkpoint.json; do
  kill -0 "$PREPARE_100K_JOB_PID"
  sleep 2
done

"$PYTHON_BIN" scripts/attest_embedding_endpoint.py capture \
  --endpoint "$EMBEDDING_URL" \
  --expected-artifact "$MODEL_PATH" \
  --expected-build-info "$LLAMA_BUILD_INFO" \
  --expected-pid "$LEANN_LLAMA_SERVER_PID" \
  --checkpoint work/bench-nq/100k/corpus.leannbc2.checkpoint.json \
  --tier 100k \
  --require-process-proof \
  --output work/bench-nq/100k/endpoint-capture.json

wait "$PREPARE_100K_JOB_PID"
```

### 8.2 1M shared artifacts

```bash
mkdir -p work/bench-nq/1m

set -o pipefail
"$PYTHON_BIN" -u scripts/run_large_scale_benchmark.py prepare \
  --documents work/datasets/nq-scale/documents-1m.txt \
  --queries work/datasets/nq-scale/queries-test-300.txt \
  --corpus-cache work/bench-nq/1m/corpus.leannbc2 \
  --query-cache work/bench-nq/1m/queries.leannbc2 \
  --ground-truth work/bench-nq/1m/ground-truth.txt \
  --top-k 10 \
  --recall-k 3 10 \
  --embedding-url "$EMBEDDING_URL" \
  --embedding-model nomic-embed-text \
  --fingerprint "$EMBED_FINGERPRINT" \
  --model-identity "$MODEL_IDENTITY" \
  --model-artifact "$MODEL_PATH" \
  --embedding-batch-size 256 \
  --dimension 768 \
  --prefix-cache work/bench-nq/100k/corpus.leannbc2 \
  2>&1 | tee work/bench-nq/1m/prepare.log &

export PREPARE_1M_JOB_PID="$!"

until test -s work/bench-nq/1m/corpus.leannbc2.checkpoint.json; do
  kill -0 "$PREPARE_1M_JOB_PID"
  sleep 2
done

"$PYTHON_BIN" scripts/attest_embedding_endpoint.py capture \
  --endpoint "$EMBEDDING_URL" \
  --expected-artifact "$MODEL_PATH" \
  --expected-build-info "$LLAMA_BUILD_INFO" \
  --expected-pid "$LEANN_LLAMA_SERVER_PID" \
  --checkpoint work/bench-nq/1m/corpus.leannbc2.checkpoint.json \
  --tier 1m \
  --require-process-proof \
  --output work/bench-nq/1m/endpoint-capture.json

wait "$PREPARE_1M_JOB_PID"
```

If capture misses the checkpoint, do not fabricate an attestation. A strict
publication run must regenerate that tier with a live capture.

## 9. Run native sweeps

Run the block first with `TIER=100k`, inspect it, then repeat with `TIER=1m`.
Do not run native and official measurements concurrently.

Stop the external server before native timing so `leann.cpp` has exclusive GPU
access:

```bash
kill -TERM "$LEANN_LLAMA_SERVER_PID"
wait "$LEANN_LLAMA_SERVER_PID" || true
```

```bash
export TIER="100k"
export DOCUMENTS="work/datasets/nq-scale/documents-$TIER.txt"
export CORPUS_CACHE="work/bench-nq/$TIER/corpus.leannbc2"
export QUERY_CACHE="work/bench-nq/$TIER/queries.leannbc2"
export GROUND_TRUTH="work/bench-nq/$TIER/ground-truth.txt"

"$PYTHON_BIN" -u scripts/run_large_scale_benchmark.py orchestrate \
  --documents "$DOCUMENTS" \
  --queries work/datasets/nq-scale/queries-test-300.txt \
  --corpus-cache "$CORPUS_CACHE" \
  --query-cache "$QUERY_CACHE" \
  --ground-truth "$GROUND_TRUTH" \
  --top-k 10 \
  --recall-k 3 10 \
  --output "work/bench-nq/$TIER/native" \
  --embedding-url "$EMBEDDING_URL" \
  --server-ctx-size 32768 \
  --build-warmups 0 \
  --build-repetitions 1 \
  --search-warmups 1 \
  --search-repetitions 3 \
  --skip-official \
  --native-binary "$NATIVE_BUILD/leann" \
  --native-index "work/bench-nq/$TIER/index/native" \
  --native-model "$MODEL_PATH" \
  --native-gpu-layers 99 \
  --native-parallel 16 \
  --native-ctx 2048 \
  --native-batch-tokens 32768 \
  --native-role sweep \
  --native-ef-search 32 64 128 256 512 \
  --native-scan-limit 0 \
  --native-rerank-ratio 0.25 \
  --native-recompute-batch 16 \
  --native-build-extra \
  '["--graph-degree","32","--ef-construction","200","--low-degree","3","--hub-ratio","0.02","--approx","pq","--pq-subquantizers","64","--pq-bits","4","--pq-iterations","10","--pq-training-samples","4096"]' \
  2>&1 | tee "work/bench-nq/$TIER/native-run.log"
```

The native build currently publishes progress only at stage boundaries. It
does not expose an inserted-node percentage and cannot resume in the middle of
HNSW construction. Re-running the exact command reuses verified completed
stages but restarts an interrupted stage.

## 10. Run official cached-recompute sweeps

The cached endpoint returns the exact shared vectors. It is an algorithmic
control, not an inference-latency baseline.

Before starting this section, restart llama-server with the exact command in
section 7 and refresh `LEANN_LLAMA_SERVER_PID` and `LLAMA_BUILD_INFO`. The
strict collector requires a healthy, identity-matched `/props` observation even
for the cached official control. Keep this fresh server running through the
cached and real official profiles.

Run the block once for each tier after its native sweep:

```bash
export TIER="100k"
export DOCUMENTS="work/datasets/nq-scale/documents-$TIER.txt"
export CORPUS_CACHE="work/bench-nq/$TIER/corpus.leannbc2"
export QUERY_CACHE="work/bench-nq/$TIER/queries.leannbc2"
export GROUND_TRUTH="work/bench-nq/$TIER/ground-truth.txt"

"$PYTHON_BIN" -u scripts/cached_embedding_server.py \
  --host 127.0.0.1 \
  --port 18082 \
  --documents "$DOCUMENTS" \
  --cache "$CORPUS_CACHE" \
  --model nomic-embed-text \
  > "work/bench-nq/$TIER/cached-server.log" 2>&1 &

export CACHED_SERVER_PID="$!"

until curl --fail --silent "$CACHED_URL/metrics" >/dev/null; do
  sleep 2
done

"$PYTHON_BIN" -u scripts/run_large_scale_benchmark.py orchestrate \
  --documents "$DOCUMENTS" \
  --queries work/datasets/nq-scale/queries-test-300.txt \
  --corpus-cache "$CORPUS_CACHE" \
  --query-cache "$QUERY_CACHE" \
  --ground-truth "$GROUND_TRUTH" \
  --top-k 10 \
  --recall-k 3 10 \
  --output "work/bench-nq/$TIER/official-cached" \
  --embedding-url "$EMBEDDING_URL" \
  --proxy-url "$CACHED_URL" \
  --server-ctx-size 32768 \
  --build-warmups 0 \
  --build-repetitions 1 \
  --search-warmups 1 \
  --search-repetitions 3 \
  --skip-native \
  --official-python "$OFFICIAL_PY" \
  --official-script "$REPO_ROOT/scripts/compare_official_leann.py" \
  --official-repo "$OFFICIAL_REPO" \
  --official-index "work/bench-nq/$TIER/index/official/benchmark.leann" \
  --official-m 32 \
  --official-ef-construction 200 \
  --official-complexities 32 64 128 256 512 \
  --official-batch-sizes 0 16 \
  --official-recompute-mode cached \
  2>&1 | tee "work/bench-nq/$TIER/official-cached-run.log"

kill -TERM "$CACHED_SERVER_PID"
wait "$CACHED_SERVER_PID" || true
```

## 11. Run official real-GGUF sweeps

Use the exact cached-build index. The reuse runner verifies its artifact set
before searching, so the real profile cannot silently rebuild or substitute an
index.

Start a fresh pinned llama-server before timed official-real work, using the
same executable, GGUF, context, slot count, pooling, and normalization settings.
Verify `/health` and `/props` again.

Do not build a server-rotation mechanism around lifetime throughput decay. The
decay had a single cause — the default server prompt cache described in section
7 — and `--cache-ram 0` removes it, with throughput measured flat across the
whole 1M corpus. Rotating servers mid-generation cannot be attested anyway: the
capture in section 8 proves the process that *started* the generation, and
`attest_embedding_endpoint.py` rejects any later process because it did not
start before the checkpoint. A tier whose generation outlived its proven server
must be regenerated, not patched with per-rotation evidence.

For each tier:

```bash
export TIER="100k"
export DOCUMENTS="work/datasets/nq-scale/documents-$TIER.txt"
export CORPUS_CACHE="work/bench-nq/$TIER/corpus.leannbc2"
export QUERY_CACHE="work/bench-nq/$TIER/queries.leannbc2"
export GROUND_TRUTH="work/bench-nq/$TIER/ground-truth.txt"

"$PYTHON_BIN" -u scripts/openai_embedding_proxy.py \
  --listen-host 127.0.0.1 \
  --listen-port 18081 \
  --upstream "$EMBEDDING_URL" \
  > "work/bench-nq/$TIER/real-proxy.log" 2>&1 &

export REAL_PROXY_PID="$!"

until curl --fail --silent "$REAL_PROXY_URL/metrics" >/dev/null; do
  sleep 2
done

"$PYTHON_BIN" -u scripts/run_official_real_search_reuse.py \
  --documents "$DOCUMENTS" \
  --queries work/datasets/nq-scale/queries-test-300.txt \
  --corpus-cache "$CORPUS_CACHE" \
  --query-cache "$QUERY_CACHE" \
  --ground-truth "$GROUND_TRUTH" \
  --top-k 10 \
  --recall-k 3 10 \
  --output "work/bench-nq/$TIER/official-real" \
  --cached-build-manifest \
    "work/bench-nq/$TIER/official-cached/manifest.json" \
  --embedding-url "$EMBEDDING_URL" \
  --embedding-model nomic-embed-text \
  --proxy-url "$REAL_PROXY_URL" \
  --server-ctx-size 32768 \
  --native-parallel 16 \
  --native-ctx 2048 \
  --search-warmups 1 \
  --search-repetitions 3 \
  --official-python "$OFFICIAL_PY" \
  --official-script "$REPO_ROOT/scripts/compare_official_leann.py" \
  --official-repo "$OFFICIAL_REPO" \
  --official-index "work/bench-nq/$TIER/index/official/benchmark.leann" \
  --official-m 32 \
  --official-ef-construction 200 \
  --official-complexities 32 64 128 256 512 \
  --official-batch-sizes 16 \
  2>&1 | tee "work/bench-nq/$TIER/official-real-run.log"

kill -TERM "$REAL_PROXY_PID"
wait "$REAL_PROXY_PID" || true
```

Running all five real complexities avoids selecting a latency point after
observing latency. The collector chooses the matched pair using the declared
recall threshold and tolerance.

## 12. Validate embedding parity and finalize endpoint evidence

Run for both tiers:

```bash
export TIER="100k"

"$PYTHON_BIN" scripts/validate_embedding_parity.py \
  --documents "work/datasets/nq-scale/documents-$TIER.txt" \
  --queries work/datasets/nq-scale/queries-test-300.txt \
  --corpus-cache "work/bench-nq/$TIER/corpus.leannbc2" \
  --query-cache "work/bench-nq/$TIER/queries.leannbc2" \
  --dataset-manifest work/datasets/nq-scale/manifest.json \
  --source-archive work/datasets/nq.zip \
  --document-ids "work/datasets/nq-scale/document_ids-$TIER.tsv" \
  --tier "$TIER" \
  --native-binary "$NATIVE_BUILD/leann" \
  --native-model "$MODEL_PATH" \
  --work-dir "work/bench-nq/$TIER/parity-work" \
  --output "work/bench-nq/$TIER/parity.json" \
  --expected-queries 300 \
  --minimum-cosine 0.9999 \
  --ctx 2048 \
  --batch-tokens 32768 \
  --parallel 16 \
  --threads "$CPU_THREADS" \
  --gpu-layers 99

"$PYTHON_BIN" scripts/attest_embedding_endpoint.py finalize \
  --capture "work/bench-nq/$TIER/endpoint-capture.json" \
  --parity-report "work/bench-nq/$TIER/parity.json" \
  --output "work/bench-nq/$TIER/endpoint-attestation.json"
```

## 13. Strict collection

Do not use `--allow-incomplete`, `--allow-missing-parity`,
`--allow-missing-endpoint-attestation`, or
`--no-verify-artifact-hashes` for the launch report.

```bash
"$PYTHON_BIN" scripts/collect_large_scale_results.py \
  --manifest 100k=work/bench-nq/100k/native/manifest.json \
  --manifest 100k=work/bench-nq/100k/official-cached/manifest.json \
  --manifest 100k=work/bench-nq/100k/official-real/manifest.json \
  --manifest 1m=work/bench-nq/1m/native/manifest.json \
  --manifest 1m=work/bench-nq/1m/official-cached/manifest.json \
  --manifest 1m=work/bench-nq/1m/official-real/manifest.json \
  --parity 100k=work/bench-nq/100k/parity.json \
  --parity 1m=work/bench-nq/1m/parity.json \
  --endpoint-attestation \
    100k=work/bench-nq/100k/endpoint-attestation.json \
  --endpoint-attestation \
    1m=work/bench-nq/1m/endpoint-attestation.json \
  --required-tier 100k \
  --required-tier 1m \
  --required-profile native \
  --required-profile official-cached \
  --required-profile official-real \
  --required-native-ef 32 64 128 256 512 \
  --required-cached-complexity 32 64 128 256 512 \
  --required-cached-batch-size 0 16 \
  --minimum-real-points 1 \
  --recall-match-cutoff 3 \
  --minimum-matched-recall 0.90 \
  --recall-match-tolerance 0.01 \
  --minimum-parity-cosine 0.9999 \
  --output-prefix outputs/nq-scale-gpu
```

Expected deliverables:

- `outputs/nq-scale-gpu.json`
- `outputs/nq-scale-gpu.md`
- `outputs/nq-scale-gpu.csv`

The collector must exit successfully and report every required profile as
complete before any number is copied into `README.md`.

## 14. Monitoring and recovery

Use a separate terminal for low-overhead monitoring:

```bash
watch -n 5 nvidia-smi
```

Inspect stage completion without changing benchmark state:

```bash
find work/bench-nq -path '*/stages/*.json' -type f -print |
  sort |
  while read -r stage_file; do
    jq -r \
      '(.name // "?") + " " + (.status // "?") +
       " measurements=" + ((.measurements | length // 0) | tostring)' \
      "$stage_file"
  done
```

Operational rules:

- Poll at five-minute intervals unless a process exits or a stage completes.
- Do not run native and official benchmarks concurrently.
- Do not compile, install packages, or generate caches during timed searches.
- Keep CPU governor, GPU power limit, clock policy, and server arguments fixed.
- Record thermal throttling, ECC errors, GPU resets, OOM kills, HTTP errors, and
  any server restart.
- If a run is interrupted, rerun the exact command. Completed verified stages
  are reused; the interrupted stage restarts.
- Do not manually mark a stale `running` manifest as complete.
- Preserve partial raw files and logs for diagnosis, but the collector must
  ignore incomplete observations.

## 15. Reporting rules

The final English report and README update must include:

- exact hardware, OS, CUDA, compiler, Python, and commit identities
- dataset and model hashes
- vector-index, vector-serving, text-store, and total storage separately
- Recall@3 and Recall@10
- mean, p50, and p95 real-GGUF latency
- candidate embeddings and embedding batches per query
- build time and peak RSS, with the different RSS scopes disclosed
- the selected recall-matched pair and selection rule
- cached-mode results labeled non-latency-comparable
- all failures, restarts, exclusions, and limitations

Do not headline a speedup when no pair satisfies the recall floor and tolerance.
Do not present cache-only official latency as real-GGUF latency. Do not describe
the report as complete when the strict collector rejects it.
