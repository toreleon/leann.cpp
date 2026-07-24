#!/usr/bin/env python3
"""Reproducible large-scale leann.cpp versus official LEANN benchmark runner.

The expensive stages are restartable.  Corpus and query embeddings are shared,
normalized FP32 vectors bound to the exact input bytes; exact ground truth is
computed once in bounded NumPy blocks.  Command observations retain argv,
stdout/stderr, wall time, sampled peak RSS, artifact hashes, and provenance.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from benchmark_cache import (
    atomic_write_json,
    cache_v2_header,
    canonical_hash,
    iter_nonempty_lines,
    normalize_vectors,
    read_cache,
    read_ground_truth,
    sha256_file,
    snapshot_files,
    source_identity,
    validate_cache_source,
)


SCHEMA = "leann-large-scale-benchmark-v1"
CACHE_META_SCHEMA = "leann-shared-embedding-cache-v1"
GT_META_SCHEMA = "leann-exact-ground-truth-v1"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def sidecar_path(path: Path) -> Path:
    return path.with_name(path.name + ".meta.json")


def checkpoint_path(path: Path) -> Path:
    return path.with_name(path.name + ".checkpoint.json")


def partial_path(path: Path) -> Path:
    return path.with_name(path.name + ".partial")


def checkpoint_created_at(state: dict[str, Any], context: str) -> str:
    created_at = state.get("created_at")
    if not isinstance(created_at, str) or not created_at.strip():
        raise RuntimeError(
            f"{context} checkpoint predates durable start-time recording; "
            "refusing to invent generation provenance"
        )
    return created_at


def parse_json_array(value: str) -> list[str]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise argparse.ArgumentTypeError("expected a JSON array of strings")
    return parsed


def embedding_endpoint(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return base + "/embeddings"
    return base + "/v1/embeddings"


def model_descriptor(args: argparse.Namespace) -> dict[str, Any]:
    artifact: dict[str, Any] | None = None
    if getattr(args, "model_artifact", None):
        path = args.model_artifact.resolve()
        artifact = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    descriptor = {
        "embedding_model": args.embedding_model,
        "native_fingerprint": args.fingerprint,
        "declared_identity": args.model_identity,
        "artifact": artifact,
    }
    descriptor["sha256"] = canonical_hash(descriptor)
    descriptor["identity_strength"] = (
        "artifact-sha256"
        if artifact is not None
        else "declared"
        if args.model_identity
        else "fingerprint-only"
    )
    return descriptor


class EmbeddingClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        timeout: int,
        retries: int,
        api_key: str | None,
    ) -> None:
        self.url = embedding_endpoint(base_url)
        self.model = model
        self.timeout = timeout
        self.retries = retries
        self.api_key = api_key

    def embed(self, texts: Sequence[str], expected_dimension: int | None) -> np.ndarray:
        if not texts:
            raise ValueError("cannot request an empty embedding batch")
        body = json.dumps({"model": self.model, "input": list(texts)}).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        last_error: BaseException | None = None
        for attempt in range(self.retries + 1):
            request = urllib.request.Request(
                self.url, data=body, headers=headers, method="POST"
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = json.load(response)
                data = payload.get("data") if isinstance(payload, dict) else None
                if not isinstance(data, list) or len(data) != len(texts):
                    raise ValueError("embedding endpoint returned the wrong row count")
                by_index: dict[int, Any] = {}
                for item in data:
                    if not isinstance(item, dict) or not isinstance(item.get("index"), int):
                        raise ValueError("embedding response has an invalid index")
                    index = item["index"]
                    if index in by_index or index < 0 or index >= len(texts):
                        raise ValueError("embedding response indices are not a permutation")
                    by_index[index] = item.get("embedding")
                if set(by_index) != set(range(len(texts))):
                    raise ValueError("embedding response indices are incomplete")
                matrix = normalize_vectors([by_index[index] for index in range(len(texts))])
                if expected_dimension is not None and matrix.shape[1] != expected_dimension:
                    raise ValueError(
                        f"embedding dimension changed: {matrix.shape[1]} vs "
                        f"{expected_dimension}"
                    )
                return matrix
            except urllib.error.HTTPError as error:
                last_error = error
                retryable = error.code == 429 or 500 <= error.code < 600
                if not retryable or attempt == self.retries:
                    detail = error.read(4096).decode("utf-8", errors="replace")
                    raise RuntimeError(
                        f"embedding HTTP {error.code}: {detail}"
                    ) from error
            except (urllib.error.URLError, TimeoutError) as error:
                last_error = error
                if attempt == self.retries:
                    raise RuntimeError(f"embedding request failed: {error}") from error
            if attempt < self.retries:
                time.sleep(min(30.0, 0.5 * (2**attempt)))
        raise RuntimeError(f"embedding request failed: {last_error}")


def _iter_embedded_batches(
    client: EmbeddingClient,
    source_path: Path,
    completed: int,
    batch_size: int,
    dimension_holder: list[int | None],
    concurrency: int,
) -> Iterator[tuple[list[str], np.ndarray]]:
    """Yield (texts, vectors) strictly in source order.

    A sequential client leaves the embedding server idle while it parses the
    previous response, so cache generation runs far below the endpoint's
    capacity. Requests are issued up to ``concurrency`` batches ahead, but
    results are consumed in submission order, so the bytes appended to the
    cache, the running payload digest, and the checkpoint row counts are
    identical to the sequential path.
    """

    batches = _iter_batches(source_path, completed, batch_size)
    if concurrency <= 1:
        for texts in batches:
            yield texts, client.embed(texts, dimension_holder[0])
        return

    import collections
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        pending: collections.deque[
            tuple[list[str], concurrent.futures.Future[np.ndarray]]
        ] = collections.deque()
        try:
            for texts in batches:
                pending.append(
                    (texts, pool.submit(client.embed, texts, dimension_holder[0]))
                )
                if len(pending) >= concurrency:
                    ready_texts, ready = pending.popleft()
                    yield ready_texts, ready.result()
            while pending:
                ready_texts, ready = pending.popleft()
                yield ready_texts, ready.result()
        finally:
            for _, future in pending:
                future.cancel()


def _iter_batches(path: Path, start: int, batch_size: int) -> Iterator[list[str]]:
    batch: list[str] = []
    for index, line in enumerate(iter_nonempty_lines(path)):
        if index < start:
            continue
        batch.append(line)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _payload_sha(path: Path, header_size: int, rows: int, dimension: int) -> str:
    return sha256_file(path, offset=header_size, length=rows * dimension * 4)


def _validate_existing_cache(
    cache_path: Path,
    source_path: Path,
    fingerprint: str,
    model: dict[str, Any],
    role: str,
    bindings: dict[str, Any],
    expected_input_hash: str,
) -> dict[str, Any]:
    vectors, metadata = read_cache(cache_path)
    validate_cache_source(metadata, source_path, require_integrity=True)
    if metadata["fingerprint"] != fingerprint:
        raise ValueError(f"{cache_path}: native fingerprint mismatch")
    meta_path = sidecar_path(cache_path)
    if meta_path.exists():
        declared = load_json(meta_path)
        if declared.get("schema") != CACHE_META_SCHEMA:
            raise ValueError(f"{cache_path}: unknown cache sidecar schema")
        if declared.get("role") != role:
            raise ValueError(f"{cache_path}: cache role mismatch")
        if declared.get("model", {}).get("sha256") != model["sha256"]:
            raise ValueError(f"{cache_path}: model identity sidecar mismatch")
        if declared.get("bindings") != bindings:
            raise ValueError(f"{cache_path}: cache bindings mismatch")
        actual_sha = sha256_file(cache_path)
        if declared.get("cache", {}).get("sha256") != actual_sha:
            raise ValueError(f"{cache_path}: cache sidecar SHA-256 mismatch")
        if declared.get("source", {}).get("sha256") != metadata["source_sha256"]:
            raise ValueError(f"{cache_path}: cache sidecar source mismatch")
    else:
        state_path = checkpoint_path(cache_path)
        if not state_path.exists():
            raise ValueError(
                f"{cache_path}: complete cache has no provenance sidecar; "
                "refusing to infer a model identity"
            )
        state = load_json(state_path)
        if state.get("status") != "ready-to-publish":
            raise ValueError(f"{cache_path}: incomplete cache provenance checkpoint")
        if state.get("input_hash") != expected_input_hash:
            raise ValueError(f"{cache_path}: publication checkpoint input mismatch")
        if state.get("cache_sha256") != sha256_file(cache_path):
            raise ValueError(f"{cache_path}: publication checkpoint checksum mismatch")
        declared = {
            "schema": CACHE_META_SCHEMA,
            "created_at": utc_now(),
            "role": role,
            "cache": {
                **metadata,
                "sha256": sha256_file(cache_path),
            },
            "source": source_identity(source_path),
            "model": model,
            "bindings": bindings,
            "note": "Sidecar recovered from a verified ready-to-publish checkpoint.",
        }
        atomic_write_json(meta_path, declared)
        state_path.unlink()
    del vectors
    return declared


def generate_cache(
    *,
    source_path: Path,
    cache_path: Path,
    client: EmbeddingClient,
    fingerprint: str,
    model: dict[str, Any],
    role: str,
    batch_size: int,
    dimension: int | None,
    bindings: dict[str, Any],
    prefix_cache: Path | None = None,
    concurrency: int = 1,
) -> dict[str, Any]:
    source_path = source_path.resolve()
    cache_path = cache_path.resolve()
    source = source_identity(source_path)
    if source["nonempty_lines"] <= 0:
        raise ValueError(f"{source_path}: no non-empty input lines")
    stable = {
        "schema": "leann-cache-generation-checkpoint-v1",
        "source": source,
        "cache_path": str(cache_path),
        "fingerprint": fingerprint,
        "model": model,
        "role": role,
        "bindings": bindings,
        "embedding_endpoint": client.url,
        "batch_size": batch_size,
        "requested_dimension": dimension,
        "prefix_cache_sha256": (
            sha256_file(prefix_cache.resolve()) if prefix_cache is not None else None
        ),
    }
    input_hash = canonical_hash(stable)
    if cache_path.exists():
        return _validate_existing_cache(
            cache_path,
            source_path,
            fingerprint,
            model,
            role,
            bindings,
            input_hash,
        )
    partial = partial_path(cache_path)
    checkpoint = checkpoint_path(cache_path)
    completed = 0
    payload_digest = hashlib.sha256()
    header_size: int | None = None
    prefix_seed_info: dict[str, Any] | None = None
    generation_started_at = utc_now()

    if checkpoint.exists() or partial.exists():
        if not checkpoint.exists() or not partial.exists():
            raise RuntimeError("cache checkpoint and partial file must exist together")
        state = load_json(checkpoint)
        if state.get("input_hash") != input_hash:
            raise RuntimeError("cache checkpoint belongs to different inputs")
        generation_started_at = checkpoint_created_at(state, "cache generation")
        dimension = int(state["dimension"])
        completed = int(state["completed_rows"])
        header_size = int(state["header_size"])
        prefix_seed_info = state.get("prefix_seed")
        expected_prefix = header_size + completed * dimension * 4
        observed = partial.stat().st_size
        if observed < expected_prefix:
            raise RuntimeError("partial cache is shorter than its checkpoint")
        if _payload_sha(partial, header_size, completed, dimension) != state["payload_sha256"]:
            raise RuntimeError("partial cache payload checksum mismatch")
        if observed > expected_prefix:
            with partial.open("r+b") as target:
                target.truncate(expected_prefix)
                target.flush()
                os.fsync(target.fileno())
        with partial.open("rb") as source_file:
            source_file.seek(header_size)
            for block in iter(lambda: source_file.read(1024 * 1024), b""):
                payload_digest.update(block)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if (
        prefix_cache is not None
        and completed == 0
        and not partial.exists()
        and not checkpoint.exists()
    ):
        prefix_cache = prefix_cache.resolve()
        prefix_vectors, prefix_meta = read_cache(prefix_cache)
        if prefix_meta["version"] != 2:
            raise ValueError("--prefix-cache must use LEANNBC2")
        if prefix_meta["count"] >= source["nonempty_lines"]:
            raise ValueError("--prefix-cache must be smaller than the target corpus")
        if prefix_meta["fingerprint"] != fingerprint:
            raise ValueError("--prefix-cache fingerprint mismatch")
        if dimension is not None and prefix_meta["dimensions"] != dimension:
            raise ValueError("--prefix-cache dimension mismatch")
        prefix_declared = load_json(sidecar_path(prefix_cache))
        if (
            prefix_declared.get("schema") != CACHE_META_SCHEMA
            or prefix_declared.get("role") != "corpus"
            or prefix_declared.get("model", {}).get("sha256") != model["sha256"]
            or prefix_declared.get("cache", {}).get("sha256")
            != sha256_file(prefix_cache)
        ):
            raise ValueError("--prefix-cache provenance mismatch")
        prefix_size = int(prefix_meta["source_size_bytes"])
        if (
            source["size_bytes"] <= prefix_size
            or sha256_file(source_path, length=prefix_size)
            != prefix_meta["source_sha256"]
        ):
            raise ValueError(
                "--prefix-cache source bytes are not an exact prefix of target source"
            )
        dimension = int(prefix_meta["dimensions"])
        header = cache_v2_header(
            dimension=dimension,
            count=source["nonempty_lines"],
            fingerprint=fingerprint,
            source_size_bytes=source["size_bytes"],
            source_sha256=source["sha256"],
        )
        header_size = len(header)
        with partial.open("xb") as target, prefix_cache.open("rb") as prefix_source:
            target.write(header)
            prefix_source.seek(prefix_meta["vector_offset"])
            remaining = prefix_meta["vector_bytes"]
            while remaining:
                block = prefix_source.read(min(1024 * 1024, remaining))
                if not block:
                    raise RuntimeError("truncated prefix-cache vector payload")
                target.write(block)
                payload_digest.update(block)
                remaining -= len(block)
            target.flush()
            os.fsync(target.fileno())
        completed = int(prefix_meta["count"])
        prefix_seed_info = {
            "cache": str(prefix_cache),
            "cache_sha256": sha256_file(prefix_cache),
            "rows": completed,
            "source_prefix_size_bytes": prefix_size,
            "source_prefix_sha256": prefix_meta["source_sha256"],
            "vectors_copied_bitwise": True,
        }
        atomic_write_json(
            checkpoint,
            {
                **stable,
                "input_hash": input_hash,
                "created_at": generation_started_at,
                "status": "running",
                "dimension": dimension,
                "header_size": header_size,
                "completed_rows": completed,
                "payload_sha256": payload_digest.hexdigest(),
                "prefix_seed": prefix_seed_info,
                "updated_at": utc_now(),
            },
        )
        del prefix_vectors
    if dimension is not None and header_size is None:
        header = cache_v2_header(
            dimension=dimension,
            count=source["nonempty_lines"],
            fingerprint=fingerprint,
            source_size_bytes=source["size_bytes"],
            source_sha256=source["sha256"],
        )
        header_size = len(header)
        with partial.open("xb") as target:
            target.write(header)
            target.flush()
            os.fsync(target.fileno())
        atomic_write_json(
            checkpoint,
            {
                **stable,
                "input_hash": input_hash,
                "created_at": generation_started_at,
                "status": "running",
                "dimension": dimension,
                "header_size": header_size,
                "completed_rows": 0,
                "payload_sha256": payload_digest.hexdigest(),
                "prefix_seed": prefix_seed_info,
                "updated_at": utc_now(),
            },
        )
    # The header cannot be written until the first response reveals the
    # dimension, so stay sequential until it is known and only then let the
    # client run ahead of the writer.
    dimension_holder: list[int | None] = [dimension]
    for texts, matrix in _iter_embedded_batches(
        client,
        source_path,
        completed,
        batch_size,
        dimension_holder,
        1 if dimension is None else concurrency,
    ):
        if dimension is None:
            dimension = int(matrix.shape[1])
            dimension_holder[0] = dimension
            header = cache_v2_header(
                dimension=dimension,
                count=source["nonempty_lines"],
                fingerprint=fingerprint,
                source_size_bytes=source["size_bytes"],
                source_sha256=source["sha256"],
            )
            header_size = len(header)
            with partial.open("xb") as target:
                target.write(header)
                target.flush()
                os.fsync(target.fileno())
            atomic_write_json(
                checkpoint,
                {
                    **stable,
                    "input_hash": input_hash,
                    "created_at": generation_started_at,
                    "status": "running",
                    "dimension": dimension,
                    "header_size": header_size,
                    "completed_rows": 0,
                    "payload_sha256": payload_digest.hexdigest(),
                    "prefix_seed": prefix_seed_info,
                    "updated_at": utc_now(),
                },
            )
        assert dimension is not None and header_size is not None
        payload = matrix.astype("<f4", copy=False).tobytes(order="C")
        with partial.open("ab") as target:
            target.write(payload)
            target.flush()
            os.fsync(target.fileno())
        payload_digest.update(payload)
        completed += len(texts)
        atomic_write_json(
            checkpoint,
            {
                **stable,
                "input_hash": input_hash,
                "created_at": generation_started_at,
                "status": "running",
                "dimension": dimension,
                "header_size": header_size,
                "completed_rows": completed,
                "payload_sha256": payload_digest.hexdigest(),
                "prefix_seed": prefix_seed_info,
                "updated_at": utc_now(),
            },
        )
        print(
            f"{role} cache: {completed}/{source['nonempty_lines']} rows",
            flush=True,
        )

    if completed != source["nonempty_lines"] or dimension is None:
        raise RuntimeError("cache generation ended before all source rows")
    vectors, cache = read_cache(partial)
    validate_cache_source(cache, source_path, require_integrity=True)
    del vectors
    cache_sha = sha256_file(partial)
    atomic_write_json(
        checkpoint,
        {
            **stable,
            "input_hash": input_hash,
            "created_at": generation_started_at,
            "status": "ready-to-publish",
            "dimension": dimension,
            "header_size": header_size,
            "completed_rows": completed,
            "payload_sha256": payload_digest.hexdigest(),
            "prefix_seed": prefix_seed_info,
            "cache_sha256": cache_sha,
            "updated_at": utc_now(),
        },
    )
    os.replace(partial, cache_path)
    _, cache = read_cache(cache_path)
    validate_cache_source(cache, source_path, require_integrity=True)
    cache["sha256"] = cache_sha
    declared = {
        "schema": CACHE_META_SCHEMA,
        "created_at": utc_now(),
        "role": role,
        "cache": cache,
        "source": source,
        "model": model,
        "bindings": bindings,
        "generation": {
            "started_at": generation_started_at,
            "endpoint": client.url,
            "batch_size": batch_size,
            "request_concurrency": concurrency,
            "normalization": "L2 float32 per row before persistence",
            "payload_sha256": payload_digest.hexdigest(),
            "prefix_seed": prefix_seed_info,
        },
    }
    atomic_write_json(sidecar_path(cache_path), declared)
    checkpoint.unlink(missing_ok=True)
    return declared


def _select_topk(
    scores: np.ndarray, ids: np.ndarray, top_k: int
) -> tuple[np.ndarray, np.ndarray]:
    if scores.shape != ids.shape:
        raise ValueError("score/ID candidate shape mismatch")
    take = min(top_k, scores.shape[1])
    ordered_scores = np.empty((scores.shape[0], take), dtype=scores.dtype)
    ordered_ids = np.empty((ids.shape[0], take), dtype=ids.dtype)
    for row in range(scores.shape[0]):
        row_scores = scores[row]
        row_ids = ids[row]
        if take == row_scores.size:
            candidates = np.arange(row_scores.size)
        else:
            provisional = np.argpartition(row_scores, row_scores.size - take)[-take:]
            threshold = np.min(row_scores[provisional])
            strict = np.flatnonzero(row_scores > threshold)
            ties = np.flatnonzero(row_scores == threshold)
            needed = take - strict.size
            tie_order = np.argsort(row_ids[ties], kind="stable")
            candidates = np.concatenate((strict, ties[tie_order[:needed]]))
        order = np.lexsort((row_ids[candidates], -row_scores[candidates]))
        selected = candidates[order[:take]]
        ordered_scores[row] = row_scores[selected]
        ordered_ids[row] = row_ids[selected]
    return ordered_scores, ordered_ids


def exact_topk_block(
    corpus: np.ndarray,
    queries: np.ndarray,
    *,
    top_k: int,
    corpus_block_rows: int,
) -> np.ndarray:
    best_scores = np.empty((queries.shape[0], 0), dtype=np.float32)
    best_ids = np.empty((queries.shape[0], 0), dtype=np.uint32)
    for begin in range(0, corpus.shape[0], corpus_block_rows):
        end = min(corpus.shape[0], begin + corpus_block_rows)
        block = np.asarray(corpus[begin:end], dtype=np.float32)
        scores = np.asarray(queries @ block.T, dtype=np.float32)
        ids = np.broadcast_to(
            np.arange(begin, end, dtype=np.uint32), scores.shape
        )
        local_scores, local_ids = _select_topk(scores, ids, top_k)
        best_scores, best_ids = _select_topk(
            np.concatenate((best_scores, local_scores), axis=1),
            np.concatenate((best_ids, local_ids), axis=1),
            top_k,
        )
    return best_ids


def generate_ground_truth(
    *,
    corpus_cache: Path,
    query_cache: Path,
    documents: Path,
    queries: Path,
    output: Path,
    top_k: int,
    query_block_rows: int,
    corpus_block_rows: int,
) -> dict[str, Any]:
    corpus, corpus_meta = read_cache(corpus_cache)
    query_vectors, query_meta = read_cache(query_cache)
    corpus_source = validate_cache_source(
        corpus_meta, documents, require_integrity=True
    )
    query_source = validate_cache_source(query_meta, queries, require_integrity=True)
    if corpus.shape[1] != query_vectors.shape[1]:
        raise ValueError("corpus/query vector dimensions differ")
    if corpus_meta["fingerprint"] != query_meta["fingerprint"]:
        raise ValueError("corpus/query native fingerprints differ")
    if top_k <= 0 or top_k > corpus.shape[0]:
        raise ValueError("top-k must be in [1, corpus count]")

    corpus_cache_sha = sha256_file(corpus_cache)
    query_cache_sha = sha256_file(query_cache)
    corpus_sidecar = load_json(sidecar_path(corpus_cache))
    query_sidecar = load_json(sidecar_path(query_cache))
    if corpus_sidecar["model"]["sha256"] != query_sidecar["model"]["sha256"]:
        raise ValueError("corpus/query model identity hashes differ")
    if query_sidecar.get("bindings", {}).get("corpus_cache_sha256") != corpus_cache_sha:
        raise ValueError("query vectors are not bound to this corpus cache")

    stable = {
        "schema": GT_META_SCHEMA,
        "corpus_cache_sha256": corpus_cache_sha,
        "query_cache_sha256": query_cache_sha,
        "corpus_source_sha256": corpus_source["sha256"],
        "query_source_sha256": query_source["sha256"],
        "model_sha256": corpus_sidecar["model"]["sha256"],
        "query_count": int(query_vectors.shape[0]),
        "corpus_count": int(corpus.shape[0]),
        "dimensions": int(corpus.shape[1]),
        "top_k": top_k,
        "algorithm": "blockwise float32 matrix multiplication; score desc, ID asc ties",
        "query_block_rows": query_block_rows,
        "corpus_block_rows": corpus_block_rows,
    }
    input_hash = canonical_hash(stable)
    meta_path = sidecar_path(output)
    if output.exists() and meta_path.exists():
        declared = load_json(meta_path)
        read_ground_truth(
            output,
            expected_queries=query_vectors.shape[0],
            expected_k=top_k,
            expected_corpus=corpus.shape[0],
        )
        if declared.get("input_hash") != input_hash:
            raise ValueError("existing ground truth belongs to different inputs")
        if declared.get("ground_truth", {}).get("sha256") != sha256_file(output):
            raise ValueError("existing ground-truth SHA-256 mismatch")
        return declared
    if output.exists() and not meta_path.exists():
        state_path = checkpoint_path(output)
        if not state_path.exists():
            raise ValueError(
                "existing ground truth has no provenance sidecar/checkpoint"
            )
        state = load_json(state_path)
        if (
            state.get("status") != "ready-to-publish"
            or state.get("input_hash") != input_hash
            or state.get("ground_truth_sha256") != sha256_file(output)
        ):
            raise ValueError("ground-truth publication checkpoint mismatch")
        _, truth = read_ground_truth(
            output,
            expected_queries=query_vectors.shape[0],
            expected_k=top_k,
            expected_corpus=corpus.shape[0],
        )
        declared = {
            **stable,
            "input_hash": input_hash,
            "created_at": utc_now(),
            "generation_started_at": checkpoint_created_at(
                state, "ground-truth generation"
            ),
            "ground_truth": truth,
            "note": "Sidecar recovered from a verified ready-to-publish checkpoint.",
        }
        atomic_write_json(meta_path, declared)
        state_path.unlink()
        return declared

    partial = partial_path(output)
    checkpoint = checkpoint_path(output)
    completed = 0
    generation_started_at = utc_now()
    header = (
        f"LEANN_GT1 {query_vectors.shape[0]} {top_k} {corpus.shape[0]}\n"
    ).encode("ascii")
    if checkpoint.exists() or partial.exists():
        if not checkpoint.exists() or not partial.exists():
            raise RuntimeError("ground-truth checkpoint and partial must coexist")
        state = load_json(checkpoint)
        if state.get("input_hash") != input_hash:
            raise RuntimeError("ground-truth checkpoint belongs to different inputs")
        generation_started_at = checkpoint_created_at(
            state, "ground-truth generation"
        )
        completed = int(state["completed_queries"])
        committed_length = state.get("committed_length_bytes")
        committed_sha = state.get(
            "committed_prefix_sha256", state.get("partial_sha256")
        )
        if committed_length is None:
            committed_length = partial.stat().st_size
        committed_length = int(committed_length)
        observed_length = partial.stat().st_size
        if observed_length < committed_length:
            raise RuntimeError(
                "ground-truth partial is shorter than its committed checkpoint"
            )
        if (
            sha256_file(partial, length=committed_length)
            != committed_sha
        ):
            raise RuntimeError("ground-truth committed-prefix checksum mismatch")
        if observed_length > committed_length:
            with partial.open("r+b") as target:
                target.truncate(committed_length)
                target.flush()
                os.fsync(target.fileno())
        lines = list(iter_nonempty_lines(partial))
        if len(lines) != completed + 1 or lines[0] != header.decode().strip():
            raise RuntimeError("ground-truth partial row count/header mismatch")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        with partial.open("xb") as target:
            target.write(header)
            target.flush()
            os.fsync(target.fileno())
        atomic_write_json(
            checkpoint,
            {
                **stable,
                "input_hash": input_hash,
                "created_at": generation_started_at,
                "status": "running",
                "completed_queries": 0,
                "committed_length_bytes": len(header),
                "committed_prefix_sha256": sha256_file(partial),
                "updated_at": utc_now(),
            },
        )

    for begin in range(completed, query_vectors.shape[0], query_block_rows):
        end = min(query_vectors.shape[0], begin + query_block_rows)
        identifiers = exact_topk_block(
            corpus,
            np.asarray(query_vectors[begin:end], dtype=np.float32),
            top_k=top_k,
            corpus_block_rows=corpus_block_rows,
        )
        payload = "".join(
            " ".join(str(int(identifier)) for identifier in row) + "\n"
            for row in identifiers
        ).encode("ascii")
        with partial.open("ab") as target:
            target.write(payload)
            target.flush()
            os.fsync(target.fileno())
        completed = end
        committed_length = partial.stat().st_size
        committed_sha = sha256_file(partial, length=committed_length)
        atomic_write_json(
            checkpoint,
            {
                **stable,
                "input_hash": input_hash,
                "created_at": generation_started_at,
                "status": "running",
                "completed_queries": completed,
                "committed_length_bytes": committed_length,
                "committed_prefix_sha256": committed_sha,
                "updated_at": utc_now(),
            },
        )
        print(
            f"exact ground truth: {completed}/{query_vectors.shape[0]} queries",
            flush=True,
        )

    _, truth = read_ground_truth(
        partial,
        expected_queries=query_vectors.shape[0],
        expected_k=top_k,
        expected_corpus=corpus.shape[0],
    )
    ground_truth_sha = sha256_file(partial)
    atomic_write_json(
        checkpoint,
        {
            **stable,
            "input_hash": input_hash,
            "created_at": generation_started_at,
            "status": "ready-to-publish",
            "completed_queries": completed,
            "committed_length_bytes": partial.stat().st_size,
            "committed_prefix_sha256": ground_truth_sha,
            "ground_truth_sha256": ground_truth_sha,
            "updated_at": utc_now(),
        },
    )
    os.replace(partial, output)
    truth["path"] = str(output.resolve())
    declared = {
        **stable,
        "input_hash": input_hash,
        "created_at": utc_now(),
        "generation_started_at": generation_started_at,
        "ground_truth": truth,
    }
    atomic_write_json(meta_path, declared)
    checkpoint.unlink(missing_ok=True)
    return declared


def run_observation(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str] | None = None,
) -> dict[str, Any]:
    if not command or any(not isinstance(item, str) for item in command):
        raise ValueError("command must be a non-empty argv string list")
    started_at = utc_now()
    started = time.perf_counter()
    usage = None
    with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=None if environment is None else {**os.environ, **environment},
            stdout=stdout_file,
            stderr=stderr_file,
        )
        if hasattr(os, "wait4"):
            _, status, usage = os.wait4(process.pid, 0)
            process.returncode = os.waitstatus_to_exitcode(status)
        else:
            process.wait()
        stdout_file.seek(0)
        stderr_file.seek(0)
        stdout = stdout_file.read().decode("utf-8", errors="replace")
        stderr = stderr_file.read().decode("utf-8", errors="replace")
    wall = time.perf_counter() - started
    peak_rss = None
    rss_source = "unavailable"
    if usage is not None:
        peak_rss = int(
            usage.ru_maxrss if sys.platform == "darwin" else usage.ru_maxrss * 1024
        )
        rss_source = (
            "wait4 direct-child ru_maxrss "
            + ("(bytes)" if sys.platform == "darwin" else "(KiB converted to bytes)")
        )
    executable = Path(command[0])
    executable_snapshot = None
    if executable.is_file():
        executable_snapshot = snapshot_files([executable])[0]
    return {
        "command": command,
        "command_display": shlex.join(command),
        "measurement_wrapper": None,
        "cwd": str(cwd.resolve()),
        "environment_overrides": environment or {},
        "started_at": started_at,
        "finished_at": utc_now(),
        "wall_seconds": wall,
        "peak_rss_bytes": peak_rss,
        "peak_rss_source": rss_source,
        "exit_code": process.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "executable": executable_snapshot,
    }


def parse_native_metrics(output: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for line in output.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        try:
            metrics[key] = int(value)
            continue
        except ValueError:
            pass
        try:
            metrics[key] = float(value)
        except ValueError:
            metrics[key] = value
    return metrics


def prefix_artifacts(prefix: Path, *, official: bool = False) -> list[Path]:
    prefix = prefix.resolve()
    pattern = f"{prefix.stem}*" if official else f"{prefix.name}.*"
    return [
        path
        for path in prefix.parent.glob(pattern)
        if path.is_file() and not path.name.endswith((".tmp", ".bak"))
    ]


def run_repeated_stage(
    *,
    name: str,
    state_dir: Path,
    input_payload: dict[str, Any],
    command_factory: Any,
    artifact_factory: Any,
    cwd: Path,
    warmups: int,
    repetitions: int,
) -> dict[str, Any]:
    if warmups < 0 or repetitions <= 0:
        raise ValueError("warmups must be non-negative and repetitions positive")
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / f"{name}.json"
    execution_protocol = {
        "warmups": warmups,
        "repetitions": repetitions,
    }
    if (
        "execution_protocol" in input_payload
        and input_payload["execution_protocol"] != execution_protocol
    ):
        raise ValueError(
            f"{name}: input execution protocol disagrees with requested counts"
        )
    stage_inputs = {
        **input_payload,
        "execution_protocol": execution_protocol,
    }
    input_hash = canonical_hash(stage_inputs)
    if state_path.exists():
        state = load_json(state_path)
        if (
            state.get("schema") != "leann-command-stage-v1"
            or state.get("name") != name
            or state.get("inputs") != stage_inputs
        ):
            raise RuntimeError(f"{name}: existing stage state identity mismatch")
        if state.get("input_hash") != input_hash:
            raise RuntimeError(
                f"{name}: existing stage state belongs to different inputs"
            )
        if state.get("status") not in {"running", "failed", "complete"}:
            raise RuntimeError(f"{name}: existing stage status is invalid")
        for kind, wanted in (("warmups", warmups), ("measurements", repetitions)):
            observations = state.get(kind)
            if not isinstance(observations, list):
                raise RuntimeError(f"{name}: {kind} state is not an array")
            if len(observations) > wanted:
                raise RuntimeError(
                    f"{name}: checkpoint contains more {kind} than requested"
                )
            for ordinal, observation in enumerate(observations):
                if (
                    not isinstance(observation, dict)
                    or observation.get("kind") != kind
                    or observation.get("ordinal") != ordinal
                ):
                    raise RuntimeError(
                        f"{name}: {kind} observation {ordinal} identity mismatch"
                    )
        if state.get("status") == "complete":
            if (
                len(state["warmups"]) != warmups
                or len(state["measurements"]) != repetitions
            ):
                raise RuntimeError(
                    f"{name}: completed stage has incomplete observation counts"
                )
            recorded = state["measurements"][-1].get("artifacts", [])
            current = snapshot_files(artifact_factory())
            if current != recorded:
                raise RuntimeError(
                    f"{name}: completed-stage artifacts changed; refuse stale reuse"
                )
            return state
    else:
        state = {
            "schema": "leann-command-stage-v1",
            "name": name,
            "input_hash": input_hash,
            "inputs": stage_inputs,
            "status": "running",
            "warmups": [],
            "measurements": [],
            "failed_attempts": [],
            "created_at": utc_now(),
        }

    for kind, wanted in (("warmups", warmups), ("measurements", repetitions)):
        while len(state[kind]) < wanted:
            ordinal = len(state[kind])
            command, output_path = command_factory(kind, ordinal)
            observation = run_observation(command, cwd=cwd)
            observation["kind"] = kind
            observation["ordinal"] = ordinal
            if output_path is not None and output_path.exists():
                observation["result_file"] = snapshot_files([output_path])[0]
                try:
                    observation["result"] = load_json(output_path)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    observation["raw_result"] = output_path.read_text(
                        encoding="utf-8", errors="replace"
                    )
            if name.startswith("native-"):
                observation["parsed_stdout"] = parse_native_metrics(
                    observation["stdout"]
                )
            observation["artifacts"] = snapshot_files(artifact_factory())
            if observation["exit_code"] != 0:
                state["failed_attempts"].append(observation)
                state["status"] = "failed"
                state["updated_at"] = utc_now()
                atomic_write_json(state_path, state)
                raise RuntimeError(
                    f"{name} failed with exit {observation['exit_code']}; "
                    f"see {state_path}"
                )
            state[kind].append(observation)
            state["status"] = "running"
            state["updated_at"] = utc_now()
            atomic_write_json(state_path, state)

    state["status"] = "complete"
    state["completed_at"] = utc_now()
    atomic_write_json(state_path, state)
    return state


def git_provenance(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path.resolve())}
    for key, command in {
        "commit": ["rev-parse", "HEAD"],
        "commit_date": ["show", "-s", "--format=%cI", "HEAD"],
        "branch": ["rev-parse", "--abbrev-ref", "HEAD"],
        "status_porcelain": ["status", "--short"],
        "remote": ["remote", "get-url", "origin"],
    }.items():
        try:
            result[key] = subprocess.check_output(
                ["git", "-C", str(path), *command],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            result[key] = None
    return result


def official_runtime_provenance(
    python: Path, repository: Path
) -> dict[str, Any]:
    probe = r'''
import hashlib, importlib, importlib.metadata, json, pathlib, sys

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

result = {"python": sys.version, "executable": str(pathlib.Path(sys.executable).resolve()), "modules": {}, "distributions": {}}
for name in (
    "leann",
    "leann_backend_hnsw",
    "leann_backend_hnsw.faiss",
    "leann_backend_hnsw._swigfaiss",
    "leann_backend_hnsw.hnsw_backend",
):
    module = importlib.import_module(name)
    path = pathlib.Path(module.__file__).resolve()
    root = path.parent if path.name == "__init__.py" else path
    files = [path]
    if root.is_dir():
        files = sorted(
            item for item in root.rglob("*")
            if item.is_file() and item.suffix in {".py", ".so", ".dylib", ".pyd"}
        )
    result["modules"][name] = {
        "path": str(path),
        "files": [
            {"path": str(item), "size_bytes": item.stat().st_size, "sha256": digest(item)}
            for item in files
        ],
    }
faiss_module = importlib.import_module("leann_backend_hnsw.faiss")
faiss_path = pathlib.Path(faiss_module.__file__).resolve()
result["backend_faiss"] = {
    "path": str(faiss_path),
    "size_bytes": faiss_path.stat().st_size,
    "sha256": digest(faiss_path),
}
packages = importlib.metadata.packages_distributions()
for import_name in ("leann", "leann_backend_hnsw"):
    for distribution in packages.get(import_name, []):
        try:
            result["distributions"][distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            pass
print("LEANN_RUNTIME_PROVENANCE=" + json.dumps(result, sort_keys=True))
'''
    launcher = Path(os.path.abspath(python))
    completed = subprocess.run(
        [str(launcher), "-c", probe],
        cwd=repository.resolve(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "official runtime provenance probe failed: "
            + completed.stderr[-4000:]
        )
    marker = "LEANN_RUNTIME_PROVENANCE="
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(marker):
            result = json.loads(line[len(marker) :])
            result["launcher"] = str(launcher)
            result["launcher_resolved"] = str(launcher.resolve())
            return result
    raise RuntimeError(
        "official runtime provenance probe returned no structured result: "
        + completed.stderr[-1000:]
    )


def endpoint_provenance(base_url: str) -> dict[str, Any]:
    normalized = normalized_base_url(base_url)
    result: dict[str, Any] = {"base_url": normalized}
    for endpoint in ("/health", "/props", "/metrics"):
        try:
            with urllib.request.urlopen(
                normalized + endpoint, timeout=5
            ) as response:
                body = response.read(1024 * 1024)
                try:
                    payload: Any = json.loads(body)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    payload = body.decode("utf-8", errors="replace")
                result[endpoint] = {
                    "status": response.status,
                    "payload": payload,
                }
        except (urllib.error.URLError, TimeoutError) as error:
            result[endpoint] = {"error": str(error)}
    return result


def normalized_base_url(value: str) -> str:
    result = value.rstrip("/")
    if result.endswith("/v1"):
        result = result[:-3]
    return result.rstrip("/")


def endpoint_payload(observation: dict[str, Any], path: str) -> Any:
    record = observation.get(path)
    if not isinstance(record, dict) or record.get("status") != 200:
        raise ValueError(f"endpoint identity probe {path} did not return HTTP 200")
    return record.get("payload")


def validate_endpoint_identity(
    args: argparse.Namespace,
    shared: dict[str, Any],
    *,
    embedding_observation: dict[str, Any],
    candidate_observation: dict[str, Any],
    tooling_by_name: dict[str, dict[str, Any]],
) -> None:
    candidate_metrics = endpoint_payload(candidate_observation, "/metrics")
    if not isinstance(candidate_metrics, dict):
        raise ValueError("candidate endpoint metrics payload is not an object")
    if args.official_recompute_mode == "cached":
        if candidate_metrics.get("mode") != "cache":
            raise ValueError("cached run requires a direct cached endpoint")
        identity = candidate_metrics.get("cache_identity")
        if not isinstance(identity, dict):
            raise ValueError("cached endpoint did not expose cache identity")
        expected = {
            "cache_sha256": shared["corpus_cache"]["sha256"],
            "source_sha256": shared["documents"]["sha256"],
            "fingerprint": shared["corpus_cache"]["fingerprint"],
            "dimensions": shared["corpus_cache"]["dimensions"],
            "count": shared["corpus_cache"]["count"],
            "model_sha256": shared["model"]["sha256"],
            "server_script_sha256": tooling_by_name[
                "cached_embedding_server.py"
            ]["sha256"],
        }
        for key, wanted in expected.items():
            if identity.get(key) != wanted:
                raise ValueError(
                    f"cached endpoint {key} mismatch: "
                    f"{identity.get(key)!r} vs {wanted!r}"
                )
        return

    if candidate_metrics.get("mode") != "proxy":
        raise ValueError("real recompute run requires the counting proxy")
    if normalized_base_url(str(candidate_metrics.get("upstream", ""))) != (
        normalized_base_url(args.embedding_url)
    ):
        raise ValueError("real recompute proxy upstream does not match embedding URL")
    if candidate_metrics.get("proxy_script_sha256") != tooling_by_name[
        "openai_embedding_proxy.py"
    ]["sha256"]:
        raise ValueError("running proxy script does not match benchmark tooling")
    artifact = shared["model"].get("artifact")
    if not isinstance(artifact, dict) or not artifact.get("path"):
        raise ValueError("real matched run requires a SHA-bound model artifact")
    props = endpoint_payload(embedding_observation, "/props")
    if not isinstance(props, dict) or not props.get("model_path"):
        raise ValueError("real embedding endpoint did not expose model_path")
    if Path(str(props["model_path"])).resolve() != Path(artifact["path"]).resolve():
        raise ValueError("real embedding endpoint model_path differs from shared model")
    if "total_slots" in props and int(props["total_slots"]) != args.native_parallel:
        raise ValueError("real embedding endpoint parallel-slot count mismatch")
    settings = props.get("default_generation_settings")
    if isinstance(settings, dict) and "n_ctx" in settings:
        if int(settings["n_ctx"]) != args.native_ctx:
            raise ValueError("real embedding endpoint per-slot context mismatch")


def validate_shared_artifacts(args: argparse.Namespace) -> dict[str, Any]:
    corpus, corpus_meta = read_cache(args.corpus_cache)
    query_vectors, query_meta = read_cache(args.query_cache)
    documents = validate_cache_source(
        corpus_meta, args.documents, require_integrity=True
    )
    queries = validate_cache_source(query_meta, args.queries, require_integrity=True)
    if corpus.shape[1] != query_vectors.shape[1]:
        raise ValueError("shared cache dimensions do not match")
    if corpus_meta["fingerprint"] != query_meta["fingerprint"]:
        raise ValueError("shared cache native fingerprints do not match")
    corpus_sha = sha256_file(args.corpus_cache)
    query_sha = sha256_file(args.query_cache)
    corpus_sidecar = load_json(sidecar_path(args.corpus_cache))
    query_sidecar = load_json(sidecar_path(args.query_cache))
    for label, declared, expected_role, actual_sha, source in (
        ("corpus", corpus_sidecar, "corpus", corpus_sha, documents),
        ("query", query_sidecar, "queries", query_sha, queries),
    ):
        if (
            declared.get("schema") != CACHE_META_SCHEMA
            or declared.get("role") != expected_role
            or declared.get("cache", {}).get("sha256") != actual_sha
            or declared.get("source", {}).get("sha256") != source["sha256"]
        ):
            raise ValueError(f"{label} cache provenance sidecar mismatch")
    if corpus_sidecar["model"]["sha256"] != query_sidecar["model"]["sha256"]:
        raise ValueError("shared cache model hashes do not match")
    if query_sidecar.get("bindings", {}).get("corpus_cache_sha256") != corpus_sha:
        raise ValueError("query cache is not bound to selected corpus cache")
    _, truth = read_ground_truth(
        args.ground_truth,
        expected_queries=query_vectors.shape[0],
        expected_k=args.top_k,
        expected_corpus=corpus.shape[0],
    )
    truth_sidecar = load_json(sidecar_path(args.ground_truth))
    if (
        truth_sidecar.get("schema") != GT_META_SCHEMA
        or truth_sidecar.get("ground_truth", {}).get("sha256")
        != truth["sha256"]
    ):
        raise ValueError("ground-truth provenance sidecar mismatch")
    required = {
        "corpus_cache_sha256": corpus_sha,
        "query_cache_sha256": query_sha,
        "corpus_source_sha256": documents["sha256"],
        "query_source_sha256": queries["sha256"],
        "model_sha256": corpus_sidecar["model"]["sha256"],
    }
    for key, expected in required.items():
        if truth_sidecar.get(key) != expected:
            raise ValueError(f"ground-truth {key} binding mismatch")
    del corpus, query_vectors
    return {
        "documents": documents,
        "queries": queries,
        "corpus_cache": {**corpus_meta, "sha256": corpus_sha},
        "query_cache": {**query_meta, "sha256": query_sha},
        "ground_truth": truth,
        "model": corpus_sidecar["model"],
    }


def native_commands(
    args: argparse.Namespace,
) -> tuple[list[str], dict[int, list[str]]]:
    common = [
        "--embedder",
        "llama",
        "--model",
        str(args.native_model.resolve()),
        "--gpu-layers",
        str(args.native_gpu_layers),
        "--parallel",
        str(args.native_parallel),
        "--ctx",
        str(args.native_ctx),
        "--batch-tokens",
        str(args.native_batch_tokens),
    ]
    build = [
        str(args.native_binary.resolve()),
        "build",
        "--docs",
        str(args.documents.resolve()),
        "--index",
        str(args.native_index.resolve()),
        "--embedder",
        "cache",
        "--embedding-cache",
        str(args.corpus_cache.resolve()),
        *args.native_build_extra,
    ]
    additional_cutoffs = [value for value in args.recall_k if value != args.top_k]
    if len(additional_cutoffs) > 1:
        raise ValueError(
            "native CLI supports one --report-k cutoff in addition to --top-k"
        )
    report_arguments = (
        ["--report-k", str(additional_cutoffs[0])] if additional_cutoffs else []
    )
    searches = {
        ef_search: [
            str(args.native_binary.resolve()),
            "bench",
            "--index",
            str(args.native_index.resolve()),
            "--queries",
            str(args.queries.resolve()),
            "--query-embedding-cache",
            str(args.query_cache.resolve()),
            "--ground-truth",
            str(args.ground_truth.resolve()),
            "--top-k",
            str(args.top_k),
            *report_arguments,
            "--ef-search",
            str(ef_search),
            "--scan-limit",
            str(args.native_scan_limit),
            "--rerank-ratio",
            str(args.native_rerank_ratio),
            "--recompute-batch",
            str(args.native_recompute_batch),
            "--dense-baseline",
            "0",
            "--warmup-queries",
            str(args.native_warmup_queries),
            *common,
            *args.native_search_extra,
        ]
        for ef_search in args.native_ef_search
    }
    return build, searches


def official_command(
    args: argparse.Namespace,
    *,
    mode: str,
    output: Path,
) -> list[str]:
    command = [
        str(Path(os.path.abspath(args.official_python))),
        str(args.official_script.resolve()),
        f"--{mode}",
        "--documents",
        str(args.documents.resolve()),
        "--queries",
        str(args.queries.resolve()),
        "--corpus-cache",
        str(args.corpus_cache.resolve()),
        "--query-cache",
        str(args.query_cache.resolve()),
        "--ground-truth",
        str(args.ground_truth.resolve()),
        "--top-k",
        str(args.top_k),
        "--report-k",
        *[str(value) for value in args.recall_k],
        "--index",
        str(args.official_index.resolve()),
        "--output",
        str(output.resolve()),
        "--official-repo",
        str(args.official_repo.resolve()),
        "--llama-url",
        args.embedding_url,
        "--proxy-url",
        args.proxy_url,
        "--recompute-mode",
        args.official_recompute_mode,
        "--model-alias",
        args.embedding_model,
        "--m",
        str(args.official_m),
        "--ef-construction",
        str(args.official_ef_construction),
        "--complexities",
        *[str(value) for value in args.official_complexities],
        "--batch-sizes",
        *[str(value) for value in args.official_batch_sizes],
        *args.official_extra,
    ]
    return command


def benchmark_run_role(args: argparse.Namespace) -> str:
    components: list[str] = []
    if not args.skip_native:
        if args.native_role == "gate" and len(args.native_ef_search) != 1:
            raise ValueError(
                "--native-role gate requires exactly one --native-ef-search value"
            )
        components.append(f"native-{args.native_role}")
    if not args.skip_official:
        components.append(
            "official-real-run"
            if args.official_recompute_mode == "real"
            else "official-cached-sweep"
        )
    if not components:
        raise ValueError("benchmark must enable at least one implementation")
    return "+".join(components)


def orchestrate(args: argparse.Namespace) -> dict[str, Any]:
    run_role = benchmark_run_role(args)
    shared = validate_shared_artifacts(args)
    if args.native_ctx * args.native_parallel > args.server_ctx_size:
        raise ValueError(
            "server context is undersized: require --server-ctx-size >= "
            "--native-ctx * --native-parallel (2048 * 16 = 32768 recommended)"
        )
    output = args.output.resolve()
    stages_dir = output / "stages"
    results_dir = output / "raw-results"
    results_dir.mkdir(parents=True, exist_ok=True)
    stages: dict[str, Any] = {}
    root = Path(__file__).resolve().parent.parent
    tooling_paths = [
        Path(__file__).resolve(),
        (root / "scripts/benchmark_cache.py").resolve(),
        args.official_script.resolve(),
        (root / "scripts/cached_embedding_server.py").resolve(),
        (root / "scripts/openai_embedding_proxy.py").resolve(),
    ]
    tooling = snapshot_files(tooling_paths)
    tooling_by_name = {
        Path(item["path"]).name: item for item in tooling
    }
    if not args.skip_native:
        native_model_sha256 = sha256_file(args.native_model)
        artifact = shared["model"].get("artifact")
        if (
            not isinstance(artifact, dict)
            or artifact.get("sha256") != native_model_sha256
        ):
            raise ValueError(
                "--native-model SHA-256 differs from the shared model artifact"
            )
    else:
        native_model_sha256 = None
    embedding_observation = endpoint_provenance(args.embedding_url)
    candidate_observation = endpoint_provenance(args.proxy_url)
    if not args.skip_official:
        validate_endpoint_identity(
            args,
            shared,
            embedding_observation=embedding_observation,
            candidate_observation=candidate_observation,
            tooling_by_name=tooling_by_name,
        )
        official_runtime = official_runtime_provenance(
            args.official_python, args.official_repo
        )
    else:
        official_runtime = None
    provenance = {
        "leann_cpp": git_provenance(root),
        "official_leann": git_provenance(args.official_repo),
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": sys.version,
            "numpy": np.__version__,
        },
        "embedding_endpoint": args.embedding_url,
        "proxy_endpoint": args.proxy_url,
        "embedding_endpoint_observation": embedding_observation,
        "candidate_endpoint_observation": candidate_observation,
        "tooling": tooling,
        "server_context": {
            "server_ctx_size": args.server_ctx_size,
            "parallel": args.native_parallel,
            "per_slot_ctx": args.native_ctx,
            "required_product": args.native_ctx * args.native_parallel,
        },
        "official_candidate_recompute_mode": args.official_recompute_mode,
        "official_runtime": official_runtime,
    }
    protocol = {
        "run_role": run_role,
        "build_warmups": args.build_warmups,
        "build_repetitions": args.build_repetitions,
        "search_warmups": args.search_warmups,
        "search_repetitions": args.search_repetitions,
        "recall_cutoffs": args.recall_k,
        "timing": (
            "Primary search latency is the implementation's point-internal "
            "per-query timing (native raw CSV and official raw_per_query). "
            "Whole-subprocess wall time is orchestration telemetry only and "
            "is not comparable because commands group sweep points differently."
        ),
        "rss": (
            "Direct-client ru_maxrss only. Native search includes its in-process "
            "GGUF model; official recompute uses external server/daemon memory. "
            "Do not compare or headline cross-system search RSS from this field. "
            "Cache-only build RSS is more comparable."
        ),
        "latency_comparability": (
            "Native candidate recompute is always real GGUF. Official cached "
            "mode is an algorithmic recall/candidate Pareto sweep only; no "
            "cross-system latency speedup may be derived from it. Headline "
            "latency requires an official real-mode run against the attested model."
        ),
    }
    planned_stages: list[str] = []
    if not args.skip_native:
        planned_stages.extend(
            ["native-build", "native-stats"]
            + [f"native-search-ef{value}" for value in args.native_ef_search]
        )
    if not args.skip_official:
        planned_stages.extend(
            ["official-build", f"official-search-{args.official_recompute_mode}"]
        )
    started_at = utc_now()
    atomic_write_json(
        output / "manifest.json",
        {
            "schema": SCHEMA,
            "created_at": started_at,
            "status": "running",
            "run_role": run_role,
            "shared_artifacts": shared,
            "provenance": provenance,
            "protocol": protocol,
            "planned_stages": planned_stages,
            "stages_directory": str(stages_dir.resolve()),
        },
    )

    if not args.skip_native:
        build_command, search_commands = native_commands(args)
        native_inputs = {
            "shared": shared,
            "tooling": tooling,
            "candidate_recompute_mode": "real-gguf-in-process",
            "binary_sha256": sha256_file(args.native_binary),
            "model_artifact_sha256": native_model_sha256,
            "server_context": {
                "server_ctx_size": args.server_ctx_size,
                "parallel": args.native_parallel,
                "per_slot_ctx": args.native_ctx,
                "required_product": args.native_ctx * args.native_parallel,
            },
        }
        stages["native-build"] = run_repeated_stage(
            name="native-build",
            state_dir=stages_dir,
            input_payload={**native_inputs, "command": build_command},
            command_factory=lambda _kind, _ordinal: (build_command, None),
            artifact_factory=lambda: prefix_artifacts(args.native_index),
            cwd=root,
            warmups=args.build_warmups,
            repetitions=args.build_repetitions,
        )
        stats_command = [
            str(args.native_binary.resolve()),
            "stats",
            "--index",
            str(args.native_index.resolve()),
        ]
        stages["native-stats"] = run_repeated_stage(
            name="native-stats",
            state_dir=stages_dir,
            input_payload={**native_inputs, "command": stats_command},
            command_factory=lambda _kind, _ordinal: (stats_command, None),
            artifact_factory=lambda: prefix_artifacts(args.native_index),
            cwd=root,
            warmups=0,
            repetitions=1,
        )
        for ef_search, search_command in search_commands.items():
            stage_name = f"native-search-ef{ef_search}"

            def native_search_factory(
                kind: str,
                ordinal: int,
                *,
                command: list[str] = search_command,
                ef_value: int = ef_search,
            ) -> tuple[list[str], Path]:
                raw_path = (
                    results_dir
                    / (
                        f"native-search-ef{ef_value}-"
                        f"{kind}-{ordinal}.csv"
                    )
                )
                return [
                    *command,
                    "--raw-latencies",
                    str(raw_path.resolve()),
                ], raw_path

            stages[stage_name] = run_repeated_stage(
                name=stage_name,
                state_dir=stages_dir,
                input_payload={
                    **native_inputs,
                    "recall_k": args.recall_k,
                    "ef_search": ef_search,
                    "ground_truth_k": args.top_k,
                    "command_template": [
                        *search_command,
                        "--raw-latencies",
                        str(
                            (
                                results_dir
                                / (
                                    f"native-search-ef{ef_search}-"
                                    "{kind}-{ordinal}.csv"
                                )
                            ).resolve()
                        ),
                    ],
                },
                command_factory=native_search_factory,
                artifact_factory=lambda: prefix_artifacts(args.native_index),
                cwd=root,
                warmups=args.search_warmups,
                repetitions=args.search_repetitions,
            )

    if not args.skip_official:
        official_inputs = {
            "shared": shared,
            "tooling": tooling,
            "candidate_recompute_mode": args.official_recompute_mode,
            "official_repo": git_provenance(args.official_repo),
            "official_runtime": official_runtime,
            "python": snapshot_files([args.official_python])[0]
            if args.official_python.is_file()
            else str(args.official_python),
        }

        def official_factory(mode: str, output_label: str) -> Any:
            def factory(kind: str, ordinal: int) -> tuple[list[str], Path]:
                result_path = (
                    results_dir
                    / f"official-{output_label}-{kind}-{ordinal}.json"
                )
                return official_command(
                    args, mode=mode, output=result_path
                ), result_path

            return factory

        stages["official-build"] = run_repeated_stage(
            name="official-build",
            state_dir=stages_dir,
            input_payload={
                **official_inputs,
                "command_template": official_command(
                    args,
                    mode="build",
                    output=results_dir / "official-build-{kind}-{ordinal}.json",
                ),
            },
            command_factory=official_factory("build", "build"),
            artifact_factory=lambda: prefix_artifacts(
                args.official_index, official=True
            ),
            cwd=args.official_repo,
            warmups=args.build_warmups,
            repetitions=args.build_repetitions,
        )
        official_stage_name = (
            f"official-search-{args.official_recompute_mode}"
        )
        stages[official_stage_name] = run_repeated_stage(
            name=official_stage_name,
            state_dir=stages_dir,
            input_payload={
                **official_inputs,
                "candidate_recompute_mode": args.official_recompute_mode,
                "command_template": official_command(
                    args,
                    mode="benchmark",
                    output=results_dir / "official-search-{kind}-{ordinal}.json",
                ),
            },
            command_factory=official_factory("benchmark", "search"),
            artifact_factory=lambda: prefix_artifacts(
                args.official_index, official=True
            ),
            cwd=args.official_repo,
            warmups=args.search_warmups,
            repetitions=args.search_repetitions,
        )

    manifest = {
        "schema": SCHEMA,
        "created_at": started_at,
        "completed_at": utc_now(),
        "status": "complete",
        "run_role": run_role,
        "shared_artifacts": shared,
        "provenance": provenance,
        "protocol": protocol,
        "planned_stages": planned_stages,
        "stages": {
            name: str((stages_dir / f"{name}.json").resolve()) for name in stages
        },
    }
    atomic_write_json(output / "manifest.json", manifest)
    return manifest


def add_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--corpus-cache", type=Path, required=True)
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Encoded exact-truth width; use 10 to report Recall@3 and Recall@10",
    )
    parser.add_argument(
        "--recall-k",
        type=int,
        nargs="+",
        default=[3, 10],
        help="Recall cutoffs run from prefixes of the single top-k truth",
    )


def add_embedding_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--embedding-url",
        default="http://127.0.0.1:18080",
        help="OpenAI-compatible base URL (the /v1 suffix is optional)",
    )
    parser.add_argument("--embedding-model", default="nomic-embed-text")
    parser.add_argument(
        "--fingerprint",
        required=True,
        help="Exact native llama.cpp embedder fingerprint stored in LEANNBC2",
    )
    parser.add_argument(
        "--model-identity",
        default="",
        help="Human-readable immutable model/build identity",
    )
    parser.add_argument(
        "--model-artifact",
        type=Path,
        help="Optional GGUF artifact; its SHA-256 becomes the strongest model binding",
    )
    parser.add_argument("--api-key")
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument(
        "--embedding-concurrency",
        type=int,
        default=1,
        help=(
            "batches to keep in flight against the embedding endpoint. Results "
            "are still consumed in source order, so the cache bytes are "
            "unchanged; this only stops the endpoint idling between requests. "
            "Requires an explicit --dimension."
        ),
    )
    parser.add_argument("--embedding-timeout", type=int, default=600)
    parser.add_argument("--embedding-retries", type=int, default=4)
    parser.add_argument("--dimension", type=int)
    parser.add_argument(
        "--prefix-cache",
        type=Path,
        help=(
            "Seed a larger corpus cache from a proven LEANNBC2 prefix cache; "
            "source byte prefix/model/fingerprint/dimension are all verified"
        ),
    )


def add_orchestration_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--embedding-url", default="http://127.0.0.1:18080")
    parser.add_argument("--embedding-model", default="nomic-embed-text")
    parser.add_argument(
        "--proxy-url",
        default="http://127.0.0.1:18082",
        help=(
            "Official candidate endpoint. Default is direct cached server; "
            "use the counting real-GGUF proxy (typically :18081) with "
            "--official-recompute-mode real."
        ),
    )
    parser.add_argument(
        "--server-ctx-size",
        type=int,
        default=32768,
        help="llama-server total context; must cover native ctx * parallel",
    )
    parser.add_argument("--build-warmups", type=int, default=0)
    parser.add_argument("--build-repetitions", type=int, default=1)
    parser.add_argument("--search-warmups", type=int, default=1)
    parser.add_argument("--search-repetitions", type=int, default=3)
    parser.add_argument("--skip-native", action="store_true")
    parser.add_argument("--native-binary", type=Path, default=Path("build-llama/leann"))
    parser.add_argument("--native-index", type=Path, default=Path("out/large/native"))
    parser.add_argument("--native-model", type=Path)
    parser.add_argument("--native-gpu-layers", type=int, default=99)
    parser.add_argument("--native-parallel", type=int, default=16)
    parser.add_argument("--native-ctx", type=int, default=2048)
    parser.add_argument("--native-batch-tokens", type=int, default=32768)
    parser.add_argument("--native-warmup-queries", type=int, default=1)
    parser.add_argument(
        "--native-role",
        choices=("gate", "sweep"),
        default="sweep",
        help=(
            "Declare whether this run is a one-point smoke gate or a publication "
            "sweep; the role is recorded explicitly and never inferred."
        ),
    )
    parser.add_argument(
        "--native-ef-search",
        type=int,
        nargs="+",
        default=[32, 64, 128, 256, 512],
    )
    parser.add_argument("--native-scan-limit", type=int, default=0)
    parser.add_argument("--native-rerank-ratio", type=float, default=0.25)
    parser.add_argument("--native-recompute-batch", type=int, default=16)
    parser.add_argument(
        "--native-build-extra",
        type=parse_json_array,
        default=["--graph-degree", "32", "--ef-construction", "200"],
        metavar="JSON_ARRAY",
        help=(
            "JSON argv appended to native build; default matches official "
            "M=32/efConstruction=200 while retaining compact-index defaults"
        ),
    )
    parser.add_argument(
        "--native-search-extra",
        type=parse_json_array,
        default=[],
        metavar="JSON_ARRAY",
    )
    parser.add_argument("--skip-official", action="store_true")
    parser.add_argument(
        "--official-python",
        type=Path,
        default=Path("work/official-leann-venv/bin/python"),
    )
    parser.add_argument(
        "--official-script",
        type=Path,
        default=Path("scripts/compare_official_leann.py"),
    )
    parser.add_argument(
        "--official-repo", type=Path, default=Path("work/reference/LEANN")
    )
    parser.add_argument(
        "--official-index",
        type=Path,
        default=Path("out/large/official/benchmark.leann"),
    )
    parser.add_argument("--official-m", type=int, default=32)
    parser.add_argument("--official-ef-construction", type=int, default=200)
    parser.add_argument(
        "--official-complexities",
        type=int,
        nargs="+",
        default=[32, 64, 128, 256, 512],
    )
    parser.add_argument(
        "--official-batch-sizes", type=int, nargs="+", default=[0, 16]
    )
    parser.add_argument(
        "--official-extra",
        type=parse_json_array,
        default=[],
        metavar="JSON_ARRAY",
    )
    parser.add_argument(
        "--official-recompute-mode",
        choices=("cached", "real"),
        default="cached",
        help=(
            "Explicitly label candidate recompute timing. Cached mode is an "
            "algorithmic Pareto sweep; real mode is end-to-end GGUF recompute."
        ),
    )


def make_client(args: argparse.Namespace) -> EmbeddingClient:
    if args.embedding_batch_size <= 0:
        raise ValueError("--embedding-batch-size must be positive")
    if args.dimension is not None and args.dimension <= 0:
        raise ValueError("--dimension must be positive")
    concurrency = getattr(args, "embedding_concurrency", 1)
    if concurrency <= 0:
        raise ValueError("--embedding-concurrency must be positive")
    if concurrency > 1 and args.dimension is None:
        raise ValueError(
            "--embedding-concurrency above 1 requires an explicit --dimension"
        )
    return EmbeddingClient(
        args.embedding_url,
        args.embedding_model,
        timeout=args.embedding_timeout,
        retries=args.embedding_retries,
        api_key=args.api_key,
    )


def prepare_shared(args: argparse.Namespace) -> dict[str, Any]:
    client = make_client(args)
    model = model_descriptor(args)
    corpus_declared = generate_cache(
        source_path=args.documents,
        cache_path=args.corpus_cache,
        client=client,
        fingerprint=args.fingerprint,
        model=model,
        role="corpus",
        batch_size=args.embedding_batch_size,
        dimension=args.dimension,
        bindings={},
        prefix_cache=args.prefix_cache,
        concurrency=args.embedding_concurrency,
    )
    corpus_sha = corpus_declared["cache"]["sha256"]
    query_declared = generate_cache(
        source_path=args.queries,
        cache_path=args.query_cache,
        client=client,
        fingerprint=args.fingerprint,
        model=model,
        role="queries",
        batch_size=args.embedding_batch_size,
        dimension=corpus_declared["cache"]["dimensions"],
        bindings={
            "corpus_cache_sha256": corpus_sha,
            "corpus_source_sha256": corpus_declared["source"]["sha256"],
            "model_sha256": model["sha256"],
        },
    )
    truth = generate_ground_truth(
        corpus_cache=args.corpus_cache,
        query_cache=args.query_cache,
        documents=args.documents,
        queries=args.queries,
        output=args.ground_truth,
        top_k=args.top_k,
        query_block_rows=args.query_block_rows,
        corpus_block_rows=args.corpus_block_rows,
    )
    return {
        "corpus_cache": corpus_declared,
        "query_cache": query_declared,
        "ground_truth": truth,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate integrity-bound shared vectors/exact truth and run a "
            "resumable matched leann.cpp versus official LEANN benchmark."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect = subparsers.add_parser("inspect-cache")
    inspect.add_argument("--cache", type=Path, required=True)
    inspect.add_argument("--source", type=Path)
    inspect.add_argument("--allow-v1", action="store_true")

    corpus = subparsers.add_parser("cache-corpus")
    corpus.add_argument("--documents", type=Path, required=True)
    corpus.add_argument("--corpus-cache", type=Path, required=True)
    add_embedding_arguments(corpus)

    query = subparsers.add_parser("cache-queries")
    query.add_argument("--queries", type=Path, required=True)
    query.add_argument("--corpus-cache", type=Path, required=True)
    query.add_argument("--query-cache", type=Path, required=True)
    add_embedding_arguments(query)

    truth = subparsers.add_parser("ground-truth")
    add_data_arguments(truth)
    truth.add_argument("--query-block-rows", type=int, default=32)
    truth.add_argument("--corpus-block-rows", type=int, default=32768)

    prepare = subparsers.add_parser("prepare")
    add_data_arguments(prepare)
    add_embedding_arguments(prepare)
    prepare.add_argument("--query-block-rows", type=int, default=32)
    prepare.add_argument("--corpus-block-rows", type=int, default=32768)

    run = subparsers.add_parser("orchestrate")
    add_data_arguments(run)
    add_orchestration_arguments(run)

    pipeline = subparsers.add_parser("pipeline")
    add_data_arguments(pipeline)
    add_embedding_arguments(pipeline)
    pipeline.add_argument("--query-block-rows", type=int, default=32)
    pipeline.add_argument("--corpus-block-rows", type=int, default=32768)
    # Avoid duplicate endpoint/model flags from orchestration.
    orchestration_parent = argparse.ArgumentParser(add_help=False)
    add_orchestration_arguments(orchestration_parent)
    for action in orchestration_parent._actions:
        if not action.option_strings:
            continue
        if action.dest in {"embedding_url", "embedding_model"}:
            continue
        pipeline._add_action(action)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if hasattr(args, "top_k"):
        if args.top_k <= 0:
            raise ValueError("--top-k must be positive")
        if any(value <= 0 or value > args.top_k for value in args.recall_k):
            raise ValueError("--recall-k values must be positive and <= --top-k")
        args.recall_k = sorted(set(args.recall_k))
    if hasattr(args, "native_ef_search"):
        if any(value <= 0 for value in args.native_ef_search):
            raise ValueError("--native-ef-search values must be positive")
        if args.native_scan_limit < 0:
            raise ValueError("--native-scan-limit must be non-negative")
        if not 0.0 <= args.native_rerank_ratio <= 1.0:
            raise ValueError("--native-rerank-ratio must be in [0,1]")
    if args.command == "inspect-cache":
        vectors, metadata = read_cache(args.cache)
        if args.source:
            metadata["source_validation"] = validate_cache_source(
                metadata,
                args.source,
                require_integrity=not args.allow_v1,
            )
        metadata["sha256"] = sha256_file(args.cache)
        del vectors
        print(json.dumps(metadata, indent=2, sort_keys=True))
        return
    if args.command == "cache-corpus":
        result = generate_cache(
            source_path=args.documents,
            cache_path=args.corpus_cache,
            client=make_client(args),
            fingerprint=args.fingerprint,
            model=model_descriptor(args),
            role="corpus",
            batch_size=args.embedding_batch_size,
            dimension=args.dimension,
            bindings={},
            prefix_cache=args.prefix_cache,
        )
    elif args.command == "cache-queries":
        _, corpus_meta = read_cache(args.corpus_cache)
        corpus_declared = load_json(sidecar_path(args.corpus_cache))
        declared_model = model_descriptor(args)
        if declared_model["sha256"] != corpus_declared["model"]["sha256"]:
            raise ValueError("query model identity differs from corpus model identity")
        if args.fingerprint != corpus_meta["fingerprint"]:
            raise ValueError("query fingerprint differs from corpus fingerprint")
        result = generate_cache(
            source_path=args.queries,
            cache_path=args.query_cache,
            client=make_client(args),
            fingerprint=args.fingerprint,
            model=declared_model,
            role="queries",
            batch_size=args.embedding_batch_size,
            dimension=corpus_meta["dimensions"],
            bindings={
                "corpus_cache_sha256": sha256_file(args.corpus_cache),
                "corpus_source_sha256": corpus_meta["source_sha256"],
                "model_sha256": corpus_declared["model"]["sha256"],
            },
        )
    elif args.command == "ground-truth":
        result = generate_ground_truth(
            corpus_cache=args.corpus_cache,
            query_cache=args.query_cache,
            documents=args.documents,
            queries=args.queries,
            output=args.ground_truth,
            top_k=args.top_k,
            query_block_rows=args.query_block_rows,
            corpus_block_rows=args.corpus_block_rows,
        )
    elif args.command == "prepare":
        result = prepare_shared(args)
    elif args.command == "orchestrate":
        if not args.skip_native and args.native_model is None:
            raise ValueError("--native-model is required unless --skip-native")
        result = orchestrate(args)
    elif args.command == "pipeline":
        if not args.skip_native and args.native_model is None:
            raise ValueError("--native-model is required unless --skip-native")
        prepared = prepare_shared(args)
        result = {"prepared": prepared, "benchmark": orchestrate(args)}
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
