#!/usr/bin/env python3
"""Validate native GGUF embeddings against official LEANN benchmark vectors.

The official comparison consumes integrity-bound LEANNBC2 corpus and query
caches.  This tool selects all query vectors and a bounded, deterministic
document sample from those caches, builds a tiny cache-backed leann.cpp index,
then asks the native in-process llama.cpp embedder to regenerate every fixture
vector through ``leann bench --ground-truth-cache``.

The selected rows are joined back to the published BEIR corpus by ID and
re-normalized from the source archive. This proves which sampled documents
were actually truncated instead of treating a longest-row heuristic as proof.
"""

from __future__ import annotations

import argparse
import datetime as dt
import heapq
import json
import os
import subprocess
import time
import zipfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from benchmark_cache import (
    atomic_write_json,
    canonical_hash,
    iter_nonempty_lines,
    read_cache,
    sha256_file,
    source_identity,
    validate_cache_source,
    validate_vector_matrix,
    write_cache_v2,
)
from prepare_beir_nq_scale import document_text


SCHEMA = "leann-embedding-parity-v1"
DEFAULT_MIN_COSINE = 0.9999


def evenly_spaced_indices(count: int, requested: int) -> list[int]:
    """Return deterministic endpoints-inclusive indices without floating point."""

    if count <= 0:
        raise ValueError("count must be positive")
    if requested < 0:
        raise ValueError("requested must be non-negative")
    if requested == 0:
        return []
    if requested >= count:
        return list(range(count))
    if requested == 1:
        return [0]
    return [
        position * (count - 1) // (requested - 1)
        for position in range(requested)
    ]


def select_document_sample(
    documents: Path,
    *,
    expected_count: int,
    evenly_spaced: int,
    longest: int,
    truncated_indices: Sequence[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select and materialize a bounded document sample in one streaming pass."""

    if expected_count <= 0:
        raise ValueError("expected_count must be positive")
    if longest < 0:
        raise ValueError("longest must be non-negative")
    truncation_candidates = sorted(set(truncated_indices))
    if truncation_candidates and (
        truncation_candidates[0] < 0
        or truncation_candidates[-1] >= expected_count
    ):
        raise ValueError("truncated document index is outside the corpus")

    spaced = evenly_spaced_indices(expected_count, evenly_spaced)
    direct_indices = set(spaced) | set(truncation_candidates)
    direct_rows: dict[int, tuple[str, int]] = {}
    # The root is the least desirable retained row: shortest first, then the
    # highest row index for equal byte lengths.
    longest_heap: list[tuple[int, int, str]] = []
    observed_count = 0

    for index, text in enumerate(iter_nonempty_lines(documents)):
        observed_count += 1
        utf8_bytes = len(text.encode("utf-8"))
        if index in direct_indices:
            direct_rows[index] = (text, utf8_bytes)
        if longest:
            candidate = (utf8_bytes, -index, text)
            if len(longest_heap) < longest:
                heapq.heappush(longest_heap, candidate)
            elif candidate[:2] > longest_heap[0][:2]:
                heapq.heapreplace(longest_heap, candidate)

    if observed_count != expected_count:
        raise ValueError(
            f"documents/cache row mismatch: {observed_count} vs {expected_count}"
        )
    if set(direct_rows) != direct_indices:
        missing = sorted(direct_indices - set(direct_rows))
        raise ValueError(f"failed to load selected document rows: {missing}")

    longest_rows: dict[int, tuple[str, int]] = {}
    for utf8_bytes, negative_index, text in longest_heap:
        longest_rows[-negative_index] = (text, utf8_bytes)

    categories: dict[int, set[str]] = {}
    for index in spaced:
        categories.setdefault(index, set()).add("evenly_spaced")
    for index in longest_rows:
        categories.setdefault(index, set()).add("longest")
    for index in truncation_candidates:
        categories.setdefault(index, set()).add(
            "explicit_truncation_candidate"
        )

    materialized = {**longest_rows, **direct_rows}
    rows = [
        {
            "kind": "document",
            "source_index": index,
            "text": materialized[index][0],
            "utf8_bytes": materialized[index][1],
            "categories": sorted(categories[index]),
        }
        for index in sorted(categories)
    ]
    return rows, {
        "algorithm": (
            "union of integer-arithmetic evenly spaced rows, longest UTF-8 "
            "rows (ties by lowest source index), and explicit truncation "
            "candidates; final order is ascending source index"
        ),
        "requested_evenly_spaced": evenly_spaced,
        "requested_longest": longest,
        "explicit_truncation_candidate_indices": truncation_candidates,
        "selected_documents": len(rows),
        "category_counts": {
            category: sum(category in row["categories"] for row in rows)
            for category in (
                "evenly_spaced",
                "longest",
                "explicit_truncation_candidate",
            )
        },
    }


def load_selected_document_ids(
    document_ids: Path,
    *,
    expected_count: int,
    selected_indices: set[int],
) -> dict[int, str]:
    selected: dict[int, str] = {}
    observed = 0
    with document_ids.open("r", encoding="utf-8", newline="") as source:
        for line_number, line in enumerate(source, start=1):
            line = line.rstrip("\n")
            if line.endswith("\r"):
                line = line[:-1]
            fields = line.split("\t")
            if len(fields) != 2:
                raise ValueError(
                    f"{document_ids}: invalid TSV row {line_number}"
                )
            try:
                position = int(fields[0])
            except ValueError as error:
                raise ValueError(
                    f"{document_ids}: non-integer row {line_number}"
                ) from error
            if position != observed:
                raise ValueError(
                    f"{document_ids}: expected position {observed}, got {position}"
                )
            if not fields[1]:
                raise ValueError(f"{document_ids}: empty corpus ID")
            if position in selected_indices:
                selected[position] = fields[1]
            observed += 1
    if observed != expected_count:
        raise ValueError(
            f"document ID/cache row mismatch: {observed} vs {expected_count}"
        )
    if set(selected) != selected_indices:
        raise ValueError("document ID file omitted selected rows")
    return selected


def verify_document_origins(
    archive: Path,
    document_ids: Path,
    rows: list[dict[str, Any]],
    *,
    expected_count: int,
    max_document_bytes: int,
) -> dict[str, Any]:
    """Join selected rows to BEIR records and prove truncation from raw text."""

    selected_indices = {int(row["source_index"]) for row in rows}
    ids_by_index = load_selected_document_ids(
        document_ids,
        expected_count=expected_count,
        selected_indices=selected_indices,
    )
    index_by_id = {
        identifier: index for index, identifier in ids_by_index.items()
    }
    if len(index_by_id) != len(ids_by_index):
        raise ValueError("selected corpus IDs are not unique")
    rows_by_index = {int(row["source_index"]): row for row in rows}
    found: set[str] = set()

    with zipfile.ZipFile(archive) as compressed:
        candidates = [
            member
            for member in compressed.infolist()
            if not member.is_dir()
            and Path(member.filename).name == "corpus.jsonl"
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"{archive}: expected exactly one corpus.jsonl member"
            )
        member = candidates[0]
        with compressed.open(member) as source:
            for line_number, raw in enumerate(source, start=1):
                try:
                    record = json.loads(raw)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise ValueError(
                        f"{archive}:{member.filename}:{line_number}: "
                        "invalid JSONL"
                    ) from error
                if not isinstance(record, dict) or "_id" not in record:
                    raise ValueError(
                        f"{archive}:{member.filename}:{line_number}: "
                        "record has no _id"
                    )
                identifier = str(record["_id"])
                if identifier not in index_by_id:
                    continue
                if identifier in found:
                    raise ValueError(
                        f"{archive}: duplicate selected corpus ID {identifier!r}"
                    )
                found.add(identifier)
                position = index_by_id[identifier]
                prepared = document_text(
                    record,
                    identifier,
                    max_document_bytes=max_document_bytes,
                )
                row = rows_by_index[position]
                if prepared.text != row["text"]:
                    raise ValueError(
                        f"prepared document differs from source at row {position}"
                    )
                row["corpus_id"] = identifier
                row["original_prepared_utf8_bytes"] = (
                    prepared.original_prepared_bytes
                )
                row["source_verified_truncated"] = prepared.truncated
                if prepared.truncated:
                    row["categories"] = sorted(
                        set(row["categories"])
                        | {"source_verified_truncated"}
                    )
                if len(found) == len(index_by_id):
                    break

    missing = sorted(set(index_by_id) - found)
    if missing:
        raise ValueError(
            f"{archive}: selected corpus IDs were not found: {missing[:8]}"
        )
    truncated = [
        int(row["source_index"])
        for row in rows
        if row["source_verified_truncated"]
    ]
    if not truncated:
        raise ValueError(
            "deterministic document sample contains no source-verified "
            "truncated row; increase --longest-documents or add an explicit "
            "--truncated-document-index candidate"
        )
    return {
        "method": (
            "join prepared row to published BEIR corpus.jsonl by corpus ID, "
            "rerun the dataset normalizer at the manifest byte ceiling, "
            "require exact prepared-text equality, and inspect its truncated flag"
        ),
        "archive_member": member.filename,
        "checked_documents": len(rows),
        "source_verified_truncated_documents": len(truncated),
        "source_verified_truncated_indices": truncated,
    }


def compare_embeddings(
    reference: np.ndarray,
    native: np.ndarray,
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Return cosine and absolute-error metrics with deterministic worst rows."""

    if reference.shape != native.shape:
        raise ValueError(
            f"embedding shape mismatch: {reference.shape} vs {native.shape}"
        )
    if reference.ndim != 2 or reference.shape[0] != len(rows):
        raise ValueError("row metadata does not match the embedding matrices")
    validate_vector_matrix(reference, require_unit=False, label="reference")
    validate_vector_matrix(native, require_unit=False, label="native")

    reference64 = np.asarray(reference, dtype=np.float64)
    native64 = np.asarray(native, dtype=np.float64)
    denominator = np.linalg.norm(reference64, axis=1) * np.linalg.norm(
        native64, axis=1
    )
    cosine = np.einsum("ij,ij->i", reference64, native64) / denominator
    cosine = np.clip(cosine, -1.0, 1.0)
    absolute = np.abs(reference64 - native64)
    row_max_absolute = np.max(absolute, axis=1)

    worst_cosine_index = int(np.argmin(cosine))
    worst_absolute_flat = int(np.argmax(absolute))
    worst_absolute_row, worst_absolute_dimension = np.unravel_index(
        worst_absolute_flat, absolute.shape
    )

    by_kind: dict[str, Any] = {}
    kinds = sorted({str(row["kind"]) for row in rows})
    for kind in kinds:
        indices = [
            index for index, row in enumerate(rows) if row["kind"] == kind
        ]
        by_kind[kind] = {
            "rows": len(indices),
            "minimum_cosine_similarity": float(np.min(cosine[indices])),
            "maximum_absolute_difference": float(
                np.max(row_max_absolute[indices])
            ),
        }

    return {
        "rows": int(reference.shape[0]),
        "dimensions": int(reference.shape[1]),
        "minimum_cosine_similarity": float(cosine[worst_cosine_index]),
        "mean_cosine_similarity": float(np.mean(cosine)),
        "maximum_absolute_difference": float(
            absolute[worst_absolute_row, worst_absolute_dimension]
        ),
        "by_kind": by_kind,
        "worst_cosine_row": {
            **{
                key: value
                for key, value in rows[worst_cosine_index].items()
                if key != "text"
            },
            "fixture_index": worst_cosine_index,
            "cosine_similarity": float(cosine[worst_cosine_index]),
            "maximum_absolute_difference": float(
                row_max_absolute[worst_cosine_index]
            ),
        },
        "worst_absolute_component": {
            **{
                key: value
                for key, value in rows[int(worst_absolute_row)].items()
                if key != "text"
            },
            "fixture_index": int(worst_absolute_row),
            "dimension": int(worst_absolute_dimension),
            "absolute_difference": float(
                absolute[worst_absolute_row, worst_absolute_dimension]
            ),
        },
    }


def load_sidecar(
    cache: Path, expected_role: str, actual_cache_sha256: str
) -> dict[str, Any]:
    sidecar = cache.with_name(cache.name + ".meta.json")
    if not sidecar.is_file():
        raise ValueError(f"{cache}: missing provenance sidecar")
    with sidecar.open(encoding="utf-8") as source:
        declared = json.load(source)
    if declared.get("schema") != "leann-shared-embedding-cache-v1":
        raise ValueError(f"{sidecar}: unexpected schema")
    if declared.get("role") != expected_role:
        raise ValueError(f"{sidecar}: expected {expected_role} role")
    if declared.get("cache", {}).get("sha256") != actual_cache_sha256:
        raise ValueError(f"{sidecar}: cache SHA-256 mismatch")
    return declared


def command_observation(argv: list[str]) -> dict[str, Any]:
    started = time.perf_counter()
    completed = subprocess.run(
        argv,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    observation = {
        "argv": argv,
        "elapsed_seconds": time.perf_counter() - started,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): "
            f"{' '.join(argv)}\n{completed.stderr[-4000:]}"
        )
    return observation


def write_line_file(path: Path, values: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as target:
        for value in values:
            if not value or "\n" in value or "\r" in value:
                raise ValueError("fixture values must be non-empty single lines")
            target.write(value)
            target.write("\n")
        target.flush()
        os.fsync(target.fileno())
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare native in-process llama.cpp embeddings with the exact "
            "LEANNBC2 vectors used by the official LEANN benchmark."
        )
    )
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--corpus-cache", type=Path, required=True)
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--document-ids", type=Path, required=True)
    parser.add_argument("--tier", required=True)
    parser.add_argument("--native-binary", type=Path, required=True)
    parser.add_argument("--native-model", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-queries", type=int, default=300)
    parser.add_argument("--evenly-spaced-documents", type=int, default=16)
    parser.add_argument("--longest-documents", type=int, default=8)
    parser.add_argument(
        "--truncated-document-index",
        type=int,
        action="append",
        default=[],
        help=(
            "Additional zero-based corpus row to include as a truncation "
            "candidate; repeat for multiple rows. The source archive verifies it."
        ),
    )
    parser.add_argument("--minimum-cosine", type=float, default=DEFAULT_MIN_COSINE)
    parser.add_argument("--ctx", type=int, default=2048)
    parser.add_argument("--batch-tokens", type=int, default=32768)
    parser.add_argument("--parallel", type=int, default=16)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--gpu-layers", type=int, default=99)
    parser.add_argument("--ground-truth-batch", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.expected_queries <= 0:
        raise ValueError("--expected-queries must be positive")
    if not 0.0 < args.minimum_cosine <= 1.0:
        raise ValueError("--minimum-cosine must be in (0, 1]")
    if (
        args.ctx <= 0
        or args.batch_tokens <= 0
        or args.parallel <= 0
        or args.ground_truth_batch <= 0
    ):
        raise ValueError("native embedding sizes must be positive")
    native_binary = args.native_binary.resolve()
    native_model = args.native_model.resolve()
    if not native_binary.is_file() or not os.access(native_binary, os.X_OK):
        raise ValueError("--native-binary must be an executable file")
    if not native_model.is_file():
        raise ValueError("--native-model must be a file")

    documents = args.documents.resolve()
    queries = args.queries.resolve()
    corpus_cache_path = args.corpus_cache.resolve()
    query_cache_path = args.query_cache.resolve()
    corpus, corpus_meta = read_cache(corpus_cache_path, validate_vectors=False)
    query_vectors, query_meta = read_cache(
        query_cache_path, validate_vectors=False
    )
    documents_identity = validate_cache_source(
        corpus_meta, documents, require_integrity=True
    )
    queries_identity = validate_cache_source(
        query_meta,
        queries,
        expected_count=args.expected_queries,
        require_integrity=True,
    )
    if corpus_meta["version"] != 2 or query_meta["version"] != 2:
        raise ValueError("embedding parity requires integrity-bound LEANNBC2 caches")
    if (
        corpus_meta["dimensions"] != query_meta["dimensions"]
        or corpus_meta["fingerprint"] != query_meta["fingerprint"]
    ):
        raise ValueError("corpus/query cache model metadata differs")

    corpus_sha256 = sha256_file(corpus_cache_path)
    query_cache_sha256 = sha256_file(query_cache_path)
    corpus_declared = load_sidecar(
        corpus_cache_path, "corpus", corpus_sha256
    )
    query_declared = load_sidecar(
        query_cache_path, "queries", query_cache_sha256
    )
    if query_declared.get("bindings", {}).get(
        "corpus_cache_sha256"
    ) != corpus_sha256:
        raise ValueError("query cache is not bound to the selected corpus cache")
    corpus_model = corpus_declared.get("model", {})
    query_model = query_declared.get("model", {})
    if corpus_model.get("sha256") != query_model.get("sha256"):
        raise ValueError("query/corpus cache model identities differ")
    declared_model_sha256 = corpus_model.get("artifact", {}).get("sha256")
    native_model_sha256 = sha256_file(native_model)
    if not declared_model_sha256 or declared_model_sha256 != native_model_sha256:
        raise ValueError(
            "native GGUF SHA-256 differs from the official-path cache model"
        )

    with args.dataset_manifest.resolve().open(encoding="utf-8") as source:
        dataset_manifest = json.load(source)
    dataset_manifest_sha256 = sha256_file(args.dataset_manifest.resolve())
    tier = dataset_manifest.get("tiers", {}).get(args.tier)
    if not isinstance(tier, dict):
        raise ValueError(f"dataset manifest does not contain tier {args.tier!r}")
    if tier.get("documents") != corpus_meta["count"]:
        raise ValueError("dataset manifest tier/cache row count differs")
    if tier.get("truncation", {}).get("documents", 0) <= 0:
        raise ValueError("selected dataset tier does not report truncated documents")

    document_rows, document_selection = select_document_sample(
        documents,
        expected_count=int(corpus_meta["count"]),
        evenly_spaced=args.evenly_spaced_documents,
        longest=args.longest_documents,
        truncated_indices=args.truncated_document_index,
    )
    maximum_document_bytes = tier.get("maximum_prepared_document_bytes")
    if not isinstance(maximum_document_bytes, int) or maximum_document_bytes <= 0:
        raise ValueError("dataset manifest has no valid document byte ceiling")
    for row in document_rows:
        if row["utf8_bytes"] > maximum_document_bytes:
            raise ValueError("sampled document exceeds the manifest byte ceiling")
    source_archive = args.source_archive.resolve()
    document_ids = args.document_ids.resolve()
    archive_declared = dataset_manifest.get("dataset", {}).get("archive")
    if not isinstance(archive_declared, dict):
        raise ValueError("dataset manifest has no source archive identity")
    source_archive_sha256 = sha256_file(source_archive)
    if source_archive_sha256 != archive_declared.get("sha256"):
        raise ValueError("source archive SHA-256 differs from dataset manifest")
    declared_files = dataset_manifest.get("files", {})
    document_id_declarations = [
        value
        for value in declared_files.values()
        if isinstance(value, dict) and value.get("path") == document_ids.name
    ]
    if len(document_id_declarations) != 1:
        raise ValueError("dataset manifest does not bind the document ID file")
    document_ids_sha256 = sha256_file(document_ids)
    if document_ids_sha256 != document_id_declarations[0].get("sha256"):
        raise ValueError("document ID file SHA-256 differs from dataset manifest")
    truncation_proof = verify_document_origins(
        source_archive,
        document_ids,
        document_rows,
        expected_count=int(corpus_meta["count"]),
        max_document_bytes=maximum_document_bytes,
    )
    document_selection["truncation_proof"] = truncation_proof

    query_texts = list(iter_nonempty_lines(queries))
    if len(query_texts) != args.expected_queries:
        raise ValueError(
            f"expected {args.expected_queries} queries, got {len(query_texts)}"
        )
    rows: list[dict[str, Any]] = [
        {
            "kind": "query",
            "source_index": index,
            "text": text,
            "utf8_bytes": len(text.encode("utf-8")),
            "categories": ["all_queries"],
        }
        for index, text in enumerate(query_texts)
    ]
    rows.extend(document_rows)
    reference = np.ascontiguousarray(
        np.concatenate(
            (
                np.asarray(query_vectors, dtype=np.float32),
                np.asarray(
                    corpus[
                        [int(row["source_index"]) for row in document_rows]
                    ],
                    dtype=np.float32,
                ),
            ),
            axis=0,
        ),
        dtype=np.float32,
    )

    stable = {
        "schema": SCHEMA,
        "documents": documents_identity,
        "queries": queries_identity,
        "corpus_cache_sha256": corpus_sha256,
        "query_cache_sha256": query_cache_sha256,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "source_archive_sha256": source_archive_sha256,
        "document_ids_sha256": document_ids_sha256,
        "tier": args.tier,
        "rows": [
            {key: value for key, value in row.items() if key != "text"}
            for row in rows
        ],
        "fixture_text_sha256": canonical_hash([row["text"] for row in rows]),
        "native_binary_sha256": sha256_file(native_binary),
        "native_model_sha256": native_model_sha256,
        "native_config": {
            "ctx": args.ctx,
            "batch_tokens": args.batch_tokens,
            "parallel": args.parallel,
            "threads": args.threads,
            "gpu_layers": args.gpu_layers,
            "ground_truth_batch": args.ground_truth_batch,
        },
    }
    input_hash = canonical_hash(stable)
    run_dir = args.work_dir.resolve() / input_hash[:16]
    run_dir.mkdir(parents=True, exist_ok=True)
    fixture_path = run_dir / "fixture.txt"
    reference_path = run_dir / "official-reference.leannbc2"
    native_cache_path = run_dir / "native.leannbc1"
    query_probe_path = run_dir / "query-probe.txt"
    index_prefix = run_dir / "fixture-index"
    state_path = run_dir / "native-run.json"

    write_line_file(fixture_path, [str(row["text"]) for row in rows])
    write_line_file(query_probe_path, [str(rows[0]["text"])])
    reference_meta = write_cache_v2(
        reference_path,
        reference,
        fingerprint=str(corpus_meta["fingerprint"]),
        source_path=fixture_path,
    )

    commands: list[dict[str, Any]]
    reuse = False
    if state_path.is_file() and native_cache_path.is_file():
        with state_path.open(encoding="utf-8") as source:
            state = json.load(source)
        if (
            state.get("input_hash") == input_hash
            and state.get("native_cache_sha256") == sha256_file(native_cache_path)
        ):
            commands = state.get("commands", [])
            reuse = True
    if not reuse:
        native_cache_path.unlink(missing_ok=True)
        native_cache_path.with_name(native_cache_path.name + ".tmp").unlink(
            missing_ok=True
        )
        print(
            f"building parity fixture with {len(rows)} rows in {run_dir}",
            flush=True,
        )
        build_argv = [
            str(native_binary),
            "build",
            "--docs",
            str(fixture_path),
            "--index",
            str(index_prefix),
            "--embedder",
            "cache",
            "--embedding-cache",
            str(reference_path),
            "--graph-degree",
            "8",
            "--ef-construction",
            "40",
            "--low-degree",
            "3",
            "--hub-ratio",
            "0.02",
            "--approx",
            "simhash",
            "--sketch-bits",
            "64",
            "--embedding-batch",
            "64",
        ]
        build_observation = command_observation(build_argv)
        print("regenerating fixture vectors with native llama.cpp", flush=True)
        bench_argv = [
            str(native_binary),
            "bench",
            "--index",
            str(index_prefix),
            "--queries",
            str(query_probe_path),
            "--embedder",
            "llama",
            "--model",
            str(native_model),
            "--ctx",
            str(args.ctx),
            "--batch-tokens",
            str(args.batch_tokens),
            "--parallel",
            str(args.parallel),
            "--threads",
            str(args.threads),
            "--gpu-layers",
            str(args.gpu_layers),
            "--top-k",
            "1",
            "--ef-search",
            "8",
            "--scan-limit",
            "0",
            "--recompute-batch",
            "16",
            "--rerank-ratio",
            "0.25",
            "--ground-truth-cache",
            str(native_cache_path),
            "--ground-truth-batch",
            str(args.ground_truth_batch),
            "--warmup-queries",
            "0",
            "--max-queries",
            "1",
        ]
        bench_observation = command_observation(bench_argv)
        commands = [build_observation, bench_observation]
        atomic_write_json(
            state_path,
            {
                "schema": "leann-native-parity-run-v1",
                "input_hash": input_hash,
                "native_cache_sha256": sha256_file(native_cache_path),
                "commands": commands,
            },
        )

    native, native_meta = read_cache(native_cache_path, require_unit=True)
    if (
        native_meta["count"] != len(rows)
        or native_meta["dimensions"] != reference.shape[1]
        or native_meta["fingerprint"] != corpus_meta["fingerprint"]
    ):
        raise ValueError("native cache does not match the parity fixture/model")
    metrics = compare_embeddings(reference, np.asarray(native), rows)
    passed = metrics["minimum_cosine_similarity"] >= args.minimum_cosine

    report = {
        "schema": SCHEMA,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "input_hash": input_hash,
        "passed": passed,
        "acceptance": {
            "minimum_cosine_similarity": args.minimum_cosine,
            "rule": "fail if any selected row has cosine similarity below threshold",
        },
        "coverage": {
            "queries": {
                "selected": len(query_texts),
                "source_rows": query_meta["count"],
                "all_queries": len(query_texts) == query_meta["count"],
            },
            "documents": document_selection,
            "total_rows": len(rows),
        },
        "dataset": {
            "tier": args.tier,
            "documents": documents_identity,
            "queries": queries_identity,
            "manifest": {
                "path": str(args.dataset_manifest.resolve()),
                "sha256": dataset_manifest_sha256,
                "tier_truncation": tier.get("truncation"),
                "maximum_prepared_document_bytes": maximum_document_bytes,
            },
            "source_archive": {
                "path": str(source_archive),
                "size_bytes": source_archive.stat().st_size,
                "sha256": source_archive_sha256,
            },
            "document_ids": {
                "path": str(document_ids),
                "size_bytes": document_ids.stat().st_size,
                "sha256": document_ids_sha256,
            },
        },
        "official_comparison_reference": {
            "description": (
                "Integrity-bound normalized LEANNBC2 vectors consumed by "
                "compare_official_leann.py."
            ),
            "corpus_cache": {
                **corpus_meta,
                "sha256": corpus_sha256,
                "sidecar_sha256": sha256_file(
                    corpus_cache_path.with_name(
                        corpus_cache_path.name + ".meta.json"
                    )
                ),
                "generation": corpus_declared.get("generation"),
            },
            "query_cache": {
                **query_meta,
                "sha256": query_cache_sha256,
                "sidecar_sha256": sha256_file(
                    query_cache_path.with_name(
                        query_cache_path.name + ".meta.json"
                    )
                ),
                "generation": query_declared.get("generation"),
            },
            "model": corpus_model,
        },
        "native": {
            "binary": {
                "path": str(native_binary),
                "sha256": stable["native_binary_sha256"],
            },
            "model": {
                "path": str(native_model),
                "sha256": native_model_sha256,
            },
            "config": stable["native_config"],
            "cache": {
                **native_meta,
                "sha256": sha256_file(native_cache_path),
            },
            "commands_reused": reuse,
            "commands": commands,
        },
        "fixture": {
            "directory": str(run_dir),
            "source": source_identity(fixture_path),
            "reference_cache": {
                **reference_meta,
                "sha256": sha256_file(reference_path),
            },
            "rows": [
                {key: value for key, value in row.items() if key != "text"}
                for row in rows
            ],
        },
        "metrics": metrics,
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output.resolve(), report)
    print(
        f"embedding parity: min_cosine="
        f"{metrics['minimum_cosine_similarity']:.9f}, "
        f"max_abs={metrics['maximum_absolute_difference']:.9g}, "
        f"passed={str(passed).lower()}",
        flush=True,
    )
    print(f"wrote {args.output.resolve()}", flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
