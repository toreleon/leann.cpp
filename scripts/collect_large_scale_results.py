#!/usr/bin/env python3
"""Validate and consolidate large-scale leann.cpp benchmark evidence.

The collector consumes manifests produced by ``run_large_scale_benchmark.py``.
It does not run benchmarks and never substitutes estimates for missing points.
By default, an incomplete matrix is an error.  ``--allow-incomplete`` writes an
explicitly incomplete report whose completeness table names every missing
profile or configuration. Publication mode also requires per-tier native versus
reference embedding parity and independently recomputes Recall@k from ranked
result IDs for both implementations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from benchmark_cache import (
    count_nonempty_lines,
    read_cache,
    read_ground_truth,
    sha256_file,
)


MANIFEST_SCHEMA = "leann-large-scale-benchmark-v1"
STAGE_SCHEMA = "leann-command-stage-v1"
OFFICIAL_SCHEMA = "leann.cpp-official-comparison-v1"
REPORT_SCHEMA = "leann-large-scale-results-v1"
PARITY_SCHEMA = "leann-embedding-parity-v1"
REUSE_SCHEMA = "leann-official-index-reuse-v1"
OUTPUT_COMMIT_SCHEMA = "leann-large-scale-output-commit-v1"
ENDPOINT_ATTESTATION_SCHEMA = "leann-embedding-endpoint-attestation-v1"
DECLARED_TIER_CARDINALITIES = {"100k": 100_000, "1m": 1_000_000}
PROFILES = ("native", "official-cached", "official-real")
OFFICIAL_RUNTIME_MODULES = {
    "leann",
    "leann_backend_hnsw",
    "leann_backend_hnsw.faiss",
    "leann_backend_hnsw._swigfaiss",
    "leann_backend_hnsw.hnsw_backend",
}
CSV_COLUMNS = (
    "report_id",
    "tier",
    "corpus_count",
    "system",
    "recompute_mode",
    "latency_comparable",
    "search_parameter",
    "search_value",
    "batch_size",
    "repetitions",
    "queries_per_repetition",
    "query_observations",
    "recall_at_3",
    "recall_at_10",
    "latency_ms_mean",
    "latency_ms_p50",
    "latency_ms_p95",
    "latency_ms_min",
    "latency_ms_max",
    "candidate_embeddings_mean_per_query",
    "embedding_batches_mean_per_query",
    "approximate_distances_mean_per_query",
    "upper_layer_hops_mean_per_query",
    "result_ids_sha256",
    "index_build_status",
    "index_artifact_set_sha256",
    "build_source_manifest_sha256",
    "matched_pair_selected",
    "matched_pair_id",
    "source_manifest_sha256",
)


class CollectionError(RuntimeError):
    """Raised when benchmark evidence is malformed or incomplete in strict mode."""


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as source:
            return json.load(source)
    except (OSError, json.JSONDecodeError) as error:
        raise CollectionError(f"{path}: cannot read JSON: {error}") from error


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _number(value: Any, context: str) -> float:
    if not _is_number(value):
        raise CollectionError(f"{context}: expected a finite number, got {value!r}")
    return float(value)


def _integer(value: Any, context: str, *, minimum: int = 0) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < minimum
    ):
        raise CollectionError(
            f"{context}: expected an integer >= {minimum}, got {value!r}"
        )
    return value


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CollectionError(f"{context}: expected an object")
    return value


def _sequence(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise CollectionError(f"{context}: expected an array")
    return value


def _required(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise CollectionError(f"{context}: missing required field {key!r}")
    return mapping[key]


def _close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-6)


def _nearest_rank(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise CollectionError("cannot calculate a percentile over no values")
    position = max(0, math.ceil(fraction * len(ordered)) - 1)
    return float(ordered[min(position, len(ordered) - 1)])


def _linear_percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise CollectionError("cannot calculate a percentile over no values")
    return float(np.percentile(np.asarray(values, dtype=np.float64), fraction * 100))


def _argv_value(command: Any, option: str, context: str) -> str | None:
    if not isinstance(command, list) or not all(
        isinstance(value, str) for value in command
    ):
        raise CollectionError(f"{context}: command is not an argv string array")
    positions = [index for index, value in enumerate(command) if value == option]
    if not positions:
        return None
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise CollectionError(f"{context}: malformed or repeated {option}")
    return command[positions[0] + 1]


def _argv_values(command: Any, option: str, context: str) -> list[str] | None:
    if not isinstance(command, list) or not all(
        isinstance(value, str) for value in command
    ):
        raise CollectionError(f"{context}: command is not an argv string array")
    positions = [index for index, value in enumerate(command) if value == option]
    if not positions:
        return None
    if len(positions) != 1:
        raise CollectionError(f"{context}: repeated {option}")
    values: list[str] = []
    for value in command[positions[0] + 1 :]:
        if value.startswith("--"):
            break
        values.append(value)
    if not values:
        raise CollectionError(f"{context}: {option} has no values")
    return values


def _snapshot_identity(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": snapshot.get("path"),
        "size_bytes": snapshot.get("size_bytes"),
        "sha256": snapshot.get("sha256"),
    }


def _artifact_set(artifacts: Any, context: str) -> dict[str, Any]:
    values = _sequence(artifacts, context)
    if not values:
        raise CollectionError(f"{context}: index artifact set is empty")
    normalized: list[dict[str, Any]] = []
    names: set[str] = set()
    for ordinal, value in enumerate(values):
        snapshot = _mapping(value, f"{context} artifact {ordinal}")
        identity = _snapshot_identity(snapshot)
        name = Path(str(identity["path"])).name
        if name in names:
            raise CollectionError(f"{context}: duplicate artifact basename {name!r}")
        names.add(name)
        normalized.append(
            {
                "name": name,
                "size_bytes": identity["size_bytes"],
                "sha256": identity["sha256"],
                "path": identity["path"],
            }
        )
    normalized.sort(key=lambda item: item["name"])
    portable = [
        {
            "name": item["name"],
            "size_bytes": item["size_bytes"],
            "sha256": item["sha256"],
        }
        for item in normalized
    ]
    return {
        "sha256": canonical_hash(portable),
        "artifacts": normalized,
        "identity_scope": "artifact basename, byte size, and SHA-256",
    }


class FileVerifier:
    """Verify recorded file snapshots once per unique path and digest."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._verified: set[tuple[str, int, str]] = set()

    @property
    def verified_count(self) -> int:
        return len(self._verified)

    def verify(self, snapshot: Any, context: str) -> Path:
        record = _mapping(snapshot, context)
        raw_path = _required(record, "path", context)
        size = _integer(_required(record, "size_bytes", context), context)
        digest = _required(record, "sha256", context)
        if not isinstance(raw_path, str) or not raw_path:
            raise CollectionError(f"{context}: invalid snapshot path")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise CollectionError(f"{context}: invalid SHA-256")
        path = Path(raw_path)
        if not path.is_file():
            raise CollectionError(f"{context}: recorded file does not exist: {path}")
        if path.stat().st_size != size:
            raise CollectionError(
                f"{context}: size changed: {path.stat().st_size} vs {size}"
            )
        key = (str(path.resolve()), size, digest)
        if self.enabled and key not in self._verified:
            actual = sha256_file(path)
            if actual != digest:
                raise CollectionError(
                    f"{context}: SHA-256 changed: {actual} vs {digest}"
                )
            self._verified.add(key)
        return path


def _shared_identity(shared: dict[str, Any]) -> dict[str, Any]:
    documents = _mapping(_required(shared, "documents", "shared artifacts"), "documents")
    queries = _mapping(_required(shared, "queries", "shared artifacts"), "queries")
    corpus = _mapping(
        _required(shared, "corpus_cache", "shared artifacts"), "corpus cache"
    )
    query_cache = _mapping(
        _required(shared, "query_cache", "shared artifacts"), "query cache"
    )
    truth = _mapping(
        _required(shared, "ground_truth", "shared artifacts"), "ground truth"
    )
    model = _mapping(_required(shared, "model", "shared artifacts"), "model")
    artifact = _mapping(_required(model, "artifact", "model"), "model artifact")
    identity = {
        "corpus_count": _integer(corpus.get("count"), "corpus count", minimum=1),
        "query_count": _integer(query_cache.get("count"), "query count", minimum=1),
        "dimensions": _integer(corpus.get("dimensions"), "dimensions", minimum=1),
        "documents_sha256": documents.get("sha256"),
        "queries_sha256": queries.get("sha256"),
        "corpus_cache_sha256": corpus.get("sha256"),
        "query_cache_sha256": query_cache.get("sha256"),
        "ground_truth_sha256": truth.get("sha256"),
        "ground_truth_top_k": truth.get("top_k"),
        "model_sha256": model.get("sha256"),
        "model_artifact_sha256": artifact.get("sha256"),
        "fingerprint": corpus.get("fingerprint"),
    }
    for key, value in identity.items():
        if key in {"corpus_count", "query_count", "dimensions"}:
            continue
        if key == "ground_truth_top_k":
            _integer(value, key, minimum=1)
        elif key.endswith("sha256"):
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise CollectionError(
                    f"shared artifacts: invalid identity field {key}"
                )
        elif not isinstance(value, str) or not value:
            raise CollectionError(f"shared artifacts: missing identity field {key}")
    if query_cache.get("dimensions") != identity["dimensions"]:
        raise CollectionError("shared artifacts: corpus/query dimensions differ")
    if query_cache.get("fingerprint") != identity["fingerprint"]:
        raise CollectionError("shared artifacts: corpus/query fingerprints differ")
    if truth.get("corpus_count") != identity["corpus_count"]:
        raise CollectionError("shared artifacts: truth corpus count differs")
    if truth.get("query_count") != identity["query_count"]:
        raise CollectionError("shared artifacts: truth query count differs")
    if documents.get("nonempty_lines") != identity["corpus_count"]:
        raise CollectionError("shared artifacts: document line count differs")
    if queries.get("nonempty_lines") != identity["query_count"]:
        raise CollectionError("shared artifacts: query line count differs")
    if model.get("identity_strength") != "artifact-sha256":
        raise CollectionError(
            "shared artifacts: benchmark model must be bound by artifact SHA-256"
        )
    return identity


def _dataset_record(tier: str, shared: dict[str, Any]) -> dict[str, Any]:
    identity = _shared_identity(shared)
    documents = shared["documents"]
    corpus = shared["corpus_cache"]
    truth = shared["ground_truth"]
    return {
        "tier": tier,
        **identity,
        "raw_text_file_bytes": _integer(
            documents.get("size_bytes"), f"{tier} document bytes", minimum=1
        ),
        "dense_fp32_bytes": _integer(
            corpus.get("vector_bytes"), f"{tier} dense bytes", minimum=1
        ),
        "documents_path": documents.get("path"),
        "corpus_cache_path": corpus.get("path"),
        "ground_truth_path": truth.get("path"),
    }


def _nonempty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise CollectionError(f"{context}: expected a non-empty string")
    return value


def _validate_dataset_derivation(
    *,
    parity_path: Path,
    tier: str,
    report_dataset: dict[str, Any],
    dataset: dict[str, Any],
) -> dict[str, Any]:
    manifest_record = _mapping(
        report_dataset.get("manifest"), f"{parity_path} dataset manifest"
    )
    recorded_path = _nonempty_string(
        manifest_record.get("path"), f"{parity_path} dataset manifest path"
    )
    manifest_path = Path(recorded_path)
    if not manifest_path.is_absolute():
        manifest_path = parity_path.parent / manifest_path
    manifest_path = manifest_path.resolve()
    if not manifest_path.is_file():
        raise CollectionError(
            f"{parity_path}: dataset manifest does not exist: {manifest_path}"
        )
    recorded_sha256 = manifest_record.get("sha256")
    if not isinstance(recorded_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", recorded_sha256
    ):
        raise CollectionError(f"{parity_path}: dataset manifest hash is invalid")
    if sha256_file(manifest_path) != recorded_sha256:
        raise CollectionError(
            f"{parity_path}: dataset manifest SHA-256 changed"
        )
    manifest = _mapping(
        load_json(manifest_path), f"{parity_path} dataset manifest payload"
    )
    _integer(
        manifest.get("schema_version"),
        f"{parity_path} dataset manifest schema",
        minimum=1,
    )

    tiers = _mapping(manifest.get("tiers"), f"{parity_path} dataset tiers")
    tier_record = _mapping(
        tiers.get(tier), f"{parity_path} dataset tier {tier}"
    )
    if tier_record.get("documents") != dataset["corpus_count"]:
        raise CollectionError(
            f"{parity_path}: dataset manifest tier/document count differs"
        )
    maximum_bytes = _integer(
        tier_record.get("maximum_prepared_document_bytes"),
        f"{parity_path} maximum prepared document bytes",
        minimum=1,
    )
    if manifest_record.get("maximum_prepared_document_bytes") != maximum_bytes:
        raise CollectionError(
            f"{parity_path}: parity/manifest document byte ceilings differ"
        )
    truncation = _mapping(
        tier_record.get("truncation"), f"{parity_path} tier truncation"
    )
    if manifest_record.get("tier_truncation") != truncation:
        raise CollectionError(
            f"{parity_path}: parity/manifest tier truncation differs"
        )
    truncated_documents = _integer(
        truncation.get("documents"),
        f"{parity_path} truncated documents",
        minimum=1,
    )
    truncated_bytes = _integer(
        truncation.get("bytes_removed"),
        f"{parity_path} truncated bytes",
        minimum=1,
    )

    duplicates = _mapping(
        tier_record.get("text_duplicates"),
        f"{parity_path} tier text duplicates",
    )
    if (
        duplicates.get("rows") != dataset["corpus_count"]
        or duplicates.get("unique_prepared_texts") != dataset["corpus_count"]
        or duplicates.get("duplicate_rows") != 0
        or _number(
            duplicates.get("duplicate_rate"),
            f"{parity_path} tier duplicate rate",
        )
        != 0.0
    ):
        raise CollectionError(
            f"{parity_path}: selected tier is not unique after text deduplication"
        )

    query_record = _mapping(
        manifest.get("queries"), f"{parity_path} dataset queries"
    )
    if query_record.get("queries") != dataset["query_count"]:
        raise CollectionError(
            f"{parity_path}: dataset manifest query count differs"
        )
    positive_qrel_rows = _integer(
        query_record.get("positive_qrel_rows"),
        f"{parity_path} positive qrel rows",
        minimum=1,
    )
    if query_record.get("qrel_rows") != positive_qrel_rows:
        raise CollectionError(
            f"{parity_path}: selected qrels contain non-positive rows"
        )

    files = _mapping(manifest.get("files"), f"{parity_path} dataset files")
    documents_file = _mapping(
        files.get(f"documents_{tier}"),
        f"{parity_path} dataset document file",
    )
    queries_file = _mapping(
        files.get("queries"), f"{parity_path} dataset query file"
    )
    document_ids_file = _mapping(
        files.get(f"document_ids_{tier}"),
        f"{parity_path} dataset document ID file",
    )
    report_document_ids = _mapping(
        report_dataset.get("document_ids"), f"{parity_path} document IDs"
    )
    document_ids_path = Path(
        _nonempty_string(
            report_document_ids.get("path"),
            f"{parity_path} document ID path",
        )
    )
    if not document_ids_path.is_absolute():
        document_ids_path = parity_path.parent / document_ids_path
    document_ids_path = document_ids_path.resolve()
    document_ids_size = _integer(
        report_document_ids.get("size_bytes"),
        f"{parity_path} document ID size",
        minimum=1,
    )
    if (
        not document_ids_path.is_file()
        or document_ids_path.stat().st_size != document_ids_size
        or sha256_file(document_ids_path)
        != report_document_ids.get("sha256")
    ):
        raise CollectionError(
            f"{parity_path}: document ID file snapshot changed"
        )
    if (
        documents_file.get("sha256") != dataset["documents_sha256"]
        or documents_file.get("lines") != dataset["corpus_count"]
        or queries_file.get("sha256") != dataset["queries_sha256"]
        or queries_file.get("lines") != dataset["query_count"]
        or document_ids_file.get("sha256")
        != report_document_ids.get("sha256")
        or document_ids_file.get("lines") != dataset["corpus_count"]
    ):
        raise CollectionError(
            f"{parity_path}: dataset manifest file bindings differ"
        )

    selection = _mapping(
        manifest.get("selection"), f"{parity_path} dataset selection"
    )
    query_selection = _nonempty_string(
        selection.get("queries"), f"{parity_path} query selection"
    )
    document_selection = _nonempty_string(
        selection.get("documents"), f"{parity_path} document selection"
    )
    rank = _nonempty_string(
        selection.get("rank"), f"{parity_path} selection rank"
    )
    seed = _nonempty_string(
        selection.get("seed"), f"{parity_path} selection seed"
    )

    qrel_coverage = _mapping(
        manifest.get("qrel_coverage"), f"{parity_path} qrel coverage"
    )
    required_positive_documents = _integer(
        qrel_coverage.get("required_positive_documents"),
        f"{parity_path} required positive documents",
        minimum=1,
    )
    covered_key = f"covered_positive_documents_{tier}"
    document_coverage_key = f"document_coverage_{tier}"
    judgment_coverage_key = f"positive_judgment_coverage_{tier}"
    if (
        qrel_coverage.get(covered_key) != required_positive_documents
        or not _close(
            _number(
                qrel_coverage.get(document_coverage_key),
                f"{parity_path} document qrel coverage",
            ),
            1.0,
        )
        or not _close(
            _number(
                qrel_coverage.get(judgment_coverage_key),
                f"{parity_path} positive judgment coverage",
            ),
            1.0,
        )
    ):
        raise CollectionError(
            f"{parity_path}: selected tier does not cover every positive qrel"
        )

    invariants = _mapping(
        manifest.get("invariants"), f"{parity_path} dataset invariants"
    )
    if invariants.get("100k_is_exact_prefix_of_1m") is not True:
        raise CollectionError(
            f"{parity_path}: nested 100K/1M corpus invariant is not proven"
        )
    if invariants.get("document_ids_are_unique") is not True:
        raise CollectionError(
            f"{parity_path}: unique document ID invariant is not proven"
        )
    if (
        invariants.get(
            "original_query_document_ids_and_qrel_scores_preserved"
        )
        is not True
    ):
        raise CollectionError(
            f"{parity_path}: source ID/qrel preservation is not proven"
        )

    normalization = _mapping(
        manifest.get("normalization"), f"{parity_path} normalization"
    )
    truncation_policy = _mapping(
        normalization.get("document_truncation"),
        f"{parity_path} document truncation policy",
    )
    if (
        truncation_policy.get("maximum_utf8_bytes_including_prefix")
        != maximum_bytes
        or truncation_policy.get(
            "applied_before_text_hashing_deduplication_and_selection"
        )
        is not True
    ):
        raise CollectionError(
            f"{parity_path}: normalization/truncation policy differs"
        )

    source_dataset = _mapping(
        manifest.get("dataset"), f"{parity_path} source dataset"
    )
    source_archive = _mapping(
        source_dataset.get("archive"), f"{parity_path} source archive"
    )
    report_archive = _mapping(
        report_dataset.get("source_archive"),
        f"{parity_path} parity source archive",
    )
    if source_archive.get("sha256") != report_archive.get("sha256"):
        raise CollectionError(
            f"{parity_path}: dataset/source archive binding differs"
        )

    return {
        "manifest": {
            "path": str(manifest_path),
            "size_bytes": manifest_path.stat().st_size,
            "sha256": recorded_sha256,
        },
        "source_dataset": {
            "name": source_dataset.get("name"),
            "archive_sha256": source_archive.get("sha256"),
        },
        "selection": {
            "queries": query_selection,
            "documents": document_selection,
            "rank": rank,
            "seed": seed,
        },
        "queries": {
            "selected": dataset["query_count"],
            "positive_qrel_rows": positive_qrel_rows,
        },
        "qrel_coverage": {
            "required_positive_documents": required_positive_documents,
            "covered_positive_documents": required_positive_documents,
            "document_coverage": 1.0,
            "positive_judgment_coverage": 1.0,
        },
        "invariants": {
            "100k_is_exact_prefix_of_1m": True,
            "document_ids_are_unique": True,
            "original_query_document_ids_and_qrel_scores_preserved": True,
        },
        "normalization": {
            "document_format": normalization.get("document_format"),
            "query_format": normalization.get("query_format"),
            "line_cleaning": normalization.get("line_cleaning"),
            "truncation_algorithm": truncation_policy.get("algorithm"),
            "maximum_utf8_bytes_including_prefix": maximum_bytes,
            "truncation_before_hashing_deduplication_and_selection": True,
        },
        "tier": {
            "name": tier,
            "documents": dataset["corpus_count"],
            "unique_prepared_texts": dataset["corpus_count"],
            "duplicate_rows_after_selection": 0,
            "truncated_documents": truncated_documents,
            "truncated_bytes_removed": truncated_bytes,
            "document_ids": {
                "path": str(document_ids_path),
                "size_bytes": document_ids_size,
                "sha256": report_document_ids.get("sha256"),
            },
        },
    }


def _validate_parity_report(
    *,
    tier: str,
    path: Path,
    dataset: dict[str, Any],
    minimum_cosine: float,
    native_binary_sha256: str | None,
) -> dict[str, Any]:
    path = path.resolve()
    report = _mapping(load_json(path), f"{path}")
    if report.get("schema") != PARITY_SCHEMA:
        raise CollectionError(f"{path}: unknown embedding parity schema")
    if report.get("passed") is not True:
        raise CollectionError(f"{path}: embedding parity did not pass")
    acceptance = _mapping(report.get("acceptance"), f"{path} acceptance")
    declared_threshold = _number(
        acceptance.get("minimum_cosine_similarity"),
        f"{path} declared cosine threshold",
    )
    metrics = _mapping(report.get("metrics"), f"{path} metrics")
    observed_minimum = _number(
        metrics.get("minimum_cosine_similarity"),
        f"{path} minimum cosine",
    )
    if declared_threshold < minimum_cosine:
        raise CollectionError(
            f"{path}: parity threshold {declared_threshold} is below required "
            f"{minimum_cosine}"
        )
    if observed_minimum < declared_threshold or observed_minimum < minimum_cosine:
        raise CollectionError(f"{path}: minimum cosine fails the declared gate")
    report_dataset = _mapping(report.get("dataset"), f"{path} dataset")
    if report_dataset.get("tier") != tier:
        raise CollectionError(f"{path}: parity tier differs")
    documents = _mapping(report_dataset.get("documents"), f"{path} documents")
    queries = _mapping(report_dataset.get("queries"), f"{path} queries")
    if (
        documents.get("sha256") != dataset["documents_sha256"]
        or documents.get("nonempty_lines") != dataset["corpus_count"]
    ):
        raise CollectionError(f"{path}: parity document binding differs")
    if (
        queries.get("sha256") != dataset["queries_sha256"]
        or queries.get("nonempty_lines") != dataset["query_count"]
    ):
        raise CollectionError(f"{path}: parity query binding differs")
    coverage = _mapping(report.get("coverage"), f"{path} coverage")
    query_coverage = _mapping(
        coverage.get("queries"), f"{path} query coverage"
    )
    if (
        query_coverage.get("selected") != dataset["query_count"]
        or query_coverage.get("source_rows") != dataset["query_count"]
        or query_coverage.get("all_queries") is not True
    ):
        raise CollectionError(f"{path}: parity does not cover every query")
    if coverage.get("total_rows") != metrics.get("rows"):
        raise CollectionError(f"{path}: parity coverage/metric row counts differ")
    if metrics.get("dimensions") != dataset["dimensions"]:
        raise CollectionError(f"{path}: parity embedding dimensions differ")
    _number(
        metrics.get("maximum_absolute_difference"),
        f"{path} maximum absolute difference",
    )
    documents_coverage = _mapping(
        coverage.get("documents"), f"{path} document coverage"
    )
    selected_documents = _integer(
        documents_coverage.get("selected_documents"),
        f"{path} selected documents",
        minimum=1,
    )
    truncation_proof = _mapping(
        documents_coverage.get("truncation_proof"),
        f"{path} truncation proof",
    )
    if (
        truncation_proof.get("checked_documents") != selected_documents
        or _integer(
            truncation_proof.get("source_verified_truncated_documents"),
            f"{path} verified truncated documents",
            minimum=1,
        )
        <= 0
    ):
        raise CollectionError(f"{path}: parity truncation proof is incomplete")
    by_kind = _mapping(metrics.get("by_kind"), f"{path} metrics by kind")
    if _mapping(by_kind.get("query"), f"{path} query metrics").get(
        "rows"
    ) != dataset["query_count"]:
        raise CollectionError(f"{path}: parity query metric row count differs")
    if _mapping(by_kind.get("document"), f"{path} document metrics").get(
        "rows"
    ) != selected_documents:
        raise CollectionError(f"{path}: parity document metric row count differs")

    reference = _mapping(
        report.get("official_comparison_reference"),
        f"{path} official comparison reference",
    )
    corpus_cache = _mapping(
        reference.get("corpus_cache"), f"{path} corpus cache"
    )
    query_cache = _mapping(
        reference.get("query_cache"), f"{path} query cache"
    )
    model = _mapping(reference.get("model"), f"{path} reference model")
    if corpus_cache.get("sha256") != dataset["corpus_cache_sha256"]:
        raise CollectionError(f"{path}: parity corpus cache binding differs")
    if query_cache.get("sha256") != dataset["query_cache_sha256"]:
        raise CollectionError(f"{path}: parity query cache binding differs")
    if model.get("sha256") != dataset["model_sha256"]:
        raise CollectionError(f"{path}: parity model identity differs")
    model_artifact = _mapping(
        model.get("artifact"), f"{path} reference model artifact"
    )
    if model_artifact.get("sha256") != dataset["model_artifact_sha256"]:
        raise CollectionError(f"{path}: parity model artifact binding differs")
    for label, cache, expected_count in (
        ("corpus", corpus_cache, dataset["corpus_count"]),
        ("query", query_cache, dataset["query_count"]),
    ):
        if (
            cache.get("count") != expected_count
            or cache.get("dimensions") != dataset["dimensions"]
            or cache.get("fingerprint") != dataset["fingerprint"]
        ):
            raise CollectionError(f"{path}: parity {label} cache metadata differs")
        sidecar_hash = cache.get("sidecar_sha256")
        if not isinstance(sidecar_hash, str) or not re.fullmatch(
            r"[0-9a-f]{64}", sidecar_hash
        ):
            raise CollectionError(f"{path}: parity {label} sidecar hash is invalid")
    native = _mapping(report.get("native"), f"{path} native")
    native_model = _mapping(native.get("model"), f"{path} native model")
    if native_model.get("sha256") != dataset["model_artifact_sha256"]:
        raise CollectionError(f"{path}: parity native GGUF binding differs")
    native_binary = _mapping(native.get("binary"), f"{path} native binary")
    if (
        native_binary_sha256 is not None
        and native_binary.get("sha256") != native_binary_sha256
    ):
        raise CollectionError(f"{path}: parity native binary binding differs")

    fixture = _mapping(report.get("fixture"), f"{path} fixture")
    fixture_rows = _sequence(fixture.get("rows"), f"{path} fixture rows")
    if len(fixture_rows) != coverage.get("total_rows"):
        raise CollectionError(f"{path}: parity fixture row count differs")
    fixture_source = _mapping(
        fixture.get("source"), f"{path} fixture source"
    )
    reference_cache = _mapping(
        fixture.get("reference_cache"), f"{path} fixture reference cache"
    )
    native_cache = _mapping(native.get("cache"), f"{path} native cache")
    parity_verifier = FileVerifier(enabled=True)
    fixture_source_path = parity_verifier.verify(
        fixture_source, f"{path} fixture source"
    )
    reference_cache_path = parity_verifier.verify(
        reference_cache, f"{path} fixture reference cache"
    )
    native_cache_path = parity_verifier.verify(
        native_cache, f"{path} native cache"
    )
    try:
        reference_vectors, reference_metadata = read_cache(
            reference_cache_path, require_unit=True
        )
        native_vectors, native_metadata = read_cache(
            native_cache_path, require_unit=True
        )
    except (OSError, ValueError) as error:
        raise CollectionError(
            f"{path}: cannot independently read parity vector evidence: {error}"
        ) from error
    expected_rows = len(fixture_rows)
    expected_dimensions = dataset["dimensions"]
    expected_fingerprint = dataset["fingerprint"]
    for label, metadata in (
        ("reference", reference_metadata),
        ("native", native_metadata),
    ):
        if (
            metadata.get("count") != expected_rows
            or metadata.get("dimensions") != expected_dimensions
            or metadata.get("fingerprint") != expected_fingerprint
        ):
            raise CollectionError(
                f"{path}: parity {label} vector evidence metadata differs"
            )
    if (
        reference_metadata.get("version") != 2
        or reference_metadata.get("source_size_bytes")
        != fixture_source_path.stat().st_size
        or reference_metadata.get("source_sha256")
        != sha256_file(fixture_source_path)
    ):
        raise CollectionError(
            f"{path}: parity reference cache is not bound to the fixture source"
        )
    reference64 = np.asarray(reference_vectors, dtype=np.float64)
    native64 = np.asarray(native_vectors, dtype=np.float64)
    denominator = np.linalg.norm(reference64, axis=1) * np.linalg.norm(
        native64, axis=1
    )
    if np.any(denominator <= np.finfo(np.float64).tiny):
        raise CollectionError(f"{path}: parity evidence contains a zero vector")
    cosine_values = np.clip(
        np.einsum("ij,ij->i", reference64, native64) / denominator,
        -1.0,
        1.0,
    )
    absolute_values = np.abs(reference64 - native64)
    recomputed_minimum = float(np.min(cosine_values))
    recomputed_maximum_absolute = float(np.max(absolute_values))
    if not math.isclose(
        observed_minimum, recomputed_minimum, rel_tol=0.0, abs_tol=1e-12
    ):
        raise CollectionError(
            f"{path}: reported minimum cosine differs from vector evidence"
        )
    reported_maximum_absolute = _number(
        metrics.get("maximum_absolute_difference"),
        f"{path} maximum absolute difference",
    )
    if not math.isclose(
        reported_maximum_absolute,
        recomputed_maximum_absolute,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise CollectionError(
            f"{path}: reported maximum absolute difference differs from "
            "vector evidence"
        )
    kind_counts: dict[str, int] = defaultdict(int)
    kind_indices: dict[str, list[int]] = defaultdict(list)
    for ordinal, row in enumerate(fixture_rows):
        record = _mapping(row, f"{path} fixture row {ordinal}")
        kind = _nonempty_string(
            record.get("kind"), f"{path} fixture row {ordinal} kind"
        )
        kind_counts[kind] += 1
        kind_indices[kind].append(ordinal)
    if kind_counts != {
        "query": dataset["query_count"],
        "document": selected_documents,
    }:
        raise CollectionError(f"{path}: parity fixture row kinds differ")
    if recomputed_minimum < declared_threshold or recomputed_minimum < minimum_cosine:
        raise CollectionError(
            f"{path}: independently recomputed cosine fails the parity gate"
        )
    reported_mean = _number(
        metrics.get("mean_cosine_similarity"), f"{path} mean cosine"
    )
    if not math.isclose(
        reported_mean,
        float(np.mean(cosine_values)),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise CollectionError(
            f"{path}: reported mean cosine differs from vector evidence"
        )
    for kind, indices in sorted(kind_indices.items()):
        reported_kind = _mapping(
            by_kind.get(kind), f"{path} {kind} metrics"
        )
        expected_kind_minimum = float(np.min(cosine_values[indices]))
        expected_kind_maximum = float(
            np.max(np.max(absolute_values[indices], axis=1))
        )
        if (
            reported_kind.get("rows") != len(indices)
            or not math.isclose(
                _number(
                    reported_kind.get("minimum_cosine_similarity"),
                    f"{path} {kind} minimum cosine",
                ),
                expected_kind_minimum,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                _number(
                    reported_kind.get("maximum_absolute_difference"),
                    f"{path} {kind} maximum absolute difference",
                ),
                expected_kind_maximum,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise CollectionError(
                f"{path}: reported {kind} metrics differ from vector evidence"
            )

    for context, record in (
        ("dataset manifest", report_dataset.get("manifest")),
        ("source archive", report_dataset.get("source_archive")),
        ("document IDs", report_dataset.get("document_ids")),
        ("native cache", native_cache),
        (
            "fixture reference cache",
            reference_cache,
        ),
    ):
        item = _mapping(record, f"{path} {context}")
        digest = item.get("sha256")
        if not isinstance(digest, str) or not re.fullmatch(
            r"[0-9a-f]{64}", digest
        ):
            raise CollectionError(f"{path}: {context} hash is invalid")
    if not isinstance(report.get("input_hash"), str) or not re.fullmatch(
        r"[0-9a-f]{64}", report["input_hash"]
    ):
        raise CollectionError(f"{path}: parity input hash is invalid")
    dataset_derivation = _validate_dataset_derivation(
        parity_path=path,
        tier=tier,
        report_dataset=report_dataset,
        dataset=dataset,
    )
    return {
        "tier": tier,
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "input_hash": report.get("input_hash"),
        "passed": True,
        "acceptance": acceptance,
        "metrics": metrics,
        "coverage": coverage,
        "dataset_derivation": dataset_derivation,
        "bindings": {
            "documents_sha256": documents.get("sha256"),
            "queries_sha256": queries.get("sha256"),
            "corpus_cache_sha256": corpus_cache.get("sha256"),
            "query_cache_sha256": query_cache.get("sha256"),
            "model_sha256": model.get("sha256"),
            "model_artifact_sha256": native_model.get("sha256"),
            "native_binary_sha256": native_binary.get("sha256"),
        },
    }


def _validate_endpoint_attestation(
    *,
    tier: str,
    path: Path,
    dataset: dict[str, Any],
    parity: dict[str, Any],
    verifier: FileVerifier,
) -> dict[str, Any]:
    path = path.resolve()
    report = _mapping(load_json(path), f"{path}")
    if (
        report.get("schema") != ENDPOINT_ATTESTATION_SCHEMA
        or report.get("phase") != "finalized"
        or _mapping(
            report.get("post_run_native_parity"),
            f"{path} post-run native parity",
        ).get("status")
        != "verified"
    ):
        raise CollectionError(f"{path}: endpoint attestation is not finalized")
    evidence = _mapping(report.get("evidence"), f"{path} evidence")
    if report.get("attestation_id") != canonical_hash(evidence):
        raise CollectionError(f"{path}: endpoint attestation ID differs")

    live = _mapping(evidence.get("live_capture"), f"{path} live capture")
    capture_path = verifier.verify(
        live.get("artifact"), f"{path} live capture artifact"
    ).resolve()
    capture = _mapping(load_json(capture_path), f"{capture_path}")
    embedded_capture = _mapping(live.get("report"), f"{path} embedded live capture")
    if capture != embedded_capture:
        raise CollectionError(
            f"{path}: embedded live capture differs from its immutable artifact"
        )
    if (
        capture.get("schema") != ENDPOINT_ATTESTATION_SCHEMA
        or capture.get("phase") != "live-capture"
    ):
        raise CollectionError(f"{path}: live capture schema/phase differs")
    capture_evidence = _mapping(
        capture.get("evidence"), f"{path} live capture evidence"
    )
    capture_id = capture.get("attestation_id")
    if (
        capture_id != canonical_hash(capture_evidence)
        or live.get("attestation_id") != capture_id
    ):
        raise CollectionError(f"{path}: live capture attestation ID differs")

    final_binding = _mapping(evidence.get("binding"), f"{path} final binding")
    capture_binding = _mapping(
        capture_evidence.get("required_post_run_native_parity"),
        f"{path} captured parity binding",
    )
    if (
        _mapping(
            capture_evidence.get("source"), f"{path} captured source"
        )
        != _mapping(
            capture_binding.get("source"),
            f"{path} captured binding source",
        )
        or capture_evidence.get("source_unchanged") is not True
        or capture_evidence.get("model_artifact_unchanged") is not True
    ):
        raise CollectionError(
            f"{path}: live capture source/model stability proof differs"
        )
    prefix_scope = final_binding.get("scope") == "prefix-seed"
    if final_binding != capture_binding:
        if not prefix_scope:
            raise CollectionError(
                f"{path}: finalized binding differs from the live capture"
            )
        prefix = _mapping(
            capture_evidence.get("prefix_seed"),
            f"{path} captured prefix seed",
        )
        prefix_source = _mapping(
            prefix.get("source"), f"{path} captured prefix source"
        )
        prefix_cache = _mapping(
            prefix.get("cache"), f"{path} captured prefix cache"
        )
        prefix_metadata = _mapping(
            prefix.get("cache_metadata"),
            f"{path} captured prefix cache metadata",
        )
        expected_prefix_binding = {
            "tier": tier,
            "expected_source_count": prefix.get("rows"),
            "source": prefix_source,
            "model_descriptor_sha256": capture_binding.get(
                "model_descriptor_sha256"
            ),
            "model_artifact_sha256": capture_binding.get(
                "model_artifact_sha256"
            ),
            "fingerprint": prefix_metadata.get("fingerprint"),
            "dimension": prefix_metadata.get("dimensions"),
            "checkpoint_input_hash": capture_binding.get(
                "checkpoint_input_hash"
            ),
            "scope": "prefix-seed",
            "parent_tier": capture_binding.get("tier"),
            "prefix_cache_sha256": prefix_cache.get("sha256"),
        }
        if final_binding != expected_prefix_binding:
            raise CollectionError(
                f"{path}: finalized prefix binding differs from the live "
                "capture prefix seed"
            )
    if final_binding.get("tier") != tier:
        raise CollectionError(f"{path}: endpoint attestation tier differs")
    source = _mapping(final_binding.get("source"), f"{path} source binding")
    source_path = verifier.verify(source, f"{path} source binding").resolve()
    if (
        source_path != Path(dataset["documents_path"]).resolve()
        or source.get("sha256") != dataset["documents_sha256"]
        or source.get("size_bytes") != dataset["raw_text_file_bytes"]
        or source.get("nonempty_lines") != dataset["corpus_count"]
        or final_binding.get("expected_source_count") != dataset["corpus_count"]
    ):
        raise CollectionError(f"{path}: endpoint attestation source differs")
    expected_binding = {
        "model_descriptor_sha256": dataset["model_sha256"],
        "model_artifact_sha256": dataset["model_artifact_sha256"],
        "fingerprint": dataset["fingerprint"],
        "dimension": dataset["dimensions"],
    }
    for field, expected in expected_binding.items():
        if final_binding.get(field) != expected:
            raise CollectionError(
                f"{path}: endpoint attestation {field} differs"
            )

    model_artifact = _mapping(
        capture_evidence.get("model_artifact"),
        f"{path} captured model artifact",
    )
    verified_model = verifier.verify(
        model_artifact, f"{path} captured model artifact"
    ).resolve()
    model_artifact_after = _mapping(
        capture_evidence.get("model_artifact_after"),
        f"{path} captured model artifact after",
    )
    verifier.verify(
        model_artifact_after, f"{path} captured model artifact after"
    )
    if (
        model_artifact.get("sha256") != dataset["model_artifact_sha256"]
        or model_artifact_after.get("sha256")
        != model_artifact.get("sha256")
        or verified_model
        != Path(
            _nonempty_string(
                model_artifact.get("path"), f"{path} model artifact path"
            )
        ).resolve()
    ):
        raise CollectionError(f"{path}: captured model artifact differs")

    endpoint = _mapping(
        capture_evidence.get("endpoint"), f"{path} captured endpoint"
    )
    for moment in ("before", "after"):
        health = _mapping(
            endpoint.get(f"health_{moment}"),
            f"{path} endpoint health {moment}",
        )
        if (
            health.get("status") != 200
            or _mapping(
                health.get("payload"), f"{path} health payload {moment}"
            ).get("status")
            != "ok"
        ):
            raise CollectionError(
                f"{path}: captured endpoint health {moment} was not healthy"
            )
    props_identity = _mapping(
        endpoint.get("props_identity"), f"{path} endpoint props identity"
    )
    if (
        props_identity.get("build_info")
        != capture_evidence.get("expected_build_info")
        or Path(
            _nonempty_string(
                props_identity.get("model_path"), f"{path} endpoint model path"
            )
        ).resolve()
        != verified_model
    ):
        raise CollectionError(f"{path}: endpoint props identity differs")
    checkpoint = _mapping(
        capture_evidence.get("checkpoint"), f"{path} captured checkpoint"
    )
    stable_checkpoint = _mapping(
        checkpoint.get("stable_identity"), f"{path} checkpoint stable identity"
    )
    checkpoint_model = _mapping(
        stable_checkpoint.get("model"), f"{path} checkpoint model"
    )
    checkpoint_source = _mapping(
        stable_checkpoint.get("source"), f"{path} checkpoint source"
    )
    checkpoint_identity_differs = (
        stable_checkpoint.get("fingerprint") != dataset["fingerprint"]
        or stable_checkpoint.get("dimension") != dataset["dimensions"]
        or checkpoint_model.get("sha256") != dataset["model_sha256"]
        or stable_checkpoint.get("embedding_endpoint")
        != endpoint.get("embedding_url")
    )
    if prefix_scope:
        checkpoint_prefix = _mapping(
            stable_checkpoint.get("prefix_seed"),
            f"{path} checkpoint prefix seed",
        )
        checkpoint_identity_differs = checkpoint_identity_differs or (
            checkpoint_prefix.get("cache_sha256")
            != final_binding.get("prefix_cache_sha256")
            or checkpoint_prefix.get("rows") != dataset["corpus_count"]
            or checkpoint_prefix.get("source_prefix_sha256")
            != dataset["documents_sha256"]
            or checkpoint_prefix.get("source_prefix_size_bytes")
            != dataset["raw_text_file_bytes"]
        )
    else:
        checkpoint_identity_differs = checkpoint_identity_differs or (
            checkpoint_source.get("sha256") != dataset["documents_sha256"]
        )
    if checkpoint_identity_differs:
        raise CollectionError(f"{path}: captured checkpoint identity differs")

    process = _mapping(
        capture_evidence.get("process_proof"), f"{path} process proof"
    )
    process_before = _mapping(
        process.get("before"), f"{path} process proof before"
    )
    process_after = _mapping(
        process.get("after"), f"{path} process proof after"
    )
    if (
        process.get("status") != "verified"
        or process.get("required") is not True
        or process.get("unchanged") is not True
        or process.get("server_started_before_checkpoint") is not True
        or process_before.get("identity_sha256")
        != process_after.get("identity_sha256")
    ):
        raise CollectionError(f"{path}: live server process proof is not verified")
    for moment, record in (("before", process_before), ("after", process_after)):
        stable_process = {
            key: value
            for key, value in record.items()
            if key != "identity_sha256"
        }
        if record.get("identity_sha256") != canonical_hash(stable_process):
            raise CollectionError(
                f"{path}: process identity hash {moment} differs"
            )
        executable = _mapping(
            record.get("executable"), f"{path} process executable {moment}"
        )
        verifier.verify(executable, f"{path} process executable {moment}")
        if Path(
            _nonempty_string(
                record.get("command_model_path"),
                f"{path} process model path {moment}",
            )
        ).resolve() != verified_model:
            raise CollectionError(
                f"{path}: process model argument differs from the captured GGUF"
            )

    native_parity = _mapping(
        evidence.get("native_parity"), f"{path} finalized native parity"
    )
    parity_artifact = _mapping(
        native_parity.get("artifact"), f"{path} finalized parity artifact"
    )
    parity_path = verifier.verify(
        parity_artifact, f"{path} finalized parity artifact"
    ).resolve()
    if (
        parity_path != Path(parity["path"]).resolve()
        or parity_artifact.get("sha256") != parity["sha256"]
        or native_parity.get("input_hash") != parity["input_hash"]
        or native_parity.get("minimum_cosine_similarity")
        != parity["metrics"]["minimum_cosine_similarity"]
        or native_parity.get("required_minimum_cosine_similarity")
        != parity["acceptance"]["minimum_cosine_similarity"]
        or _mapping(
            native_parity.get("native_model"),
            f"{path} finalized native model",
        ).get("sha256")
        != dataset["model_artifact_sha256"]
        or native_parity.get("corpus_cache_sha256")
        != dataset["corpus_cache_sha256"]
    ):
        raise CollectionError(
            f"{path}: finalized endpoint attestation differs from validated parity"
        )
    native_binary = _mapping(
        native_parity.get("native_binary"),
        f"{path} finalized native binary",
    )
    if native_binary.get("sha256") != parity["bindings"][
        "native_binary_sha256"
    ]:
        raise CollectionError(
            f"{path}: endpoint attestation native binary differs from parity"
        )
    return {
        "tier": tier,
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "attestation_id": report["attestation_id"],
        "phase": "finalized",
        "live_capture": {
            "path": str(capture_path),
            "sha256": live["artifact"]["sha256"],
            "attestation_id": capture_id,
        },
        "bindings": {
            "documents_sha256": dataset["documents_sha256"],
            "model_sha256": dataset["model_sha256"],
            "model_artifact_sha256": dataset["model_artifact_sha256"],
            "fingerprint": dataset["fingerprint"],
            "dimensions": dataset["dimensions"],
            "corpus_cache_sha256": dataset["corpus_cache_sha256"],
            "parity_sha256": parity["sha256"],
            "native_binary_sha256": native_binary.get("sha256"),
            "server_build_info": props_identity.get("build_info"),
            "server_executable_sha256": process_before["executable"].get(
                "sha256"
            ),
        },
        "process_proof": {
            "status": "verified",
            "provider": process_before.get("provider"),
            "pid": process_before.get("pid"),
            "identity_sha256": process_before.get("identity_sha256"),
        },
    }


def _resolve_stage_path(
    manifest_path: Path, stage_name: str, recorded: Any
) -> Path:
    candidates: list[Path] = []
    if isinstance(recorded, str) and recorded:
        candidate = Path(recorded)
        candidates.append(
            candidate if candidate.is_absolute() else manifest_path.parent / candidate
        )
    candidates.append(manifest_path.parent / "stages" / f"{stage_name}.json")
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0].resolve()


def _validate_document_id_prefix(
    dataset_derivation: dict[str, Any],
) -> dict[str, Any] | None:
    tiers = {
        item["name"]: item
        for item in _sequence(
            dataset_derivation.get("tiers"),
            "dataset derivation tiers",
        )
    }
    if not set(DECLARED_TIER_CARDINALITIES).issubset(tiers):
        return None
    smaller = _mapping(
        tiers["100k"].get("document_ids"), "100k document IDs"
    )
    larger = _mapping(
        tiers["1m"].get("document_ids"), "1m document IDs"
    )
    smaller_size = _integer(
        smaller.get("size_bytes"), "100k document ID bytes", minimum=1
    )
    if _integer(
        larger.get("size_bytes"), "1m document ID bytes", minimum=1
    ) <= smaller_size:
        raise CollectionError(
            "1m document ID mapping is not larger than 100k"
        )
    prefix_sha256 = sha256_file(
        Path(_nonempty_string(larger.get("path"), "1m document ID path")),
        length=smaller_size,
    )
    if prefix_sha256 != smaller.get("sha256"):
        raise CollectionError(
            "100k document ID mapping is not the exact byte prefix of 1m"
        )
    return {
        "document_ids_prefix_size_bytes": smaller_size,
        "document_ids_prefix_sha256": prefix_sha256,
        "document_ids_exact_byte_prefix": True,
    }


def _validate_declared_run_role(
    declared: Any, planned_stages: Sequence[str], context: str
) -> str:
    run_role = _nonempty_string(declared, f"{context} run role")
    components = run_role.split("+")
    allowed = {
        "native-gate",
        "native-sweep",
        "official-cached-sweep",
        "official-real-run",
        "official-real-search-reuse",
    }
    if (
        any(component not in allowed for component in components)
        or len(set(components)) != len(components)
    ):
        raise CollectionError(f"{context}: invalid explicit run role {run_role!r}")
    planned = set(planned_stages)
    native_present = any(name.startswith("native-") for name in planned)
    native_roles = {
        component for component in components if component.startswith("native-")
    }
    if native_present != (len(native_roles) == 1):
        raise CollectionError(
            f"{context}: explicit native run role does not match planned stages"
        )
    expected_official: set[str] = set()
    if "official-search-cached" in planned:
        expected_official.add("official-cached-sweep")
    if "official-search-real" in planned:
        expected_official.add(
            "official-real-run"
            if "official-build" in planned
            else "official-real-search-reuse"
        )
    declared_official = {
        component
        for component in components
        if component.startswith("official-")
    }
    if declared_official != expected_official:
        raise CollectionError(
            f"{context}: explicit official run role does not match planned stages"
        )
    if not native_present and not expected_official:
        raise CollectionError(f"{context}: run role has no executable stage family")
    canonical_components = [
        component
        for component in (
            "native-gate",
            "native-sweep",
            "official-cached-sweep",
            "official-real-run",
            "official-real-search-reuse",
        )
        if component in components
    ]
    canonical = "+".join(canonical_components)
    if canonical != run_role:
        raise CollectionError(
            f"{context}: run role components are not in canonical order"
        )
    return run_role


def _expected_observation_command(
    inputs: dict[str, Any],
    *,
    kind: str,
    ordinal: int,
    context: str,
) -> list[str]:
    raw = inputs.get("command_template", inputs.get("command"))
    command = _sequence(raw, f"{context} planned command")
    if not command or not all(isinstance(value, str) for value in command):
        raise CollectionError(f"{context}: planned command is not an argv string array")
    return [
        value.replace("{kind}", kind).replace("{ordinal}", str(ordinal))
        for value in command
    ]


def _recognized_stage_name(name: str) -> bool:
    return name in {
        "native-build",
        "native-stats",
        "official-build",
        "official-search-cached",
        "official-search-real",
    } or re.fullmatch(r"native-search-ef[1-9][0-9]*", name) is not None


def _recorded_endpoint_payload(
    observation: dict[str, Any], endpoint: str, context: str
) -> Any:
    record = _mapping(
        observation.get(endpoint), f"{context} endpoint {endpoint}"
    )
    if record.get("status") != 200:
        raise CollectionError(f"{context}: endpoint {endpoint} was not HTTP 200")
    return record.get("payload")


def _normalize_url(value: Any) -> str:
    if not isinstance(value, str) or not value:
        return ""
    normalized = value.rstrip("/")
    if normalized.endswith("/v1"):
        normalized = normalized[:-3]
    return normalized.rstrip("/")


def _validate_recorded_official_attestation(
    *,
    provenance: dict[str, Any],
    shared: dict[str, Any],
    dataset: dict[str, Any],
    tooling_by_name: dict[str, dict[str, Any]],
    context: str,
) -> dict[str, Any]:
    runtime = _mapping(
        provenance.get("official_runtime"), f"{context} official runtime"
    )
    embedding = _mapping(
        provenance.get("embedding_endpoint_observation"),
        f"{context} embedding endpoint observation",
    )
    candidate = _mapping(
        provenance.get("candidate_endpoint_observation"),
        f"{context} candidate endpoint observation",
    )
    props = _mapping(
        _recorded_endpoint_payload(embedding, "/props", context),
        f"{context} embedding props",
    )
    artifact = _mapping(
        _mapping(shared.get("model"), f"{context} model").get("artifact"),
        f"{context} model artifact",
    )
    model_path = artifact.get("path")
    if not isinstance(model_path, str) or not model_path:
        raise CollectionError(f"{context}: model artifact path is missing")
    if Path(str(props.get("model_path", ""))).resolve() != Path(model_path).resolve():
        raise CollectionError(f"{context}: recorded endpoint model path differs")
    server = _mapping(
        provenance.get("server_context"), f"{context} server context"
    )
    parallel = _integer(
        server.get("parallel"), f"{context} server parallelism", minimum=1
    )
    per_slot_ctx = _integer(
        server.get("per_slot_ctx"),
        f"{context} per-slot context",
        minimum=1,
    )
    required_product = _integer(
        server.get("required_product"),
        f"{context} required context product",
        minimum=1,
    )
    server_ctx_size = _integer(
        server.get("server_ctx_size"),
        f"{context} server context size",
        minimum=1,
    )
    if props.get("total_slots") != parallel:
        raise CollectionError(f"{context}: recorded endpoint slot count differs")
    settings = _mapping(
        props.get("default_generation_settings"),
        f"{context} endpoint generation settings",
    )
    if settings.get("n_ctx") != per_slot_ctx:
        raise CollectionError(f"{context}: recorded endpoint context differs")
    if required_product != parallel * per_slot_ctx:
        raise CollectionError(f"{context}: server context product is inconsistent")
    if server_ctx_size < required_product:
        raise CollectionError(f"{context}: total server context is undersized")

    metrics = _mapping(
        _recorded_endpoint_payload(candidate, "/metrics", context),
        f"{context} candidate endpoint metrics",
    )
    mode = provenance.get("official_candidate_recompute_mode")
    expected_mode = "cache" if mode == "cached" else "proxy"
    if mode not in ("cached", "real") or metrics.get("mode") != expected_mode:
        raise CollectionError(f"{context}: recorded candidate endpoint mode differs")
    if mode == "cached":
        identity = _mapping(
            metrics.get("cache_identity"), f"{context} cached endpoint identity"
        )
        expected = {
            "cache_sha256": dataset["corpus_cache_sha256"],
            "source_sha256": dataset["documents_sha256"],
            "fingerprint": dataset["fingerprint"],
            "dimensions": dataset["dimensions"],
            "count": dataset["corpus_count"],
            "model_sha256": dataset["model_sha256"],
        }
        for key, value in expected.items():
            if identity.get(key) != value:
                raise CollectionError(
                    f"{context}: recorded cached endpoint {key} differs"
                )
        server_tool = tooling_by_name.get("cached_embedding_server.py")
        if server_tool is not None and identity.get(
            "server_script_sha256"
        ) != server_tool.get("sha256"):
            raise CollectionError(
                f"{context}: recorded cached server script hash differs"
            )
    else:
        if _normalize_url(metrics.get("upstream")) != _normalize_url(
            provenance.get("embedding_endpoint")
        ):
            raise CollectionError(f"{context}: recorded proxy upstream differs")
        proxy_tool = tooling_by_name.get("openai_embedding_proxy.py")
        if proxy_tool is not None and metrics.get(
            "proxy_script_sha256"
        ) != proxy_tool.get("sha256"):
            raise CollectionError(
                f"{context}: recorded proxy script hash differs"
            )
    return {
        "official_runtime_sha256": canonical_hash(runtime),
        "machine_sha256": canonical_hash(
            _mapping(provenance.get("environment"), f"{context} environment")
        ),
        "server_context_sha256": canonical_hash(server),
        "embedding_endpoint_observation_sha256": canonical_hash(embedding),
        "candidate_endpoint_observation_sha256": canonical_hash(candidate),
    }


def _read_native_csv(
    path: Path,
    *,
    query_count: int,
    corpus_count: int,
    recall_cutoffs: Sequence[int],
    top_k: int,
    truth: np.ndarray,
    parsed: dict[str, Any],
    context: str,
) -> dict[str, Any]:
    try:
        with path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            rows = list(reader)
            fields = reader.fieldnames or []
    except (OSError, csv.Error) as error:
        raise CollectionError(f"{context}: cannot parse raw CSV: {error}") from error
    required = {
        "query_index",
        "recall",
        "latency_ms",
        "exact_recomputations",
        "approximate_distances",
        "upper_layer_hops",
        "embedding_batches",
        "result_ids",
    }
    for cutoff in recall_cutoffs:
        if cutoff != top_k:
            required.add(f"recall_at_{cutoff}")
    missing = sorted(required - set(fields))
    if missing:
        raise CollectionError(f"{context}: raw CSV missing columns {missing}")
    if len(rows) != query_count:
        raise CollectionError(
            f"{context}: raw CSV has {len(rows)} rows, expected {query_count}"
        )
    metrics: dict[str, Any] = defaultdict(list)
    result_ids: list[list[int]] = []
    for expected_index, row in enumerate(rows):
        try:
            observed_index = int(row["query_index"])
        except (TypeError, ValueError) as error:
            raise CollectionError(
                f"{context}: invalid query_index at row {expected_index + 2}"
            ) from error
        if observed_index != expected_index:
            raise CollectionError(
                f"{context}: query indices must be exactly 0..{query_count - 1}"
            )
        raw_result_ids = row.get("result_ids")
        if not isinstance(raw_result_ids, str) or not re.fullmatch(
            r"(?:0|[1-9][0-9]*)(?: (?:0|[1-9][0-9]*))*",
            raw_result_ids,
        ):
            raise CollectionError(
                f"{context}: result_ids at query {expected_index} are not "
                "canonical space-separated unsigned decimals"
            )
        labels = [int(value) for value in raw_result_ids.split(" ")]
        if len(labels) != top_k:
            raise CollectionError(
                f"{context}: result_ids at query {expected_index} contain "
                f"{len(labels)} IDs, expected exactly {top_k}"
            )
        if len(set(labels)) != len(labels):
            raise CollectionError(
                f"{context}: result_ids at query {expected_index} contain "
                "duplicates"
            )
        if any(label >= corpus_count for label in labels):
            raise CollectionError(
                f"{context}: result_ids at query {expected_index} are outside "
                "the corpus"
            )
        result_ids.append(labels)
        for counter in (
            "exact_recomputations",
            "approximate_distances",
            "upper_layer_hops",
            "embedding_batches",
        ):
            raw_counter = row.get(counter)
            if not isinstance(raw_counter, str) or not re.fullmatch(
                r"0|[1-9][0-9]*", raw_counter
            ):
                raise CollectionError(
                    f"{context}: {counter} at query {expected_index} must be "
                    "a canonical nonnegative integer"
                )
            metrics[counter].append(int(raw_counter))
        columns = {"latency_ms": "latency_ms"}
        for cutoff in recall_cutoffs:
            columns[f"recall_at_{cutoff}"] = (
                "recall" if cutoff == top_k else f"recall_at_{cutoff}"
            )
        for metric, column in columns.items():
            try:
                value = float(row[column])
            except (TypeError, ValueError) as error:
                raise CollectionError(
                    f"{context}: invalid {column!r} at query {expected_index}"
                ) from error
            if not math.isfinite(value):
                raise CollectionError(
                    f"{context}: non-finite {column!r} at query {expected_index}"
                )
            if metric == "latency_ms" and value < 0.0:
                raise CollectionError(
                    f"{context}: negative latency at query {expected_index}"
                )
            if metric.startswith("recall_at_") and not 0.0 <= value <= 1.0:
                raise CollectionError(
                    f"{context}: recall outside [0,1] at query {expected_index}"
                )
            metrics[metric].append(value)
        truth_row = truth[expected_index]
        for cutoff in recall_cutoffs:
            expected_recall = len(
                set(labels[:cutoff])
                & {int(value) for value in truth_row[:cutoff]}
            ) / cutoff
            actual_recall = metrics[f"recall_at_{cutoff}"][-1]
            if not _close(actual_recall, expected_recall):
                raise CollectionError(
                    f"{context}: native Recall@{cutoff} at query "
                    f"{expected_index}={actual_recall} does not match "
                    f"result IDs/truth {expected_recall}"
                )

    expected_summaries: dict[str, float] = {
        "latency_ms_mean": float(np.mean(metrics["latency_ms"])),
        "latency_ms_p50": _nearest_rank(metrics["latency_ms"], 0.50),
        "latency_ms_p95": _nearest_rank(metrics["latency_ms"], 0.95),
        "exact_recomputations_mean": float(
            np.mean(metrics["exact_recomputations"])
        ),
        "approximate_distances_mean": float(
            np.mean(metrics["approximate_distances"])
        ),
        "upper_layer_hops_mean": float(np.mean(metrics["upper_layer_hops"])),
    }
    for cutoff in recall_cutoffs:
        expected_summaries[f"recall_at_{cutoff}"] = float(
            np.mean(metrics[f"recall_at_{cutoff}"])
        )
    if parsed.get("queries") != query_count:
        raise CollectionError(
            f"{context}: parsed stdout query count does not match raw CSV"
        )
    for key, expected in expected_summaries.items():
        actual = _number(parsed.get(key), f"{context} parsed stdout {key}")
        if not _close(actual, expected):
            raise CollectionError(
                f"{context}: parsed stdout {key}={actual} does not match "
                f"raw data {expected}"
            )
    return {
        **dict(metrics),
        "result_ids_sha256": canonical_hash(result_ids),
    }


def _validate_official_point(
    point: dict[str, Any],
    *,
    mode: str,
    dataset: dict[str, Any],
    query_count: int,
    report_ks: Sequence[int],
    truth: np.ndarray,
    query_indices: Sequence[int],
    top_k: int,
    context: str,
) -> dict[str, Any]:
    complexity = _integer(point.get("complexity"), f"{context} complexity", minimum=1)
    batch_size = _integer(point.get("batch_size"), f"{context} batch size")
    if point.get("queries") != query_count:
        raise CollectionError(f"{context}: point query count differs")
    latency = _mapping(point.get("latency_ms"), f"{context} latency")
    raw_latency = [
        _number(value, f"{context} raw latency")
        for value in _sequence(latency.get("raw_per_query"), f"{context} raw latency")
    ]
    if len(raw_latency) != query_count:
        raise CollectionError(f"{context}: raw latency count differs")
    if any(value < 0.0 for value in raw_latency):
        raise CollectionError(f"{context}: raw latency contains a negative value")
    expected_latency = {
        "mean": float(np.mean(raw_latency)),
        "p50": _linear_percentile(raw_latency, 0.50),
        "p95": _linear_percentile(raw_latency, 0.95),
        "min": min(raw_latency),
        "max": max(raw_latency),
    }
    for key, expected in expected_latency.items():
        actual = _number(latency.get(key), f"{context} latency {key}")
        if not _close(actual, expected):
            raise CollectionError(
                f"{context}: latency {key}={actual} does not match raw {expected}"
            )

    result_ids = _sequence(point.get("result_ids"), f"{context} result IDs")
    if len(result_ids) != query_count:
        raise CollectionError(f"{context}: result ID row count differs")
    recall_sums = {cutoff: 0 for cutoff in report_ks}
    for local_index, labels_value in enumerate(result_ids):
        labels = _sequence(labels_value, f"{context} result IDs row {local_index}")
        if len(labels) != top_k or any(
            not isinstance(label, int)
            or isinstance(label, bool)
            or label < 0
            or label >= dataset["corpus_count"]
            for label in labels
        ):
            raise CollectionError(
                f"{context}: result row {local_index} must contain exactly "
                f"{top_k} in-range integer IDs"
            )
        if len(set(labels)) != len(labels):
            raise CollectionError(
                f"{context}: result row {local_index} contains duplicate IDs"
            )
        truth_row = truth[query_indices[local_index]]
        for cutoff in report_ks:
            recall_sums[cutoff] += len(
                set(labels[:cutoff])
                & {int(value) for value in truth_row[:cutoff]}
            )
    recalls: dict[str, list[float]] = {}
    for cutoff in report_ks:
        expected = recall_sums[cutoff] / (query_count * cutoff)
        actual = _number(point.get(f"recall_at_{cutoff}"), f"{context} recall")
        if not _close(actual, expected):
            raise CollectionError(
                f"{context}: Recall@{cutoff}={actual} does not match IDs/truth "
                f"{expected}"
            )
        recalls[f"recall_at_{cutoff}"] = [actual] * query_count

    total = _integer(
        point.get("candidate_embeddings_total"),
        f"{context} candidate embeddings",
        minimum=1,
    )
    candidate_mean = _number(
        point.get("candidate_embeddings_mean_per_query"),
        f"{context} candidate embedding mean",
    )
    if not _close(candidate_mean, total / query_count):
        raise CollectionError(f"{context}: candidate embedding mean differs")
    proxy = _mapping(
        point.get("embedding_proxy_metrics"), f"{context} endpoint metrics"
    )
    expected_endpoint_mode = "cache" if mode == "cached" else "proxy"
    if proxy.get("mode") != expected_endpoint_mode:
        raise CollectionError(
            f"{context}: endpoint mode {proxy.get('mode')!r} is not "
            f"{expected_endpoint_mode!r}"
        )
    if proxy.get("inputs") != total:
        raise CollectionError(f"{context}: endpoint input count differs")
    requests = _integer(
        point.get("candidate_embedding_requests"),
        f"{context} candidate embedding requests",
        minimum=1,
    )
    if proxy.get("requests") != requests:
        raise CollectionError(f"{context}: endpoint request count differs")
    if int(proxy.get("http_errors", 0)) != 0 or int(
        proxy.get("network_errors", 0)
    ) != 0:
        raise CollectionError(f"{context}: endpoint reported recompute errors")
    if mode == "cached":
        identity = _mapping(
            proxy.get("cache_identity"), f"{context} cached endpoint identity"
        )
        expected_identity = {
            "cache_sha256": dataset["corpus_cache_sha256"],
            "source_sha256": dataset["documents_sha256"],
            "fingerprint": dataset["fingerprint"],
            "dimensions": dataset["dimensions"],
            "count": dataset["corpus_count"],
            "model_sha256": dataset["model_sha256"],
        }
        for key, expected in expected_identity.items():
            if identity.get(key) != expected:
                raise CollectionError(
                    f"{context}: cached endpoint {key} identity differs"
                )
    return {
        **recalls,
        "latency_ms": raw_latency,
        "candidate_embeddings": [candidate_mean] * query_count,
        "result_ids_sha256": canonical_hash(result_ids),
        "complexity": complexity,
        "batch_size": batch_size,
    }


class Collector:
    def __init__(self, *, verify_artifacts: bool = True) -> None:
        self.verifier = FileVerifier(verify_artifacts)
        self.datasets: dict[str, dict[str, Any]] = {}
        self.sources: list[dict[str, Any]] = []
        self.samples: list[dict[str, Any]] = []
        self.build_samples: list[dict[str, Any]] = []
        self.issues: list[dict[str, Any]] = []
        self.warnings: list[dict[str, Any]] = []
        self._truth_cache: dict[str, np.ndarray] = {}
        self._native_binary_hashes: set[str] = set()
        self._official_commits: set[str] = set()
        self._leann_commits: set[str] = set()
        self._tooling_hashes: dict[str, set[str]] = defaultdict(set)
        self._model_hashes: set[str] = set()
        self._query_hashes: set[str] = set()
        self._query_counts: set[int] = set()
        self._dimensions: set[int] = set()
        self._source_context: dict[str, dict[str, str]] = {}
        self._official_runtime_hashes: set[str] = set()
        self._machine_hashes: set[str] = set()
        self._server_context_hashes: set[str] = set()
        self._validated_caches: dict[
            tuple[str, int, str], dict[str, Any]
        ] = {}
        self._source_line_counts: dict[tuple[str, int, str], int] = {}
        self.tier_prefix_identity: dict[str, Any] | None = None

    def issue(self, code: str, tier: str, profile: str, detail: str) -> None:
        self.issues.append(
            {"code": code, "tier": tier, "profile": profile, "detail": detail}
        )

    def _line_count(
        self, record: dict[str, Any], context: str
    ) -> int:
        path = Path(_nonempty_string(record.get("path"), f"{context} path"))
        key = (
            str(path.resolve()),
            _integer(record.get("size_bytes"), f"{context} size"),
            _nonempty_string(record.get("sha256"), f"{context} SHA-256"),
        )
        if key not in self._source_line_counts:
            try:
                self._source_line_counts[key] = count_nonempty_lines(path)
            except (OSError, UnicodeError) as error:
                raise CollectionError(
                    f"{context}: cannot count source lines: {error}"
                ) from error
        return self._source_line_counts[key]

    def _validate_shared_cache(
        self,
        *,
        cache_record: dict[str, Any],
        source_record: dict[str, Any],
        context: str,
    ) -> dict[str, Any]:
        cache_path = Path(
            _nonempty_string(cache_record.get("path"), f"{context} path")
        ).resolve()
        cache_key = (
            str(cache_path),
            _integer(cache_record.get("size_bytes"), f"{context} size"),
            _nonempty_string(
                cache_record.get("sha256"), f"{context} SHA-256"
            ),
        )
        if cache_key not in self._validated_caches:
            try:
                vectors, metadata = read_cache(
                    cache_path,
                    validate_vectors=True,
                    require_unit=True,
                )
                del vectors
            except (OSError, ValueError) as error:
                raise CollectionError(
                    f"{context}: invalid LEANNBC2 cache: {error}"
                ) from error
            if metadata.get("schema") != "LEANNBC2" or metadata.get(
                "version"
            ) != 2:
                raise CollectionError(
                    f"{context}: publication caches must use LEANNBC2"
                )
            self._validated_caches[cache_key] = metadata
        metadata = self._validated_caches[cache_key]
        required_header_fields = (
            "schema",
            "version",
            "count",
            "dimensions",
            "fingerprint",
            "fingerprint_sha256",
            "vector_offset",
            "vector_bytes",
            "size_bytes",
            "source_size_bytes",
            "source_sha256",
        )
        for field in required_header_fields:
            if field not in cache_record:
                raise CollectionError(
                    f"{context}: shared cache metadata omits {field}"
                )
            if cache_record[field] != metadata[field]:
                raise CollectionError(
                    f"{context}: declared {field} differs from parsed "
                    "LEANNBC2 header"
                )
        source_count = self._line_count(
            source_record, f"{context} source"
        )
        declared_source_count = _integer(
            source_record.get("nonempty_lines"),
            f"{context} declared source line count",
            minimum=1,
        )
        if source_count != declared_source_count:
            raise CollectionError(
                f"{context}: source has {source_count} nonempty lines, "
                f"manifest declares {declared_source_count}"
            )
        if metadata["count"] != source_count:
            raise CollectionError(
                f"{context}: cache/source line counts differ"
            )
        if metadata["source_size_bytes"] != source_record["size_bytes"]:
            raise CollectionError(
                f"{context}: LEANNBC2 source byte-size binding differs"
            )
        if metadata["source_sha256"] != source_record["sha256"]:
            raise CollectionError(
                f"{context}: LEANNBC2 source SHA-256 binding differs"
            )
        return metadata

    def _register_shared(
        self, tier: str, shared: dict[str, Any], manifest_path: Path
    ) -> dict[str, Any]:
        for name in (
            "documents",
            "queries",
            "corpus_cache",
            "query_cache",
            "ground_truth",
        ):
            self.verifier.verify(
                shared.get(name), f"{manifest_path} shared {name}"
            )
        model_record = _mapping(
            shared.get("model"), f"{manifest_path} shared model"
        )
        self.verifier.verify(
            model_record.get("artifact"),
            f"{manifest_path} shared model artifact",
        )
        dataset = _dataset_record(tier, shared)
        if (
            tier in DECLARED_TIER_CARDINALITIES
            and dataset["corpus_count"]
            != DECLARED_TIER_CARDINALITIES[tier]
        ):
            raise CollectionError(
                f"{manifest_path}: tier {tier} must contain exactly "
                f"{DECLARED_TIER_CARDINALITIES[tier]:,} documents, got "
                f"{dataset['corpus_count']:,}"
            )
        corpus_metadata = self._validate_shared_cache(
            cache_record=_mapping(
                shared.get("corpus_cache"),
                f"{manifest_path} corpus cache",
            ),
            source_record=_mapping(
                shared.get("documents"), f"{manifest_path} documents"
            ),
            context=f"{manifest_path} corpus cache",
        )
        query_metadata = self._validate_shared_cache(
            cache_record=_mapping(
                shared.get("query_cache"), f"{manifest_path} query cache"
            ),
            source_record=_mapping(
                shared.get("queries"), f"{manifest_path} queries"
            ),
            context=f"{manifest_path} query cache",
        )
        dataset["corpus_cache_header"] = corpus_metadata
        dataset["query_cache_header"] = query_metadata
        existing = self.datasets.get(tier)
        if existing is not None and existing != dataset:
            differing = sorted(
                key for key in dataset if existing.get(key) != dataset.get(key)
            )
            raise CollectionError(
                f"{manifest_path}: shared artifact identity differs within tier "
                f"{tier}: {differing}"
            )
        self.datasets[tier] = dataset
        self._model_hashes.add(dataset["model_sha256"])
        self._query_hashes.add(dataset["queries_sha256"])
        self._query_counts.add(dataset["query_count"])
        self._dimensions.add(dataset["dimensions"])
        return dataset

    def _load_truth(self, dataset: dict[str, Any]) -> np.ndarray:
        path = Path(str(dataset["ground_truth_path"]))
        key = str(path.resolve())
        if key not in self._truth_cache:
            truth, metadata = read_ground_truth(
                path,
                expected_queries=dataset["query_count"],
                expected_k=dataset["ground_truth_top_k"],
                expected_corpus=dataset["corpus_count"],
            )
            if metadata.get("sha256") != dataset["ground_truth_sha256"]:
                raise CollectionError(f"{path}: ground-truth SHA-256 changed")
            self._truth_cache[key] = np.asarray(truth)
        return self._truth_cache[key]

    def _measurement(
        self,
        observation: Any,
        *,
        context: str,
        require_result: bool,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        item = _mapping(observation, context)
        if item.get("exit_code") != 0:
            raise CollectionError(f"{context}: observation exit code is not zero")
        wall = _number(item.get("wall_seconds"), f"{context} wall time")
        if wall < 0:
            raise CollectionError(f"{context}: negative wall time")
        result: dict[str, Any] | None = None
        if "result_file" in item:
            result_path = self.verifier.verify(
                item["result_file"], f"{context} result file"
            )
            loaded = load_json(result_path) if result_path.suffix == ".json" else None
            if loaded is not None:
                result = _mapping(loaded, f"{context} result")
                if "result" in item and item["result"] != result:
                    raise CollectionError(
                        f"{context}: embedded result differs from result file"
                    )
        if result is None and isinstance(item.get("result"), dict):
            result = item["result"]
        if require_result and result is None:
            raise CollectionError(f"{context}: missing structured result")
        for artifact_index, artifact in enumerate(item.get("artifacts", [])):
            self.verifier.verify(
                artifact, f"{context} artifact {artifact_index}"
            )
        if "executable" in item and item["executable"] is not None:
            self.verifier.verify(item["executable"], f"{context} executable")
        return item, result

    def _validate_official_runtime_binding(
        self,
        *,
        stage: dict[str, Any],
        observation: dict[str, Any],
        context: str,
    ) -> dict[str, Any]:
        inputs = _mapping(stage.get("inputs"), f"{context} stage inputs")
        runtime = _mapping(
            inputs.get("official_runtime"), f"{context} official runtime"
        )
        python_record = _mapping(
            inputs.get("python"), f"{context} official Python snapshot"
        )
        python_path = self.verifier.verify(
            python_record, f"{context} official Python snapshot"
        ).resolve()
        for field in ("executable", "launcher", "launcher_resolved"):
            raw_path = _nonempty_string(
                runtime.get(field), f"{context} runtime {field}"
            )
            if Path(raw_path).resolve() != python_path:
                raise CollectionError(
                    f"{context}: runtime {field} differs from the staged Python "
                    "executable"
                )

        modules = _mapping(runtime.get("modules"), f"{context} runtime modules")
        if set(modules) != OFFICIAL_RUNTIME_MODULES:
            raise CollectionError(
                f"{context}: official runtime module set differs from the "
                "attested LEANN/FAISS module set"
            )
        verified_module_files: dict[str, set[tuple[str, str]]] = {}
        for module_name in sorted(OFFICIAL_RUNTIME_MODULES):
            module = _mapping(
                modules.get(module_name), f"{context} runtime module {module_name}"
            )
            module_path = Path(
                _nonempty_string(
                    module.get("path"), f"{context} runtime module {module_name} path"
                )
            ).resolve()
            files = _sequence(
                module.get("files"), f"{context} runtime module {module_name} files"
            )
            if not files:
                raise CollectionError(
                    f"{context}: runtime module {module_name} has no hashed files"
                )
            identities: set[tuple[str, str]] = set()
            for ordinal, file_record in enumerate(files):
                verified = self.verifier.verify(
                    file_record,
                    f"{context} runtime module {module_name} file {ordinal}",
                ).resolve()
                record = _mapping(
                    file_record,
                    f"{context} runtime module {module_name} file {ordinal}",
                )
                identities.add(
                    (
                        str(verified),
                        _nonempty_string(
                            record.get("sha256"),
                            f"{context} runtime module {module_name} file SHA-256",
                        ),
                    )
                )
            if not any(path == str(module_path) for path, _ in identities):
                raise CollectionError(
                    f"{context}: runtime module {module_name} path is not among "
                    "its hashed files"
                )
            verified_module_files[module_name] = identities

        backend = _mapping(
            runtime.get("backend_faiss"), f"{context} runtime FAISS backend"
        )
        backend_path = self.verifier.verify(
            backend, f"{context} runtime FAISS backend"
        ).resolve()
        backend_identity = (
            str(backend_path),
            _nonempty_string(
                backend.get("sha256"), f"{context} runtime FAISS backend SHA-256"
            ),
        )
        if backend_identity not in verified_module_files[
            "leann_backend_hnsw.faiss"
        ]:
            raise CollectionError(
                f"{context}: runtime FAISS backend differs from the imported "
                "LEANN backend module"
            )

        command = _sequence(observation.get("command"), f"{context} command")
        if not command or not all(isinstance(value, str) for value in command):
            raise CollectionError(f"{context}: command is not an argv string array")
        if Path(command[0]).resolve() != python_path:
            raise CollectionError(
                f"{context}: observed command did not use the staged Python executable"
            )
        command_template = _sequence(
            inputs.get("command_template"), f"{context} command template"
        )
        if not command_template or not isinstance(command_template[0], str):
            raise CollectionError(f"{context}: command template is invalid")
        if Path(command_template[0]).resolve() != python_path:
            raise CollectionError(
                f"{context}: planned command did not use the staged Python executable"
            )
        executable = _mapping(
            observation.get("executable"), f"{context} observed executable"
        )
        executable_path = self.verifier.verify(
            executable, f"{context} observed executable"
        ).resolve()
        if (
            executable_path != python_path
            or executable.get("sha256") != python_record.get("sha256")
        ):
            raise CollectionError(
                f"{context}: observed executable differs from the staged Python "
                "snapshot"
            )
        return runtime

    def _native_build(
        self,
        tier: str,
        source_sha: str,
        stage: dict[str, Any],
        dataset: dict[str, Any],
        context: str,
    ) -> None:
        measurements = _sequence(stage.get("measurements"), f"{context} measurements")
        for ordinal, value in enumerate(measurements):
            observation, _ = self._measurement(
                value, context=f"{context} measurement {ordinal}", require_result=False
            )
            self.build_samples.append(
                {
                    "tier": tier,
                    "system": "leann.cpp",
                    "source_manifest_sha256": source_sha,
                    **self._source_context[source_sha],
                    "wall_seconds": observation["wall_seconds"],
                    "peak_rss_bytes": observation.get("peak_rss_bytes"),
                    "algorithm_seconds": None,
                    "storage": None,
                    "profile": "native",
                    "config": {
                        "graph_degree": _argv_value(
                            observation.get("command"),
                            "--graph-degree",
                            f"{context} measurement {ordinal}",
                        ),
                        "ef_construction": _argv_value(
                            observation.get("command"),
                            "--ef-construction",
                            f"{context} measurement {ordinal}",
                        ),
                    },
                    "evidence": observation.get("artifacts", []),
                    "index_artifact_set": _artifact_set(
                        observation.get("artifacts"),
                        f"{context} measurement {ordinal} index artifacts",
                    ),
                }
            )
        if not measurements:
            self.issue("no-measurements", tier, "native", f"{context} has no measurement")

    def _native_stats(
        self,
        tier: str,
        source_sha: str,
        stage: dict[str, Any],
        dataset: dict[str, Any],
        context: str,
    ) -> None:
        measurements = _sequence(stage.get("measurements"), f"{context} measurements")
        if not measurements:
            self.issue("no-measurements", tier, "native", f"{context} has no measurement")
            return
        stats_observation, _ = self._measurement(
            measurements[-1],
            context=f"{context} measurement {len(measurements) - 1}",
            require_result=False,
        )
        parsed = _mapping(
            stats_observation.get("parsed_stdout"), f"{context} parsed stdout"
        )
        index_bytes = _integer(
            parsed.get("index_bytes"), f"{context} index bytes", minimum=1
        )
        if parsed.get("nodes") != dataset["corpus_count"]:
            raise CollectionError(f"{context}: stats node count differs")
        if parsed.get("dimension") != dataset["dimensions"]:
            raise CollectionError(f"{context}: stats dimensions differ")
        if parsed.get("dense_vector_bytes_avoided") != dataset["dense_fp32_bytes"]:
            raise CollectionError(f"{context}: dense byte baseline differs")
        artifacts = _sequence(
            stats_observation.get("artifacts"), f"{context} artifacts"
        )
        stats_artifact_set = _artifact_set(
            artifacts, f"{context} stats index artifacts"
        )
        index_artifacts = [
            artifact
            for artifact in artifacts
            if str(_mapping(artifact, context).get("path", "")).endswith(".leann")
        ]
        text_artifacts = [
            artifact
            for artifact in artifacts
            if str(_mapping(artifact, context).get("path", "")).endswith(".docs")
        ]
        if len(index_artifacts) != 1 or len(text_artifacts) != 1:
            raise CollectionError(
                f"{context}: expected one .leann and one .docs artifact"
            )
        if index_artifacts[0].get("size_bytes") != index_bytes:
            raise CollectionError(f"{context}: index artifact size differs")
        text_bytes = _integer(
            text_artifacts[0].get("size_bytes"), f"{context} text store bytes"
        )
        build_samples = [
            sample
            for sample in self.build_samples
            if sample["tier"] == tier
            and sample["system"] == "leann.cpp"
            and sample["source_manifest_sha256"] == source_sha
        ]
        if not build_samples:
            raise CollectionError(
                f"{context}: native stats has no build observation to bind"
            )
        mismatched = [
            sample
            for sample in build_samples
            if sample["index_artifact_set"]["sha256"]
            != stats_artifact_set["sha256"]
        ]
        if mismatched:
            raise CollectionError(
                f"{context}: native stats artifact set differs from its build "
                "observation artifact set"
            )
        for sample in build_samples:
            sample["storage"] = {
                "vector_index_bytes": index_bytes,
                "lookup_aux_bytes": 0,
                "vector_serving_bytes": index_bytes,
                "text_store_bytes": text_bytes,
                "total_bytes": index_bytes + text_bytes,
                "dense_fp32_bytes_omitted": dataset["dense_fp32_bytes"],
                "raw_text_file_bytes": dataset["raw_text_file_bytes"],
                "definition": (
                    "vector_serving_bytes is the native .leann graph/PQ/metadata "
                    "artifact; the .docs text store is reported separately"
                ),
            }
            sample.setdefault("config", {})["approximation"] = parsed.get(
                "approximation"
            )

    def _native_search(
        self,
        tier: str,
        source_sha: str,
        stage_name: str,
        stage: dict[str, Any],
        dataset: dict[str, Any],
        recall_cutoffs: Sequence[int],
        context: str,
    ) -> None:
        matched = re.fullmatch(r"native-search-ef(\d+)", stage_name)
        if matched is None:
            raise CollectionError(f"{context}: invalid native search stage name")
        ef_search = int(matched.group(1))
        if stage.get("inputs", {}).get("ef_search") != ef_search:
            raise CollectionError(f"{context}: ef-search input differs from stage name")
        truth = self._load_truth(dataset)
        measurements = _sequence(stage.get("measurements"), f"{context} measurements")
        for ordinal, value in enumerate(measurements):
            measurement_context = f"{context} measurement {ordinal}"
            observation, _ = self._measurement(
                value, context=measurement_context, require_result=False
            )
            if "result_file" not in observation:
                raise CollectionError(f"{measurement_context}: raw CSV snapshot missing")
            csv_path = self.verifier.verify(
                observation["result_file"], f"{measurement_context} raw CSV"
            )
            parsed = _mapping(
                observation.get("parsed_stdout"),
                f"{measurement_context} parsed stdout",
            )
            metrics = _read_native_csv(
                csv_path,
                query_count=dataset["query_count"],
                corpus_count=dataset["corpus_count"],
                recall_cutoffs=recall_cutoffs,
                top_k=dataset["ground_truth_top_k"],
                truth=truth,
                parsed=parsed,
                context=measurement_context,
            )
            command = observation.get("command")
            command_ef = _argv_value(command, "--ef-search", measurement_context)
            try:
                command_ef_value = (
                    int(command_ef) if command_ef is not None else None
                )
            except ValueError as error:
                raise CollectionError(
                    f"{measurement_context}: command ef-search is not an integer"
                ) from error
            if command_ef_value != ef_search:
                raise CollectionError(
                    f"{measurement_context}: command ef-search differs from stage"
                )
            batch_value = _argv_value(command, "--recompute-batch", measurement_context)
            if batch_value is None:
                raise CollectionError(f"{measurement_context}: missing recompute batch")
            try:
                batch_size = int(batch_value)
            except ValueError as error:
                raise CollectionError(
                    f"{measurement_context}: invalid recompute batch"
                ) from error
            sample = {
                "tier": tier,
                "corpus_count": dataset["corpus_count"],
                "system": "leann.cpp",
                "recompute_mode": "real",
                "latency_comparable": True,
                "search_parameter": "ef_search",
                "search_value": ef_search,
                "batch_size": batch_size,
                "queries": dataset["query_count"],
                "source_manifest_sha256": source_sha,
                **self._source_context[source_sha],
                "evidence": _snapshot_identity(observation["result_file"]),
                "index_artifact_set": _artifact_set(
                    observation.get("artifacts"),
                    f"{measurement_context} index artifacts",
                ),
                **metrics,
            }
            sample["candidate_embeddings"] = metrics["exact_recomputations"]
            self.samples.append(sample)

    def _official_build(
        self,
        tier: str,
        profile: str,
        source_sha: str,
        stage: dict[str, Any],
        dataset: dict[str, Any],
        context: str,
    ) -> None:
        measurements = _sequence(stage.get("measurements"), f"{context} measurements")
        for ordinal, value in enumerate(measurements):
            observation, result = self._measurement(
                value,
                context=f"{context} measurement {ordinal}",
                require_result=True,
            )
            assert result is not None
            expected_runtime = self._validate_official_runtime_binding(
                stage=stage,
                observation=observation,
                context=f"{context} measurement {ordinal}",
            )
            self._validate_official_report_identity(
                result,
                dataset=dataset,
                expected_runtime=expected_runtime,
                context=f"{context} measurement {ordinal}",
            )
            build = _mapping(result.get("build"), f"{context} build result")
            storage = _mapping(build.get("storage"), f"{context} storage")
            files = _mapping(storage.get("files"), f"{context} storage files")
            normalized_files: dict[str, int] = {}
            for name, value in files.items():
                if not isinstance(name, str) or not name:
                    raise CollectionError(
                        f"{context}: official storage has an invalid file name"
                    )
                normalized_files[name] = _integer(
                    value, f"{context} storage file {name}"
                )
            artifact_records = [
                _mapping(item, f"{context} build artifact")
                for item in _sequence(
                    observation.get("artifacts"),
                    f"{context} build artifacts",
                )
            ]
            artifact_files: dict[str, int] = {}
            for artifact in artifact_records:
                name = Path(
                    _nonempty_string(
                        artifact.get("path"), f"{context} artifact path"
                    )
                ).name
                if name in artifact_files:
                    raise CollectionError(
                        f"{context}: duplicate official artifact basename {name}"
                    )
                artifact_files[name] = _integer(
                    artifact.get("size_bytes"),
                    f"{context} artifact {name} size",
                )
            if normalized_files != artifact_files:
                raise CollectionError(
                    f"{context}: official storage file entries do not exactly "
                    "match verified build artifacts"
                )
            vector_index_files = {
                name: size
                for name, size in normalized_files.items()
                if name.endswith(".index")
            }
            lookup_files = {
                name: size
                for name, size in normalized_files.items()
                if name.endswith(
                    (".ids.txt", ".passages.idx", ".meta.json")
                )
            }
            text_files = {
                name: size
                for name, size in normalized_files.items()
                if name.endswith(".passages.jsonl")
            }
            classified = (
                set(vector_index_files) | set(lookup_files) | set(text_files)
            )
            if (
                len(vector_index_files) != 1
                or len(text_files) != 1
                or classified != set(normalized_files)
            ):
                raise CollectionError(
                    f"{context}: official storage artifacts cannot be "
                    "unambiguously classified"
                )
            derived_vector_index = sum(vector_index_files.values())
            derived_lookup = sum(lookup_files.values())
            derived_text = sum(text_files.values())
            total = sum(normalized_files.values())
            for key in (
                "vector_index_bytes",
                "lookup_aux_bytes",
                "vector_serving_bytes",
                "text_store_bytes",
                "total_bytes",
            ):
                _integer(storage.get(key), f"{context} storage {key}")
            if storage["total_bytes"] != total:
                raise CollectionError(f"{context}: official storage total differs")
            expected_semantics = {
                "vector_index_bytes": derived_vector_index,
                "lookup_aux_bytes": derived_lookup,
                "vector_serving_bytes": (
                    derived_vector_index + derived_lookup
                ),
                "text_store_bytes": derived_text,
                "total_bytes": total,
            }
            if any(
                storage[key] != value
                for key, value in expected_semantics.items()
            ):
                raise CollectionError(
                    f"{context}: official storage semantic totals do not "
                    "match verified artifacts"
                )
            storage_identity_sha256 = canonical_hash(
                {"files": normalized_files, **expected_semantics}
            )
            normalized_storage = {
                **{key: storage[key] for key in (
                    "vector_index_bytes",
                    "lookup_aux_bytes",
                    "vector_serving_bytes",
                    "text_store_bytes",
                    "total_bytes",
                )},
                "dense_fp32_bytes_omitted": dataset["dense_fp32_bytes"],
                "raw_text_file_bytes": dataset["raw_text_file_bytes"],
                "definition": (
                    "vector_serving_bytes is official LEANN's vector index plus "
                    "ID/offset/metadata lookup auxiliaries; passages are separate"
                ),
            }
            self.build_samples.append(
                {
                    "tier": tier,
                    "system": "official LEANN",
                    "source_manifest_sha256": source_sha,
                    **self._source_context[source_sha],
                    "wall_seconds": observation["wall_seconds"],
                    "peak_rss_bytes": observation.get("peak_rss_bytes"),
                    "algorithm_seconds": build.get(
                        "elapsed_seconds_excluding_embedding"
                    ),
                    "storage": normalized_storage,
                    "storage_identity_sha256": storage_identity_sha256,
                    "config": result["official"].get("config"),
                    "evidence": _snapshot_identity(observation["result_file"]),
                    "profile": profile,
                    "index_artifact_set": _artifact_set(
                        observation.get("artifacts"),
                        f"{context} measurement {ordinal} index artifacts",
                    ),
                }
            )

    def _validate_official_report_identity(
        self,
        result: dict[str, Any],
        *,
        dataset: dict[str, Any],
        expected_runtime: dict[str, Any],
        context: str,
    ) -> None:
        if result.get("schema") != OFFICIAL_SCHEMA:
            raise CollectionError(f"{context}: unknown official result schema")
        official = _mapping(result.get("official"), f"{context} official provenance")
        report_runtime = _mapping(
            official.get("runtime"), f"{context} official report runtime"
        )
        expected_execution_runtime = {
            key: value
            for key, value in expected_runtime.items()
            if key not in {"launcher", "launcher_resolved"}
        }
        if report_runtime != expected_execution_runtime:
            raise CollectionError(
                f"{context}: official report runtime differs from the staged runtime"
            )
        commit = official.get("commit")
        if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise CollectionError(f"{context}: invalid official commit")
        self._official_commits.add(commit)
        result_dataset = _mapping(result.get("dataset"), f"{context} dataset")
        if result_dataset.get("documents") != dataset["corpus_count"]:
            raise CollectionError(f"{context}: official corpus count differs")
        if result_dataset.get("queries") != dataset["query_count"]:
            raise CollectionError(f"{context}: official query count differs")
        cache = _mapping(result.get("embedding_cache"), f"{context} embedding cache")
        if cache.get("sha256") != dataset["corpus_cache_sha256"]:
            raise CollectionError(f"{context}: official corpus cache SHA differs")

    def _official_search(
        self,
        tier: str,
        mode: str,
        source_sha: str,
        stage: dict[str, Any],
        dataset: dict[str, Any],
        recall_cutoffs: Sequence[int],
        context: str,
    ) -> None:
        measurements = _sequence(stage.get("measurements"), f"{context} measurements")
        truth = self._load_truth(dataset)
        command_template = stage.get("inputs", {}).get("command_template")
        expected_complexity_values = _argv_values(
            command_template, "--complexities", context
        )
        expected_batch_values = _argv_values(
            command_template, "--batch-sizes", context
        )
        try:
            expected_points = (
                {
                    (int(complexity), int(batch))
                    for batch in expected_batch_values
                    for complexity in expected_complexity_values
                }
                if expected_complexity_values is not None
                and expected_batch_values is not None
                else None
            )
        except ValueError as error:
            raise CollectionError(
                f"{context}: official command has non-integer sweep values"
            ) from error
        for ordinal, value in enumerate(measurements):
            measurement_context = f"{context} measurement {ordinal}"
            observation, result = self._measurement(
                value, context=measurement_context, require_result=True
            )
            assert result is not None
            expected_runtime = self._validate_official_runtime_binding(
                stage=stage,
                observation=observation,
                context=measurement_context,
            )
            self._validate_official_report_identity(
                result,
                dataset=dataset,
                expected_runtime=expected_runtime,
                context=measurement_context,
            )
            benchmark = _mapping(
                result.get("benchmark"), f"{measurement_context} benchmark"
            )
            if benchmark.get("candidate_recompute_mode") != mode:
                raise CollectionError(
                    f"{measurement_context}: recompute mode differs from stage"
                )
            query_count = _integer(
                benchmark.get("query_count"),
                f"{measurement_context} query count",
                minimum=1,
            )
            if query_count != dataset["query_count"]:
                raise CollectionError(
                    f"{measurement_context}: official query count is not complete"
                )
            if (
                benchmark.get("query_embedding_cache_sha256")
                != dataset["query_cache_sha256"]
            ):
                raise CollectionError(
                    f"{measurement_context}: official query cache SHA differs"
                )
            if (
                benchmark.get("corpus_embedding_cache_sha256")
                != dataset["corpus_cache_sha256"]
            ):
                raise CollectionError(
                    f"{measurement_context}: official corpus cache SHA differs"
                )
            truth_description = benchmark.get("truth")
            if (
                isinstance(truth_description, dict)
                and truth_description.get("sha256")
                != dataset["ground_truth_sha256"]
            ):
                raise CollectionError(
                    f"{measurement_context}: official ground-truth SHA differs"
                )
            query_indices = _sequence(
                benchmark.get("query_indices"), f"{measurement_context} query indices"
            )
            if query_indices != list(range(dataset["query_count"])):
                raise CollectionError(
                    f"{measurement_context}: official queries are not exactly "
                    "the complete ordered query set"
                )
            points = _sequence(
                benchmark.get("points"), f"{measurement_context} points"
            )
            seen: set[tuple[int, int]] = set()
            for point_index, raw_point in enumerate(points):
                point = _mapping(
                    raw_point, f"{measurement_context} point {point_index}"
                )
                metrics = _validate_official_point(
                    point,
                    mode=mode,
                    dataset=dataset,
                    query_count=query_count,
                    report_ks=recall_cutoffs,
                    truth=truth,
                    query_indices=query_indices,
                    top_k=dataset["ground_truth_top_k"],
                    context=f"{measurement_context} point {point_index}",
                )
                key = (metrics["complexity"], metrics["batch_size"])
                if key in seen:
                    raise CollectionError(
                        f"{measurement_context}: duplicate official point {key}"
                    )
                seen.add(key)
                sample = {
                    "tier": tier,
                    "corpus_count": dataset["corpus_count"],
                    "system": "official LEANN",
                    "recompute_mode": mode,
                    "latency_comparable": mode == "real",
                    "search_parameter": "complexity",
                    "search_value": metrics.pop("complexity"),
                    "batch_size": metrics.pop("batch_size"),
                    "queries": query_count,
                    "source_manifest_sha256": source_sha,
                    **self._source_context[source_sha],
                    "evidence": _snapshot_identity(observation["result_file"]),
                    "index_artifact_set": _artifact_set(
                        observation.get("artifacts"),
                        f"{measurement_context} index artifacts",
                    ),
                    **metrics,
                }
                self.samples.append(sample)
            if expected_points is not None and seen != expected_points:
                raise CollectionError(
                    f"{measurement_context}: official result point set {sorted(seen)} "
                    f"differs from command sweep {sorted(expected_points)}"
                )

    def ingest(self, tier: str, manifest_path: Path) -> None:
        manifest_path = manifest_path.resolve()
        manifest = _mapping(load_json(manifest_path), f"{manifest_path}")
        if manifest.get("schema") != MANIFEST_SCHEMA:
            raise CollectionError(f"{manifest_path}: unknown manifest schema")
        source_sha = sha256_file(manifest_path)
        shared = _mapping(
            manifest.get("shared_artifacts"), f"{manifest_path} shared artifacts"
        )
        dataset = self._register_shared(tier, shared, manifest_path)
        protocol = _mapping(manifest.get("protocol"), f"{manifest_path} protocol")
        planned = _sequence(
            manifest.get("planned_stages"), f"{manifest_path} planned stages"
        )
        if not all(isinstance(name, str) for name in planned):
            raise CollectionError(f"{manifest_path}: planned stage names must be strings")
        if len(set(planned)) != len(planned):
            raise CollectionError(
                f"{manifest_path}: planned stage names contain duplicates"
            )
        unrecognized_planned = sorted(
            name for name in planned if not _recognized_stage_name(name)
        )
        if unrecognized_planned:
            raise CollectionError(
                f"{manifest_path}: unrecognized planned stages "
                f"{unrecognized_planned}"
            )
        recall_cutoffs = sorted(
            {
                _integer(value, f"{manifest_path} recall cutoff", minimum=1)
                for value in _sequence(
                    protocol.get("recall_cutoffs"),
                    f"{manifest_path} recall cutoffs",
                )
            }
        )
        if not recall_cutoffs:
            raise CollectionError(f"{manifest_path}: no recall cutoffs")
        if any(value > dataset["ground_truth_top_k"] for value in recall_cutoffs):
            raise CollectionError(f"{manifest_path}: recall cutoff exceeds truth width")
        provenance = _mapping(
            manifest.get("provenance"), f"{manifest_path} provenance"
        )
        leann_provenance = _mapping(
            provenance.get("leann_cpp"), f"{manifest_path} leann.cpp provenance"
        )
        official_provenance = _mapping(
            provenance.get("official_leann"),
            f"{manifest_path} official provenance",
        )
        leann_commit = leann_provenance.get("commit")
        official_commit = official_provenance.get("commit")
        if leann_commit is not None:
            if not isinstance(leann_commit, str) or not re.fullmatch(
                r"[0-9a-f]{40}", leann_commit
            ):
                raise CollectionError(f"{manifest_path}: invalid leann.cpp commit")
            self._leann_commits.add(leann_commit)
        if official_commit is not None:
            if not isinstance(official_commit, str) or not re.fullmatch(
                r"[0-9a-f]{40}", official_commit
            ):
                raise CollectionError(
                    f"{manifest_path}: invalid official LEANN commit"
                )
            self._official_commits.add(official_commit)
        tooling = provenance.get("tooling")
        tooling_by_name: dict[str, dict[str, Any]] = {}
        if tooling is not None:
            for ordinal, item in enumerate(
                _sequence(tooling, f"{manifest_path} tooling")
            ):
                path = self.verifier.verify(
                    item, f"{manifest_path} tooling {ordinal}"
                )
                self._tooling_hashes[path.name].add(item["sha256"])
                tooling_by_name[path.name] = item
        environment = _mapping(
            provenance.get("environment"), f"{manifest_path} environment"
        )
        machine_identity = {
            key: environment.get(key)
            for key in ("platform", "machine", "processor")
        }
        if not all(
            isinstance(value, str) for value in machine_identity.values()
        ):
            raise CollectionError(f"{manifest_path}: incomplete machine identity")
        machine_hash = canonical_hash(machine_identity)
        self._machine_hashes.add(machine_hash)
        server_context = _mapping(
            provenance.get("server_context"), f"{manifest_path} server context"
        )
        server_context_hash = canonical_hash(server_context)
        self._server_context_hashes.add(server_context_hash)
        official_attestation: dict[str, Any] | None = None
        if any(name.startswith("official-") for name in planned):
            official_attestation = _validate_recorded_official_attestation(
                provenance=provenance,
                shared=shared,
                dataset=dataset,
                tooling_by_name=tooling_by_name,
                context=str(manifest_path),
            )
            self._official_runtime_hashes.add(
                official_attestation["official_runtime_sha256"]
            )
        run_role = _validate_declared_run_role(
            manifest.get("run_role"), planned, str(manifest_path)
        )
        if protocol.get("run_role") != run_role:
            raise CollectionError(
                f"{manifest_path}: protocol run role differs from manifest run role"
            )
        reuse_attestation: dict[str, Any] | None = None
        raw_reuse = provenance.get("official_index_reuse")
        if raw_reuse is not None:
            reuse_attestation = _mapping(
                raw_reuse, f"{manifest_path} official index reuse"
            )
            if reuse_attestation.get("schema") != REUSE_SCHEMA:
                raise CollectionError(
                    f"{manifest_path}: unknown official index reuse schema"
                )
            if run_role != "official-real-search-reuse":
                raise CollectionError(
                    f"{manifest_path}: index reuse attestation is attached to "
                    f"unexpected run role {run_role!r}"
                )
            source_manifest_snapshot = _mapping(
                reuse_attestation.get("source_manifest"),
                f"{manifest_path} reuse source manifest",
            )
            self.verifier.verify(
                source_manifest_snapshot,
                f"{manifest_path} reuse source manifest",
            )
            self.verifier.verify(
                reuse_attestation.get("source_stage"),
                f"{manifest_path} reuse source stage",
            )
            reuse_artifacts = _sequence(
                reuse_attestation.get("artifacts"),
                f"{manifest_path} reuse artifacts",
            )
            for ordinal, artifact in enumerate(reuse_artifacts):
                self.verifier.verify(
                    artifact,
                    f"{manifest_path} reuse artifact {ordinal}",
                )
            reuse_artifact_set = _artifact_set(
                reuse_artifacts, f"{manifest_path} reuse artifacts"
            )
            if (
                reuse_attestation.get("artifact_set_sha256")
                != reuse_artifact_set["sha256"]
            ):
                raise CollectionError(
                    f"{manifest_path}: reused index artifact identity differs"
                )
            storage_identity = reuse_attestation.get(
                "storage_identity_sha256"
            )
            if not isinstance(storage_identity, str) or not re.fullmatch(
                r"[0-9a-f]{64}", storage_identity
            ):
                raise CollectionError(
                    f"{manifest_path}: reused storage identity is invalid"
                )
        elif run_role == "official-real-search-reuse":
            raise CollectionError(
                f"{manifest_path}: search-only official real run has no exact "
                "cached-build reuse attestation"
            )
        protocol_identity_hash = canonical_hash(
            {
                "run_role": run_role,
                "protocol": protocol,
                "planned_stages": sorted(planned),
                "server_context": provenance.get("server_context"),
                "candidate_recompute_mode": provenance.get(
                    "official_candidate_recompute_mode"
                ),
            }
        )
        self._source_context[source_sha] = {
            "run_role": run_role,
            "protocol_identity_hash": protocol_identity_hash,
            **(
                {
                    "reuse_build_source_manifest_sha256": (
                        reuse_attestation["source_manifest"]["sha256"]
                    ),
                    "reuse_artifact_set_sha256": reuse_attestation[
                        "artifact_set_sha256"
                    ],
                    "reuse_storage_identity_sha256": reuse_attestation[
                        "storage_identity_sha256"
                    ],
                }
                if reuse_attestation is not None
                else {}
            ),
        }
        source_record = {
            "tier": tier,
            "path": str(manifest_path),
            "sha256": source_sha,
            "status": manifest.get("status"),
            "created_at": manifest.get("created_at"),
            "completed_at": manifest.get("completed_at"),
            "leann_cpp": leann_provenance,
            "leann_cpp_commit": leann_commit,
            "official_leann": official_provenance,
            "official_leann_commit": official_commit,
            "tooling": tooling,
            "environment": environment,
            "machine_identity_sha256": machine_hash,
            "server_context": server_context,
            "server_context_sha256": server_context_hash,
            "official_runtime": provenance.get("official_runtime"),
            "endpoint_attestation": official_attestation,
            "embedding_endpoint_observation": provenance.get(
                "embedding_endpoint_observation"
            ),
            "candidate_endpoint_observation": provenance.get(
                "candidate_endpoint_observation"
            ),
            "official_index_reuse": reuse_attestation,
            "run_role": run_role,
            "protocol_identity_hash": protocol_identity_hash,
            "protocol": protocol,
            "stages": [],
        }
        self.sources.append(source_record)
        if manifest.get("status") != "complete":
            self.issue(
                "manifest-incomplete",
                tier,
                "all",
                f"{manifest_path} status is {manifest.get('status')!r}",
            )

        def stage_profile(name: str) -> str:
            if name.startswith("native-"):
                return "native"
            if name.startswith("official-search-"):
                return f"official-{name.rsplit('-', 1)[-1]}"
            if name == "official-build":
                mode = provenance.get("official_candidate_recompute_mode")
                return f"official-{mode}" if mode in ("cached", "real") else "official"
            return "official"

        stage_mapping = _mapping(manifest.get("stages", {}), f"{manifest_path} stages")
        unrecognized_recorded = sorted(
            str(name)
            for name in stage_mapping
            if not isinstance(name, str) or not _recognized_stage_name(name)
        )
        if unrecognized_recorded:
            raise CollectionError(
                f"{manifest_path}: unrecognized recorded stages "
                f"{unrecognized_recorded}"
            )
        if set(stage_mapping) != set(planned):
            raise CollectionError(
                f"{manifest_path}: planned/recorded stage sets differ; "
                f"missing={sorted(set(planned) - set(stage_mapping))}, "
                f"extra={sorted(set(stage_mapping) - set(planned))}"
            )
        for stage_name in sorted(stage_mapping):
            if not isinstance(stage_name, str):
                raise CollectionError(f"{manifest_path}: non-string stage name")
            stage_path = _resolve_stage_path(
                manifest_path, stage_name, stage_mapping[stage_name]
            )
            if not stage_path.is_file():
                profile = stage_profile(stage_name)
                self.issue(
                    "missing-stage-file",
                    tier,
                    profile,
                    f"{stage_path} does not exist",
                )
                continue
            stage = _mapping(load_json(stage_path), f"{stage_path}")
            source_record["stages"].append(
                {
                    "name": stage_name,
                    "path": str(stage_path),
                    "sha256": sha256_file(stage_path),
                    "status": stage.get("status"),
                }
            )
            if stage.get("schema") != STAGE_SCHEMA or stage.get("name") != stage_name:
                raise CollectionError(f"{stage_path}: stage schema/name mismatch")
            if stage.get("status") != "complete":
                profile = stage_profile(stage_name)
                self.issue(
                    "stage-incomplete",
                    tier,
                    profile,
                    f"{stage_path} status is {stage.get('status')!r}",
                )
            inputs = _mapping(stage.get("inputs"), f"{stage_path} inputs")
            if stage.get("input_hash") != canonical_hash(inputs):
                raise CollectionError(f"{stage_path}: stage input hash differs")
            if inputs.get("shared") != shared:
                raise CollectionError(
                    f"{stage_path}: stage shared artifacts differ from manifest"
                )
            if "tooling" in inputs and inputs["tooling"] != tooling:
                raise CollectionError(
                    f"{stage_path}: stage tooling differs from manifest provenance"
                )
            if stage_name.startswith("official-") and (
                inputs.get("official_runtime")
                != provenance.get("official_runtime")
            ):
                raise CollectionError(
                    f"{stage_path}: official runtime differs from manifest provenance"
                )
            if (
                stage_name == "official-search-real"
                and reuse_attestation is not None
                and inputs.get("index_reuse") != reuse_attestation
            ):
                raise CollectionError(
                    f"{stage_path}: search stage reuse binding differs from "
                    "manifest provenance"
                )
            if (
                run_role == "native-gate"
                and stage_name.startswith("native-")
            ):
                # Gate runs are smoke tests, not publication measurements. Keep
                # the source/stage identity in the audit trail, but do not parse
                # or pool its raw results or binary identity.
                continue
            if stage_name.startswith("native-"):
                binary_hash = inputs.get("binary_sha256")
                if not isinstance(binary_hash, str) or not re.fullmatch(
                    r"[0-9a-f]{64}", binary_hash
                ):
                    raise CollectionError(f"{stage_path}: missing native binary SHA")
                self._native_binary_hashes.add(binary_hash)
            measurement_values = _sequence(
                stage.get("measurements"), f"{stage_path} measurements"
            )
            warmup_values = _sequence(
                stage.get("warmups"), f"{stage_path} warmups"
            )
            measurement_count = len(measurement_values)
            warmup_count = len(warmup_values)
            if stage_name in ("native-build", "official-build"):
                expected_measurements = _integer(
                    protocol.get("build_repetitions"),
                    f"{manifest_path} build repetitions",
                    minimum=1,
                )
                expected_warmups = _integer(
                    protocol.get("build_warmups"),
                    f"{manifest_path} build warmups",
                )
            elif stage_name == "native-stats":
                expected_measurements = 1
                expected_warmups = 0
            else:
                expected_measurements = _integer(
                    protocol.get("search_repetitions"),
                    f"{manifest_path} search repetitions",
                    minimum=1,
                )
                expected_warmups = _integer(
                    protocol.get("search_warmups"),
                    f"{manifest_path} search warmups",
                )
            execution_protocol = _mapping(
                inputs.get("execution_protocol"),
                f"{stage_path} execution protocol",
            )
            if execution_protocol != {
                "warmups": expected_warmups,
                "repetitions": expected_measurements,
            }:
                raise CollectionError(
                    f"{stage_path}: stage execution protocol differs from the "
                    "manifest protocol"
                )
            if measurement_count != expected_measurements:
                raise CollectionError(
                    f"{stage_path}: measurement repetition count differs: "
                    f"{measurement_count} != {expected_measurements}"
                )
            if warmup_count != expected_warmups:
                raise CollectionError(
                    f"{stage_path}: warmup count differs: "
                    f"{warmup_count} != {expected_warmups}"
                )
            for kind, observations in (
                ("warmups", warmup_values),
                ("measurements", measurement_values),
            ):
                for ordinal, observation_value in enumerate(observations):
                    observation = _mapping(
                        observation_value,
                        f"{stage_path} {kind} {ordinal}",
                    )
                    if (
                        observation.get("kind") != kind
                        or observation.get("ordinal") != ordinal
                    ):
                        raise CollectionError(
                            f"{stage_path}: {kind} observation {ordinal} has "
                            "a mismatched kind/ordinal"
                        )
                    expected_command = _expected_observation_command(
                        inputs,
                        kind=kind,
                        ordinal=ordinal,
                        context=f"{stage_path} {kind} {ordinal}",
                    )
                    if observation.get("command") != expected_command:
                        raise CollectionError(
                            f"{stage_path}: {kind} observation {ordinal} command "
                            "differs from the stage command template"
                        )
                    if stage_name.startswith("official-"):
                        self._validate_official_runtime_binding(
                            stage=stage,
                            observation=observation,
                            context=f"{stage_path} {kind} {ordinal}",
                        )
            if stage_name.startswith("native-"):
                expected_binary_hash = inputs["binary_sha256"]
                for group_name, observations in (
                    ("warmup", warmup_values),
                    ("measurement", measurement_values),
                ):
                    for ordinal, observation_value in enumerate(observations):
                        observation = _mapping(
                            observation_value,
                            f"{stage_path} {group_name} {ordinal}",
                        )
                        executable = _mapping(
                            observation.get("executable"),
                            f"{stage_path} {group_name} {ordinal} executable",
                        )
                        self.verifier.verify(
                            executable,
                            f"{stage_path} {group_name} {ordinal} executable",
                        )
                        if executable.get("sha256") != expected_binary_hash:
                            raise CollectionError(
                                f"{stage_path}: {group_name} {ordinal} "
                                "executable SHA-256 differs from binary_sha256"
                            )
            context = str(stage_path)
            if stage_name == "native-build":
                self._native_build(
                    tier, source_sha, stage, dataset, context
                )
            elif stage_name == "native-stats":
                self._native_stats(tier, source_sha, stage, dataset, context)
            elif stage_name.startswith("native-search-"):
                self._native_search(
                    tier,
                    source_sha,
                    stage_name,
                    stage,
                    dataset,
                    recall_cutoffs,
                    context,
                )
            elif stage_name == "official-build":
                mode = str(
                    inputs.get(
                        "candidate_recompute_mode",
                        provenance.get("official_candidate_recompute_mode", ""),
                    )
                )
                if mode not in ("cached", "real"):
                    raise CollectionError(
                        f"{stage_path}: invalid official build recompute mode"
                    )
                profile = f"official-{mode}"
                self._official_build(
                    tier, profile, source_sha, stage, dataset, context
                )
            elif stage_name in ("official-search-cached", "official-search-real"):
                mode = stage_name.rsplit("-", 1)[-1]
                if inputs.get("candidate_recompute_mode") != mode:
                    raise CollectionError(
                        f"{stage_path}: candidate recompute mode differs from stage name"
                    )
                if provenance.get("official_candidate_recompute_mode") != mode:
                    raise CollectionError(
                        f"{stage_path}: candidate recompute mode differs from manifest"
                    )
                self._official_search(
                    tier,
                    mode,
                    source_sha,
                    stage,
                    dataset,
                    recall_cutoffs,
                    context,
                )
            else:
                raise CollectionError(
                    f"{stage_path}: recognized stage was not dispatched"
                )

    def validate_cross_source_identities(self) -> None:
        for label, values in (
            ("model identity", self._model_hashes),
            ("query source identity", self._query_hashes),
            ("query count", self._query_counts),
            ("embedding dimensions", self._dimensions),
            ("native binary identity", self._native_binary_hashes),
            ("leann.cpp commit", self._leann_commits),
            ("official LEANN commit", self._official_commits),
            ("official runtime identity", self._official_runtime_hashes),
            ("machine/platform identity", self._machine_hashes),
            ("server context identity", self._server_context_hashes),
        ):
            if len(values) > 1:
                raise CollectionError(
                    f"cross-source {label} mismatch: {sorted(values, key=str)}"
                )
        for name, hashes in sorted(self._tooling_hashes.items()):
            if len(hashes) > 1:
                raise CollectionError(
                    f"cross-source tooling mismatch for {name}: {sorted(hashes)}"
                )

    def validate_declared_tier_prefix(self) -> None:
        if not set(DECLARED_TIER_CARDINALITIES).issubset(self.datasets):
            return
        smaller = self.datasets["100k"]
        larger = self.datasets["1m"]
        shared_fields = (
            "query_count",
            "dimensions",
            "queries_sha256",
            "query_cache_sha256",
            "model_sha256",
            "model_artifact_sha256",
            "fingerprint",
        )
        differing = [
            field
            for field in shared_fields
            if smaller.get(field) != larger.get(field)
        ]
        if differing:
            raise CollectionError(
                "100k/1m shared query/model identities differ: "
                f"{differing}"
            )
        smaller_documents = Path(smaller["documents_path"])
        larger_documents = Path(larger["documents_path"])
        if larger["raw_text_file_bytes"] <= smaller["raw_text_file_bytes"]:
            raise CollectionError(
                "1m prepared document file is not larger than 100k"
            )
        document_prefix_sha256 = sha256_file(
            larger_documents, length=smaller["raw_text_file_bytes"]
        )
        if document_prefix_sha256 != smaller["documents_sha256"]:
            raise CollectionError(
                "100k prepared documents are not the exact byte prefix of 1m"
            )
        smaller_header = smaller["corpus_cache_header"]
        larger_header = larger["corpus_cache_header"]
        if (
            smaller_header["dimensions"] != larger_header["dimensions"]
            or smaller_header["fingerprint"] != larger_header["fingerprint"]
            or smaller_header["vector_bytes"]
            != (
                DECLARED_TIER_CARDINALITIES["100k"]
                * smaller_header["dimensions"]
                * 4
            )
        ):
            raise CollectionError(
                "100k/1m corpus cache vector envelopes are inconsistent"
            )
        smaller_vector_sha256 = sha256_file(
            Path(smaller["corpus_cache_path"]),
            offset=smaller_header["vector_offset"],
            length=smaller_header["vector_bytes"],
        )
        larger_vector_prefix_sha256 = sha256_file(
            Path(larger["corpus_cache_path"]),
            offset=larger_header["vector_offset"],
            length=smaller_header["vector_bytes"],
        )
        if smaller_vector_sha256 != larger_vector_prefix_sha256:
            raise CollectionError(
                "100k corpus vectors are not the exact bitwise prefix of 1m"
            )
        self.tier_prefix_identity = {
            "schema": "leann-declared-tier-prefix-v1",
            "smaller_tier": "100k",
            "larger_tier": "1m",
            "rows": DECLARED_TIER_CARDINALITIES["100k"],
            "documents_prefix_size_bytes": smaller["raw_text_file_bytes"],
            "documents_prefix_sha256": document_prefix_sha256,
            "vector_prefix_bytes": smaller_header["vector_bytes"],
            "vector_prefix_sha256": smaller_vector_sha256,
            "documents_exact_byte_prefix": True,
            "vectors_exact_bitwise_prefix": True,
        }


def _profile_for_sample(sample: dict[str, Any]) -> str:
    return (
        "native"
        if sample["system"] == "leann.cpp"
        else f"official-{sample['recompute_mode']}"
    )


def _bind_search_indexes(collector: Collector) -> None:
    """Bind search samples to measured builds by exact artifact identity.

    Official real-mode search may intentionally reuse the index built by the
    cached sweep.  That relationship is marked as inherited; build timings are
    not counted again.
    """

    for search in collector.samples:
        artifact_set = _mapping(
            search.get("index_artifact_set"), "search index artifact set"
        )
        signature = artifact_set.get("sha256")
        reuse_signature = search.get("reuse_artifact_set_sha256")
        if reuse_signature is not None and reuse_signature != signature:
            raise CollectionError(
                f"{search['tier']} official real search artifact identity "
                "differs from its cached-build reuse attestation"
            )
        expected_reuse_source = search.get(
            "reuse_build_source_manifest_sha256"
        )
        expected_reuse_storage = search.get(
            "reuse_storage_identity_sha256"
        )
        matches = [
            build
            for build in collector.build_samples
            if build["tier"] == search["tier"]
            and build["system"] == search["system"]
            and build.get("index_artifact_set", {}).get("sha256") == signature
            and (
                expected_reuse_source is None
                or build["source_manifest_sha256"] == expected_reuse_source
            )
            and (
                expected_reuse_storage is None
                or build.get("storage_identity_sha256")
                == expected_reuse_storage
            )
        ]
        profile = _profile_for_sample(search)
        if not matches:
            collector.issue(
                "unbound-index",
                search["tier"],
                profile,
                (
                    "search artifact hashes do not match the exact attested "
                    "cached-build source/storage supplied to the collector"
                    if expected_reuse_source is not None
                    else "search artifact hashes do not match any measured build"
                ),
            )
            search["index_build_provenance"] = {
                "status": "unbound",
                "artifact_set_sha256": signature,
                "build_source_manifest_sha256": [],
            }
            continue
        same_run = [
            build
            for build in matches
            if build["source_manifest_sha256"]
            == search["source_manifest_sha256"]
        ]
        selected = same_run or matches
        status = "measured-in-same-run" if same_run else "inherited"
        if not any(build.get("storage") is not None for build in selected):
            collector.issue(
                "missing-build-metrics",
                search["tier"],
                profile,
                "index artifacts are bound, but no matching build has validated "
                "storage/build metrics",
            )
        search["index_build_provenance"] = {
            "status": status,
            "artifact_set_sha256": signature,
            "build_source_manifest_sha256": sorted(
                {build["source_manifest_sha256"] for build in selected}
            ),
            "note": (
                "Build metrics were measured in this search run."
                if status == "measured-in-same-run"
                else (
                    "Search reused the exact SHA-verified index built by another "
                    "source run; build metrics are inherited and were not rerun."
                )
            ),
        }
        if status == "inherited":
            for build in selected:
                build.setdefault("inherited_by_profiles", set()).add(profile)


def _aggregate_points(samples: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        key = (
            sample["tier"],
            sample["corpus_count"],
            sample["system"],
            sample["recompute_mode"],
            sample["search_parameter"],
            sample["search_value"],
            sample["batch_size"],
        )
        grouped[key].append(sample)
    output: list[dict[str, Any]] = []
    for key in sorted(grouped):
        group = grouped[key]
        source_hashes = {sample["source_manifest_sha256"] for sample in group}
        protocol_hashes = {sample["protocol_identity_hash"] for sample in group}
        run_roles = {sample["run_role"] for sample in group}
        if len(source_hashes) != 1 or len(protocol_hashes) != 1 or len(run_roles) != 1:
            raise CollectionError(
                f"point {key}: duplicate measurements from separate benchmark "
                "runs/roles cannot be pooled; keep all repetitions in one stage"
            )
        query_counts = {sample["queries"] for sample in group}
        if len(query_counts) != 1:
            raise CollectionError(f"point {key}: query count differs across repetitions")
        latency = [
            value for sample in group for value in sample["latency_ms"]
        ]
        artifact_signatures = {
            sample["index_build_provenance"]["artifact_set_sha256"]
            for sample in group
        }
        if len(artifact_signatures) != 1:
            raise CollectionError(
                f"point {key}: index artifact identity differs across repetitions"
            )
        binding_statuses = sorted(
            {sample["index_build_provenance"]["status"] for sample in group}
        )
        query_observations = sum(sample["queries"] for sample in group)
        recall_keys = sorted(
            key_name
            for key_name in group[0]
            if re.fullmatch(r"recall_at_\d+", key_name)
        )
        result: dict[str, Any] = {
            "tier": key[0],
            "corpus_count": key[1],
            "system": key[2],
            "recompute_mode": key[3],
            "latency_comparable": all(
                sample["latency_comparable"] for sample in group
            ),
            "search_parameter": key[4],
            "search_value": key[5],
            "batch_size": key[6],
            "repetitions": len(group),
            "queries_per_repetition": next(iter(query_counts)),
            "query_observations": query_observations,
            "latency_ms": {
                "mean": float(np.mean(latency)),
                "p50": _linear_percentile(latency, 0.50),
                "p95": _linear_percentile(latency, 0.95),
                "min": min(latency),
                "max": max(latency),
                "aggregation": (
                    "NumPy linear percentiles over pooled point-internal per-query "
                    "measurements for a common cross-system display estimator; "
                    "source summaries were separately validated with each runner's "
                    "native estimator"
                ),
            },
            "evidence": sorted(
                [sample["evidence"] for sample in group],
                key=lambda value: (
                    value.get("sha256", ""),
                    value.get("path", ""),
                ),
            ),
            "source_manifest_sha256": sorted(
                source_hashes
            ),
            "run_role": next(iter(run_roles)),
            "protocol_identity_hash": next(iter(protocol_hashes)),
            "index_build_provenance": {
                "statuses": binding_statuses,
                "artifact_set_sha256": next(iter(artifact_signatures)),
                "build_source_manifest_sha256": sorted(
                    {
                        source
                        for sample in group
                        for source in sample["index_build_provenance"][
                            "build_source_manifest_sha256"
                        ]
                    }
                ),
                "build_metrics_inherited": "inherited" in binding_statuses,
                "note": (
                    "Inherited build metrics were not rerun for this search point."
                    if "inherited" in binding_statuses
                    else "Build and search share a source run."
                ),
            },
        }
        for recall_key in recall_keys:
            result[recall_key] = sum(
                sum(sample[recall_key]) for sample in group
            ) / query_observations
        weighted_fields = (
            ("candidate_embeddings", "candidate_embeddings_mean_per_query"),
            ("embedding_batches", "embedding_batches_mean_per_query"),
            ("approximate_distances", "approximate_distances_mean_per_query"),
            ("upper_layer_hops", "upper_layer_hops_mean_per_query"),
        )
        for sample_field, result_field in weighted_fields:
            if all(sample_field in sample for sample in group):
                result[result_field] = sum(
                    sum(sample[sample_field]) for sample in group
                ) / query_observations
            else:
                result[result_field] = None
        if all("result_ids_sha256" in sample for sample in group):
            result_hashes = {
                sample["result_ids_sha256"] for sample in group
            }
            if len(result_hashes) != 1:
                raise CollectionError(
                    f"point {key}: ranked result IDs changed across repetitions"
                )
            result["result_ids_sha256"] = next(iter(result_hashes))
        else:
            result["result_ids_sha256"] = None
        output.append(result)
    return output


def _aggregate_builds(samples: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    usable = [sample for sample in samples if sample.get("storage") is not None]
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for sample in usable:
        storage_hash = canonical_hash(sample["storage"])
        grouped[(sample["tier"], sample["system"], storage_hash)].append(sample)
    systems_per_tier: dict[tuple[str, str], set[str]] = defaultdict(set)
    for tier, system, storage_hash in grouped:
        systems_per_tier[(tier, system)].add(storage_hash)
    for key, hashes in systems_per_tier.items():
        if len(hashes) > 1:
            raise CollectionError(
                f"{key[0]} {key[1]} storage differs across source runs"
            )
    output: list[dict[str, Any]] = []
    for (tier, system, _), group in sorted(grouped.items()):
        storage = group[0]["storage"]
        dense = storage["dense_fp32_bytes_omitted"]
        raw = storage["raw_text_file_bytes"]
        peak_values = [
            value
            for value in (sample.get("peak_rss_bytes") for sample in group)
            if _is_number(value)
        ]
        algorithm_values = [
            float(value)
            for value in (sample.get("algorithm_seconds") for sample in group)
            if _is_number(value)
        ]
        evidence: list[dict[str, Any]] = []
        for sample in group:
            raw_evidence = sample["evidence"]
            if isinstance(raw_evidence, list):
                evidence.extend(
                    _snapshot_identity(_mapping(item, "build evidence"))
                    for item in raw_evidence
                )
            else:
                evidence.append(
                    _snapshot_identity(_mapping(raw_evidence, "build evidence"))
                )
        artifact_sets_by_sha = {
            sample["index_artifact_set"]["sha256"]: sample["index_artifact_set"]
            for sample in group
        }
        measured_profiles = sorted(
            {
                sample["profile"]
                for sample in group
                if sample.get("profile")
            }
        )
        inherited_profiles = sorted(
            {
                profile
                for sample in group
                for profile in sample.get("inherited_by_profiles", set())
            }
        )
        output.append(
            {
                "tier": tier,
                "system": system,
                "repetitions": len(group),
                "build_wall_seconds_mean": float(
                    np.mean([sample["wall_seconds"] for sample in group])
                ),
                "build_peak_rss_bytes_max": (
                    int(max(peak_values)) if peak_values else None
                ),
                "algorithm_build_seconds_mean": (
                    float(np.mean(algorithm_values))
                    if algorithm_values
                    else None
                ),
                "storage": {
                    **storage,
                    "vector_serving_over_dense_percent": (
                        100.0 * storage["vector_serving_bytes"] / dense
                    ),
                    "vector_serving_over_raw_text_percent": (
                        100.0 * storage["vector_serving_bytes"] / raw
                    ),
                },
                "config": group[0].get("config"),
                "measured_profiles": measured_profiles,
                "inherited_by_profiles": inherited_profiles,
                "build_metric_provenance": (
                    "Build metrics are counted only for measured_profiles. "
                    "Profiles in inherited_by_profiles reused an exact "
                    "SHA-verified artifact set and did not rerun construction."
                ),
                "index_artifact_sets": [
                    artifact_sets_by_sha[digest]
                    for digest in sorted(artifact_sets_by_sha)
                ],
                "source_manifest_sha256": sorted(
                    {sample["source_manifest_sha256"] for sample in group}
                ),
                "evidence": sorted(evidence, key=lambda value: canonical_hash(value)),
            }
        )
    return output


def _matched_comparisons(
    collector: Collector,
    points: Sequence[dict[str, Any]],
    *,
    required_tiers: Sequence[str],
    recall_cutoff: int,
    minimum_recall: float,
    recall_tolerance: float,
) -> list[dict[str, Any]]:
    recall_key = f"recall_at_{recall_cutoff}"
    output: list[dict[str, Any]] = []
    for point in points:
        point["matched_pair_selected"] = False
        point["matched_pair_id"] = None
    for tier in sorted(set(required_tiers)):
        tier_points = [point for point in points if point["tier"] == tier]
        native = [
            point
            for point in tier_points
            if point["system"] == "leann.cpp"
            and point["recompute_mode"] == "real"
        ]
        official_real = [
            point
            for point in tier_points
            if point["system"] == "official LEANN"
            and point["recompute_mode"] == "real"
        ]
        cached_by_config = {
            (point["search_value"], point["batch_size"]): point
            for point in tier_points
            if point["system"] == "official LEANN"
            and point["recompute_mode"] == "cached"
        }
        verified_real: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for real in official_real:
            cached = cached_by_config.get(
                (real["search_value"], real["batch_size"])
            )
            mismatch: list[str] = []
            if cached is None:
                mismatch.append("configuration absent from cached sweep")
            else:
                if (
                    cached["queries_per_repetition"]
                    != real["queries_per_repetition"]
                ):
                    mismatch.append("complete-query counts differ")
                if cached.get("result_ids_sha256") != real.get(
                    "result_ids_sha256"
                ):
                    mismatch.append("all-query result IDs differ")
                for key in sorted(
                    set(cached) & set(real)
                ):
                    if re.fullmatch(r"recall_at_\d+", key) and not _close(
                        float(cached[key]), float(real[key])
                    ):
                        mismatch.append(f"{key} differs")
                if not _close(
                    float(cached["candidate_embeddings_mean_per_query"]),
                    float(real["candidate_embeddings_mean_per_query"]),
                ):
                    mismatch.append("candidate counts differ")
            if mismatch:
                collector.issue(
                    "real-cached-counterpart-mismatch",
                    tier,
                    "official-real",
                    f"complexity={real['search_value']},batch_size="
                    f"{real['batch_size']}: {', '.join(mismatch)}",
                )
            else:
                assert cached is not None
                verified_real.append((real, cached))

        candidates: list[
            tuple[
                tuple[float, float, float, int, int, int],
                dict[str, Any],
                dict[str, Any],
                dict[str, Any],
            ]
        ] = []
        for real, cached in verified_real:
            official_recall = real.get(recall_key)
            if not _is_number(official_recall):
                continue
            for native_point in native:
                native_recall = native_point.get(recall_key)
                if not _is_number(native_recall):
                    continue
                native_value = float(native_recall)
                official_value = float(official_recall)
                gap = abs(native_value - official_value)
                if (
                    native_value < minimum_recall
                    or official_value < minimum_recall
                    or gap > recall_tolerance
                ):
                    continue
                native_candidates = _number(
                    native_point.get(
                        "candidate_embeddings_mean_per_query"
                    ),
                    f"{tier} native candidate count",
                )
                official_candidates = _number(
                    real.get("candidate_embeddings_mean_per_query"),
                    f"{tier} official candidate count",
                )
                rank = (
                    gap,
                    native_candidates + official_candidates,
                    -min(native_value, official_value),
                    int(native_point["search_value"]),
                    int(real["search_value"]),
                    int(real["batch_size"]),
                )
                candidates.append((rank, native_point, real, cached))
        if not candidates:
            collector.issue(
                "no-predeclared-matched-pair",
                tier,
                "official-real",
                f"requires both real Recall@{recall_cutoff}>={minimum_recall:.2f}, "
                f"gap<={recall_tolerance:.3f}, and an exact cached counterpart",
            )
            output.append(
                {
                    "tier": tier,
                    "status": "missing",
                    "recall_cutoff": recall_cutoff,
                    "minimum_recall": minimum_recall,
                    "maximum_recall_gap": recall_tolerance,
                }
            )
            continue
        rank, native_point, real, cached = min(
            candidates, key=lambda value: value[0]
        )
        pair_core = {
            "tier": tier,
            "native": {
                "ef_search": native_point["search_value"],
                "batch_size": native_point["batch_size"],
                "source_manifest_sha256": native_point[
                    "source_manifest_sha256"
                ],
            },
            "official_real": {
                "complexity": real["search_value"],
                "batch_size": real["batch_size"],
                "source_manifest_sha256": real[
                    "source_manifest_sha256"
                ],
            },
        }
        pair_id = canonical_hash(pair_core)
        native_point["matched_pair_selected"] = True
        native_point["matched_pair_id"] = pair_id
        real["matched_pair_selected"] = True
        real["matched_pair_id"] = pair_id
        cached["matched_pair_id"] = pair_id
        native_latency = native_point["latency_ms"]["mean"]
        official_latency = real["latency_ms"]["mean"]
        native_candidates = native_point[
            "candidate_embeddings_mean_per_query"
        ]
        official_candidates = real[
            "candidate_embeddings_mean_per_query"
        ]
        if (
            native_latency <= 0.0
            or official_latency <= 0.0
            or native_candidates <= 0.0
            or official_candidates <= 0.0
        ):
            raise CollectionError(
                f"{tier}: matched comparison ratios require positive latency "
                "and candidate counts"
            )
        output.append(
            {
                "tier": tier,
                "status": "complete",
                "pair_id": pair_id,
                "selection_policy": (
                    "Eligible pairs require both real-mode Recall@"
                    f"{recall_cutoff}>={minimum_recall:.2f}, recall gap<="
                    f"{recall_tolerance:.3f}, and an official cached point with "
                    "the same complexity/batch, complete-query result-ID hash, "
                    "recall, and candidate count. Rank by smallest recall gap, "
                    "then lowest combined candidate count, then highest minimum "
                    "recall, then numeric configuration."
                ),
                "native": {
                    "ef_search": native_point["search_value"],
                    "batch_size": native_point["batch_size"],
                    recall_key: native_point[recall_key],
                    "latency_ms_mean": native_latency,
                    "candidate_embeddings_mean_per_query": native_candidates,
                    "point_source_manifest_sha256": native_point[
                        "source_manifest_sha256"
                    ],
                },
                "official_real": {
                    "complexity": real["search_value"],
                    "batch_size": real["batch_size"],
                    recall_key: real[recall_key],
                    "latency_ms_mean": official_latency,
                    "candidate_embeddings_mean_per_query": official_candidates,
                    "point_source_manifest_sha256": real[
                        "source_manifest_sha256"
                    ],
                },
                "official_cached_counterpart": {
                    "complexity": cached["search_value"],
                    "batch_size": cached["batch_size"],
                    "queries": cached["queries_per_repetition"],
                    "result_ids_sha256": cached["result_ids_sha256"],
                    recall_key: cached[recall_key],
                    "candidate_embeddings_mean_per_query": cached[
                        "candidate_embeddings_mean_per_query"
                    ],
                    "verified_equal_to_real": True,
                },
                "recall_gap": rank[0],
                "ratios": {
                    "official_over_native_mean_latency": (
                        official_latency / native_latency
                    ),
                    "native_speedup_x": official_latency / native_latency,
                    "official_over_native_candidate_count": (
                        official_candidates / native_candidates
                    ),
                },
            }
        )
    return output


def _completeness(
    collector: Collector,
    points: Sequence[dict[str, Any]],
    *,
    required_tiers: Sequence[str],
    required_profiles: Sequence[str],
    required_native_efs: Sequence[int],
    required_cached_complexities: Sequence[int],
    required_cached_batches: Sequence[int],
    min_real_points: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    point_lookup: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for point in points:
        profile = (
            "native"
            if point["system"] == "leann.cpp"
            else f"official-{point['recompute_mode']}"
        )
        point_lookup[(point["tier"], profile)].append(point)
    for tier in required_tiers:
        if tier not in collector.datasets:
            collector.issue(
                "missing-tier", tier, "all", f"no manifest supplied for tier {tier}"
            )
        for profile in required_profiles:
            observed = point_lookup.get((tier, profile), [])
            missing: list[str] = []
            if profile == "native":
                observed_values = {point["search_value"] for point in observed}
                missing = [
                    f"ef_search={value}"
                    for value in required_native_efs
                    if value not in observed_values
                ]
                required_points = [
                    point
                    for point in observed
                    if point["search_value"] in required_native_efs
                ]
            elif profile == "official-cached":
                observed_values = {
                    (point["search_value"], point["batch_size"]) for point in observed
                }
                missing = [
                    f"complexity={complexity},batch_size={batch}"
                    for batch in required_cached_batches
                    for complexity in required_cached_complexities
                    if (complexity, batch) not in observed_values
                ]
                required_points = [
                    point
                    for point in observed
                    if point["search_value"] in required_cached_complexities
                    and point["batch_size"] in required_cached_batches
                ]
            elif profile == "official-real":
                required_points = observed
                if len(observed) < min_real_points:
                    missing = [
                        f"{min_real_points - len(observed)} additional real point(s)"
                    ]
            else:
                required_points = observed
            if not observed:
                collector.issue(
                    "missing-profile",
                    tier,
                    profile,
                    f"no measured {profile} point",
                )
            for detail in missing:
                collector.issue(
                    "missing-configuration", tier, profile, detail
                )
            if not missing and required_points and profile in (
                "native",
                "official-cached",
            ):
                common_sources = set(required_points[0]["source_manifest_sha256"])
                for point in required_points[1:]:
                    common_sources &= set(point["source_manifest_sha256"])
                if not common_sources:
                    collector.issue(
                        "split-sweep",
                        tier,
                        profile,
                        "required sweep points do not come from one protocol-bound "
                        "benchmark manifest",
                    )
            blocking_codes = sorted(
                {
                    issue["code"]
                    for issue in collector.issues
                    if issue["tier"] == tier
                    and issue["profile"] in (profile, "all")
                }
            )
            records.append(
                {
                    "tier": tier,
                    "profile": profile,
                    "status": (
                        "complete"
                        if observed and not missing and not blocking_codes
                        else "incomplete"
                    ),
                    "measured_points": len(observed),
                    "missing": missing,
                    "blocking_issue_codes": blocking_codes,
                }
            )
    return records


def collect_results(
    manifest_specs: Sequence[tuple[str, Path]],
    *,
    parity_specs: Sequence[tuple[str, Path]] = (),
    endpoint_attestation_specs: Sequence[tuple[str, Path]] = (),
    required_tiers: Sequence[str] = ("100k", "1m"),
    required_profiles: Sequence[str] = PROFILES,
    required_native_efs: Sequence[int] = (32, 64, 128, 256, 512),
    required_cached_complexities: Sequence[int] = (32, 64, 128, 256, 512),
    required_cached_batches: Sequence[int] = (0, 16),
    min_real_points: int = 1,
    recall_match_cutoff: int = 3,
    minimum_matched_recall: float = 0.90,
    recall_match_tolerance: float = 0.01,
    minimum_parity_cosine: float = 0.9999,
    require_parity: bool = True,
    require_endpoint_attestation: bool | None = None,
    allow_incomplete: bool = False,
    verify_artifacts: bool = True,
) -> dict[str, Any]:
    if not manifest_specs:
        raise CollectionError("at least one manifest is required")
    required_tier_set = set(required_tiers)
    if required_tier_set & set(DECLARED_TIER_CARDINALITIES) and (
        required_tier_set != set(DECLARED_TIER_CARDINALITIES)
    ):
        raise CollectionError(
            "the declared publication matrix requires exactly tiers 100k and 1m"
        )
    if any(profile not in PROFILES for profile in required_profiles):
        raise CollectionError(f"required profiles must be selected from {PROFILES}")
    if min_real_points < 1:
        raise CollectionError("min_real_points must be positive")
    if recall_match_cutoff <= 0:
        raise CollectionError("recall_match_cutoff must be positive")
    if not 0.0 <= minimum_matched_recall <= 1.0:
        raise CollectionError("minimum_matched_recall must be in [0,1]")
    if recall_match_tolerance < 0.0:
        raise CollectionError("recall_match_tolerance must be non-negative")
    if not 0.0 < minimum_parity_cosine <= 1.0:
        raise CollectionError("minimum_parity_cosine must be in (0,1]")
    if require_endpoint_attestation is None:
        require_endpoint_attestation = (
            set(required_tiers) == set(DECLARED_TIER_CARDINALITIES)
        )
    collector = Collector(verify_artifacts=verify_artifacts)
    if not verify_artifacts:
        collector.issue(
            "artifact-hash-verification-disabled",
            "all",
            "all",
            "recorded artifact hashes were trusted instead of recomputed",
        )
    seen_specs: set[tuple[str, str]] = set()
    for tier, raw_path in sorted(
        manifest_specs, key=lambda value: (value[0], str(value[1]))
    ):
        if not tier or any(character.isspace() for character in tier):
            raise CollectionError(f"invalid tier label {tier!r}")
        if (
            re.fullmatch(r"[1-9][0-9]*[km]", tier)
            and tier not in DECLARED_TIER_CARDINALITIES
        ):
            raise CollectionError(
                f"tier {tier!r} is outside the declared 100k/1m matrix"
            )
        path = raw_path.resolve()
        key = (tier, str(path))
        if key in seen_specs:
            raise CollectionError(f"duplicate manifest specification {tier}={path}")
        seen_specs.add(key)
        collector.ingest(tier, path)
    collector.validate_cross_source_identities()
    collector.validate_declared_tier_prefix()
    parity_by_tier: dict[str, dict[str, Any]] = {}
    native_binary_sha256 = next(
        iter(collector._native_binary_hashes), None
    )
    for tier, path in sorted(
        parity_specs, key=lambda value: (value[0], str(value[1]))
    ):
        if tier in parity_by_tier:
            raise CollectionError(f"duplicate parity result for tier {tier}")
        if tier not in collector.datasets:
            raise CollectionError(
                f"parity result supplied for unknown tier {tier}"
            )
        parity_by_tier[tier] = _validate_parity_report(
            tier=tier,
            path=path,
            dataset=collector.datasets[tier],
            minimum_cosine=minimum_parity_cosine,
            native_binary_sha256=native_binary_sha256,
        )
    dataset_derivation: dict[str, Any] | None = None
    if parity_by_tier:
        ordered_derivations = [
            parity_by_tier[tier]["dataset_derivation"]
            for tier in sorted(parity_by_tier)
        ]
        manifest_hashes = {
            value["manifest"]["sha256"] for value in ordered_derivations
        }
        if len(manifest_hashes) != 1:
            raise CollectionError(
                "embedding parity results do not share one dataset manifest"
            )
        common_keys = (
            "source_dataset",
            "selection",
            "queries",
            "qrel_coverage",
            "invariants",
            "normalization",
        )
        for key in common_keys:
            if len(
                {canonical_hash(value[key]) for value in ordered_derivations}
            ) != 1:
                raise CollectionError(
                    f"embedding parity dataset derivations differ in {key}"
                )
        first_derivation = ordered_derivations[0]
        dataset_derivation = {
            "manifest": first_derivation["manifest"],
            **{
                key: first_derivation[key]
                for key in common_keys
            },
            "tiers": [
                value["tier"]
                for value in sorted(
                    ordered_derivations,
                    key=lambda item: item["tier"]["name"],
                )
            ],
        }
        document_id_prefix = _validate_document_id_prefix(
            dataset_derivation
        )
        if document_id_prefix is not None:
            if collector.tier_prefix_identity is None:
                raise CollectionError(
                    "document ID prefix evidence exists without declared "
                    "document/vector tier-prefix evidence"
                )
            collector.tier_prefix_identity.update(document_id_prefix)
    if require_parity:
        for tier in sorted(set(required_tiers)):
            if tier not in parity_by_tier:
                collector.issue(
                    "missing-embedding-parity",
                    tier,
                    "all",
                    "no passed, identity-bound embedding parity JSON supplied",
                )
    endpoint_attestations: dict[str, dict[str, Any]] = {}
    for tier, path in sorted(
        endpoint_attestation_specs,
        key=lambda value: (value[0], str(value[1])),
    ):
        if tier in endpoint_attestations:
            raise CollectionError(
                f"duplicate endpoint attestation for tier {tier}"
            )
        if tier not in collector.datasets:
            raise CollectionError(
                f"endpoint attestation supplied for unknown tier {tier}"
            )
        if tier not in parity_by_tier:
            raise CollectionError(
                f"endpoint attestation for tier {tier} has no validated parity "
                "report"
            )
        endpoint_attestations[tier] = _validate_endpoint_attestation(
            tier=tier,
            path=path,
            dataset=collector.datasets[tier],
            parity=parity_by_tier[tier],
            verifier=collector.verifier,
        )
    if require_endpoint_attestation:
        for tier in sorted(set(required_tiers)):
            if tier not in endpoint_attestations:
                collector.issue(
                    "missing-endpoint-attestation",
                    tier,
                    "all",
                    "no finalized process/model/cache/parity-bound endpoint "
                    "attestation supplied",
                )
    native_gate_source_hashes = {
        source["sha256"]
        for source in collector.sources
        if source["run_role"] == "native-gate"
    }
    if native_gate_source_hashes:
        collector.samples = [
            sample
            for sample in collector.samples
            if sample["source_manifest_sha256"] not in native_gate_source_hashes
        ]
        collector.build_samples = [
            sample
            for sample in collector.build_samples
            if sample["source_manifest_sha256"] not in native_gate_source_hashes
        ]
        collector.warnings.append(
            {
                "code": "native-gate-excluded",
                "tier": sorted(
                    {
                        source["tier"]
                        for source in collector.sources
                        if source["sha256"] in native_gate_source_hashes
                    }
                ),
                "detail": (
                    "native gate runs are diagnostic smoke checks and are "
                    "excluded from publication points and build evidence"
                ),
            }
        )
    _bind_search_indexes(collector)
    points = _aggregate_points(collector.samples)
    builds = _aggregate_builds(collector.build_samples)
    matched_comparisons = _matched_comparisons(
        collector,
        points,
        required_tiers=(
            required_tiers
            if {
                "native",
                "official-cached",
                "official-real",
            }.issubset(set(required_profiles))
            else []
        ),
        recall_cutoff=recall_match_cutoff,
        minimum_recall=minimum_matched_recall,
        recall_tolerance=recall_match_tolerance,
    )
    completeness = _completeness(
        collector,
        points,
        required_tiers=sorted(set(required_tiers)),
        required_profiles=sorted(set(required_profiles), key=PROFILES.index),
        required_native_efs=sorted(set(required_native_efs)),
        required_cached_complexities=sorted(set(required_cached_complexities)),
        required_cached_batches=sorted(set(required_cached_batches)),
        min_real_points=min_real_points,
    )
    issues = sorted(
        {
            (
                issue["code"],
                issue["tier"],
                issue["profile"],
                issue["detail"],
            )
            for issue in collector.issues
        }
    )
    normalized_issues = [
        {"code": code, "tier": tier, "profile": profile, "detail": detail}
        for code, tier, profile, detail in issues
    ]
    if normalized_issues and not allow_incomplete:
        details = "\n".join(
            f"- [{issue['tier']}/{issue['profile']}] "
            f"{issue['code']}: {issue['detail']}"
            for issue in normalized_issues
        )
        raise CollectionError(f"benchmark matrix is incomplete:\n{details}")
    sources = sorted(
        collector.sources, key=lambda value: (value["tier"], value["sha256"])
    )
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "status": "incomplete" if normalized_issues else "complete",
        "source_completed_at_max": max(
            (
                source["completed_at"]
                for source in sources
                if isinstance(source.get("completed_at"), str)
            ),
            default=None,
        ),
        "requirements": {
            "tiers": sorted(set(required_tiers)),
            "profiles": sorted(set(required_profiles), key=PROFILES.index),
            "native_ef_search": sorted(set(required_native_efs)),
            "official_cached_complexities": sorted(
                set(required_cached_complexities)
            ),
            "official_cached_batch_sizes": sorted(set(required_cached_batches)),
            "minimum_official_real_points_per_tier": min_real_points,
            "matched_comparison": {
                "recall_cutoff": recall_match_cutoff,
                "minimum_recall_for_both_real_points": minimum_matched_recall,
                "maximum_recall_gap": recall_match_tolerance,
                "cached_counterpart_required": True,
            },
            "embedding_parity": {
                "required_per_tier": require_parity,
                "minimum_cosine_similarity": minimum_parity_cosine,
                "all_queries_required": True,
                "exact_model_cache_query_bindings_required": True,
            },
            "embedding_endpoint_attestation": {
                "required_per_tier": require_endpoint_attestation,
                "finalized_phase_required": True,
                "verified_live_process_proof_required": True,
                "exact_source_model_cache_parity_bindings_required": True,
            },
        },
        "validation": {
            "integrity": (
                "passed"
                if verify_artifacts
                else "artifact-hash-verification-disabled"
            ),
            "verified_file_snapshots": collector.verifier.verified_count,
            "shared_cache_validation": {
                "format": "LEANNBC2 required",
                "headers_parsed": True,
                "source_line_counts_recomputed": True,
                "source_size_and_sha256_bindings_verified": True,
                "all_vector_rows_checked_finite_nonzero_unit_normalized": True,
            },
            "declared_matrix_validation": {
                "tiers": dict(DECLARED_TIER_CARDINALITIES),
                "exact_prefix_required": True,
            },
            "incomplete_issues": normalized_issues,
            "warnings": sorted(
                collector.warnings,
                key=lambda value: canonical_hash(value),
            ),
        },
        "identities": {
            "model_sha256": next(iter(collector._model_hashes), None),
            "query_source_sha256": next(iter(collector._query_hashes), None),
            "query_count": next(iter(collector._query_counts), None),
            "dimensions": next(iter(collector._dimensions), None),
            "native_binary_sha256": next(
                iter(collector._native_binary_hashes), None
            ),
            "leann_cpp_commit": next(iter(collector._leann_commits), None),
            "official_leann_commit": next(
                iter(collector._official_commits), None
            ),
            "official_runtime_sha256": next(
                iter(collector._official_runtime_hashes), None
            ),
            "machine_identity_sha256": next(
                iter(collector._machine_hashes), None
            ),
            "server_context_sha256": next(
                iter(collector._server_context_hashes), None
            ),
            "tooling_sha256": {
                name: next(iter(hashes))
                for name, hashes in sorted(collector._tooling_hashes.items())
            },
            "embedding_endpoint_attestation_ids": {
                tier: endpoint_attestations[tier]["attestation_id"]
                for tier in sorted(endpoint_attestations)
            },
        },
        "datasets": [
            collector.datasets[tier] for tier in sorted(collector.datasets)
        ],
        "declared_tier_prefix_identity": collector.tier_prefix_identity,
        "dataset_derivation": dataset_derivation,
        "builds": builds,
        "points": points,
        "matched_comparisons": matched_comparisons,
        "embedding_parity": [
            parity_by_tier[tier] for tier in sorted(parity_by_tier)
        ],
        "embedding_endpoint_attestations": [
            endpoint_attestations[tier]
            for tier in sorted(endpoint_attestations)
        ],
        "completeness": completeness,
        "sources": sources,
        "measurement_policy": {
            "search_latency": (
                "Pooled point-internal per-query measurements; query embedding "
                "and process cold start are excluded by the source harness. "
                "Displayed p50/p95 use one NumPy-linear estimator across systems; "
                "source summaries are validated with their native estimators."
            ),
            "official_cached_latency": (
                "Recorded for audit but not cross-system comparable and omitted "
                "from the Markdown latency table"
            ),
            "subprocess_wall_time": (
                "Build orchestration telemetry only; official search sweeps group "
                "multiple points in one process and are not converted to latency"
            ),
            "storage": (
                "Vector-serving bytes are compared separately from text stores; "
                "dense FP32 omitted bytes and prepared raw text file bytes are "
                "reported as distinct denominators"
            ),
        },
    }
    report["report_id"] = canonical_hash(report)
    return report


def _format_float(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def _mib(value: Any) -> str:
    if value is None:
        return "—"
    return f"{float(value) / (1024 * 1024):.2f}"


def report_csv(report: dict[str, Any]) -> str:
    target = io.StringIO(newline="")
    writer = csv.DictWriter(target, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for point in report["points"]:
        latency = point["latency_ms"]
        row = {
            **{column: point.get(column) for column in CSV_COLUMNS},
            "report_id": report["report_id"],
            "recall_at_3": point.get("recall_at_3"),
            "recall_at_10": point.get("recall_at_10"),
            "latency_ms_mean": latency["mean"],
            "latency_ms_p50": latency["p50"],
            "latency_ms_p95": latency["p95"],
            "latency_ms_min": latency["min"],
            "latency_ms_max": latency["max"],
            "index_build_status": ";".join(
                point["index_build_provenance"]["statuses"]
            ),
            "index_artifact_set_sha256": point[
                "index_build_provenance"
            ]["artifact_set_sha256"],
            "build_source_manifest_sha256": ";".join(
                point["index_build_provenance"][
                    "build_source_manifest_sha256"
                ]
            ),
            "source_manifest_sha256": ";".join(
                point["source_manifest_sha256"]
            ),
            "matched_pair_selected": point["matched_pair_selected"],
            "matched_pair_id": point["matched_pair_id"],
        }
        writer.writerow(row)
    return target.getvalue()


def report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Natural Questions large-scale benchmark",
        "",
        f"Status: **{report['status'].upper()}**. "
        f"Evidence ID: `{report['report_id']}`.",
        "",
    ]
    if report["status"] == "incomplete":
        lines.extend(
            [
                "> This report is intentionally incomplete. Missing measurements "
                "are listed below; no values were estimated or extrapolated.",
                "",
            ]
        )
    lines.extend(
        [
            "## Dataset and identity",
            "",
            "| Tier | Documents | Queries | Dimensions | Raw text MiB | Dense FP32 MiB |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset in report["datasets"]:
        lines.append(
            f"| {dataset['tier']} | {dataset['corpus_count']:,} | "
            f"{dataset['query_count']:,} | {dataset['dimensions']:,} | "
            f"{_mib(dataset['raw_text_file_bytes'])} | "
            f"{_mib(dataset['dense_fp32_bytes'])} |"
        )
    lines.extend(
        [
            "",
            f"- Model identity: `{report['identities']['model_sha256']}`",
            f"- leann.cpp commit: `{report['identities']['leann_cpp_commit']}`",
            f"- Native binary: `{report['identities']['native_binary_sha256']}`",
            f"- Official LEANN commit: `{report['identities']['official_leann_commit']}`",
        ]
    )
    if any(
        source.get("leann_cpp", {}).get("status_porcelain")
        for source in report["sources"]
    ):
        lines.append(
            "- leann.cpp working tree: dirty in at least one source run; "
            "the exact measured executable and tooling SHA-256 identities are "
            "preserved in the JSON evidence."
        )
    lines.append("")
    derivation = report.get("dataset_derivation")
    lines.extend(["## Derived benchmark scope and limitations", ""])
    if derivation is None:
        lines.extend(
            [
                "Dataset derivation evidence is unavailable because no validated "
                "embedding-parity report was supplied.",
                "",
                "Recall@k in this report measures ANN agreement with exact dense "
                "top-k neighbors over the shared embedding matrix. It is not "
                "qrel effectiveness and is not end-to-end RAG answer quality.",
                "",
            ]
        )
    else:
        queries = derivation["queries"]
        coverage = derivation["qrel_coverage"]
        selection = derivation["selection"]
        normalization = derivation["normalization"]
        lines.extend(
            [
                f"- Query set: {queries['selected']:,} qrel-bearing queries "
                f"({queries['positive_qrel_rows']:,} positive qrel rows). "
                f"Selection rule: {selection['queries']}.",
                f"- Corpus construction: {selection['documents']}. "
                f"All {coverage['required_positive_documents']:,} required "
                "positive documents are covered in every reported tier.",
                *(
                    [
                        "- Tier nesting: collector verification proved that the "
                        "100K document-ID mapping, prepared-document bytes, and "
                        "normalized FP32 vectors are exact prefixes of the 1M tier "
                        f"(`{report['declared_tier_prefix_identity']['documents_prefix_sha256']}`, "
                        f"`{report['declared_tier_prefix_identity']['vector_prefix_sha256']}`)."
                    ]
                    if report.get("declared_tier_prefix_identity")
                    else []
                ),
                f"- Normalization: documents use a "
                f"{normalization['maximum_utf8_bytes_including_prefix']:,}-byte "
                "UTF-8 ceiling. Truncation occurs before hashing, text "
                "deduplication, and deterministic selection.",
                "",
                "| Tier | Selected documents | Unique prepared texts | "
                "Truncated documents | Bytes removed by truncation |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for tier_record in derivation["tiers"]:
            lines.append(
                f"| {tier_record['name']} | "
                f"{tier_record['documents']:,} | "
                f"{tier_record['unique_prepared_texts']:,} | "
                f"{tier_record['truncated_documents']:,} | "
                f"{tier_record['truncated_bytes_removed']:,} |"
            )
        lines.extend(
            [
                "",
                "Recall@k in this report measures ANN agreement with exact dense "
                "top-k neighbors over the shared embedding matrix. Qrels define "
                "the derived query/corpus scope; these recall values are not BEIR "
                "qrel effectiveness and are not end-to-end RAG answer quality.",
                "",
            ]
        )
    lines.extend(
        [
            "## Embedding parity gate",
            "",
            "| Tier | Status | Rows | All queries | Minimum cosine | Required cosine |",
            "|---|---|---:|---|---:|---:|",
        ]
    )
    for parity in report["embedding_parity"]:
        lines.append(
            f"| {parity['tier']} | passed | {parity['metrics']['rows']} | "
            f"{str(parity['coverage']['queries']['all_queries']).lower()} | "
            f"{_format_float(parity['metrics']['minimum_cosine_similarity'], 7)} | "
            f"{_format_float(parity['acceptance']['minimum_cosine_similarity'], 7)} |"
        )
    if not report["embedding_parity"]:
        lines.append("| — | missing | — | — | — | — |")
    lines.extend(
        [
            "",
            "## Live embedding endpoint attestation",
            "",
            "| Tier | Status | Server build | Process proof | Attestation ID |",
            "|---|---|---|---|---|",
        ]
    )
    for attestation in report["embedding_endpoint_attestations"]:
        lines.append(
            f"| {attestation['tier']} | finalized | "
            f"`{attestation['bindings']['server_build_info']}` | "
            f"{attestation['process_proof']['status']} | "
            f"`{attestation['attestation_id']}` |"
        )
    if not report["embedding_endpoint_attestations"]:
        lines.append("| — | missing or disabled | — | — | — |")
    lines.extend(
        [
            "",
            "A finalized row is emitted only after the collector binds the live "
            "server process, executable, build identity, GGUF bytes, source/cache "
            "identity, and independently validated native parity evidence.",
            "",
            "## Build and storage",
            "",
            "| Tier | System | Measured for | Inherited by | Build wall s | "
            "Peak RSS MiB | Vector-serving MiB | Text store MiB | "
            "Vector / dense | Vector / raw text |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for build in report["builds"]:
        storage = build["storage"]
        measured = ", ".join(build["measured_profiles"]) or "—"
        inherited = ", ".join(build["inherited_by_profiles"]) or "—"
        lines.append(
            f"| {build['tier']} | {build['system']} | "
            f"{measured} | {inherited} | "
            f"{_format_float(build['build_wall_seconds_mean'])} | "
            f"{_mib(build['build_peak_rss_bytes_max'])} | "
            f"{_mib(storage['vector_serving_bytes'])} | "
            f"{_mib(storage['text_store_bytes'])} | "
            f"{_format_float(storage['vector_serving_over_dense_percent'], 2)}% | "
            f"{_format_float(storage['vector_serving_over_raw_text_percent'], 2)}% |"
        )
    lines.extend(
        [
            "",
            "Vector-serving artifacts and text stores are separated. The dense "
            "denominator is the omitted normalized FP32 embedding matrix; the raw "
            "denominator is the prepared UTF-8 document file. An inherited profile "
            "reused the exact SHA-verified index and did not rerun construction; "
            "its build metrics come from the listed measured profile.",
            "",
            "## Predeclared matched real-recompute comparison",
            "",
            "| Tier | Native setting | Official setting | Native Recall | "
            "Official Recall | Gap | Native mean ms | Official mean ms | "
            "Native speedup |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for match in report["matched_comparisons"]:
        if match["status"] != "complete":
            lines.append(
                f"| {match['tier']} | — | No eligible matched pair | — | — | "
                "— | — | — | — |"
            )
            continue
        native_match = match["native"]
        official_match = match["official_real"]
        recall_keys = sorted(
            key for key in native_match if re.fullmatch(r"recall_at_\d+", key)
        )
        recall_key = recall_keys[0]
        lines.append(
            f"| {match['tier']} | ef_search={native_match['ef_search']} | "
            f"complexity={official_match['complexity']},"
            f"batch={official_match['batch_size']} | "
            f"{_format_float(native_match[recall_key], 4)} | "
            f"{_format_float(official_match[recall_key], 4)} | "
            f"{_format_float(match['recall_gap'], 4)} | "
            f"{_format_float(native_match['latency_ms_mean'])} | "
            f"{_format_float(official_match['latency_ms_mean'])} | "
            f"{_format_float(match['ratios']['native_speedup_x'], 2)}x |"
        )
    lines.extend(
        [
            "",
            "Ratios are emitted only for pairs satisfying the declared recall floor "
            "and gap and whose official real point exactly matches a complete-query "
            "cached counterpart in result IDs, recall, and candidate count.",
            "",
            "## Search with real GGUF candidate recomputation",
            "",
            "| Tier | System | Search setting | Batch | Build provenance | Repeats | "
            "Recall@3 | Recall@10 | Mean ms | p50 ms | p95 ms | Candidates/query |",
            "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    real_points = [
        point
        for point in report["points"]
        if point["recompute_mode"] == "real" and point["latency_comparable"]
    ]
    for point in real_points:
        lines.append(
            f"| {point['tier']} | {point['system']} | "
            f"{point['search_parameter']}={point['search_value']} | "
            f"{point['batch_size']} | "
            f"{'/'.join(point['index_build_provenance']['statuses'])} | "
            f"{point['repetitions']} | "
            f"{_format_float(point.get('recall_at_3'), 4)} | "
            f"{_format_float(point.get('recall_at_10'), 4)} | "
            f"{_format_float(point['latency_ms']['mean'])} | "
            f"{_format_float(point['latency_ms']['p50'])} | "
            f"{_format_float(point['latency_ms']['p95'])} | "
            f"{_format_float(point.get('candidate_embeddings_mean_per_query'), 1)} |"
        )
    if not real_points:
        lines.append("| — | No complete real-recompute point | — | — | — | — | — | — | — | — | — | — |")
    lines.extend(
        [
            "",
            "For both systems, every reported Recall@k is independently "
            "recomputed from the integrity-bound ranked result IDs and exact "
            "LEANN_GT1 dense ground truth; runner summaries alone are not trusted.",
            "",
            "## Official cached-recompute candidate sweep",
            "",
            "Cached mode isolates the official graph/candidate Pareto frontier. Its "
            "latency is not compared with native real-GGUF recomputation.",
            "",
            "| Tier | Complexity | Batch | Repeats | Recall@3 | Recall@10 | "
            "Candidates/query |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    cached_points = [
        point
        for point in report["points"]
        if point["system"] == "official LEANN"
        and point["recompute_mode"] == "cached"
    ]
    for point in cached_points:
        lines.append(
            f"| {point['tier']} | {point['search_value']} | "
            f"{point['batch_size']} | {point['repetitions']} | "
            f"{_format_float(point.get('recall_at_3'), 4)} | "
            f"{_format_float(point.get('recall_at_10'), 4)} | "
            f"{_format_float(point.get('candidate_embeddings_mean_per_query'), 1)} |"
        )
    if not cached_points:
        lines.append("| — | — | — | — | — | — | No complete cached point |")
    lines.extend(
        [
            "",
            "## Completeness",
            "",
            "| Tier | Profile | Status | Measured points | Missing |",
            "|---|---|---|---:|---|",
        ]
    )
    for item in report["completeness"]:
        missing = "; ".join(item["missing"]) if item["missing"] else "—"
        lines.append(
            f"| {item['tier']} | {item['profile']} | {item['status']} | "
            f"{item['measured_points']} | {missing} |"
        )
    if report["validation"]["incomplete_issues"]:
        lines.extend(["", "Incomplete evidence:"])
        for issue in report["validation"]["incomplete_issues"]:
            lines.append(
                f"- `{issue['tier']}/{issue['profile']}` "
                f"{issue['code']}: {issue['detail']}"
            )
    if report["validation"]["warnings"]:
        lines.extend(["", "Validation notes:"])
        for warning in report["validation"]["warnings"]:
            tier_value = warning.get("tier", "all")
            tier_text = (
                ",".join(str(value) for value in tier_value)
                if isinstance(tier_value, list)
                else str(tier_value)
            )
            lines.append(
                f"- `{tier_text}` {warning['code']}: {warning['detail']}"
            )
    lines.extend(
        [
            "",
            "## Evidence",
            "",
            "| Tier | Manifest SHA-256 | Manifest |",
            "|---|---|---|",
        ]
    )
    for source in report["sources"]:
        lines.append(
            f"| {source['tier']} | `{source['sha256']}` | `{source['path']}` |"
        )
    lines.extend(
        [
            "",
            "All displayed values are derived from integrity-checked stage JSON and "
            "raw per-query result files. See the JSON output for artifact identities, "
            "measurement definitions, and the complete normalized point set. The "
            "output commit marker binds this Markdown, CSV, and JSON generation by "
            "report ID, byte size, and SHA-256.",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as target:
            target.write(payload)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_report_identifier(report: dict[str, Any]) -> str:
    report_id = report.get("report_id")
    if not isinstance(report_id, str) or not re.fullmatch(
        r"[0-9a-f]{64}", report_id
    ):
        raise CollectionError("report has no valid report_id")
    expected = canonical_hash(
        {key: value for key, value in report.items() if key != "report_id"}
    )
    if report_id != expected:
        raise CollectionError(
            f"report_id {report_id} does not match report payload {expected}"
        )
    return report_id


def _output_snapshot(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def validate_output_commit(commit_path: Path) -> dict[str, Any]:
    commit_path = commit_path.resolve()
    commit = _mapping(load_json(commit_path), str(commit_path))
    if commit.get("schema") != OUTPUT_COMMIT_SCHEMA:
        raise CollectionError(f"{commit_path}: unknown output commit schema")
    commit_id = commit.get("commit_id")
    if not isinstance(commit_id, str) or not re.fullmatch(
        r"[0-9a-f]{64}", commit_id
    ):
        raise CollectionError(f"{commit_path}: invalid output commit ID")
    core = {
        key: value for key, value in commit.items() if key != "commit_id"
    }
    if canonical_hash(core) != commit_id:
        raise CollectionError(f"{commit_path}: output commit ID differs")
    report_id = commit.get("report_id")
    if not isinstance(report_id, str) or not re.fullmatch(
        r"[0-9a-f]{64}", report_id
    ):
        raise CollectionError(f"{commit_path}: invalid committed report ID")
    files = _mapping(commit.get("files"), f"{commit_path} files")
    if set(files) != {"json", "csv", "markdown"}:
        raise CollectionError(
            f"{commit_path}: committed output set must be JSON/CSV/Markdown"
        )
    verifier = FileVerifier(enabled=True)
    resolved: dict[str, Path] = {}
    for name in ("json", "csv", "markdown"):
        resolved[name] = verifier.verify(
            files[name], f"{commit_path} committed {name}"
        )
    report = _mapping(
        load_json(resolved["json"]), f"{commit_path} committed report"
    )
    if _validate_report_identifier(report) != report_id:
        raise CollectionError(
            f"{commit_path}: JSON report ID differs from commit marker"
        )
    try:
        with resolved["csv"].open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            rows = list(reader)
            fields = reader.fieldnames or []
    except (OSError, csv.Error) as error:
        raise CollectionError(
            f"{commit_path}: cannot parse committed CSV: {error}"
        ) from error
    if "report_id" not in fields or any(
        row.get("report_id") != report_id for row in rows
    ):
        raise CollectionError(
            f"{commit_path}: CSV report IDs differ from commit marker"
        )
    try:
        markdown = resolved["markdown"].read_text(encoding="utf-8")
    except OSError as error:
        raise CollectionError(
            f"{commit_path}: cannot read committed Markdown: {error}"
        ) from error
    marker = f"Evidence ID: `{report_id}`."
    if markdown.count(marker) != 1:
        raise CollectionError(
            f"{commit_path}: Markdown report ID differs from commit marker"
        )
    return {
        "schema": OUTPUT_COMMIT_SCHEMA,
        "commit_path": str(commit_path),
        "commit_id": commit_id,
        "report_id": report_id,
        "files": {name: str(path) for name, path in resolved.items()},
    }


def write_outputs(report: dict[str, Any], prefix: Path) -> dict[str, Path]:
    report_id = _validate_report_identifier(report)
    prefix = prefix.resolve()
    generation_directory = (
        prefix.parent / f"{prefix.name}.generations"
    )
    paths = {
        "json": generation_directory / f"{report_id}.json",
        "csv": generation_directory / f"{report_id}.csv",
        "markdown": generation_directory / f"{report_id}.md",
    }
    _atomic_write(
        paths["json"],
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    _atomic_write(paths["csv"], report_csv(report))
    _atomic_write(paths["markdown"], report_markdown(report))
    core = {
        "schema": OUTPUT_COMMIT_SCHEMA,
        "report_id": report_id,
        "files": {
            name: _output_snapshot(paths[name])
            for name in ("json", "csv", "markdown")
        },
    }
    commit = {**core, "commit_id": canonical_hash(core)}
    commit_path = prefix.parent / f"{prefix.name}.commit.json"
    _atomic_write(
        commit_path,
        json.dumps(commit, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    validated = validate_output_commit(commit_path)
    if validated["report_id"] != report_id:
        raise CollectionError("published output commit report ID differs")
    return {**paths, "commit": commit_path}


def _manifest_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected TIER=PATH")
    tier, path = value.split("=", 1)
    if not tier or not path:
        raise argparse.ArgumentTypeError("expected non-empty TIER=PATH")
    return tier, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        action="append",
        required=True,
        type=_manifest_spec,
        metavar="TIER=PATH",
        help="Benchmark manifest; repeat when a tier uses separate native/cached/real runs",
    )
    parser.add_argument(
        "--parity",
        action="append",
        type=_manifest_spec,
        metavar="TIER=PATH",
        help=(
            "Passed leann-embedding-parity-v1 JSON; one is required for every "
            "required tier unless --allow-missing-parity is explicit"
        ),
    )
    parser.add_argument(
        "--endpoint-attestation",
        action="append",
        type=_manifest_spec,
        metavar="TIER=PATH",
        help=(
            "Finalized leann-embedding-endpoint-attestation-v1 JSON; one is "
            "required for every declared publication tier unless "
            "--allow-missing-endpoint-attestation is explicit"
        ),
    )
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--required-tier", action="append", dest="required_tiers")
    parser.add_argument(
        "--required-profile",
        action="append",
        choices=PROFILES,
        dest="required_profiles",
    )
    parser.add_argument(
        "--required-native-ef",
        type=int,
        nargs="+",
        default=[32, 64, 128, 256, 512],
    )
    parser.add_argument(
        "--required-cached-complexity",
        type=int,
        nargs="+",
        default=[32, 64, 128, 256, 512],
    )
    parser.add_argument(
        "--required-cached-batch-size",
        type=int,
        nargs="+",
        default=[0, 16],
    )
    parser.add_argument("--minimum-real-points", type=int, default=1)
    parser.add_argument(
        "--recall-match-cutoff",
        type=int,
        default=3,
        help="Recall cutoff used by the predeclared real-mode pair selector",
    )
    parser.add_argument(
        "--minimum-matched-recall",
        type=float,
        default=0.90,
        help="Both native and official real points must meet this recall",
    )
    parser.add_argument(
        "--recall-match-tolerance",
        type=float,
        default=0.01,
        help="Maximum absolute recall gap for a matched real-mode pair",
    )
    parser.add_argument(
        "--minimum-parity-cosine",
        type=float,
        default=0.9999,
    )
    parser.add_argument(
        "--allow-missing-parity",
        action="store_true",
        help="Disable the publication parity gate (report records that it was disabled)",
    )
    parser.add_argument(
        "--allow-missing-endpoint-attestation",
        action="store_true",
        help=(
            "Disable the publication live endpoint/process attestation gate "
            "(report records that it was disabled)"
        ),
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument(
        "--no-verify-artifact-hashes",
        action="store_true",
        help="Verify paths/sizes but trust recorded hashes (not recommended for publication)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = collect_results(
        args.manifest,
        parity_specs=args.parity or [],
        endpoint_attestation_specs=args.endpoint_attestation or [],
        required_tiers=args.required_tiers or ["100k", "1m"],
        required_profiles=args.required_profiles or list(PROFILES),
        required_native_efs=args.required_native_ef,
        required_cached_complexities=args.required_cached_complexity,
        required_cached_batches=args.required_cached_batch_size,
        min_real_points=args.minimum_real_points,
        recall_match_cutoff=args.recall_match_cutoff,
        minimum_matched_recall=args.minimum_matched_recall,
        recall_match_tolerance=args.recall_match_tolerance,
        minimum_parity_cosine=args.minimum_parity_cosine,
        require_parity=not args.allow_missing_parity,
        require_endpoint_attestation=not args.allow_missing_endpoint_attestation,
        allow_incomplete=args.allow_incomplete,
        verify_artifacts=not args.no_verify_artifact_hashes,
    )
    outputs = write_outputs(report, args.output_prefix)
    print(
        json.dumps(
            {
                "status": report["status"],
                "report_id": report["report_id"],
                "outputs": {key: str(value.resolve()) for key, value in outputs.items()},
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
