#!/usr/bin/env python3
"""Focused tests for deterministic large-scale result collection."""

from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import struct
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_cache import MAGIC_V1, write_cache_v2  # noqa: E402
import collect_large_scale_results as collector_module  # noqa: E402
from collect_large_scale_results import (  # noqa: E402
    CollectionError,
    Collector,
    canonical_hash,
    collect_results,
    report_csv,
    report_markdown,
    validate_output_commit,
    write_outputs,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": digest(path),
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


class Fixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.query_count = 2
        self.corpus_count = 10
        self.dimensions = 4
        self.truth = root / "truth.txt"
        self.truth.write_text(
            "LEANN_GT1 2 3 10\n0 1 2\n3 4 5\n", encoding="ascii"
        )
        self.documents = root / "documents.txt"
        self.documents.write_text(
            "".join(f"document-{index}\n" for index in range(self.corpus_count)),
            encoding="utf-8",
        )
        self.queries = root / "queries.txt"
        self.queries.write_text(
            "".join(f"query-{index}\n" for index in range(self.query_count)),
            encoding="utf-8",
        )
        self.corpus_cache = root / "corpus.leannbc2"
        corpus_vectors = np.zeros(
            (self.corpus_count, self.dimensions), dtype=np.float32
        )
        for index in range(self.corpus_count):
            corpus_vectors[index, index % self.dimensions] = 1.0
        corpus_metadata = write_cache_v2(
            self.corpus_cache,
            corpus_vectors,
            fingerprint="synthetic-fingerprint",
            source_path=self.documents,
        )
        self.query_cache = root / "queries.leannbc2"
        query_vectors = np.zeros(
            (self.query_count, self.dimensions), dtype=np.float32
        )
        for index in range(self.query_count):
            query_vectors[index, index % self.dimensions] = 1.0
        query_metadata = write_cache_v2(
            self.query_cache,
            query_vectors,
            fingerprint="synthetic-fingerprint",
            source_path=self.queries,
        )
        self.model = root / "model.gguf"
        self.model.write_bytes(b"synthetic-model")
        self.native_binary = root / "leann"
        self.native_binary.write_bytes(b"synthetic-native-binary")
        self.llama_server = root / "llama-server"
        self.llama_server.write_bytes(b"synthetic-llama-server")
        self.official_python = root / "official-python"
        self.official_python.write_bytes(b"synthetic-python")
        self.official_module = root / "leann-runtime.py"
        self.official_module.write_bytes(b"synthetic-leann-runtime")
        self.official_faiss = root / "faiss-runtime.so"
        self.official_faiss.write_bytes(b"synthetic-faiss-runtime")
        module_snapshot = snapshot(self.official_module)
        faiss_snapshot = snapshot(self.official_faiss)
        self.official_runtime = {
            "python": "synthetic-python",
            "executable": str(self.official_python.resolve()),
            "launcher": str(self.official_python.resolve()),
            "launcher_resolved": str(self.official_python.resolve()),
            "modules": {
                name: {
                    "path": (
                        str(self.official_faiss.resolve())
                        if name.endswith(".faiss") or name.endswith("._swigfaiss")
                        else str(self.official_module.resolve())
                    ),
                    "files": [
                        faiss_snapshot
                        if name.endswith(".faiss") or name.endswith("._swigfaiss")
                        else module_snapshot
                    ],
                }
                for name in (
                    "leann",
                    "leann_backend_hnsw",
                    "leann_backend_hnsw.faiss",
                    "leann_backend_hnsw._swigfaiss",
                    "leann_backend_hnsw.hnsw_backend",
                )
            },
            "backend_faiss": faiss_snapshot,
            "distributions": {
                "leann-core": "synthetic",
                "leann-backend-hnsw": "synthetic",
            },
        }
        self.shared = {
            "documents": {
                **snapshot(self.documents),
                "nonempty_lines": self.corpus_count,
            },
            "queries": {
                **snapshot(self.queries),
                "nonempty_lines": self.query_count,
            },
            "corpus_cache": {
                **corpus_metadata,
            },
            "query_cache": {
                **query_metadata,
            },
            "ground_truth": {
                **snapshot(self.truth),
                "corpus_count": self.corpus_count,
                "query_count": self.query_count,
                "top_k": 3,
            },
            "model": {
                "sha256": "5" * 64,
                "identity_strength": "artifact-sha256",
                "artifact": {
                    **snapshot(self.model),
                },
            },
        }
        self.leann_commit = "a" * 40
        self.official_commit = "b" * 40
        self.native_csv: Path | None = None

    def protocol(self, run_role: str) -> dict[str, Any]:
        return {
            "run_role": run_role,
            "build_warmups": 0,
            "build_repetitions": 1,
            "search_warmups": 0,
            "search_repetitions": 1,
            "recall_cutoffs": [3],
        }

    def parity(self, tier: str = "tiny") -> Path:
        path = self.root / f"parity-{tier}.json"
        dataset_manifest_path = self.root / "dataset-manifest.json"
        document_ids_path = self.root / f"document-ids-{tier}.tsv"
        fixture_source_path = self.root / f"parity-fixture-{tier}.txt"
        reference_cache_path = self.root / f"parity-reference-{tier}.leannbc2"
        native_cache_path = self.root / f"parity-native-{tier}.leannbc1"
        fixture_rows = [
            {"kind": "query", "source_index": 0},
            {"kind": "query", "source_index": 1},
            {"kind": "document", "source_index": 0},
        ]
        fixture_source_path.write_text(
            "query-0\nquery-1\ndocument-0\n", encoding="utf-8"
        )
        fixture_vectors = np.zeros(
            (len(fixture_rows), self.dimensions), dtype=np.float32
        )
        for index in range(len(fixture_rows)):
            fixture_vectors[index, index] = 1.0
        reference_cache = write_cache_v2(
            reference_cache_path,
            fixture_vectors,
            fingerprint="synthetic-fingerprint",
            source_path=fixture_source_path,
        )
        fingerprint = b"synthetic-fingerprint"
        with native_cache_path.open("wb") as target:
            target.write(
                struct.pack(
                    "<8sIQI",
                    MAGIC_V1,
                    self.dimensions,
                    len(fixture_rows),
                    len(fingerprint),
                )
            )
            target.write(fingerprint)
            target.write(
                np.asarray(fixture_vectors, dtype="<f4").tobytes(order="C")
            )
        native_cache = {
            **snapshot(native_cache_path),
            "schema": "LEANNBC1",
            "version": 1,
            "count": len(fixture_rows),
            "dimensions": self.dimensions,
            "fingerprint": "synthetic-fingerprint",
        }
        document_ids_path.write_text(
            "".join(
                f"{index}\tdocument-{index}\n"
                for index in range(self.corpus_count)
            ),
            encoding="utf-8",
        )
        tier_truncation = {"documents": 1, "bytes_removed": 7}
        dataset_manifest = {
            "schema_version": 1,
            "dataset": {
                "name": "Synthetic Natural Questions",
                "archive": {"sha256": "a" * 64},
            },
            "files": {
                f"documents_{tier}": {
                    "lines": self.corpus_count,
                    "sha256": self.shared["documents"]["sha256"],
                },
                f"document_ids_{tier}": {
                    "lines": self.corpus_count,
                    "sha256": digest(document_ids_path),
                },
                "queries": {
                    "lines": self.query_count,
                    "sha256": self.shared["queries"]["sha256"],
                },
            },
            "invariants": {
                "100k_is_exact_prefix_of_1m": True,
                "document_ids_are_unique": True,
                "original_query_document_ids_and_qrel_scores_preserved": True,
            },
            "normalization": {
                "document_format": "search_document: <text>",
                "query_format": "search_query: <query>",
                "line_cleaning": "synthetic whitespace normalization",
                "document_truncation": {
                    "algorithm": "synthetic UTF-8 prefix",
                    "applied_before_text_hashing_deduplication_and_selection": True,
                    "maximum_utf8_bytes_including_prefix": 2000,
                },
            },
            "qrel_coverage": {
                "required_positive_documents": 3,
                f"covered_positive_documents_{tier}": 3,
                f"document_coverage_{tier}": 1.0,
                f"positive_judgment_coverage_{tier}": 1.0,
            },
            "queries": {
                "queries": self.query_count,
                "qrel_rows": 3,
                "positive_qrel_rows": 3,
            },
            "selection": {
                "queries": "lowest-ranked qrel-bearing query IDs",
                "documents": (
                    "all selected-query positive documents in the 100K tier; "
                    "unique-text fillers in rank order"
                ),
                "rank": "SHA-256(seed || original_id)",
                "seed": "synthetic-seed",
            },
            "tiers": {
                tier: {
                    "documents": self.corpus_count,
                    "maximum_prepared_document_bytes": 2000,
                    "text_duplicates": {
                        "rows": self.corpus_count,
                        "unique_prepared_texts": self.corpus_count,
                        "duplicate_rows": 0,
                        "duplicate_rate": 0.0,
                    },
                    "truncation": tier_truncation,
                }
            },
        }
        write_json(dataset_manifest_path, dataset_manifest)
        report = {
            "schema": "leann-embedding-parity-v1",
            "created_at": "2026-01-01T00:03:00+00:00",
            "input_hash": "8" * 64,
            "passed": True,
            "acceptance": {
                "minimum_cosine_similarity": 0.9999,
                "rule": "synthetic threshold fixture",
            },
            "coverage": {
                "queries": {
                    "selected": self.query_count,
                    "source_rows": self.query_count,
                    "all_queries": True,
                },
                "documents": {
                    "selected_documents": 1,
                    "truncation_proof": {
                        "checked_documents": 1,
                        "source_verified_truncated_documents": 1,
                    },
                },
                "total_rows": self.query_count + 1,
            },
            "dataset": {
                "tier": tier,
                "documents": {
                    "sha256": self.shared["documents"]["sha256"],
                    "nonempty_lines": self.corpus_count,
                },
                "queries": {
                    "sha256": self.shared["queries"]["sha256"],
                    "nonempty_lines": self.query_count,
                },
                "manifest": {
                    "path": str(dataset_manifest_path.resolve()),
                    "sha256": digest(dataset_manifest_path),
                    "tier_truncation": tier_truncation,
                    "maximum_prepared_document_bytes": 2000,
                },
                "source_archive": {"sha256": "a" * 64},
                "document_ids": snapshot(document_ids_path),
            },
            "official_comparison_reference": {
                "corpus_cache": {
                    "sha256": self.shared["corpus_cache"]["sha256"],
                    "sidecar_sha256": "c" * 64,
                    "count": self.corpus_count,
                    "dimensions": self.dimensions,
                    "fingerprint": "synthetic-fingerprint",
                },
                "query_cache": {
                    "sha256": self.shared["query_cache"]["sha256"],
                    "sidecar_sha256": "d" * 64,
                    "count": self.query_count,
                    "dimensions": self.dimensions,
                    "fingerprint": "synthetic-fingerprint",
                },
                "model": {
                    "sha256": self.shared["model"]["sha256"],
                    "artifact": {
                        "sha256": self.shared["model"]["artifact"][
                            "sha256"
                        ]
                    },
                },
            },
            "native": {
                "binary": {"sha256": digest(self.native_binary)},
                "model": {
                    "sha256": self.shared["model"]["artifact"]["sha256"]
                },
                "cache": native_cache,
            },
            "fixture": {
                "source": {
                    **snapshot(fixture_source_path),
                    "nonempty_lines": len(fixture_rows),
                },
                "reference_cache": reference_cache,
                "rows": fixture_rows,
            },
            "metrics": {
                "rows": self.query_count + 1,
                "dimensions": self.dimensions,
                "minimum_cosine_similarity": 1.0,
                "mean_cosine_similarity": 1.0,
                "maximum_absolute_difference": 0.0,
                "by_kind": {
                    "query": {
                        "rows": self.query_count,
                        "minimum_cosine_similarity": 1.0,
                        "maximum_absolute_difference": 0.0,
                    },
                    "document": {
                        "rows": 1,
                        "minimum_cosine_similarity": 1.0,
                        "maximum_absolute_difference": 0.0,
                    },
                },
            },
        }
        write_json(path, report)
        return path

    def endpoint_attestation(self, parity_path: Path, tier: str = "tiny") -> Path:
        parity = json.loads(parity_path.read_text(encoding="utf-8"))
        source = {
            **snapshot(self.documents),
            "nonempty_lines": self.corpus_count,
        }
        binding = {
            "tier": tier,
            "expected_source_count": self.corpus_count,
            "source": source,
            "model_descriptor_sha256": self.shared["model"]["sha256"],
            "model_artifact_sha256": self.shared["model"]["artifact"]["sha256"],
            "fingerprint": self.shared["corpus_cache"]["fingerprint"],
            "dimension": self.dimensions,
            "checkpoint_input_hash": "1" * 64,
        }
        process_stable = {
            "provider": "synthetic-process-proof-v1",
            "pid": 4242,
            "executable": snapshot(self.llama_server),
            "command_model_path": str(self.model.resolve()),
        }
        process_stable["identity_sha256"] = canonical_hash(process_stable)
        capture_evidence = {
            "endpoint": {
                "embedding_url": "http://127.0.0.1:18080/v1/embeddings",
                "health_before": {
                    "status": 200,
                    "payload": {"status": "ok"},
                },
                "health_after": {
                    "status": 200,
                    "payload": {"status": "ok"},
                },
                "props_identity": {
                    "model_path": str(self.model.resolve()),
                    "build_info": "synthetic-build",
                },
            },
            "model_artifact": snapshot(self.model),
            "model_artifact_after": snapshot(self.model),
            "model_artifact_unchanged": True,
            "expected_build_info": "synthetic-build",
            "checkpoint": {
                "stable_identity": {
                    "source": source,
                    "model": {
                        "sha256": self.shared["model"]["sha256"],
                    },
                    "fingerprint": self.shared["corpus_cache"]["fingerprint"],
                    "dimension": self.dimensions,
                    "embedding_endpoint": (
                        "http://127.0.0.1:18080/v1/embeddings"
                    ),
                }
            },
            "source": source,
            "source_unchanged": True,
            "process_proof": {
                "status": "verified",
                "required": True,
                "unchanged": True,
                "server_started_before_checkpoint": True,
                "before": process_stable,
                "after": process_stable,
            },
            "required_post_run_native_parity": binding,
        }
        capture = {
            "schema": "leann-embedding-endpoint-attestation-v1",
            "phase": "live-capture",
            "created_at": "2026-01-01T00:02:00+00:00",
            "attestation_id": canonical_hash(capture_evidence),
            "evidence": capture_evidence,
            "post_run_native_parity": {"status": "required-pending"},
        }
        capture_path = self.root / f"endpoint-capture-{tier}.json"
        write_json(capture_path, capture)
        final_evidence = {
            "live_capture": {
                "artifact": snapshot(capture_path),
                "attestation_id": capture["attestation_id"],
                "report": capture,
            },
            "native_parity": {
                "artifact": snapshot(parity_path),
                "input_hash": parity["input_hash"],
                "minimum_cosine_similarity": parity["metrics"][
                    "minimum_cosine_similarity"
                ],
                "required_minimum_cosine_similarity": parity["acceptance"][
                    "minimum_cosine_similarity"
                ],
                "native_binary": parity["native"]["binary"],
                "native_model": parity["native"]["model"],
                "corpus_cache_sha256": parity[
                    "official_comparison_reference"
                ]["corpus_cache"]["sha256"],
            },
            "binding": binding,
        }
        final = {
            "schema": "leann-embedding-endpoint-attestation-v1",
            "phase": "finalized",
            "created_at": "2026-01-01T00:04:00+00:00",
            "attestation_id": canonical_hash(final_evidence),
            "evidence": final_evidence,
            "post_run_native_parity": {"status": "verified"},
        }
        final_path = self.root / f"endpoint-final-{tier}.json"
        write_json(final_path, final)
        return final_path

    def provenance(self, mode: str) -> dict[str, Any]:
        candidate_metrics: dict[str, Any]
        if mode == "cached":
            candidate_metrics = {
                "mode": "cache",
                "cache_identity": {
                    "cache_sha256": self.shared["corpus_cache"]["sha256"],
                    "source_sha256": self.shared["documents"]["sha256"],
                    "fingerprint": "synthetic-fingerprint",
                    "dimensions": self.dimensions,
                    "count": self.corpus_count,
                    "model_sha256": self.shared["model"]["sha256"],
                },
            }
        else:
            candidate_metrics = {
                "mode": "proxy",
                "upstream": "http://127.0.0.1:18080",
            }
        return {
            "leann_cpp": {"commit": self.leann_commit},
            "official_leann": {"commit": self.official_commit},
            "official_candidate_recompute_mode": mode,
            "tooling": [],
            "environment": {
                "platform": "synthetic-os",
                "machine": "synthetic-machine",
                "processor": "synthetic-cpu",
                "python": "synthetic-python",
                "numpy": "synthetic-numpy",
            },
            "server_context": {
                "server_ctx_size": 32768,
                "parallel": 16,
                "per_slot_ctx": 2048,
                "required_product": 32768,
            },
            "embedding_endpoint": "http://127.0.0.1:18080",
            "proxy_endpoint": (
                "http://127.0.0.1:18082"
                if mode == "cached"
                else "http://127.0.0.1:18081"
            ),
            "embedding_endpoint_observation": {
                "/props": {
                    "status": 200,
                    "payload": {
                        "model_path": str(self.model.resolve()),
                        "total_slots": 16,
                        "default_generation_settings": {"n_ctx": 2048},
                    },
                }
            },
            "candidate_endpoint_observation": {
                "/metrics": {"status": 200, "payload": candidate_metrics}
            },
            "official_runtime": self.official_runtime,
        }

    def stage(
        self,
        name: str,
        *,
        inputs: dict[str, Any],
        measurements: list[dict[str, Any]],
    ) -> dict[str, Any]:
        combined_inputs = {
            "shared": self.shared,
            "execution_protocol": {"warmups": 0, "repetitions": 1},
            **inputs,
        }
        return {
            "schema": "leann-command-stage-v1",
            "name": name,
            "status": "complete",
            "input_hash": canonical_hash(combined_inputs),
            "inputs": combined_inputs,
            "warmups": [],
            "measurements": measurements,
            "failed_attempts": [],
        }

    def observation(
        self,
        *,
        command: list[str],
        artifacts: list[dict[str, Any]],
        result_path: Path | None = None,
        result: dict[str, Any] | None = None,
        parsed: dict[str, Any] | None = None,
        wall: float = 1.0,
    ) -> dict[str, Any]:
        value: dict[str, Any] = {
            "command": command,
            "exit_code": 0,
            "wall_seconds": wall,
            "peak_rss_bytes": 1024,
            "kind": "measurements",
            "ordinal": 0,
            "artifacts": artifacts,
        }
        if result_path is not None:
            value["result_file"] = snapshot(result_path)
        if result is not None:
            value["result"] = result
        if parsed is not None:
            value["parsed_stdout"] = parsed
        if command and command[0] == "leann":
            value["executable"] = snapshot(self.native_binary)
        elif command and Path(command[0]).is_file():
            value["executable"] = snapshot(Path(command[0]))
        return value

    def native_manifest(self) -> Path:
        output = self.root / "native"
        stages = output / "stages"
        raw = output / "raw-results"
        stages.mkdir(parents=True)
        raw.mkdir()
        index = self.root / "native.leann"
        documents = self.root / "native.docs"
        index.write_bytes(b"i" * 12)
        documents.write_bytes(b"d" * 30)
        artifacts = [snapshot(documents), snapshot(index)]
        build_command = [
            "leann",
            "build",
            "--graph-degree",
            "32",
            "--ef-construction",
            "200",
        ]
        build = self.stage(
            "native-build",
            inputs={
                "binary_sha256": digest(self.native_binary),
                "command": build_command,
            },
            measurements=[
                self.observation(
                    command=build_command, artifacts=artifacts, wall=2.0
                )
            ],
        )
        stats = self.stage(
            "native-stats",
            inputs={
                "binary_sha256": digest(self.native_binary),
                "command": ["leann", "stats"],
            },
            measurements=[
                self.observation(
                    command=["leann", "stats"],
                    artifacts=artifacts,
                    parsed={
                        "nodes": self.corpus_count,
                        "dimension": self.dimensions,
                        "index_bytes": 12,
                        "dense_vector_bytes_avoided": 160,
                        "approximation": "pq",
                    },
                )
            ],
        )
        self.native_csv = raw / "native-search-ef32-measurements-0.csv"
        self.native_csv.write_text(
            "query_index,recall,latency_ms,exact_recomputations,"
            "approximate_distances,upper_layer_hops,embedding_batches,"
            "result_ids\n"
            "0,1,10,32,100,2,2,0 1 2\n"
            "1,1,20,32,120,4,2,3 4 5\n",
            encoding="utf-8",
        )
        search_command = [
            "leann",
            "bench",
            "--ef-search",
            "32",
            "--recompute-batch",
            "16",
        ]
        search = self.stage(
            "native-search-ef32",
            inputs={
                "binary_sha256": digest(self.native_binary),
                "ef_search": 32,
                "command": search_command,
            },
            measurements=[
                self.observation(
                    command=search_command,
                    artifacts=artifacts,
                    result_path=self.native_csv,
                    parsed={
                        "queries": 2,
                        "recall_at_3": 1.0,
                        "latency_ms_mean": 15.0,
                        "latency_ms_p50": 10.0,
                        "latency_ms_p95": 20.0,
                        "exact_recomputations_mean": 32.0,
                        "approximate_distances_mean": 110.0,
                        "upper_layer_hops_mean": 3.0,
                    },
                )
            ],
        )
        native_csv_64 = raw / "native-search-ef64-measurements-0.csv"
        native_csv_64.write_text(
            "query_index,recall,latency_ms,exact_recomputations,"
            "approximate_distances,upper_layer_hops,embedding_batches,"
            "result_ids\n"
            "0,1,12,64,140,2,4,0 1 2\n"
            "1,1,24,64,160,4,4,3 4 5\n",
            encoding="utf-8",
        )
        search_command_64 = [
            "leann",
            "bench",
            "--ef-search",
            "64",
            "--recompute-batch",
            "16",
        ]
        search_64 = self.stage(
            "native-search-ef64",
            inputs={
                "binary_sha256": digest(self.native_binary),
                "ef_search": 64,
                "command": search_command_64,
            },
            measurements=[
                self.observation(
                    command=search_command_64,
                    artifacts=artifacts,
                    result_path=native_csv_64,
                    parsed={
                        "queries": 2,
                        "recall_at_3": 1.0,
                        "latency_ms_mean": 18.0,
                        "latency_ms_p50": 12.0,
                        "latency_ms_p95": 24.0,
                        "exact_recomputations_mean": 64.0,
                        "approximate_distances_mean": 150.0,
                        "upper_layer_hops_mean": 3.0,
                    },
                )
            ],
        )
        stage_values = {
            "native-build": build,
            "native-search-ef32": search,
            "native-search-ef64": search_64,
            "native-stats": stats,
        }
        stage_paths: dict[str, str] = {}
        for name, value in stage_values.items():
            path = stages / f"{name}.json"
            write_json(path, value)
            stage_paths[name] = str(path.resolve())
        manifest = {
            "schema": "leann-large-scale-benchmark-v1",
            "run_role": "native-sweep",
            "status": "complete",
            "created_at": "2026-01-01T00:00:00+00:00",
            "completed_at": "2026-01-01T00:01:00+00:00",
            "shared_artifacts": self.shared,
            "provenance": self.provenance("cached"),
            "protocol": self.protocol("native-sweep"),
            "planned_stages": list(stage_values),
            "stages": stage_paths,
        }
        manifest_path = output / "manifest.json"
        write_json(manifest_path, manifest)
        return manifest_path

    def official_report(
        self, mode: str, *, include_benchmark: bool
    ) -> dict[str, Any]:
        storage = {
            "files": {
                "fixture.index": 10,
                "fixture.ids.txt": 2,
                "fixture.passages.idx": 3,
                "fixture.meta.json": 4,
                "fixture.passages.jsonl": 20,
            },
            "vector_index_bytes": 10,
            "lookup_aux_bytes": 9,
            "vector_serving_bytes": 19,
            "text_store_bytes": 20,
            "total_bytes": 39,
        }
        report: dict[str, Any] = {
            "schema": "leann.cpp-official-comparison-v1",
            "official": {
                "commit": self.official_commit,
                "runtime": {
                    key: value
                    for key, value in self.official_runtime.items()
                    if key not in {"launcher", "launcher_resolved"}
                },
                "config": {
                    "M": 32,
                    "efConstruction": 200,
                    "is_compact": True,
                    "is_recompute": True,
                },
            },
            "dataset": {
                "documents": self.corpus_count,
                "queries": self.query_count,
            },
            "embedding_cache": {
                "sha256": self.shared["corpus_cache"]["sha256"]
            },
            "build": {
                "elapsed_seconds_excluding_embedding": 0.5,
                "storage": storage,
            },
        }
        if include_benchmark:
            complexity = 64
            batch_size = 16
            report["benchmark"] = {
                "candidate_recompute_mode": mode,
                "query_count": 2,
                "query_indices": [0, 1],
                "query_embedding_cache_sha256": self.shared["query_cache"][
                    "sha256"
                ],
                "corpus_embedding_cache_sha256": self.shared[
                    "corpus_cache"
                ]["sha256"],
                "truth": {"sha256": digest(self.truth)},
                "points": [
                    {
                        "complexity": complexity,
                        "batch_size": batch_size,
                        "queries": 2,
                        "latency_ms": {
                            "mean": 15.0,
                            "p50": 15.0,
                            "p95": 19.5,
                            "min": 10.0,
                            "max": 20.0,
                            "raw_per_query": [10.0, 20.0],
                        },
                        "candidate_embedding_requests": 2,
                        "candidate_embeddings_total": 64,
                        "candidate_embeddings_mean_per_query": 32.0,
                        "embedding_proxy_metrics": {
                            "mode": "cache" if mode == "cached" else "proxy",
                            "requests": 2,
                            "inputs": 64,
                            "http_errors": 0,
                            "network_errors": 0,
                            **(
                                {
                                    "cache_identity": {
                                        "cache_sha256": self.shared[
                                            "corpus_cache"
                                        ]["sha256"],
                                        "source_sha256": self.shared[
                                            "documents"
                                        ]["sha256"],
                                        "fingerprint": "synthetic-fingerprint",
                                        "dimensions": self.dimensions,
                                        "count": self.corpus_count,
                                        "model_sha256": self.shared["model"][
                                            "sha256"
                                        ],
                                    }
                                }
                                if mode == "cached"
                                else {}
                            ),
                        },
                        "result_ids": [[0, 1, 2], [3, 4, 5]],
                        "recall_at_3": 1.0,
                    }
                ],
            }
        return report

    def official_manifest(
        self,
        mode: str,
        *,
        include_build: bool = True,
        exact_artifacts: bool = True,
    ) -> Path:
        output = self.root / f"official-{mode}"
        stages = output / "stages"
        raw = output / "raw-results"
        artifacts_root = output / "index"
        stages.mkdir(parents=True)
        raw.mkdir()
        artifacts_root.mkdir()
        artifact_sizes = {
            "fixture.index": 10,
            "fixture.ids.txt": 2,
            "fixture.passages.idx": 3,
            "fixture.meta.json": 4,
            "fixture.passages.jsonl": 20,
        }
        artifact_snapshots: list[dict[str, Any]] = []
        for name, size in artifact_sizes.items():
            path = artifacts_root / name
            fill = name[:1].encode("ascii") if exact_artifacts else b"z"
            path.write_bytes(fill * size)
            artifact_snapshots.append(snapshot(path))

        benchmark_result = self.official_report(mode, include_benchmark=True)
        benchmark_result_path = raw / "official-search-measurements-0.json"
        write_json(benchmark_result_path, benchmark_result)
        stage_values: list[tuple[str, dict[str, Any]]] = []
        if include_build:
            build_command = [str(self.official_python.resolve()), "--build"]
            build_result = self.official_report(mode, include_benchmark=False)
            build_result_path = raw / "official-build-measurements-0.json"
            write_json(build_result_path, build_result)
            build = self.stage(
                "official-build",
                inputs={
                    "candidate_recompute_mode": mode,
                    "official_runtime": self.official_runtime,
                    "python": snapshot(self.official_python),
                    "command_template": build_command,
                },
                measurements=[
                    self.observation(
                        command=build_command,
                        artifacts=artifact_snapshots,
                        result_path=build_result_path,
                        result=build_result,
                        wall=3.0,
                    )
                ],
            )
            stage_values.append(("official-build", build))
        search_name = f"official-search-{mode}"
        search_command = [str(self.official_python.resolve()), "--benchmark"]
        search = self.stage(
            search_name,
            inputs={
                "candidate_recompute_mode": mode,
                "official_runtime": self.official_runtime,
                "python": snapshot(self.official_python),
                "command_template": search_command,
            },
            measurements=[
                self.observation(
                    command=search_command,
                    artifacts=artifact_snapshots,
                    result_path=benchmark_result_path,
                    result=benchmark_result,
                    wall=4.0,
                )
            ],
        )
        stage_values.append((search_name, search))
        stage_paths: dict[str, str] = {}
        for name, value in stage_values:
            path = stages / f"{name}.json"
            write_json(path, value)
            stage_paths[name] = str(path.resolve())
        provenance = self.provenance(mode)
        if mode == "real" and not include_build:
            cached_manifest_path = (
                self.root / "official-cached" / "manifest.json"
            )
            if not cached_manifest_path.is_file():
                raise AssertionError(
                    "synthetic search-only real run requires cached build first"
                )
            cached_manifest = json.loads(
                cached_manifest_path.read_text(encoding="utf-8")
            )
            cached_stage_path = Path(
                cached_manifest["stages"]["official-build"]
            )
            cached_stage = json.loads(
                cached_stage_path.read_text(encoding="utf-8")
            )
            cached_artifacts = cached_stage["measurements"][-1]["artifacts"]
            cached_result = cached_stage["measurements"][-1]["result"]
            cached_storage = cached_result["build"]["storage"]
            storage_identity = canonical_hash(
                {
                    "files": cached_storage["files"],
                    **{
                        key: cached_storage[key]
                        for key in (
                            "vector_index_bytes",
                            "lookup_aux_bytes",
                            "vector_serving_bytes",
                            "text_store_bytes",
                            "total_bytes",
                        )
                    },
                }
            )
            portable = sorted(
                [
                    {
                        "name": Path(item["path"]).name,
                        "size_bytes": item["size_bytes"],
                        "sha256": item["sha256"],
                    }
                    for item in cached_artifacts
                ],
                key=lambda item: item["name"],
            )
            provenance["official_index_reuse"] = {
                "schema": "leann-official-index-reuse-v1",
                "source_manifest": {
                    **snapshot(cached_manifest_path),
                    "completed_at": cached_manifest["completed_at"],
                },
                "source_stage": {
                    **snapshot(cached_stage_path),
                    "input_hash": cached_stage["input_hash"],
                    "measurement_ordinal": 0,
                },
                "official_index": str(
                    (self.root / "official-cached" / "index" / "fixture.leann")
                    .resolve()
                ),
                "artifact_set_sha256": canonical_hash(portable),
                "storage_identity_sha256": storage_identity,
                "artifacts": cached_artifacts,
                "identity_scope": (
                    "artifact basename, byte size, and SHA-256"
                ),
                "build_metrics": (
                    "Inherited from source official-build; construction is "
                    "not rerun by this real-mode search-only runner."
                ),
            }
            real_search_stage = dict(stage_values)[
                "official-search-real"
            ]
            real_search_stage["inputs"]["index_reuse"] = provenance[
                "official_index_reuse"
            ]
            real_search_stage["input_hash"] = canonical_hash(
                real_search_stage["inputs"]
            )
            write_json(
                Path(stage_paths["official-search-real"]),
                real_search_stage,
            )
        run_role = (
            "official-cached-sweep"
            if mode == "cached"
            else (
                "official-real-run"
                if include_build
                else "official-real-search-reuse"
            )
        )
        manifest = {
            "schema": "leann-large-scale-benchmark-v1",
            "run_role": run_role,
            "status": "complete",
            "created_at": "2026-01-01T00:00:00+00:00",
            "completed_at": "2026-01-01T00:02:00+00:00",
            "shared_artifacts": self.shared,
            "provenance": provenance,
            "protocol": self.protocol(run_role),
            "planned_stages": [name for name, _ in stage_values],
            "stages": stage_paths,
        }
        manifest_path = output / "manifest.json"
        write_json(manifest_path, manifest)
        return manifest_path


class CollectLargeScaleResultsTest(unittest.TestCase):
    def test_complete_matrix_is_deterministic_and_inspectable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            manifests = [
                ("tiny", fixture.native_manifest()),
                ("tiny", fixture.official_manifest("cached")),
                (
                    "tiny",
                    fixture.official_manifest("real", include_build=False),
                ),
            ]
            options = {
                "parity_specs": [("tiny", fixture.parity())],
                "required_tiers": ["tiny"],
                "required_profiles": [
                    "native",
                    "official-cached",
                    "official-real",
                ],
                "required_native_efs": [32],
                "required_cached_complexities": [64],
                "required_cached_batches": [16],
                "min_real_points": 1,
            }
            first = collect_results(manifests, **options)
            second = collect_results(list(reversed(manifests)), **options)
            self.assertEqual(first, second)
            self.assertEqual(first["status"], "complete")
            self.assertEqual(len(first["points"]), 4)
            self.assertEqual(len(first["builds"]), 2)
            native = next(
                point
                for point in first["points"]
                if point["system"] == "leann.cpp"
                and point["search_value"] == 32
            )
            self.assertEqual(native["latency_ms"]["p50"], 15.0)
            self.assertEqual(
                native["candidate_embeddings_mean_per_query"], 32.0
            )
            cached = next(
                point
                for point in first["points"]
                if point["recompute_mode"] == "cached"
            )
            self.assertFalse(cached["latency_comparable"])
            real = next(
                point
                for point in first["points"]
                if point["system"] == "official LEANN"
                and point["recompute_mode"] == "real"
            )
            self.assertEqual(
                real["index_build_provenance"]["statuses"], ["inherited"]
            )
            official_build = next(
                build
                for build in first["builds"]
                if build["system"] == "official LEANN"
            )
            self.assertEqual(
                official_build["measured_profiles"], ["official-cached"]
            )
            self.assertEqual(
                official_build["inherited_by_profiles"], ["official-real"]
            )
            self.assertEqual(official_build["repetitions"], 1)
            self.assertEqual(
                first["matched_comparisons"][0]["status"], "complete"
            )
            self.assertEqual(
                first["dataset_derivation"]["queries"]["selected"], 2
            )

            prefix = Path(directory) / "result"
            paths = write_outputs(first, prefix)
            initial_bytes = {key: path.read_bytes() for key, path in paths.items()}
            committed = validate_output_commit(paths["commit"])
            self.assertEqual(committed["report_id"], first["report_id"])
            write_outputs(first, prefix)
            self.assertEqual(
                initial_bytes, {key: path.read_bytes() for key, path in paths.items()}
            )
            rows = list(csv.DictReader(io.StringIO(report_csv(first))))
            self.assertEqual(len(rows), 4)
            markdown = report_markdown(first)
            self.assertIn("Status: **COMPLETE**", markdown)
            self.assertIn(
                "Cached mode isolates the official graph/candidate Pareto frontier",
                markdown,
            )
            self.assertIn("Derived benchmark scope and limitations", markdown)
            self.assertIn(
                "not end-to-end RAG answer quality", markdown
            )
            self.assertIn(
                "runner summaries alone are not trusted", markdown
            )

    def test_output_commit_survives_interrupted_new_generation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            manifests = [
                ("tiny", fixture.native_manifest()),
                ("tiny", fixture.official_manifest("cached")),
                (
                    "tiny",
                    fixture.official_manifest("real", include_build=False),
                ),
            ]
            options = {
                "parity_specs": [("tiny", fixture.parity())],
                "required_tiers": ["tiny"],
                "required_profiles": [
                    "native",
                    "official-cached",
                    "official-real",
                ],
                "required_native_efs": [32],
                "required_cached_complexities": [64],
                "required_cached_batches": [16],
                "min_real_points": 1,
            }
            first = collect_results(manifests, **options)
            prefix = Path(directory) / "transactional-result"
            first_paths = write_outputs(first, prefix)
            marker_bytes = first_paths["commit"].read_bytes()

            second = copy.deepcopy(first)
            second["measurement_policy"]["storage"] += (
                "; interrupted-generation-test"
            )
            second.pop("report_id")
            second["report_id"] = canonical_hash(second)
            original_atomic_write = collector_module._atomic_write

            def interrupted_write(path: Path, payload: str) -> None:
                if (
                    second["report_id"] in path.name
                    and path.suffix == ".csv"
                ):
                    raise OSError("synthetic interrupted publication")
                original_atomic_write(path, payload)

            with mock.patch.object(
                collector_module,
                "_atomic_write",
                side_effect=interrupted_write,
            ):
                with self.assertRaisesRegex(
                    OSError, "synthetic interrupted publication"
                ):
                    write_outputs(second, prefix)
            self.assertEqual(
                first_paths["commit"].read_bytes(), marker_bytes
            )
            committed = validate_output_commit(first_paths["commit"])
            self.assertEqual(committed["report_id"], first["report_id"])

    def test_incomplete_matrix_is_explicit_or_fails_strictly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            manifests = [("tiny", fixture.native_manifest())]
            options = {
                "parity_specs": [("tiny", fixture.parity())],
                "required_tiers": ["tiny"],
                "required_profiles": ["native", "official-cached", "official-real"],
                "required_native_efs": [32],
                "required_cached_complexities": [64],
                "required_cached_batches": [16],
                "min_real_points": 1,
            }
            with self.assertRaisesRegex(
                CollectionError, "benchmark matrix is incomplete"
            ):
                collect_results(manifests, **options)
            report = collect_results(
                manifests, allow_incomplete=True, **options
            )
            self.assertEqual(report["status"], "incomplete")
            missing_profiles = {
                issue["profile"]
                for issue in report["validation"]["incomplete_issues"]
                if issue["code"] == "missing-profile"
            }
            self.assertEqual(
                missing_profiles, {"official-cached", "official-real"}
            )
            self.assertIn("no values were estimated", report_markdown(report))

    def test_changed_raw_result_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            manifest = fixture.native_manifest()
            assert fixture.native_csv is not None
            payload = fixture.native_csv.read_text(encoding="utf-8")
            fixture.native_csv.write_text(
                payload.replace("0,1,10,32", "0,1,11,32"),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(CollectionError, "SHA-256 changed"):
                collect_results(
                    [("tiny", manifest)],
                    parity_specs=[("tiny", fixture.parity())],
                    required_tiers=["tiny"],
                    required_profiles=["native"],
                    required_native_efs=[32],
                    required_cached_complexities=[64],
                    required_cached_batches=[16],
                    min_real_points=1,
                )

    def test_changed_shared_input_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            manifest = fixture.native_manifest()
            fixture.documents.write_text(
                fixture.documents.read_text(encoding="utf-8") + "changed\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                CollectionError, "shared documents: size changed"
            ):
                collect_results(
                    [("tiny", manifest)],
                    parity_specs=[("tiny", fixture.parity())],
                    required_tiers=["tiny"],
                    required_profiles=["native"],
                    required_native_efs=[32],
                    required_cached_complexities=[64],
                    required_cached_batches=[16],
                )

    def test_nonfinite_cache_vector_is_rejected_from_bound_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            with fixture.corpus_cache.open("r+b") as cache:
                cache.seek(fixture.shared["corpus_cache"]["vector_offset"])
                cache.write(np.float32(np.nan).tobytes())
            fixture.shared["corpus_cache"]["sha256"] = digest(
                fixture.corpus_cache
            )
            manifest = fixture.native_manifest()
            with self.assertRaisesRegex(
                CollectionError, "NaN or infinity"
            ):
                collect_results(
                    [("tiny", manifest)],
                    required_tiers=["tiny"],
                    required_profiles=["native"],
                    required_native_efs=[32],
                    required_cached_complexities=[64],
                    required_cached_batches=[16],
                    require_parity=False,
                )

    def test_declared_cache_header_is_parsed_not_trusted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            fixture.shared["corpus_cache"]["vector_offset"] += 1
            manifest = fixture.native_manifest()
            with self.assertRaisesRegex(
                CollectionError,
                "declared vector_offset differs from parsed LEANNBC2 header",
            ):
                collect_results(
                    [("tiny", manifest)],
                    required_tiers=["tiny"],
                    required_profiles=["native"],
                    required_native_efs=[32],
                    required_cached_complexities=[64],
                    required_cached_batches=[16],
                    require_parity=False,
                )

    def test_cache_source_sha_binding_is_recomputed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            payload = fixture.documents.read_bytes()
            replacement = (
                b"D" if payload[:1] != b"D" else b"E"
            ) + payload[1:]
            fixture.documents.write_bytes(replacement)
            fixture.shared["documents"] = {
                **snapshot(fixture.documents),
                "nonempty_lines": fixture.corpus_count,
            }
            manifest = fixture.native_manifest()
            with self.assertRaisesRegex(
                CollectionError, "source SHA-256 binding differs"
            ):
                collect_results(
                    [("tiny", manifest)],
                    required_tiers=["tiny"],
                    required_profiles=["native"],
                    required_native_efs=[32],
                    required_cached_complexities=[64],
                    required_cached_batches=[16],
                    require_parity=False,
                )

    def test_declared_source_line_count_is_recomputed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            fixture.documents.write_text(
                fixture.documents.read_text(encoding="utf-8") + "extra\n",
                encoding="utf-8",
            )
            fixture.shared["documents"] = {
                **snapshot(fixture.documents),
                "nonempty_lines": fixture.corpus_count,
            }
            manifest = fixture.native_manifest()
            with self.assertRaisesRegex(
                CollectionError, "manifest declares"
            ):
                collect_results(
                    [("tiny", manifest)],
                    required_tiers=["tiny"],
                    required_profiles=["native"],
                    required_native_efs=[32],
                    required_cached_complexities=[64],
                    required_cached_batches=[16],
                    require_parity=False,
                )

    def test_declared_100k_tier_requires_exact_cardinality(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            manifest = fixture.native_manifest()
            with self.assertRaisesRegex(
                CollectionError, "must contain exactly 100,000 documents"
            ):
                collect_results(
                    [("100k", manifest)],
                    required_tiers=["tiny"],
                    required_profiles=["native"],
                    required_native_efs=[32],
                    required_cached_complexities=[64],
                    required_cached_batches=[16],
                    require_parity=False,
                )

    def test_10k_tier_is_outside_declared_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            manifest = fixture.native_manifest()
            with self.assertRaisesRegex(
                CollectionError, "outside the declared 100k/1m matrix"
            ):
                collect_results(
                    [("10k", manifest)],
                    required_tiers=["tiny"],
                    required_profiles=["native"],
                    required_native_efs=[32],
                    required_cached_complexities=[64],
                    required_cached_batches=[16],
                    require_parity=False,
                )

    def test_declared_tier_prefix_is_verified_from_exact_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            small_documents = root / "documents-100k.txt"
            large_documents = root / "documents-1m.txt"
            small_documents.write_bytes(b"prefix-row\n")
            large_documents.write_bytes(
                small_documents.read_bytes() + b"larger-row\n"
            )
            small_vectors = root / "corpus-100k.bin"
            large_vectors = root / "corpus-1m.bin"
            vector_prefix = b"\0" * (100_000 * 4)
            small_vectors.write_bytes(vector_prefix)
            large_vectors.write_bytes(vector_prefix + b"\1" * 4)
            common = {
                "query_count": 2,
                "dimensions": 1,
                "queries_sha256": "1" * 64,
                "query_cache_sha256": "2" * 64,
                "model_sha256": "3" * 64,
                "model_artifact_sha256": "4" * 64,
                "fingerprint": "same-fingerprint",
            }
            smaller = {
                **common,
                "corpus_count": 100_000,
                "documents_path": str(small_documents),
                "documents_sha256": digest(small_documents),
                "raw_text_file_bytes": small_documents.stat().st_size,
                "corpus_cache_path": str(small_vectors),
                "corpus_cache_header": {
                    "dimensions": 1,
                    "fingerprint": "same-fingerprint",
                    "vector_offset": 0,
                    "vector_bytes": len(vector_prefix),
                },
            }
            larger = {
                **common,
                "corpus_count": 1_000_000,
                "documents_path": str(large_documents),
                "documents_sha256": digest(large_documents),
                "raw_text_file_bytes": large_documents.stat().st_size,
                "corpus_cache_path": str(large_vectors),
                "corpus_cache_header": {
                    "dimensions": 1,
                    "fingerprint": "same-fingerprint",
                    "vector_offset": 0,
                    "vector_bytes": 1_000_000 * 4,
                },
            }
            collector = Collector()
            collector.datasets = {"100k": smaller, "1m": larger}
            collector.validate_declared_tier_prefix()
            self.assertTrue(
                collector.tier_prefix_identity[
                    "vectors_exact_bitwise_prefix"
                ]
            )
            with large_vectors.open("r+b") as target:
                target.seek(0)
                target.write(b"\2")
            changed = Collector()
            changed.datasets = {"100k": smaller, "1m": larger}
            with self.assertRaisesRegex(
                CollectionError, "exact bitwise prefix"
            ):
                changed.validate_declared_tier_prefix()

    def test_native_recall_is_recomputed_from_ranked_result_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            manifest_path = fixture.native_manifest()
            assert fixture.native_csv is not None
            payload = fixture.native_csv.read_text(encoding="utf-8")
            fixture.native_csv.write_text(
                payload.replace("0 1 2", "0 1 3", 1),
                encoding="utf-8",
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            stage_path = Path(manifest["stages"]["native-search-ef32"])
            stage = json.loads(stage_path.read_text(encoding="utf-8"))
            stage["measurements"][0]["result_file"] = snapshot(
                fixture.native_csv
            )
            write_json(stage_path, stage)
            with self.assertRaisesRegex(
                CollectionError, "does not match result IDs/truth"
            ):
                collect_results(
                    [("tiny", manifest_path)],
                    parity_specs=[("tiny", fixture.parity())],
                    required_tiers=["tiny"],
                    required_profiles=["native"],
                    required_native_efs=[32],
                    required_cached_complexities=[64],
                    required_cached_batches=[16],
                )

    def test_native_counters_must_be_nonnegative_integers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            manifest_path = fixture.native_manifest()
            assert fixture.native_csv is not None
            payload = fixture.native_csv.read_text(encoding="utf-8")
            fixture.native_csv.write_text(
                payload.replace("0,1,10,32,100", "0,1,10,32.5,100", 1),
                encoding="utf-8",
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            stage_path = Path(manifest["stages"]["native-search-ef32"])
            stage = json.loads(stage_path.read_text(encoding="utf-8"))
            stage["measurements"][0]["result_file"] = snapshot(
                fixture.native_csv
            )
            write_json(stage_path, stage)
            with self.assertRaisesRegex(
                CollectionError, "must be a canonical nonnegative integer"
            ):
                collect_results(
                    [("tiny", manifest_path)],
                    parity_specs=[("tiny", fixture.parity())],
                    required_tiers=["tiny"],
                    required_profiles=["native"],
                    required_native_efs=[32],
                    required_cached_complexities=[64],
                    required_cached_batches=[16],
                )

    def test_native_executable_must_match_stage_binary_sha(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            manifest_path = fixture.native_manifest()
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            stage_path = Path(manifest["stages"]["native-build"])
            stage = json.loads(stage_path.read_text(encoding="utf-8"))
            stage["inputs"]["binary_sha256"] = "0" * 64
            stage["input_hash"] = canonical_hash(stage["inputs"])
            write_json(stage_path, stage)
            with self.assertRaisesRegex(
                CollectionError,
                "executable SHA-256 differs from binary_sha256",
            ):
                collect_results(
                    [("tiny", manifest_path)],
                    parity_specs=[("tiny", fixture.parity())],
                    required_tiers=["tiny"],
                    required_profiles=["native"],
                    required_native_efs=[32],
                    required_cached_complexities=[64],
                    required_cached_batches=[16],
                )

    def test_native_stats_must_bind_the_exact_build_artifact_set(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            manifest_path = fixture.native_manifest()
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            stats_path = Path(manifest["stages"]["native-stats"])
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            other_index = Path(directory) / "other.leann"
            other_documents = Path(directory) / "other.docs"
            other_index.write_bytes(b"x" * 12)
            other_documents.write_bytes(b"y" * 30)
            stats["measurements"][0]["artifacts"] = [
                snapshot(other_documents),
                snapshot(other_index),
            ]
            write_json(stats_path, stats)
            with self.assertRaisesRegex(
                CollectionError, "stats artifact set differs from its build"
            ):
                collect_results(
                    [("tiny", manifest_path)],
                    parity_specs=[("tiny", fixture.parity())],
                    required_tiers=["tiny"],
                    required_profiles=["native"],
                    required_native_efs=[32],
                    required_cached_complexities=[64],
                    required_cached_batches=[16],
                )

    def test_planned_and_recorded_stage_sets_must_match_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            manifest_path = fixture.native_manifest()
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            original = copy.deepcopy(manifest)
            manifest["planned_stages"].remove("native-search-ef64")
            write_json(manifest_path, manifest)
            with self.assertRaisesRegex(
                CollectionError, "planned/recorded stage sets differ"
            ):
                collect_results(
                    [("tiny", manifest_path)],
                    parity_specs=[("tiny", fixture.parity())],
                    required_tiers=["tiny"],
                    required_profiles=["native"],
                    required_native_efs=[32],
                    required_cached_complexities=[64],
                    required_cached_batches=[16],
                )
            original["planned_stages"].append("mystery-stage")
            original["stages"]["mystery-stage"] = original["stages"][
                "native-stats"
            ]
            write_json(manifest_path, original)
            with self.assertRaisesRegex(
                CollectionError, "unrecognized planned stages"
            ):
                collect_results(
                    [("tiny", manifest_path)],
                    parity_specs=[("tiny", fixture.parity())],
                    required_tiers=["tiny"],
                    required_profiles=["native"],
                    required_native_efs=[32],
                    required_cached_complexities=[64],
                    required_cached_batches=[16],
                )

    def test_native_gate_is_excluded_from_publication_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            manifest_path = fixture.native_manifest()
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["planned_stages"].remove("native-search-ef64")
            manifest["stages"].pop("native-search-ef64")
            manifest["run_role"] = "native-gate"
            manifest["protocol"]["run_role"] = "native-gate"
            write_json(manifest_path, manifest)
            report = collect_results(
                [("tiny", manifest_path)],
                parity_specs=[("tiny", fixture.parity())],
                required_tiers=["tiny"],
                required_profiles=["native"],
                required_native_efs=[32],
                required_cached_complexities=[64],
                required_cached_batches=[16],
                allow_incomplete=True,
            )
            self.assertEqual(report["points"], [])
            self.assertEqual(report["builds"], [])
            self.assertIn(
                "native-gate-excluded",
                {
                    warning["code"]
                    for warning in report["validation"]["warnings"]
                },
            )
            self.assertIn(
                "missing-profile",
                {
                    issue["code"]
                    for issue in report["validation"]["incomplete_issues"]
                },
            )
            self.assertIn(
                "native gate runs are diagnostic smoke checks",
                report_markdown(report),
            )

    def test_single_ef_native_sweep_is_not_misclassified_as_a_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            manifest_path = fixture.native_manifest()
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["planned_stages"].remove("native-search-ef64")
            manifest["stages"].pop("native-search-ef64")
            write_json(manifest_path, manifest)
            report = collect_results(
                [("tiny", manifest_path)],
                parity_specs=[("tiny", fixture.parity())],
                required_tiers=["tiny"],
                required_profiles=["native"],
                required_native_efs=[32],
                required_cached_complexities=[64],
                required_cached_batches=[16],
            )
            self.assertEqual(report["status"], "complete")
            self.assertEqual(
                {point["search_value"] for point in report["points"]}, {32}
            )
            self.assertNotIn(
                "native-gate-excluded",
                {
                    warning["code"]
                    for warning in report["validation"]["warnings"]
                },
            )

    def test_stage_rejects_extra_stale_repetitions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            manifest_path = fixture.native_manifest()
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            stage_path = Path(manifest["stages"]["native-search-ef32"])
            stage = json.loads(stage_path.read_text(encoding="utf-8"))
            duplicate = copy.deepcopy(stage["measurements"][0])
            duplicate["ordinal"] = 1
            stage["measurements"].append(duplicate)
            write_json(stage_path, stage)
            with self.assertRaisesRegex(
                CollectionError, "measurement repetition count differs"
            ):
                collect_results(
                    [("tiny", manifest_path)],
                    parity_specs=[("tiny", fixture.parity())],
                    required_tiers=["tiny"],
                    required_profiles=["native"],
                    required_native_efs=[32],
                    required_cached_complexities=[64],
                    required_cached_batches=[16],
                )

    def test_stage_observation_kind_and_ordinal_are_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            manifest_path = fixture.native_manifest()
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            stage_path = Path(manifest["stages"]["native-search-ef32"])
            stage = json.loads(stage_path.read_text(encoding="utf-8"))
            stage["measurements"][0]["kind"] = "warmups"
            write_json(stage_path, stage)
            with self.assertRaisesRegex(
                CollectionError, "mismatched kind/ordinal"
            ):
                collect_results(
                    [("tiny", manifest_path)],
                    parity_specs=[("tiny", fixture.parity())],
                    required_tiers=["tiny"],
                    required_profiles=["native"],
                    required_native_efs=[32],
                    required_cached_complexities=[64],
                    required_cached_batches=[16],
                )

    def test_missing_embedding_parity_blocks_publication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            manifest = fixture.native_manifest()
            options = {
                "required_tiers": ["tiny"],
                "required_profiles": ["native"],
                "required_native_efs": [32],
                "required_cached_complexities": [64],
                "required_cached_batches": [16],
            }
            with self.assertRaisesRegex(
                CollectionError, "missing-embedding-parity"
            ):
                collect_results([("tiny", manifest)], **options)
            report = collect_results(
                [("tiny", manifest)], allow_incomplete=True, **options
            )
            self.assertEqual(report["status"], "incomplete")
            self.assertIsNone(report["dataset_derivation"])

    def test_parity_vector_evidence_is_recomputed_not_trusted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            manifest = fixture.native_manifest()
            parity_path = fixture.parity()
            parity = json.loads(parity_path.read_text(encoding="utf-8"))
            native_cache = Path(parity["native"]["cache"]["path"])
            payload = bytearray(native_cache.read_bytes())
            vector_offset = struct.calcsize("<8sIQI") + len(
                "synthetic-fingerprint".encode("utf-8")
            )
            last_row_offset = vector_offset + (
                (fixture.query_count * fixture.dimensions) * 4
            )
            payload[last_row_offset : last_row_offset + fixture.dimensions * 4] = (
                np.asarray([0.0, 0.0, 0.0, 1.0], dtype="<f4").tobytes()
            )
            native_cache.write_bytes(payload)
            parity["native"]["cache"].update(snapshot(native_cache))
            write_json(parity_path, parity)
            with self.assertRaisesRegex(
                CollectionError, "reported minimum cosine differs"
            ):
                collect_results(
                    [("tiny", manifest)],
                    parity_specs=[("tiny", parity_path)],
                    required_tiers=["tiny"],
                    required_profiles=["native"],
                    required_native_efs=[32],
                    required_cached_complexities=[64],
                    required_cached_batches=[16],
                )

    def test_parity_reported_metrics_must_match_vector_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            manifest = fixture.native_manifest()
            parity_path = fixture.parity()
            parity = json.loads(parity_path.read_text(encoding="utf-8"))
            parity["metrics"]["minimum_cosine_similarity"] = 0.99999
            write_json(parity_path, parity)
            with self.assertRaisesRegex(
                CollectionError, "reported minimum cosine differs"
            ):
                collect_results(
                    [("tiny", manifest)],
                    parity_specs=[("tiny", parity_path)],
                    required_tiers=["tiny"],
                    required_profiles=["native"],
                    required_native_efs=[32],
                    required_cached_complexities=[64],
                    required_cached_batches=[16],
                )

    def test_missing_parity_cache_artifact_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            manifest = fixture.native_manifest()
            parity_path = fixture.parity()
            parity = json.loads(parity_path.read_text(encoding="utf-8"))
            Path(parity["native"]["cache"]["path"]).unlink()
            with self.assertRaisesRegex(CollectionError, "does not exist"):
                collect_results(
                    [("tiny", manifest)],
                    parity_specs=[("tiny", parity_path)],
                    required_tiers=["tiny"],
                    required_profiles=["native"],
                    required_native_efs=[32],
                    required_cached_complexities=[64],
                    required_cached_batches=[16],
                )

    def test_finalized_endpoint_attestation_is_consumed_and_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            manifest = fixture.native_manifest()
            parity = fixture.parity()
            attestation = fixture.endpoint_attestation(parity)
            report = collect_results(
                [("tiny", manifest)],
                parity_specs=[("tiny", parity)],
                endpoint_attestation_specs=[("tiny", attestation)],
                require_endpoint_attestation=True,
                required_tiers=["tiny"],
                required_profiles=["native"],
                required_native_efs=[32],
                required_cached_complexities=[64],
                required_cached_batches=[16],
            )
            self.assertEqual(
                report["embedding_endpoint_attestations"][0]["phase"],
                "finalized",
            )
            self.assertEqual(
                report["embedding_endpoint_attestations"][0]["process_proof"][
                    "status"
                ],
                "verified",
            )
            self.assertIn(
                "Live embedding endpoint attestation", report_markdown(report)
            )

    def test_endpoint_attestation_is_required_and_cannot_forge_parity_binding(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            manifest = fixture.native_manifest()
            parity = fixture.parity()
            options = {
                "parity_specs": [("tiny", parity)],
                "require_endpoint_attestation": True,
                "required_tiers": ["tiny"],
                "required_profiles": ["native"],
                "required_native_efs": [32],
                "required_cached_complexities": [64],
                "required_cached_batches": [16],
            }
            with self.assertRaisesRegex(
                CollectionError, "missing-endpoint-attestation"
            ):
                collect_results([("tiny", manifest)], **options)

            attestation_path = fixture.endpoint_attestation(parity)
            attestation = json.loads(
                attestation_path.read_text(encoding="utf-8")
            )
            attestation["evidence"]["native_parity"][
                "minimum_cosine_similarity"
            ] = 0.99999
            attestation["attestation_id"] = canonical_hash(
                attestation["evidence"]
            )
            write_json(attestation_path, attestation)
            with self.assertRaisesRegex(
                CollectionError, "differs from validated parity"
            ):
                collect_results(
                    [("tiny", manifest)],
                    endpoint_attestation_specs=[("tiny", attestation_path)],
                    **options,
                )

    def test_search_only_reuse_requires_exact_index_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            cached = fixture.official_manifest("cached")
            different_real = fixture.official_manifest(
                "real", include_build=False, exact_artifacts=False
            )
            options = {
                "parity_specs": [("tiny", fixture.parity())],
                "required_tiers": ["tiny"],
                "required_profiles": ["official-cached", "official-real"],
                "required_native_efs": [32],
                "required_cached_complexities": [64],
                "required_cached_batches": [16],
                "min_real_points": 1,
            }
            with self.assertRaisesRegex(
                CollectionError, "differs from its cached-build reuse attestation"
            ):
                collect_results(
                    [("tiny", cached), ("tiny", different_real)], **options
                )

    def test_real_point_requires_exact_cached_result_counterpart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            native = fixture.native_manifest()
            cached = fixture.official_manifest("cached")
            real = fixture.official_manifest("real", include_build=False)
            manifest = json.loads(real.read_text(encoding="utf-8"))
            stage_path = Path(manifest["stages"]["official-search-real"])
            stage = json.loads(stage_path.read_text(encoding="utf-8"))
            measurement = stage["measurements"][0]
            result_path = Path(measurement["result_file"]["path"])
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["benchmark"]["points"][0]["result_ids"] = [
                [1, 0, 2],
                [4, 3, 5],
            ]
            write_json(result_path, result)
            measurement["result"] = result
            measurement["result_file"] = snapshot(result_path)
            write_json(stage_path, stage)
            options = {
                "parity_specs": [("tiny", fixture.parity())],
                "required_tiers": ["tiny"],
                "required_profiles": [
                    "native",
                    "official-cached",
                    "official-real",
                ],
                "required_native_efs": [32],
                "required_cached_complexities": [64],
                "required_cached_batches": [16],
                "min_real_points": 1,
            }
            with self.assertRaisesRegex(
                CollectionError, "real-cached-counterpart-mismatch"
            ):
                collect_results(
                    [("tiny", native), ("tiny", cached), ("tiny", real)],
                    **options,
                )
            report = collect_results(
                [("tiny", native), ("tiny", cached), ("tiny", real)],
                allow_incomplete=True,
                **options,
            )
            self.assertEqual(
                report["matched_comparisons"][0]["status"], "missing"
            )
            self.assertNotIn(
                "ratios", report["matched_comparisons"][0]
            )

    def test_official_runtime_identity_cannot_change_between_modes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            cached = fixture.official_manifest("cached")
            real = fixture.official_manifest("real", include_build=False)
            manifest = json.loads(real.read_text(encoding="utf-8"))
            manifest["provenance"]["official_runtime"] = {
                "identity": "different-runtime"
            }
            write_json(real, manifest)
            with self.assertRaisesRegex(
                CollectionError, "official runtime differs from manifest provenance"
            ):
                collect_results(
                    [("tiny", cached), ("tiny", real)],
                    parity_specs=[("tiny", fixture.parity())],
                    required_tiers=["tiny"],
                    required_profiles=[
                        "official-cached",
                        "official-real",
                    ],
                    required_native_efs=[32],
                    required_cached_complexities=[64],
                    required_cached_batches=[16],
                )

    def test_official_report_runtime_must_match_staged_imports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            manifest_path = fixture.official_manifest("cached")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            stage_path = Path(manifest["stages"]["official-search-cached"])
            stage = json.loads(stage_path.read_text(encoding="utf-8"))
            measurement = stage["measurements"][0]
            result_path = Path(measurement["result_file"]["path"])
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["official"]["runtime"]["python"] = "forged-runtime"
            write_json(result_path, result)
            measurement["result"] = result
            measurement["result_file"] = snapshot(result_path)
            write_json(stage_path, stage)
            with self.assertRaisesRegex(
                CollectionError, "report runtime differs from the staged runtime"
            ):
                collect_results(
                    [("tiny", manifest_path)],
                    parity_specs=[("tiny", fixture.parity())],
                    required_tiers=["tiny"],
                    required_profiles=["official-cached"],
                    required_native_efs=[32],
                    required_cached_complexities=[64],
                    required_cached_batches=[16],
                )

    def test_official_observation_must_execute_staged_python(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            manifest_path = fixture.official_manifest("cached")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            stage_path = Path(manifest["stages"]["official-search-cached"])
            stage = json.loads(stage_path.read_text(encoding="utf-8"))
            other_python = Path(directory) / "other-python"
            other_python.write_bytes(b"different-python")
            measurement = stage["measurements"][0]
            measurement["command"][0] = str(other_python.resolve())
            measurement["executable"] = snapshot(other_python)
            stage["inputs"]["command_template"][0] = str(other_python.resolve())
            stage["input_hash"] = canonical_hash(stage["inputs"])
            write_json(stage_path, stage)
            with self.assertRaisesRegex(
                CollectionError, "observed command did not use the staged Python"
            ):
                collect_results(
                    [("tiny", manifest_path)],
                    parity_specs=[("tiny", fixture.parity())],
                    required_tiers=["tiny"],
                    required_profiles=["official-cached"],
                    required_native_efs=[32],
                    required_cached_complexities=[64],
                    required_cached_batches=[16],
                )

    def test_official_storage_entries_must_match_verified_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            manifest_path = fixture.official_manifest("cached")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            stage_path = Path(manifest["stages"]["official-build"])
            stage = json.loads(stage_path.read_text(encoding="utf-8"))
            measurement = stage["measurements"][0]
            result_path = Path(measurement["result_file"]["path"])
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["build"]["storage"]["files"]["fixture.index"] += 1
            write_json(result_path, result)
            measurement["result"] = result
            measurement["result_file"] = snapshot(result_path)
            write_json(stage_path, stage)
            with self.assertRaisesRegex(
                CollectionError,
                "storage file entries do not exactly match verified",
            ):
                collect_results(
                    [("tiny", manifest_path)],
                    parity_specs=[("tiny", fixture.parity())],
                    required_tiers=["tiny"],
                    required_profiles=["official-cached"],
                    required_native_efs=[32],
                    required_cached_complexities=[64],
                    required_cached_batches=[16],
                )

    def test_official_report_query_count_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            manifest_path = fixture.official_manifest("cached")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            stage_path = Path(
                manifest["stages"]["official-search-cached"]
            )
            stage = json.loads(stage_path.read_text(encoding="utf-8"))
            measurement = stage["measurements"][0]
            result_path = Path(measurement["result_file"]["path"])
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["dataset"]["queries"] = 999
            write_json(result_path, result)
            measurement["result"] = result
            measurement["result_file"] = snapshot(result_path)
            write_json(stage_path, stage)
            with self.assertRaisesRegex(
                CollectionError, "official query count differs"
            ):
                collect_results(
                    [("tiny", manifest_path)],
                    parity_specs=[("tiny", fixture.parity())],
                    required_tiers=["tiny"],
                    required_profiles=["official-cached"],
                    required_native_efs=[32],
                    required_cached_complexities=[64],
                    required_cached_batches=[16],
                )

    def test_separate_run_roles_cannot_pool_one_search_point(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            first = fixture.native_manifest()
            duplicate = Path(directory) / "second-run" / "manifest.json"
            copied = json.loads(first.read_text(encoding="utf-8"))
            copied["completed_at"] = "2026-01-02T00:00:00+00:00"
            write_json(duplicate, copied)
            with self.assertRaisesRegex(
                CollectionError, "cannot be pooled"
            ):
                collect_results(
                    [("tiny", first), ("tiny", duplicate)],
                    parity_specs=[("tiny", fixture.parity())],
                    required_tiers=["tiny"],
                    required_profiles=["native"],
                    required_native_efs=[32],
                    required_cached_complexities=[64],
                    required_cached_batches=[16],
                )


if __name__ == "__main__":
    unittest.main()
