#!/usr/bin/env python3
"""Small dependency-free tests for the large-scale benchmark tooling."""

from __future__ import annotations

import hashlib
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_cache import (  # noqa: E402
    MAGIC_V1,
    read_cache,
    read_ground_truth,
    validate_cache_source,
)
from run_large_scale_benchmark import (  # noqa: E402
    benchmark_run_role,
    exact_topk_block,
    generate_cache,
    generate_ground_truth,
    run_repeated_stage,
)
from openai_embedding_proxy import Metrics as ProxyMetrics  # noqa: E402
from compare_official_leann import exact_topk as official_exact_topk  # noqa: E402


class FakeEmbeddings:
    url = "mock://embeddings"

    def embed(
        self, texts: list[str], expected_dimension: int | None
    ) -> np.ndarray:
        vectors = np.array(
            [
                [
                    len(text) + 1,
                    sum(text.encode("utf-8")) % 17 + 1,
                    text.count("a") + 1,
                    1,
                ]
                for text in texts
            ],
            dtype=np.float32,
        )
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        if expected_dimension is not None:
            assert vectors.shape[1] == expected_dimension
        return vectors


class FailAfterOneBatch(FakeEmbeddings):
    def __init__(self) -> None:
        self.calls = 0

    def embed(
        self, texts: list[str], expected_dimension: int | None
    ) -> np.ndarray:
        self.calls += 1
        if self.calls > 1:
            raise RuntimeError("synthetic interruption")
        return super().embed(texts, expected_dimension)


class BenchmarkToolsTest(unittest.TestCase):
    def test_reads_legacy_v1(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.f32"
            fingerprint = b"legacy"
            vectors = np.eye(2, dtype="<f4")
            path.write_bytes(
                struct.pack("<8sIQI", MAGIC_V1, 2, 2, len(fingerprint))
                + fingerprint
                + vectors.tobytes()
            )
            loaded, metadata = read_cache(path)
            self.assertEqual(metadata["version"], 1)
            np.testing.assert_array_equal(loaded, vectors)

    def test_generation_v2_bindings_and_ground_truth(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            documents = root / "documents.txt"
            queries = root / "queries.txt"
            documents.write_text("alpha\nbeta\nalpha\n", encoding="utf-8")
            queries.write_text("alpha?\nbeta?\n", encoding="utf-8")
            corpus_cache = root / "corpus.f32"
            query_cache = root / "queries.f32"
            truth_path = root / "truth.txt"
            model = {
                "embedding_model": "fake",
                "native_fingerprint": "fake-fp",
                "declared_identity": "test",
                "artifact": None,
                "sha256": "a" * 64,
                "identity_strength": "declared",
            }
            client = FakeEmbeddings()
            corpus = generate_cache(
                source_path=documents,
                cache_path=corpus_cache,
                client=client,
                fingerprint="fake-fp",
                model=model,
                role="corpus",
                batch_size=2,
                dimension=4,
                bindings={},
            )
            query = generate_cache(
                source_path=queries,
                cache_path=query_cache,
                client=client,
                fingerprint="fake-fp",
                model=model,
                role="queries",
                batch_size=1,
                dimension=4,
                bindings={
                    "corpus_cache_sha256": corpus["cache"]["sha256"],
                    "corpus_source_sha256": corpus["source"]["sha256"],
                    "model_sha256": model["sha256"],
                },
            )
            self.assertEqual(query["cache"]["count"], 2)
            declared = generate_ground_truth(
                corpus_cache=corpus_cache,
                query_cache=query_cache,
                documents=documents,
                queries=queries,
                output=truth_path,
                top_k=2,
                query_block_rows=1,
                corpus_block_rows=2,
            )
            truth, metadata = read_ground_truth(
                truth_path, expected_queries=2, expected_k=2, expected_corpus=3
            )
            self.assertEqual(declared["ground_truth"]["sha256"], metadata["sha256"])
            self.assertEqual(truth.shape, (2, 2))
            original_truth = truth_path.read_bytes()
            committed = b"".join(original_truth.splitlines(keepends=True)[:2])
            truth_path.unlink()
            truth_path.with_name(truth_path.name + ".meta.json").unlink()
            partial = truth_path.with_name(truth_path.name + ".partial")
            partial.write_bytes(committed + b"uncommitted crash tail\n")
            checkpoint = truth_path.with_name(
                truth_path.name + ".checkpoint.json"
            )
            checkpoint.write_text(
                json.dumps(
                    {
                        "input_hash": declared["input_hash"],
                        "created_at": "2026-01-01T00:00:00+00:00",
                        "status": "running",
                        "completed_queries": 1,
                        "committed_length_bytes": len(committed),
                        "committed_prefix_sha256": hashlib.sha256(
                            committed
                        ).hexdigest(),
                    }
                ),
                encoding="utf-8",
            )
            generate_ground_truth(
                corpus_cache=corpus_cache,
                query_cache=query_cache,
                documents=documents,
                queries=queries,
                output=truth_path,
                top_k=2,
                query_block_rows=1,
                corpus_block_rows=2,
            )
            self.assertEqual(truth_path.read_bytes(), original_truth)
            extended_documents = root / "documents-extended.txt"
            extended_documents.write_bytes(
                documents.read_bytes() + b"gamma\n"
            )
            extended_cache = root / "corpus-extended.f32"
            generate_cache(
                source_path=extended_documents,
                cache_path=extended_cache,
                client=client,
                fingerprint="fake-fp",
                model=model,
                role="corpus",
                batch_size=2,
                dimension=4,
                bindings={},
                prefix_cache=corpus_cache,
            )
            prefix_vectors, _ = read_cache(corpus_cache)
            extended_vectors, _ = read_cache(extended_cache)
            np.testing.assert_array_equal(
                prefix_vectors, extended_vectors[: len(prefix_vectors)]
            )
            _, cache_metadata = read_cache(corpus_cache)
            validate_cache_source(cache_metadata, documents)
            documents.write_text("alpha\nzeta\nalpha\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                validate_cache_source(cache_metadata, documents)

    def test_tie_stable_topk(self) -> None:
        corpus = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
        queries = np.array([[1, 0]], dtype=np.float32)
        truth = exact_topk_block(
            corpus, queries, top_k=1, corpus_block_rows=2
        )
        self.assertEqual(int(truth[0, 0]), 0)
        official_truth = official_exact_topk(
            np.repeat(corpus[:1], 10, axis=0), queries, 3
        )
        np.testing.assert_array_equal(official_truth[0], [0, 1, 2])

    def test_proxy_error_counters_reset(self) -> None:
        metrics = ProxyMetrics("http://upstream")
        metrics.add(7)
        metrics.upstream_started()
        metrics.upstream_finished(503)
        snapshot = metrics.snapshot()
        self.assertEqual(snapshot["http_errors"], 1)
        self.assertEqual(snapshot["status_counts"], {"503": 1})
        previous = metrics.reset()
        self.assertEqual(previous["inputs"], 7)
        self.assertEqual(metrics.snapshot()["http_errors"], 0)

    def test_cache_generation_resumes_from_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "documents.txt"
            source.write_text("one\ntwo\nthree\n", encoding="utf-8")
            cache = root / "corpus.f32"
            model = {
                "embedding_model": "fake",
                "native_fingerprint": "fake-fp",
                "declared_identity": "test",
                "artifact": None,
                "sha256": "b" * 64,
                "identity_strength": "declared",
            }
            arguments = {
                "source_path": source,
                "cache_path": cache,
                "fingerprint": "fake-fp",
                "model": model,
                "role": "corpus",
                "batch_size": 1,
                "dimension": 4,
                "bindings": {},
            }
            with self.assertRaisesRegex(RuntimeError, "interruption"):
                generate_cache(client=FailAfterOneBatch(), **arguments)
            checkpoint = cache.with_name(cache.name + ".checkpoint.json")
            checkpoint_state = json.loads(checkpoint.read_text(encoding="utf-8"))
            generation_started_at = checkpoint_state["created_at"]
            completed = generate_cache(client=FakeEmbeddings(), **arguments)
            self.assertEqual(completed["cache"]["count"], 3)
            self.assertEqual(
                completed["generation"]["started_at"], generation_started_at
            )
            vectors, _ = read_cache(cache)
            self.assertEqual(vectors.shape, (3, 4))

    def test_request_concurrency_produces_identical_cache_bytes(self) -> None:
        model = {
            "embedding_model": "fake",
            "native_fingerprint": "fake-fp",
            "declared_identity": "test",
            "artifact": None,
            "sha256": "b" * 64,
            "identity_strength": "declared",
        }
        digests = []
        payloads = []
        for concurrency in (1, 8):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "documents.txt"
                source.write_text(
                    "".join(f"document number {index}\n" for index in range(64)),
                    encoding="utf-8",
                )
                cache = root / "corpus.f32"
                declared = generate_cache(
                    client=FakeEmbeddings(),
                    source_path=source,
                    cache_path=cache,
                    fingerprint="fake-fp",
                    model=model,
                    role="corpus",
                    batch_size=3,
                    dimension=4,
                    bindings={},
                    concurrency=concurrency,
                )
                self.assertEqual(declared["cache"]["count"], 64)
                self.assertEqual(
                    declared["generation"]["request_concurrency"], concurrency
                )
                digests.append(hashlib.sha256(cache.read_bytes()).hexdigest())
                payloads.append(declared["generation"]["payload_sha256"])
        self.assertEqual(digests[0], digests[1])
        self.assertEqual(payloads[0], payloads[1])

    def test_concurrent_generation_surfaces_endpoint_failures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "documents.txt"
            source.write_text(
                "".join(f"document number {index}\n" for index in range(32)),
                encoding="utf-8",
            )
            cache = root / "corpus.f32"
            with self.assertRaisesRegex(RuntimeError, "interruption"):
                generate_cache(
                    client=FailAfterOneBatch(),
                    source_path=source,
                    cache_path=cache,
                    fingerprint="fake-fp",
                    model={
                        "embedding_model": "fake",
                        "native_fingerprint": "fake-fp",
                        "declared_identity": "test",
                        "artifact": None,
                        "sha256": "b" * 64,
                        "identity_strength": "declared",
                    },
                    role="corpus",
                    batch_size=4,
                    dimension=4,
                    bindings={},
                    concurrency=4,
                )
            self.assertFalse(cache.exists())
            self.assertTrue(
                cache.with_name(cache.name + ".checkpoint.json").exists()
            )

    def test_repeated_stage_reuses_only_exact_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state_dir = root / "stages"
            artifact = root / "index.bin"
            artifact.write_bytes(b"stable artifact")
            calls: list[tuple[str, int]] = []

            def command_factory(
                kind: str, ordinal: int
            ) -> tuple[list[str], None]:
                calls.append((kind, ordinal))
                return [sys.executable, "-c", "pass"], None

            arguments = {
                "name": "native-search-ef32",
                "state_dir": state_dir,
                "input_payload": {
                    "command": [sys.executable, "-c", "pass"],
                },
                "command_factory": command_factory,
                "artifact_factory": lambda: [artifact],
                "cwd": root,
                "warmups": 1,
                "repetitions": 2,
            }
            first = run_repeated_stage(**arguments)
            self.assertEqual(calls, [("warmups", 0), ("measurements", 0), ("measurements", 1)])
            self.assertEqual(
                first["inputs"]["execution_protocol"],
                {"warmups": 1, "repetitions": 2},
            )
            completed_at = first["completed_at"]

            reused = run_repeated_stage(**arguments)
            self.assertEqual(reused["completed_at"], completed_at)
            self.assertEqual(len(calls), 3)

            changed = {**arguments, "repetitions": 3}
            with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                run_repeated_stage(**changed)

            state_path = state_dir / "native-search-ef32.json"
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state["measurements"].append(state["measurements"][-1])
            state_path.write_text(json.dumps(state), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "more measurements"):
                run_repeated_stage(**arguments)

    def test_benchmark_run_role_is_explicit_not_inferred(self) -> None:
        arguments = SimpleNamespace(
            skip_native=False,
            native_role="sweep",
            native_ef_search=[32],
            skip_official=True,
            official_recompute_mode="cached",
        )
        self.assertEqual(benchmark_run_role(arguments), "native-sweep")
        arguments.native_role = "gate"
        self.assertEqual(benchmark_run_role(arguments), "native-gate")
        arguments.native_ef_search = [32, 64]
        with self.assertRaisesRegex(ValueError, "exactly one"):
            benchmark_run_role(arguments)

        arguments.native_role = "sweep"
        arguments.skip_official = False
        self.assertEqual(
            benchmark_run_role(arguments),
            "native-sweep+official-cached-sweep",
        )
        arguments.skip_native = True
        arguments.official_recompute_mode = "real"
        self.assertEqual(benchmark_run_role(arguments), "official-real-run")
        arguments.skip_official = True
        with self.assertRaisesRegex(ValueError, "at least one"):
            benchmark_run_role(arguments)


if __name__ == "__main__":
    unittest.main()
