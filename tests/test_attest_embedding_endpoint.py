#!/usr/bin/env python3
"""Unit tests for live llama-server/cache endpoint attestation."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import pathlib
import sys
import tempfile
import time
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import attest_embedding_endpoint as attestation_module  # noqa: E402
from attest_embedding_endpoint import (  # noqa: E402
    AttestationError,
    atomic_write_json,
    canonical_hash,
    capture_attestation,
    finalize_attestation,
)


class AttestationFixture:
    def __init__(self, root: pathlib.Path) -> None:
        self.root = root
        self.model = root / "model.gguf"
        self.model.write_bytes(b"synthetic-gguf")
        old_model_time = time.time() - 300
        os.utime(self.model, (old_model_time, old_model_time))
        self.source = root / "documents.txt"
        self.source.write_text("document one\ndocument two\n", encoding="utf-8")
        self.cache = root / "corpus.leannbc2"
        self.partial = root / "corpus.leannbc2.partial"
        self.partial.write_bytes(b"partial-cache")
        self.checkpoint = root / "corpus.leannbc2.checkpoint.json"
        self.endpoint = "http://127.0.0.1:18080"
        self.build_info = "b1-test-build"
        self.fingerprint = "llama.cpp-v1:test"
        self.dimension = 3
        self.model_sha256 = hashlib.sha256(self.model.read_bytes()).hexdigest()
        artifact = {
            "path": str(self.model.resolve()),
            "size_bytes": self.model.stat().st_size,
            "sha256": self.model_sha256,
        }
        model = {
            "embedding_model": "test-model",
            "native_fingerprint": self.fingerprint,
            "declared_identity": "test identity",
            "artifact": artifact,
        }
        model["sha256"] = canonical_hash(model)
        model["identity_strength"] = "artifact-sha256"
        now = dt.datetime.now(dt.timezone.utc)
        self.checkpoint_payload = {
            "schema": "leann-cache-generation-checkpoint-v1",
            "status": "running",
            "created_at": (now - dt.timedelta(seconds=30)).isoformat(),
            "updated_at": now.isoformat(),
            "source": {
                "path": str(self.source.resolve()),
                "size_bytes": self.source.stat().st_size,
                "sha256": hashlib.sha256(self.source.read_bytes()).hexdigest(),
                "nonempty_lines": 2,
            },
            "cache_path": str(self.cache.resolve()),
            "fingerprint": self.fingerprint,
            "model": model,
            "role": "corpus",
            "bindings": {},
            "embedding_endpoint": self.endpoint + "/v1/embeddings",
            "batch_size": 2,
            "requested_dimension": self.dimension,
            "prefix_cache_sha256": None,
            "input_hash": "a" * 64,
            "dimension": self.dimension,
            "header_size": 100,
            "completed_rows": 1,
            "payload_sha256": "b" * 64,
            "prefix_seed": None,
        }
        self.write_checkpoint()
        process_start = now.timestamp() - 120
        stable_process = {
            "provider": "test-process-proof-v1",
            "pid": 1234,
            "process_start_epoch": process_start,
            "process_start_time": dt.datetime.fromtimestamp(
                process_start, dt.timezone.utc
            ).isoformat(),
            "working_directory": str(root.resolve()),
            "listener": {"host": "127.0.0.1", "port": 18080},
            "command": "llama-server --host 127.0.0.1 --port 18080",
        }
        self.process = {
            **stable_process,
            "identity_sha256": canonical_hash(stable_process),
        }

    def write_checkpoint(self) -> None:
        self.checkpoint.write_text(
            json.dumps(self.checkpoint_payload, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def add_prefix_seed(self) -> pathlib.Path:
        prefix_cache = self.root / "prefix.leannbc2"
        prefix_cache.write_bytes(b"prefix-cache-bytes")
        prefix_cache_sha = hashlib.sha256(prefix_cache.read_bytes()).hexdigest()
        source_prefix = b"document one\n"
        source_prefix_sha = hashlib.sha256(source_prefix).hexdigest()
        prefix_source_path = self.root / "prefix-documents.txt"
        prefix_source_path.write_bytes(source_prefix)
        self.checkpoint_payload["prefix_cache_sha256"] = prefix_cache_sha
        self.checkpoint_payload["prefix_seed"] = {
            "cache": str(prefix_cache.resolve()),
            "cache_sha256": prefix_cache_sha,
            "rows": 1,
            "source_prefix_size_bytes": len(source_prefix),
            "source_prefix_sha256": source_prefix_sha,
            "vectors_copied_bitwise": True,
        }
        self.write_checkpoint()
        sidecar = {
            "schema": "leann-shared-embedding-cache-v1",
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "role": "corpus",
            "cache": {
                "sha256": prefix_cache_sha,
                "count": 1,
                "dimensions": self.dimension,
                "fingerprint": self.fingerprint,
                "source_size_bytes": len(source_prefix),
                "source_sha256": source_prefix_sha,
            },
            "source": {
                "path": str(prefix_source_path.resolve()),
                "size_bytes": len(source_prefix),
                "sha256": source_prefix_sha,
                "nonempty_lines": 1,
            },
            "model": self.checkpoint_payload["model"],
            "generation": {
                "endpoint": self.endpoint + "/v1/embeddings",
            },
        }
        prefix_cache.with_name(prefix_cache.name + ".meta.json").write_text(
            json.dumps(sidecar, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return prefix_cache

    def query(self, url: str, timeout: float) -> dict[str, object]:
        del timeout
        if url.endswith("/health"):
            payload: object = {"status": "ok"}
        elif url.endswith("/props"):
            payload = {
                "model_path": str(self.model.resolve()),
                "model_alias": "test-model",
                "model_ftype": "synthetic",
                "build_info": self.build_info,
            }
        else:
            raise AssertionError(url)
        raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        return {
            "url": url,
            "status": 200,
            "payload_sha256": hashlib.sha256(raw).hexdigest(),
            "payload": payload,
        }

    def process_probe(self, **kwargs: object) -> dict[str, object]:
        self.last_process_kwargs = kwargs
        return dict(self.process)

    def capture(self, **overrides: object) -> dict[str, object]:
        arguments: dict[str, object] = {
            "endpoint_value": self.endpoint,
            "expected_artifact_path": self.model,
            "expected_build_info": self.build_info,
            "expected_pid": 1234,
            "checkpoint_path": self.checkpoint,
            "tier": "synthetic",
            "expected_source_count": 2,
            "timeout": 1.0,
            "require_process_proof": True,
            "process_probe": self.process_probe,
            "json_query": self.query,
        }
        arguments.update(overrides)
        return capture_attestation(**arguments)

    def parity_report(self, capture: dict[str, object]) -> dict[str, object]:
        evidence = capture["evidence"]
        assert isinstance(evidence, dict)
        binding = evidence["required_post_run_native_parity"]
        assert isinstance(binding, dict)
        source = binding["source"]
        assert isinstance(source, dict)
        created = dt.datetime.fromisoformat(str(capture["created_at"]))
        return {
            "schema": "leann-embedding-parity-v1",
            "created_at": (created + dt.timedelta(seconds=1)).isoformat(),
            "input_hash": "c" * 64,
            "passed": True,
            "acceptance": {"minimum_cosine_similarity": 0.9999},
            "metrics": {"minimum_cosine_similarity": 0.99995},
            "dataset": {
                "tier": binding["tier"],
                "documents": {
                    "path": source["path"],
                    "size_bytes": source["size_bytes"],
                    "sha256": source["sha256"],
                    "nonempty_lines": source["nonempty_lines"],
                },
            },
            "official_comparison_reference": {
                "corpus_cache": {
                    "sha256": "d" * 64,
                    "source_sha256": source["sha256"],
                    "source_size_bytes": source["size_bytes"],
                    "count": binding["expected_source_count"],
                    "dimensions": binding["dimension"],
                    "fingerprint": binding["fingerprint"],
                },
                "model": {
                    "sha256": binding["model_descriptor_sha256"],
                    "artifact": {
                        "sha256": binding["model_artifact_sha256"],
                    },
                },
            },
            "native": {
                "binary": {"path": "/tmp/leann", "sha256": "e" * 64},
                "model": {"sha256": binding["model_artifact_sha256"]},
                "cache": {
                    "dimensions": binding["dimension"],
                    "fingerprint": binding["fingerprint"],
                },
            },
        }


class EmbeddingEndpointAttestationTest(unittest.TestCase):
    def test_capture_binds_endpoint_checkpoint_model_source_and_process(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = AttestationFixture(pathlib.Path(temporary))
            report = fixture.capture()
        self.assertEqual(report["phase"], "live-capture")
        self.assertEqual(
            report["attestation_id"], canonical_hash(report["evidence"])
        )
        evidence = report["evidence"]
        self.assertEqual(evidence["process_proof"]["status"], "verified")
        self.assertTrue(
            evidence["process_proof"]["server_started_before_checkpoint"]
        )
        self.assertEqual(
            evidence["required_post_run_native_parity"]["model_artifact_sha256"],
            fixture.model_sha256,
        )
        self.assertEqual(
            evidence["required_post_run_native_parity"]["expected_source_count"],
            2,
        )

    def test_wrong_props_model_or_build_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = AttestationFixture(pathlib.Path(temporary))

            def wrong_model(url: str, timeout: float) -> dict[str, object]:
                result = fixture.query(url, timeout)
                if url.endswith("/props"):
                    result["payload"]["model_path"] = str(
                        (fixture.root / "other.gguf").resolve()
                    )
                return result

            with self.assertRaisesRegex(
                (AttestationError, FileNotFoundError), "model_path|No such file"
            ):
                fixture.capture(json_query=wrong_model)

            with self.assertRaisesRegex(AttestationError, "build_info"):
                fixture.capture(expected_build_info="different-build")

    def test_process_restart_and_late_start_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = AttestationFixture(pathlib.Path(temporary))
            calls = 0

            def restarted(**kwargs: object) -> dict[str, object]:
                nonlocal calls
                del kwargs
                calls += 1
                result = dict(fixture.process)
                if calls == 2:
                    result["identity_sha256"] = "f" * 64
                return result

            with self.assertRaisesRegex(AttestationError, "identity changed"):
                fixture.capture(process_probe=restarted)

            late = dict(fixture.process)
            late["process_start_epoch"] = (
                dt.datetime.now(dt.timezone.utc).timestamp() + 60
            )
            late["identity_sha256"] = canonical_hash(
                {
                    key: value
                    for key, value in late.items()
                    if key != "identity_sha256"
                }
            )
            with self.assertRaisesRegex(
                AttestationError, "did not start before cache generation"
            ):
                fixture.capture(process_probe=lambda **kwargs: dict(late))

    def test_prefix_cache_is_hashed_and_bound_to_same_model_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = AttestationFixture(pathlib.Path(temporary))
            prefix_cache = fixture.add_prefix_seed()
            report = fixture.capture()
            prefix = report["evidence"]["prefix_seed"]
            self.assertEqual(prefix["rows"], 1)
            self.assertTrue(prefix["vectors_copied_bitwise"])
            self.assertEqual(
                prefix["cache"]["sha256"],
                hashlib.sha256(prefix_cache.read_bytes()).hexdigest(),
            )

            prefix_cache.write_bytes(b"tampered")
            with self.assertRaisesRegex(AttestationError, "SHA-256 differs"):
                fixture.capture()

    def test_linux_without_birth_time_uses_proven_prefix_or_fails_closed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = AttestationFixture(pathlib.Path(temporary))
            fixture.checkpoint_payload.pop("created_at")
            fixture.write_checkpoint()
            original_stat_fields = attestation_module._stat_fields

            def without_birth(metadata: os.stat_result) -> dict[str, object]:
                result = original_stat_fields(metadata)
                result["birth_time_epoch"] = None
                result["birth_time"] = None
                return result

            with mock.patch.object(
                attestation_module, "_stat_fields", side_effect=without_birth
            ):
                with self.assertRaisesRegex(
                    AttestationError, "cannot prove cache generation start"
                ):
                    fixture.capture()
                fixture.add_prefix_seed()
                report = fixture.capture()
            self.assertEqual(
                report["evidence"]["process_proof"]["status"], "verified"
            )
            self.assertIn(
                "prefix",
                report["evidence"]["process_proof"][
                    "checkpoint_start_source"
                ],
            )

    def test_finalize_requires_matching_post_capture_native_parity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            fixture = AttestationFixture(root)
            capture = fixture.capture()
            capture_path = root / "capture.json"
            atomic_write_json(capture_path, capture)
            parity = fixture.parity_report(capture)
            parity_path = root / "parity.json"
            atomic_write_json(parity_path, parity)
            final = finalize_attestation(
                capture_path=capture_path,
                parity_path=parity_path,
            )
            self.assertEqual(final["phase"], "finalized")
            self.assertEqual(
                final["post_run_native_parity"]["status"], "verified"
            )

            parity["native"]["model"]["sha256"] = "0" * 64
            atomic_write_json(parity_path, parity)
            with self.assertRaisesRegex(AttestationError, "GGUF differs"):
                finalize_attestation(
                    capture_path=capture_path,
                    parity_path=parity_path,
                )

    def test_finalize_can_bind_the_proven_100k_prefix_of_a_1m_capture(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            fixture = AttestationFixture(root)
            fixture.add_prefix_seed()
            capture = fixture.capture(tier="1m", expected_source_count=2)
            capture_path = root / "capture.json"
            atomic_write_json(capture_path, capture)
            prefix = capture["evidence"]["prefix_seed"]
            parity = fixture.parity_report(capture)
            parity["dataset"]["tier"] = "100k"
            parity["dataset"]["documents"] = {
                key: prefix["source"][key]
                for key in ("path", "size_bytes", "sha256", "nonempty_lines")
            }
            parity["official_comparison_reference"]["corpus_cache"].update(
                {
                    "sha256": prefix["cache"]["sha256"],
                    "source_sha256": prefix["source"]["sha256"],
                    "source_size_bytes": prefix["source"]["size_bytes"],
                    "count": prefix["rows"],
                    "dimensions": prefix["cache_metadata"]["dimensions"],
                    "fingerprint": prefix["cache_metadata"]["fingerprint"],
                }
            )
            parity_path = root / "prefix-parity.json"
            atomic_write_json(parity_path, parity)
            final = finalize_attestation(
                capture_path=capture_path,
                parity_path=parity_path,
            )
            self.assertEqual(final["evidence"]["binding"]["tier"], "100k")
            self.assertEqual(
                final["evidence"]["binding"]["scope"], "prefix-seed"
            )
            self.assertEqual(
                final["evidence"]["binding"]["parent_tier"], "1m"
            )

    def test_atomic_write_preserves_previous_target_when_replace_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "attestation.json"
            path.write_text('{"old":true}\n', encoding="utf-8")
            with mock.patch(
                "attest_embedding_endpoint.os.replace",
                side_effect=OSError("injected replace failure"),
            ):
                with self.assertRaisesRegex(OSError, "injected"):
                    atomic_write_json(path, {"new": True})
            self.assertEqual(path.read_text(encoding="utf-8"), '{"old":true}\n')
            self.assertEqual(
                list(path.parent.glob(f".{path.name}.*.tmp")),
                [],
            )


if __name__ == "__main__":
    unittest.main()
