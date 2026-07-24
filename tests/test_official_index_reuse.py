#!/usr/bin/env python3
"""Tests for the official real search-only index reuse bridge."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_cache import canonical_hash, snapshot_files  # noqa: E402
from run_official_real_search_reuse import (  # noqa: E402
    REUSE_SCHEMA,
    build_manifest,
    validate_reused_official_index,
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


class ReuseFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.shared = {"identity": "exact-shared-artifacts"}
        self.runtime = {"runtime": "exact-official-runtime"}
        self.repo = {"commit": "a" * 40}
        self.tool = root / "tool.py"
        self.tool.write_text("# frozen tool\n", encoding="utf-8")
        self.tooling = snapshot_files([self.tool])
        self.index = root / "index" / "fixture.leann"
        self.index.parent.mkdir()
        self.artifact_a = self.index.parent / "fixture.index"
        self.artifact_b = self.index.parent / "fixture.ids.txt"
        self.artifact_c = (
            self.index.parent / "fixture.leann.passages.idx"
        )
        self.artifact_d = self.index.parent / "fixture.leann.meta.json"
        self.artifact_e = (
            self.index.parent / "fixture.leann.passages.jsonl"
        )
        self.artifact_a.write_bytes(b"index-bytes")
        self.artifact_b.write_bytes(b"0\n1\n")
        self.artifact_c.write_bytes(b"offsets")
        self.artifact_d.write_bytes(b"{}")
        self.artifact_e.write_bytes(b'{"text":"fixture"}\n')
        self.artifacts = snapshot_files(
            [
                self.artifact_a,
                self.artifact_b,
                self.artifact_c,
                self.artifact_d,
                self.artifact_e,
            ]
        )
        files = {
            Path(item["path"]).name: item["size_bytes"]
            for item in self.artifacts
        }
        vector_index = files["fixture.index"]
        lookup = (
            files["fixture.ids.txt"]
            + files["fixture.leann.passages.idx"]
            + files["fixture.leann.meta.json"]
        )
        text_store = files["fixture.leann.passages.jsonl"]
        result_path = root / "cached" / "raw-results" / "build.json"
        write_json(
            result_path,
            {
                "build": {
                    "storage": {
                        "files": files,
                        "vector_index_bytes": vector_index,
                        "lookup_aux_bytes": lookup,
                        "vector_serving_bytes": vector_index + lookup,
                        "text_store_bytes": text_store,
                        "total_bytes": sum(files.values()),
                    }
                }
            },
        )
        inputs = {
            "shared": self.shared,
            "tooling": self.tooling,
            "official_runtime": self.runtime,
            "official_repo": self.repo,
            "candidate_recompute_mode": "cached",
            "execution_protocol": {"warmups": 0, "repetitions": 1},
            "command_template": [
                "python",
                "compare.py",
                "--build",
                "--index",
                str(self.index.resolve()),
                "--output",
                "ignored.json",
            ],
        }
        stage = {
            "schema": "leann-command-stage-v1",
            "name": "official-build",
            "status": "complete",
            "input_hash": canonical_hash(inputs),
            "inputs": inputs,
            "warmups": [],
            "measurements": [
                {
                    "kind": "measurements",
                    "ordinal": 0,
                    "command": inputs["command_template"],
                    "exit_code": 0,
                    "result_file": snapshot_files([result_path])[0],
                    "artifacts": self.artifacts,
                }
            ],
            "failed_attempts": [],
        }
        stage_path = root / "cached" / "stages" / "official-build.json"
        write_json(stage_path, stage)
        manifest = {
            "schema": "leann-large-scale-benchmark-v1",
            "run_role": "official-cached-sweep",
            "status": "complete",
            "completed_at": "2026-01-01T00:00:00+00:00",
            "shared_artifacts": self.shared,
            "provenance": {
                "official_candidate_recompute_mode": "cached",
                "official_leann": self.repo,
                "official_runtime": self.runtime,
                "tooling": self.tooling,
            },
            "protocol": {
                "run_role": "official-cached-sweep",
                "build_warmups": 0,
                "build_repetitions": 1,
            },
            "planned_stages": ["official-build"],
            "stages": {"official-build": str(stage_path.resolve())},
        }
        self.manifest = root / "cached" / "manifest.json"
        write_json(self.manifest, manifest)

    def validate(self) -> dict[str, Any]:
        return validate_reused_official_index(
            cached_manifest_path=self.manifest,
            shared=self.shared,
            current_tooling=self.tooling,
            official_repo_provenance=self.repo,
            official_runtime=self.runtime,
            official_index=self.index,
        )


class OfficialIndexReuseTest(unittest.TestCase):
    def test_exact_cached_artifacts_produce_static_attestation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = ReuseFixture(Path(directory))
            first = fixture.validate()
            second = fixture.validate()
            self.assertEqual(first, second)
            self.assertEqual(first["schema"], REUSE_SCHEMA)
            self.assertEqual(first["artifacts"], fixture.artifacts)
            self.assertIn("not rerun", first["build_metrics"])

            manifest = build_manifest(
                created_at="2026-01-01T00:00:00+00:00",
                status="complete",
                shared=fixture.shared,
                provenance={"official_index_reuse": first},
                protocol={
                    "run_role": "official-real-search-reuse",
                    "search_repetitions": 1,
                },
                stages_dir=Path(directory) / "real" / "stages",
                stage_path=Path(directory) / "real" / "stages" / "official-search-real.json",
            )
            self.assertEqual(
                manifest["planned_stages"], ["official-search-real"]
            )
            self.assertNotIn("official-build", manifest["stages"])
            self.assertEqual(
                manifest["run_role"], "official-real-search-reuse"
            )

    def test_changed_index_is_rejected_before_search(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = ReuseFixture(Path(directory))
            fixture.artifact_a.write_bytes(b"changed----")
            with self.assertRaisesRegex(
                ValueError, "file size or SHA-256 changed"
            ):
                fixture.validate()

    def test_changed_tooling_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = ReuseFixture(Path(directory))
            other_tool = Path(directory) / "current" / "tool.py"
            other_tool.parent.mkdir()
            other_tool.write_text("# changed tool\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "tooling changed"):
                validate_reused_official_index(
                    cached_manifest_path=fixture.manifest,
                    shared=fixture.shared,
                    current_tooling=snapshot_files([other_tool]),
                    official_repo_provenance=fixture.repo,
                    official_runtime=fixture.runtime,
                    official_index=fixture.index,
                )

    def test_cached_stage_plan_must_match_recorded_stages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = ReuseFixture(Path(directory))
            manifest = json.loads(
                fixture.manifest.read_text(encoding="utf-8")
            )
            manifest["planned_stages"].append("official-search-cached")
            write_json(fixture.manifest, manifest)
            with self.assertRaisesRegex(
                ValueError, "planned/recorded stage sets differ"
            ):
                fixture.validate()

    def test_cached_storage_must_match_reused_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = ReuseFixture(Path(directory))
            manifest = json.loads(
                fixture.manifest.read_text(encoding="utf-8")
            )
            stage_path = Path(manifest["stages"]["official-build"])
            stage = json.loads(stage_path.read_text(encoding="utf-8"))
            result_path = Path(
                stage["measurements"][0]["result_file"]["path"]
            )
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["build"]["storage"]["files"]["fixture.index"] += 1
            write_json(result_path, result)
            stage["measurements"][0]["result_file"] = snapshot_files(
                [result_path]
            )[0]
            write_json(stage_path, stage)
            with self.assertRaisesRegex(
                ValueError,
                "storage files differ from verified artifacts",
            ):
                fixture.validate()


if __name__ == "__main__":
    unittest.main()
