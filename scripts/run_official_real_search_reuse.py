#!/usr/bin/env python3
"""Run official LEANN real recompute search using an attested cached-sweep index.

This is deliberately separate from ``run_large_scale_benchmark.py`` so an
active cache/ground-truth preparation keeps its frozen script identity.  The
runner refuses to search unless the existing official index exactly matches
the final artifact snapshot of a completed cached-build manifest.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

from benchmark_cache import (
    atomic_write_json,
    canonical_hash,
    sha256_file,
    snapshot_files,
)
from run_large_scale_benchmark import (
    SCHEMA,
    add_data_arguments,
    endpoint_provenance,
    git_provenance,
    load_json,
    normalized_base_url,
    official_command,
    official_runtime_provenance,
    parse_json_array,
    prefix_artifacts,
    run_repeated_stage,
    utc_now,
    validate_endpoint_identity,
    validate_shared_artifacts,
)


REUSE_SCHEMA = "leann-official-index-reuse-v1"
STAGE_SCHEMA = "leann-command-stage-v1"


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context}: expected an object")
    return value


def _sequence(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{context}: expected an array")
    return value


def _stage_path(manifest_path: Path, recorded: Any, name: str) -> Path:
    candidates: list[Path] = []
    if isinstance(recorded, str) and recorded:
        path = Path(recorded)
        candidates.append(path if path.is_absolute() else manifest_path.parent / path)
    candidates.append(manifest_path.parent / "stages" / f"{name}.json")
    for path in candidates:
        if path.is_file():
            return path.resolve()
    return candidates[0].resolve()


def _command_value(command: Any, option: str, context: str) -> str:
    values = _sequence(command, f"{context} command")
    positions = [index for index, value in enumerate(values) if value == option]
    if len(positions) != 1 or positions[0] + 1 >= len(values):
        raise ValueError(f"{context}: missing, repeated, or malformed {option}")
    value = values[positions[0] + 1]
    if not isinstance(value, str):
        raise ValueError(f"{context}: {option} value is not a string")
    return value


def _snapshot_matches(record: Any, context: str) -> dict[str, Any]:
    snapshot = _mapping(record, context)
    path_value = snapshot.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"{context}: missing file path")
    path = Path(path_value)
    current = snapshot_files([path])
    if len(current) != 1 or current[0] != snapshot:
        raise ValueError(f"{context}: file size or SHA-256 changed")
    return snapshot


def _tooling_by_name(tooling: Any, context: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for ordinal, value in enumerate(_sequence(tooling, context)):
        snapshot = _snapshot_matches(value, f"{context} item {ordinal}")
        name = Path(snapshot["path"]).name
        if name in result:
            raise ValueError(f"{context}: duplicate tooling basename {name!r}")
        result[name] = snapshot
    return result


def _portable_artifact_identity(
    artifacts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return sorted(
        (
            {
                "name": Path(item["path"]).name,
                "size_bytes": item["size_bytes"],
                "sha256": item["sha256"],
            }
            for item in artifacts
        ),
        key=lambda item: item["name"],
    )


def _recognized_stage_name(name: str) -> bool:
    return name in {
        "native-build",
        "native-stats",
        "official-build",
        "official-search-cached",
        "official-search-real",
    } or re.fullmatch(r"native-search-ef[1-9][0-9]*", name) is not None


def _nonnegative_integer(value: Any, context: str) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
    ):
        raise ValueError(f"{context}: expected a nonnegative integer")
    return value


def _validate_official_storage(
    result: dict[str, Any],
    artifacts: list[dict[str, Any]],
) -> str:
    build = _mapping(result.get("build"), "cached build result")
    storage = _mapping(build.get("storage"), "cached build storage")
    raw_files = _mapping(
        storage.get("files"), "cached build storage files"
    )
    files = {
        str(name): _nonnegative_integer(
            value, f"cached storage file {name}"
        )
        for name, value in raw_files.items()
    }
    if any(not name for name in files):
        raise ValueError("cached build storage has an empty file name")
    artifact_files = {
        Path(item["path"]).name: _nonnegative_integer(
            item.get("size_bytes"), "cached build artifact size"
        )
        for item in artifacts
    }
    if len(artifact_files) != len(artifacts) or files != artifact_files:
        raise ValueError(
            "cached build storage files differ from verified artifacts"
        )
    vector = {
        name: size for name, size in files.items() if name.endswith(".index")
    }
    lookup = {
        name: size
        for name, size in files.items()
        if name.endswith((".ids.txt", ".passages.idx", ".meta.json"))
    }
    text = {
        name: size
        for name, size in files.items()
        if name.endswith(".passages.jsonl")
    }
    if (
        len(vector) != 1
        or len(text) != 1
        or (set(vector) | set(lookup) | set(text)) != set(files)
    ):
        raise ValueError(
            "cached build storage artifacts are not unambiguously classified"
        )
    expected = {
        "vector_index_bytes": sum(vector.values()),
        "lookup_aux_bytes": sum(lookup.values()),
        "vector_serving_bytes": sum(vector.values()) + sum(lookup.values()),
        "text_store_bytes": sum(text.values()),
        "total_bytes": sum(files.values()),
    }
    for key, value in expected.items():
        if _nonnegative_integer(
            storage.get(key), f"cached build storage {key}"
        ) != value:
            raise ValueError(
                "cached build storage totals differ from verified artifacts"
            )
    return canonical_hash({"files": files, **expected})


def validate_reused_official_index(
    *,
    cached_manifest_path: Path,
    shared: dict[str, Any],
    current_tooling: list[dict[str, Any]],
    official_repo_provenance: dict[str, Any],
    official_runtime: dict[str, Any],
    official_index: Path,
) -> dict[str, Any]:
    """Validate a cached build and return a static reuse attestation."""

    cached_manifest_path = cached_manifest_path.resolve()
    manifest = _mapping(load_json(cached_manifest_path), str(cached_manifest_path))
    if manifest.get("schema") != SCHEMA or manifest.get("status") != "complete":
        raise ValueError("cached source manifest is not a completed benchmark manifest")
    if manifest.get("shared_artifacts") != shared:
        raise ValueError("cached source shared-artifact identity differs")
    provenance = _mapping(
        manifest.get("provenance"), "cached source provenance"
    )
    if provenance.get("official_candidate_recompute_mode") != "cached":
        raise ValueError("reuse source must be an official cached-recompute run")
    source_run_role = manifest.get("run_role")
    source_protocol = _mapping(
        manifest.get("protocol"), "cached source protocol"
    )
    if (
        not isinstance(source_run_role, str)
        or "official-cached-sweep" not in source_run_role.split("+")
        or source_protocol.get("run_role") != source_run_role
    ):
        raise ValueError(
            "reuse source has no explicit official-cached-sweep run role"
        )
    cached_repo = _mapping(
        provenance.get("official_leann"), "cached official repository provenance"
    )
    if cached_repo.get("commit") != official_repo_provenance.get("commit"):
        raise ValueError("official LEANN commit changed since cached build")
    if provenance.get("official_runtime") != official_runtime:
        raise ValueError("official LEANN runtime changed since cached build")

    cached_tooling = _tooling_by_name(
        provenance.get("tooling"), "cached source tooling"
    )
    current_by_name = {
        Path(item["path"]).name: item for item in current_tooling
    }
    for name, cached_snapshot in cached_tooling.items():
        current_snapshot = current_by_name.get(name)
        if current_snapshot is None or current_snapshot.get("sha256") != (
            cached_snapshot.get("sha256")
        ):
            raise ValueError(f"benchmark tooling changed since cached build: {name}")

    planned = _sequence(
        manifest.get("planned_stages"), "cached source planned stages"
    )
    if (
        not all(isinstance(name, str) for name in planned)
        or len(set(planned)) != len(planned)
        or any(not _recognized_stage_name(name) for name in planned)
    ):
        raise ValueError(
            "cached source planned stages are duplicated or unrecognized"
        )
    stages = _mapping(manifest.get("stages"), "cached source stages")
    if set(stages) != set(planned):
        raise ValueError(
            "cached source planned/recorded stage sets differ"
        )
    if "official-build" not in stages:
        raise ValueError("cached source has no official-build stage")
    build_stage_path = _stage_path(
        cached_manifest_path, stages["official-build"], "official-build"
    )
    build_stage = _mapping(load_json(build_stage_path), str(build_stage_path))
    if (
        build_stage.get("schema") != STAGE_SCHEMA
        or build_stage.get("name") != "official-build"
        or build_stage.get("status") != "complete"
    ):
        raise ValueError("cached official-build stage is not complete")
    inputs = _mapping(build_stage.get("inputs"), "cached build inputs")
    if build_stage.get("input_hash") != canonical_hash(inputs):
        raise ValueError("cached official-build input hash is invalid")
    if inputs.get("shared") != shared:
        raise ValueError("cached build shared-artifact identity differs")
    if inputs.get("tooling") != provenance.get("tooling"):
        raise ValueError("cached build tooling differs from source manifest")
    if inputs.get("official_runtime") != official_runtime:
        raise ValueError("cached build runtime differs from current runtime")
    if inputs.get("candidate_recompute_mode") != "cached":
        raise ValueError("cached build stage has the wrong recompute mode")
    source_repo = _mapping(
        inputs.get("official_repo"), "cached build repository provenance"
    )
    if source_repo.get("commit") != official_repo_provenance.get("commit"):
        raise ValueError("cached build repository commit differs")
    recorded_index = Path(
        _command_value(
            inputs.get("command_template"),
            "--index",
            "cached build command template",
        )
    ).resolve()
    if recorded_index != official_index.resolve():
        raise ValueError(
            "cached build used a different --official-index path; refusing reuse"
        )

    measurements = _sequence(
        build_stage.get("measurements"), "cached build measurements"
    )
    expected_repetitions = source_protocol.get("build_repetitions")
    expected_warmups = source_protocol.get("build_warmups")
    execution_protocol = _mapping(
        inputs.get("execution_protocol"), "cached build execution protocol"
    )
    if (
        not isinstance(expected_repetitions, int)
        or expected_repetitions <= 0
        or len(measurements) != expected_repetitions
        or not isinstance(expected_warmups, int)
        or expected_warmups < 0
        or len(_sequence(build_stage.get("warmups"), "cached build warmups"))
        != expected_warmups
        or execution_protocol
        != {
            "warmups": expected_warmups,
            "repetitions": expected_repetitions,
        }
    ):
        raise ValueError("cached build observation counts differ from protocol")
    measurement = _mapping(measurements[-1], "cached build final measurement")
    expected_command = [
        str(value)
        .replace("{kind}", "measurements")
        .replace("{ordinal}", str(expected_repetitions - 1))
        for value in _sequence(
            inputs.get("command_template"), "cached build command template"
        )
    ]
    if (
        measurement.get("kind") != "measurements"
        or measurement.get("ordinal") != expected_repetitions - 1
        or measurement.get("command") != expected_command
    ):
        raise ValueError(
            "cached build final observation kind/ordinal/command differs"
        )
    if measurement.get("exit_code") != 0:
        raise ValueError("cached build final measurement failed")
    if "result_file" not in measurement:
        raise ValueError("cached build has no structured result snapshot")
    result_snapshot = _snapshot_matches(
        measurement["result_file"], "cached build structured result"
    )
    result = _mapping(
        load_json(Path(result_snapshot["path"])),
        "cached build structured result payload",
    )
    if "result" in measurement and measurement["result"] != result:
        raise ValueError(
            "cached build embedded result differs from result snapshot"
        )
    recorded_artifacts = [
        _mapping(item, "cached build artifact")
        for item in _sequence(
            measurement.get("artifacts"), "cached build artifacts"
        )
    ]
    if not recorded_artifacts:
        raise ValueError("cached build recorded no index artifacts")
    for ordinal, artifact in enumerate(recorded_artifacts):
        _snapshot_matches(artifact, f"cached build artifact {ordinal}")
    current_artifacts = snapshot_files(
        prefix_artifacts(official_index, official=True)
    )
    if current_artifacts != recorded_artifacts:
        raise ValueError(
            "current official index artifact set differs from cached build snapshot"
        )
    storage_identity = _validate_official_storage(
        result, recorded_artifacts
    )
    portable = _portable_artifact_identity(recorded_artifacts)
    return {
        "schema": REUSE_SCHEMA,
        "source_manifest": {
            "path": str(cached_manifest_path),
            "size_bytes": cached_manifest_path.stat().st_size,
            "sha256": sha256_file(cached_manifest_path),
            "completed_at": manifest.get("completed_at"),
        },
        "source_stage": {
            "path": str(build_stage_path),
            "size_bytes": build_stage_path.stat().st_size,
            "sha256": sha256_file(build_stage_path),
            "input_hash": build_stage["input_hash"],
            "measurement_ordinal": measurement.get(
                "ordinal", len(measurements) - 1
            ),
        },
        "official_index": str(official_index.resolve()),
        "artifact_set_sha256": canonical_hash(portable),
        "storage_identity_sha256": storage_identity,
        "artifacts": recorded_artifacts,
        "identity_scope": "artifact basename, byte size, and SHA-256",
        "build_metrics": (
            "Inherited from source official-build; construction is not rerun "
            "by this real-mode search-only runner."
        ),
    }


def build_manifest(
    *,
    created_at: str,
    status: str,
    shared: dict[str, Any],
    provenance: dict[str, Any],
    protocol: dict[str, Any],
    stages_dir: Path,
    stage_path: Path | None = None,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "run_role": "official-real-search-reuse",
        "created_at": created_at,
        "status": status,
        "shared_artifacts": shared,
        "provenance": provenance,
        "protocol": protocol,
        "planned_stages": ["official-search-real"],
        "stages_directory": str(stages_dir.resolve()),
    }
    if stage_path is not None:
        manifest["completed_at"] = utc_now()
        manifest["stages"] = {
            "official-search-real": str(stage_path.resolve())
        }
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cached-build-manifest",
        type=Path,
        required=True,
        help="Completed cached-mode manifest containing official-build",
    )
    parser.add_argument("--embedding-url", default="http://127.0.0.1:18080")
    parser.add_argument("--embedding-model", default="nomic-embed-text")
    parser.add_argument("--proxy-url", default="http://127.0.0.1:18081")
    parser.add_argument("--server-ctx-size", type=int, default=32768)
    parser.add_argument("--native-parallel", type=int, default=16)
    parser.add_argument("--native-ctx", type=int, default=2048)
    parser.add_argument("--search-warmups", type=int, default=0)
    parser.add_argument("--search-repetitions", type=int, default=1)
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
        "--official-complexities", type=int, nargs="+", required=True
    )
    parser.add_argument(
        "--official-batch-sizes", type=int, nargs="+", default=[16]
    )
    parser.add_argument(
        "--official-extra",
        type=parse_json_array,
        default=[],
        metavar="JSON_ARRAY",
    )
    args = parser.parse_args()
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    if any(value <= 0 or value > args.top_k for value in args.recall_k):
        parser.error("--recall-k values must be positive and <= --top-k")
    args.recall_k = sorted(set(args.recall_k))
    if args.search_warmups < 0 or args.search_repetitions <= 0:
        parser.error("search warmups/repetitions are invalid")
    if any(value <= 0 for value in args.official_complexities):
        parser.error("official complexities must be positive")
    if any(value < 0 for value in args.official_batch_sizes):
        parser.error("official batch sizes must be non-negative")
    args.official_recompute_mode = "real"
    return args


def main() -> None:
    args = parse_args()
    if args.native_ctx * args.native_parallel > args.server_ctx_size:
        raise ValueError(
            "--server-ctx-size must cover --native-ctx * --native-parallel"
        )
    shared = validate_shared_artifacts(args)
    root = Path(__file__).resolve().parent.parent
    tooling_paths = [
        (root / "scripts/run_large_scale_benchmark.py").resolve(),
        (root / "scripts/benchmark_cache.py").resolve(),
        args.official_script.resolve(),
        (root / "scripts/cached_embedding_server.py").resolve(),
        (root / "scripts/openai_embedding_proxy.py").resolve(),
        Path(__file__).resolve(),
    ]
    tooling = snapshot_files(tooling_paths)
    tooling_by_name = {Path(item["path"]).name: item for item in tooling}
    embedding_observation = endpoint_provenance(args.embedding_url)
    candidate_observation = endpoint_provenance(args.proxy_url)
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
    official_repo = git_provenance(args.official_repo)
    reuse = validate_reused_official_index(
        cached_manifest_path=args.cached_build_manifest,
        shared=shared,
        current_tooling=tooling,
        official_repo_provenance=official_repo,
        official_runtime=official_runtime,
        official_index=args.official_index,
    )

    provenance = {
        "leann_cpp": git_provenance(root),
        "official_leann": official_repo,
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": sys.version,
            "numpy": np.__version__,
        },
        "embedding_endpoint": normalized_base_url(args.embedding_url),
        "proxy_endpoint": normalized_base_url(args.proxy_url),
        "embedding_endpoint_observation": embedding_observation,
        "candidate_endpoint_observation": candidate_observation,
        "tooling": tooling,
        "server_context": {
            "server_ctx_size": args.server_ctx_size,
            "parallel": args.native_parallel,
            "per_slot_ctx": args.native_ctx,
            "required_product": args.native_ctx * args.native_parallel,
        },
        "official_candidate_recompute_mode": "real",
        "official_runtime": official_runtime,
        "official_index_reuse": reuse,
    }
    protocol = {
        "run_role": "official-real-search-reuse",
        "build_warmups": 0,
        "build_repetitions": 1,
        "search_warmups": args.search_warmups,
        "search_repetitions": args.search_repetitions,
        "recall_cutoffs": args.recall_k,
        "timing": (
            "Primary search latency is official raw_per_query point-internal "
            "timing; subprocess wall time is orchestration telemetry only."
        ),
        "rss": (
            "Direct-client ru_maxrss only; the external llama.cpp server is not "
            "included and search RSS is not compared cross-system."
        ),
        "latency_comparability": (
            "Official candidate recomputation uses the counting real-GGUF proxy "
            "against the attested shared model."
        ),
        "build": (
            "Search-only run. Build metrics and storage are inherited from the "
            "exact cached-build source recorded in official_index_reuse."
        ),
    }
    output = args.output.resolve()
    stages_dir = output / "stages"
    results_dir = output / "raw-results"
    results_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    created_at = utc_now()
    if manifest_path.exists():
        existing = _mapping(load_json(manifest_path), str(manifest_path))
        if (
            existing.get("schema") == SCHEMA
            and existing.get("provenance", {}).get("official_index_reuse") == reuse
            and isinstance(existing.get("created_at"), str)
        ):
            created_at = existing["created_at"]
    atomic_write_json(
        manifest_path,
        build_manifest(
            created_at=created_at,
            status="running",
            shared=shared,
            provenance=provenance,
            protocol=protocol,
            stages_dir=stages_dir,
        ),
    )

    official_inputs = {
        "shared": shared,
        "tooling": tooling,
        "candidate_recompute_mode": "real",
        "official_repo": official_repo,
        "official_runtime": official_runtime,
        "python": (
            snapshot_files([args.official_python])[0]
            if args.official_python.is_file()
            else str(args.official_python)
        ),
        "index_reuse": reuse,
    }

    def command_factory(kind: str, ordinal: int) -> tuple[list[str], Path]:
        result_path = (
            results_dir / f"official-search-{kind}-{ordinal}.json"
        )
        return (
            official_command(
                args, mode="benchmark", output=result_path
            ),
            result_path,
        )

    command_template = official_command(
        args,
        mode="benchmark",
        output=results_dir / "official-search-{kind}-{ordinal}.json",
    )
    stage = run_repeated_stage(
        name="official-search-real",
        state_dir=stages_dir,
        input_payload={
            **official_inputs,
            "command_template": command_template,
        },
        command_factory=command_factory,
        artifact_factory=lambda: prefix_artifacts(
            args.official_index, official=True
        ),
        cwd=args.official_repo,
        warmups=args.search_warmups,
        repetitions=args.search_repetitions,
    )
    if stage.get("status") != "complete":
        raise RuntimeError("official real search stage did not complete")
    final_artifacts = snapshot_files(
        prefix_artifacts(args.official_index, official=True)
    )
    if final_artifacts != reuse["artifacts"]:
        raise RuntimeError("official search changed the attested index artifacts")
    stage_path = stages_dir / "official-search-real.json"
    manifest = build_manifest(
        created_at=created_at,
        status="complete",
        shared=shared,
        provenance=provenance,
        protocol=protocol,
        stages_dir=stages_dir,
        stage_path=stage_path,
    )
    atomic_write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
