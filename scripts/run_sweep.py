#!/usr/bin/env python3
"""Run repeatable leann.cpp recall/latency sweeps and emit CSV + JSON."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import subprocess
import sys
from typing import Any


def parse_key_values(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        try:
            result[key] = float(value) if "." in value else int(value)
        except ValueError:
            result[key] = value
    return result


def run(command: list[str]) -> tuple[dict[str, Any], str]:
    completed = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="")
    return parse_key_values(completed.stdout), completed.stdout


def parse_index(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--index must be LABEL=PREFIX")
    label, prefix = value.split("=", 1)
    if not label or not prefix:
        raise argparse.ArgumentTypeError("--index must be LABEL=PREFIX")
    return label, prefix


def comma_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",")]


def comma_floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=pathlib.Path, required=True)
    parser.add_argument("--index", action="append", type=parse_index, required=True)
    parser.add_argument("--queries", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--embedder", choices=("hash", "llama"), default="hash")
    parser.add_argument("--model", type=pathlib.Path)
    parser.add_argument("--hash-dim", type=int, default=256)
    parser.add_argument("--gpu-layers", type=int, default=99)
    parser.add_argument("--parallel", type=int, default=16)
    parser.add_argument("--ctx", type=int, default=2048)
    parser.add_argument("--batch-tokens", type=int, default=32768)
    parser.add_argument("--ground-truth-cache", type=pathlib.Path)
    parser.add_argument("--ef-search", type=comma_ints, default=[32, 64, 96])
    parser.add_argument("--rerank-ratio", type=comma_floats, default=[0.25])
    parser.add_argument("--recompute-batch", type=int, default=16)
    parser.add_argument("--scan-limit", type=int, default=100000)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--dense-baseline", action="store_true")
    arguments = parser.parse_args()

    if arguments.embedder == "llama" and arguments.model is None:
        parser.error("--model is required with --embedder llama")
    if any(value <= 0 for value in arguments.ef_search):
        parser.error("--ef-search values must be positive")

    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    binary = str(arguments.binary.resolve())
    queries = str(arguments.queries.resolve())

    common = ["--embedder", arguments.embedder]
    if arguments.embedder == "llama":
        common += [
            "--model",
            str(arguments.model.resolve()),
            "--gpu-layers",
            str(arguments.gpu_layers),
            "--parallel",
            str(arguments.parallel),
            "--ctx",
            str(arguments.ctx),
            "--batch-tokens",
            str(arguments.batch_tokens),
        ]
    else:
        common += ["--hash-dim", str(arguments.hash_dim)]
    if arguments.ground_truth_cache:
        common += [
            "--ground-truth-cache",
            str(arguments.ground_truth_cache.resolve()),
        ]

    records: list[dict[str, Any]] = []
    commands: list[list[str]] = []
    raw_outputs: list[dict[str, str]] = []
    for label, prefix_value in arguments.index:
        prefix = str(pathlib.Path(prefix_value).resolve())
        stats_command = [binary, "stats", "--index", prefix]
        stats, stats_text = run(stats_command)
        commands.append(stats_command)
        raw_outputs.append({"label": label, "kind": "stats", "stdout": stats_text})

        for ef_search in arguments.ef_search:
            for rerank_ratio in arguments.rerank_ratio:
                command = [
                    binary,
                    "bench",
                    "--index",
                    prefix,
                    "--queries",
                    queries,
                    "--top-k",
                    str(arguments.top_k),
                    "--ef-search",
                    str(ef_search),
                    "--recompute-batch",
                    str(arguments.recompute_batch),
                    "--rerank-ratio",
                    str(rerank_ratio),
                    "--scan-limit",
                    str(arguments.scan_limit),
                    "--max-queries",
                    str(arguments.max_queries),
                    "--dense-baseline",
                    "1" if arguments.dense_baseline else "0",
                    *common,
                ]
                metrics, metrics_text = run(command)
                commands.append(command)
                raw_outputs.append(
                    {
                        "label": label,
                        "kind": "bench",
                        "stdout": metrics_text,
                    }
                )
                records.append(
                    {
                        "label": label,
                        "prefix": prefix,
                        "ef_search": ef_search,
                        "rerank_ratio": rerank_ratio,
                        **stats,
                        **metrics,
                    }
                )
                print(
                    f"{label}: ef={ef_search} ratio={rerank_ratio} "
                    f"recall={metrics.get(f'recall_at_{arguments.top_k}')} "
                    f"mean_ms={metrics.get('latency_ms_mean')}"
                )

    fieldnames: list[str] = []
    for record in records:
        for key in record:
            if key not in fieldnames:
                fieldnames.append(key)
    with (output / "sweep.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    (output / "sweep.json").write_text(
        json.dumps(
            {
                "records": records,
                "commands": commands,
                "raw_outputs": raw_outputs,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
