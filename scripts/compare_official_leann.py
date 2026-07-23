#!/usr/bin/env python3
"""Build and benchmark official LEANN against the leann.cpp SciFact fixture."""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import resource
import struct
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np


MAGIC = b"LEANNBC1"


def read_lines(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle]


def read_leann_cache(path: Path) -> tuple[np.memmap, dict[str, Any]]:
    with path.open("rb") as handle:
        magic = handle.read(8)
        if magic != MAGIC:
            raise ValueError(f"Unexpected cache magic: {magic!r}")
        dim = struct.unpack("<I", handle.read(4))[0]
        count = struct.unpack("<Q", handle.read(8))[0]
        fingerprint_size = struct.unpack("<I", handle.read(4))[0]
        fingerprint = handle.read(fingerprint_size).decode("utf-8")
        offset = handle.tell()
    expected = offset + count * dim * 4
    if path.stat().st_size != expected:
        raise ValueError(
            f"Cache length mismatch: expected {expected}, got {path.stat().st_size}"
        )
    vectors = np.memmap(
        path,
        mode="r",
        dtype="<f4",
        offset=offset,
        shape=(count, dim),
    )
    return vectors, {
        "path": str(path),
        "count": count,
        "dimensions": dim,
        "fingerprint": fingerprint,
    }


def post_json(url: str, payload: object, timeout: int = 600) -> Any:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def get_json(url: str, timeout: int = 30) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def openai_embeddings(
    base_url: str,
    texts: list[str],
    model: str,
    batch_size: int = 64,
) -> np.ndarray:
    result: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        response = post_json(
            f"{base_url.rstrip('/')}/v1/embeddings",
            {"model": model, "input": texts[start : start + batch_size]},
        )
        ordered = sorted(response["data"], key=lambda item: item["index"])
        result.extend(item["embedding"] for item in ordered)
    return np.asarray(result, dtype=np.float32)


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def artifact_sizes(index_path: Path) -> dict[str, Any]:
    prefix = index_path.stem
    files = sorted(
        path
        for path in index_path.parent.iterdir()
        if path.is_file()
        and (
            path.name.startswith(index_path.name)
            or path.name.startswith(prefix + ".")
        )
    )
    entries = {path.name: path.stat().st_size for path in files}
    vector_index = entries.get(f"{prefix}.index", 0)
    ids = entries.get(f"{prefix}.ids.txt", 0)
    passages = entries.get(f"{index_path.name}.passages.jsonl", 0)
    offsets = entries.get(f"{index_path.name}.passages.idx", 0)
    metadata = entries.get(f"{index_path.name}.meta.json", 0)
    vector_serving = vector_index + ids + offsets + metadata
    total = sum(entries.values())
    return {
        "files": entries,
        "vector_index_bytes": vector_index,
        "lookup_aux_bytes": ids + offsets + metadata,
        "vector_serving_bytes": vector_serving,
        "text_store_bytes": passages,
        "total_bytes": total,
    }


def build(args: argparse.Namespace, corpus: np.memmap, docs: list[str]) -> dict[str, Any]:
    from leann import LeannBuilder

    index_path = args.index.resolve()
    index_path.parent.mkdir(parents=True, exist_ok=True)
    for path in index_path.parent.glob(f"{index_path.stem}*"):
        if path.is_file():
            path.unlink()

    builder = LeannBuilder(
        backend_name="hnsw",
        embedding_model=args.model_alias,
        dimensions=corpus.shape[1],
        embedding_mode="openai",
        embedding_options={
            "base_url": f"{args.proxy_url.rstrip('/')}/v1",
            "api_key": "local-leann-cpp-benchmark",
        },
        prebuild_bm25=False,
        distance_metric="cosine",
        M=args.m,
        efConstruction=args.ef_construction,
        is_compact=True,
        is_recompute=True,
    )
    for idx, text in enumerate(docs):
        builder.add_text(text, metadata={"id": str(idx)})

    started = time.perf_counter()
    builder.build_index_from_arrays(
        str(index_path),
        list(range(len(docs))),
        np.ascontiguousarray(corpus, dtype=np.float32),
    )
    elapsed = time.perf_counter() - started
    return {
        "elapsed_seconds_excluding_embedding": elapsed,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "storage": artifact_sizes(index_path),
    }


def load_or_compute_queries(
    args: argparse.Namespace,
    queries: list[str],
    dimensions: int,
) -> np.ndarray:
    cache_path = args.query_cache.resolve()
    if cache_path.exists():
        cached = np.load(cache_path)
        if cached.shape == (len(queries), dimensions):
            return np.asarray(cached, dtype=np.float32)
    query_vectors = openai_embeddings(
        args.llama_url, queries, args.model_alias, batch_size=args.query_batch_size
    )
    if query_vectors.shape != (len(queries), dimensions):
        raise ValueError(f"Unexpected query embedding shape: {query_vectors.shape}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, query_vectors)
    return query_vectors


def exact_topk(corpus: np.memmap, queries: np.ndarray, k: int) -> np.ndarray:
    output = np.empty((len(queries), k), dtype=np.int64)
    corpus_array = np.asarray(corpus)
    for idx, query in enumerate(queries):
        scores = corpus_array @ query
        candidates = np.argpartition(scores, -k)[-k:]
        output[idx] = candidates[np.argsort(scores[candidates])[::-1]]
    return output


def proxy_reset(url: str) -> None:
    post_json(f"{url.rstrip('/')}/metrics/reset", {})


def proxy_metrics(url: str) -> dict[str, int]:
    data = get_json(f"{url.rstrip('/')}/metrics")
    return {"requests": int(data["requests"]), "inputs": int(data["inputs"])}


def run_point(
    backend: Any,
    port: int,
    query_vectors: np.ndarray,
    truth: np.ndarray,
    *,
    complexity: int,
    batch_size: int,
    proxy_url: str,
) -> dict[str, Any]:
    latencies_ms: list[float] = []
    recall_total = 0
    results: list[list[int]] = []

    backend.search(
        query_vectors[0:1],
        3,
        complexity=complexity,
        beam_width=1,
        prune_ratio=0.0,
        recompute_embeddings=True,
        pruning_strategy="global",
        zmq_port=port,
        batch_size=batch_size,
    )
    proxy_reset(proxy_url)

    for idx, query in enumerate(query_vectors):
        started = time.perf_counter()
        response = backend.search(
            query.reshape(1, -1),
            3,
            complexity=complexity,
            beam_width=1,
            prune_ratio=0.0,
            recompute_embeddings=True,
            pruning_strategy="global",
            zmq_port=port,
            batch_size=batch_size,
        )
        latencies_ms.append((time.perf_counter() - started) * 1000.0)
        labels = [int(label) for label in response["labels"][0]]
        results.append(labels)
        recall_total += len(set(labels) & set(map(int, truth[idx])))
        if (idx + 1) % max(1, len(query_vectors) // 10) == 0:
            print(
                f"progress complexity={complexity} batch_size={batch_size}: "
                f"{idx + 1}/{len(query_vectors)}",
                flush=True,
            )

    metrics = proxy_metrics(proxy_url)
    count = len(query_vectors)
    return {
        "complexity": complexity,
        "batch_size": batch_size,
        "queries": count,
        "recall_at_3": recall_total / (count * 3),
        "latency_ms": {
            "mean": float(np.mean(latencies_ms)),
            "p50": percentile(latencies_ms, 50),
            "p95": percentile(latencies_ms, 95),
            "min": float(np.min(latencies_ms)),
            "max": float(np.max(latencies_ms)),
        },
        "candidate_embedding_requests": metrics["requests"],
        "candidate_embeddings_total": metrics["inputs"],
        "candidate_embeddings_mean_per_query": metrics["inputs"] / count,
        "result_ids": results,
    }


def benchmark(
    args: argparse.Namespace,
    corpus: np.memmap,
    queries: list[str],
) -> dict[str, Any]:
    from leann import LeannSearcher

    all_query_vectors = load_or_compute_queries(args, queries, corpus.shape[1])
    if args.query_sample is not None:
        selected = np.linspace(
            0, len(queries) - 1, min(args.query_sample, len(queries)), dtype=np.int64
        )
    else:
        stop = (
            len(queries)
            if args.query_limit is None
            else min(len(queries), args.query_offset + args.query_limit)
        )
        selected = np.arange(args.query_offset, stop, dtype=np.int64)
    query_vectors = all_query_vectors[selected]
    if not len(query_vectors):
        raise ValueError("Query selection is empty")
    truth = exact_topk(corpus, query_vectors, 3)
    searcher = LeannSearcher(
        str(args.index.resolve()),
        enable_warmup=False,
        recompute_embeddings=True,
        use_daemon=False,
    )
    port = searcher.backend_impl._ensure_server_running(
        searcher.meta_path_str,
        port=args.zmq_port,
        enable_warmup=False,
        use_daemon=False,
    )
    points: list[dict[str, Any]] = []
    try:
        for batch_size in args.batch_sizes:
            for complexity in args.complexities:
                point = run_point(
                    searcher.backend_impl,
                    port,
                    query_vectors,
                    truth,
                    complexity=complexity,
                    batch_size=batch_size,
                    proxy_url=args.proxy_url,
                )
                points.append(point)
                print(
                    json.dumps(
                        {
                            "complexity": complexity,
                            "batch_size": batch_size,
                            "recall_at_3": point["recall_at_3"],
                            "mean_ms": point["latency_ms"]["mean"],
                            "candidate_embeddings_mean": point[
                                "candidate_embeddings_mean_per_query"
                            ],
                        }
                    ),
                    flush=True,
                )
    finally:
        searcher.cleanup()
    return {
        "query_embedding_cache": str(args.query_cache.resolve()),
        "query_count": len(query_vectors),
        "query_offset": args.query_offset,
        "query_indices": selected.tolist(),
        "truth": "exact dense top-3 inner product over the shared normalized corpus cache",
        "timing_scope": (
            "official HNSW backend search plus candidate recompute via ZMQ/OpenAI-compatible "
            "llama.cpp; excludes query embedding, Python passage enrichment, and cold start"
        ),
        "points": points,
    }


def git_output(repo: Path, *command: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *command], text=True
    ).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument(
        "--documents", type=Path, default=Path("work/datasets/scifact/documents.txt")
    )
    parser.add_argument(
        "--queries", type=Path, default=Path("work/datasets/scifact/queries-test.txt")
    )
    parser.add_argument(
        "--corpus-cache",
        type=Path,
        default=Path("work/datasets/scifact/nomic-q4-metal.bench.f32"),
    )
    parser.add_argument(
        "--query-cache",
        type=Path,
        default=Path("work/datasets/scifact/nomic-q4-metal.queries.npy"),
    )
    parser.add_argument(
        "--index", type=Path, default=Path("work/official-leann-scifact/scifact.leann")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("work/official-leann-scifact/benchmark.json"),
    )
    parser.add_argument("--llama-url", default="http://127.0.0.1:18080")
    parser.add_argument("--proxy-url", default="http://127.0.0.1:18081")
    parser.add_argument("--model-alias", default="nomic-embed-text")
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--complexities", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[0, 16])
    parser.add_argument("--query-batch-size", type=int, default=64)
    parser.add_argument("--query-offset", type=int, default=0)
    parser.add_argument("--query-limit", type=int)
    parser.add_argument(
        "--query-sample",
        type=int,
        help="Select this many evenly spaced queries across the complete set",
    )
    parser.add_argument("--zmq-port", type=int, default=15557)
    args = parser.parse_args()
    if not args.build and not args.benchmark:
        parser.error("Select --build, --benchmark, or both")
    return args


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.WARNING)
    docs = read_lines(args.documents)
    queries = read_lines(args.queries)
    corpus, cache_meta = read_leann_cache(args.corpus_cache)
    if len(docs) != corpus.shape[0]:
        raise ValueError(f"Documents/cache mismatch: {len(docs)} vs {corpus.shape[0]}")

    official_repo = Path("work/reference/LEANN").resolve()
    report: dict[str, Any] = {
        "schema": "leann.cpp-official-comparison-v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "official": {
            "repository": "https://github.com/StarTrail-org/LEANN",
            "commit": git_output(official_repo, "rev-parse", "HEAD"),
            "commit_date": git_output(official_repo, "show", "-s", "--format=%cI", "HEAD"),
            "core_source": "official pinned commit",
            "backend_hnsw_package": "0.3.7",
            "config": {
                "backend": "hnsw",
                "M": args.m,
                "efConstruction": args.ef_construction,
                "is_compact": True,
                "is_recompute": True,
                "distance_metric": "cosine",
            },
        },
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "dataset": {
            "documents": len(docs),
            "queries": len(queries),
            "raw_text_file_bytes": args.documents.stat().st_size,
            "raw_text_payload_bytes": sum(
                len(text.encode("utf-8")) for text in docs
            ),
        },
        "embedding_cache": cache_meta,
    }
    if args.build:
        report["build"] = build(args, corpus, docs)
    elif args.index.exists():
        report["build"] = {"storage": artifact_sizes(args.index.resolve())}
    if args.benchmark:
        report["benchmark"] = benchmark(args, corpus, queries)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
