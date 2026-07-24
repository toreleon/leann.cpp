#!/usr/bin/env python3
"""Build and benchmark official LEANN against the leann.cpp SciFact fixture."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import logging
import os
import platform
import resource
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

from benchmark_cache import (
    iter_nonempty_lines,
    read_cache,
    read_ground_truth,
    sha256_file,
    validate_cache_source,
)


def read_lines(path: Path) -> list[str]:
    return list(iter_nonempty_lines(path))


def read_leann_cache(path: Path) -> tuple[np.memmap, dict[str, Any]]:
    return read_cache(path)


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


def normalized_base_url(value: str) -> str:
    result = value.rstrip("/")
    if result.endswith("/v1"):
        result = result[:-3]
    return result.rstrip("/")


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
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_bytes = int(rss if sys.platform == "darwin" else rss * 1024)
    return {
        "elapsed_seconds_excluding_embedding": elapsed,
        "peak_rss_bytes": rss_bytes,
        "peak_rss_source": (
            "getrusage(RUSAGE_SELF); process-lifetime maximum, not isolated stage delta"
        ),
        "storage": artifact_sizes(index_path),
    }


def load_or_compute_queries(
    args: argparse.Namespace,
    queries: list[str],
    dimensions: int,
) -> np.ndarray:
    cache_path = args.query_cache.resolve()
    if cache_path.exists():
        if cache_path.suffix == ".npy":
            cached = np.load(cache_path, mmap_mode="r")
        else:
            cached, metadata = read_cache(cache_path)
            validate_cache_source(
                metadata,
                args.queries.resolve(),
                expected_count=len(queries),
                require_integrity=True,
            )
            if metadata["fingerprint"] != args.cache_fingerprint:
                raise ValueError("query-cache fingerprint differs from corpus cache")
            sidecar = cache_path.with_name(cache_path.name + ".meta.json")
            corpus_sidecar = args.corpus_cache.resolve().with_name(
                args.corpus_cache.name + ".meta.json"
            )
            if not sidecar.exists() or not corpus_sidecar.exists():
                raise ValueError("V2 shared caches require provenance sidecars")
            query_declared = json.loads(sidecar.read_text(encoding="utf-8"))
            corpus_declared = json.loads(corpus_sidecar.read_text(encoding="utf-8"))
            if query_declared.get("bindings", {}).get(
                "corpus_cache_sha256"
            ) != sha256_file(args.corpus_cache):
                raise ValueError("query cache is not bound to selected corpus cache")
            if query_declared.get("model", {}).get(
                "sha256"
            ) != corpus_declared.get("model", {}).get("sha256"):
                raise ValueError("query/corpus model identity hashes differ")
        if cached.shape != (len(queries), dimensions):
            raise ValueError(
                f"Unexpected cached query embedding shape: {cached.shape}"
            )
        if not np.isfinite(cached).all():
            raise ValueError("query cache contains NaN or infinity")
        return np.asarray(cached, dtype=np.float32)
    if cache_path.suffix != ".npy":
        raise ValueError(
            "missing integrity-bound query cache; generate it with "
            "run_large_scale_benchmark.py prepare/cache-queries"
        )
    query_vectors = openai_embeddings(
        args.llama_url, queries, args.model_alias, batch_size=args.query_batch_size
    )
    if query_vectors.shape != (len(queries), dimensions):
        raise ValueError(f"Unexpected query embedding shape: {query_vectors.shape}")
    norms = np.linalg.norm(query_vectors, axis=1, keepdims=True)
    if not np.isfinite(query_vectors).all() or np.any(norms <= 0):
        raise ValueError("query embeddings contain non-finite/zero vectors")
    query_vectors = np.asarray(query_vectors / norms, dtype=np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, query_vectors)
    return query_vectors


def exact_topk(corpus: np.memmap, queries: np.ndarray, k: int) -> np.ndarray:
    output = np.empty((len(queries), k), dtype=np.int64)
    corpus_array = np.asarray(corpus)
    identifiers = np.arange(corpus_array.shape[0], dtype=np.int64)
    for idx, query in enumerate(queries):
        scores = corpus_array @ query
        if k == scores.size:
            candidates = identifiers
        else:
            provisional = np.argpartition(scores, scores.size - k)[-k:]
            threshold = np.min(scores[provisional])
            strict = np.flatnonzero(scores > threshold)
            ties = np.flatnonzero(scores == threshold)
            needed = k - strict.size
            candidates = np.concatenate((strict, ties[:needed]))
        order = np.lexsort((identifiers[candidates], -scores[candidates]))
        output[idx] = identifiers[candidates][order[:k]]
    return output


def proxy_reset(url: str) -> None:
    post_json(f"{url.rstrip('/')}/metrics/reset", {})


def proxy_metrics(url: str) -> dict[str, Any]:
    data = get_json(f"{url.rstrip('/')}/metrics")
    return data


def run_point(
    backend: Any,
    port: int,
    query_vectors: np.ndarray,
    truth: np.ndarray,
    *,
    top_k: int,
    report_ks: list[int],
    complexity: int,
    batch_size: int,
    proxy_url: str,
) -> dict[str, Any]:
    latencies_ms: list[float] = []
    recall_totals = {value: 0 for value in report_ks}
    results: list[list[int]] = []

    backend.search(
        query_vectors[0:1],
        top_k,
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
            top_k,
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
        for report_k in report_ks:
            recall_totals[report_k] += len(
                set(labels[:report_k])
                & set(map(int, truth[idx, :report_k]))
            )
        if (idx + 1) % max(1, len(query_vectors) // 10) == 0:
            print(
                f"progress complexity={complexity} batch_size={batch_size}: "
                f"{idx + 1}/{len(query_vectors)}",
                flush=True,
            )

    metrics = proxy_metrics(proxy_url)
    count = len(query_vectors)
    result: dict[str, Any] = {
        "complexity": complexity,
        "batch_size": batch_size,
        "queries": count,
        "latency_ms": {
            "mean": float(np.mean(latencies_ms)),
            "p50": percentile(latencies_ms, 50),
            "p95": percentile(latencies_ms, 95),
            "min": float(np.min(latencies_ms)),
            "max": float(np.max(latencies_ms)),
            "raw_per_query": latencies_ms,
        },
        "candidate_embedding_requests": metrics["requests"],
        "candidate_embeddings_total": metrics["inputs"],
        "candidate_embeddings_mean_per_query": metrics["inputs"] / count,
        "embedding_proxy_metrics": metrics,
        "result_ids": results,
    }
    for report_k in report_ks:
        result[f"recall_at_{report_k}"] = recall_totals[report_k] / (
            count * report_k
        )
    return result


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
    if args.ground_truth is not None:
        all_truth, truth_meta = read_ground_truth(
            args.ground_truth.resolve(),
            expected_queries=len(queries),
            expected_k=args.top_k,
            expected_corpus=corpus.shape[0],
        )
        truth = all_truth[selected]
        truth_description: Any = truth_meta
    else:
        truth = exact_topk(corpus, query_vectors, args.top_k)
        truth_description = (
            f"exact dense top-{args.top_k} inner product over the shared "
            "normalized corpus cache (computed in this process)"
        )
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
                    top_k=args.top_k,
                    report_ks=args.report_k,
                    complexity=complexity,
                    batch_size=batch_size,
                    proxy_url=args.proxy_url,
                )
                proxy_observation = point["embedding_proxy_metrics"]
                if (
                    int(proxy_observation.get("http_errors", 0)) != 0
                    or int(proxy_observation.get("network_errors", 0)) != 0
                ):
                    raise RuntimeError(
                        "embedding proxy reported upstream errors: "
                        f"{proxy_observation}"
                    )
                expected_mode = (
                    "cache" if args.recompute_mode == "cached" else "proxy"
                )
                if proxy_observation.get("mode") != expected_mode:
                    raise RuntimeError(
                        f"--recompute-mode={args.recompute_mode} requires "
                        f"metrics mode {expected_mode!r}, got "
                        f"{proxy_observation.get('mode')!r}"
                    )
                if args.recompute_mode == "cached":
                    identity = proxy_observation.get("cache_identity")
                    if not isinstance(identity, dict):
                        raise RuntimeError(
                            "cached endpoint did not expose cache identity"
                        )
                    expected_identity = {
                        "cache_sha256": args.corpus_cache_sha256,
                        "source_sha256": args.cache_source_sha256,
                        "fingerprint": args.cache_fingerprint,
                        "dimensions": int(corpus.shape[1]),
                        "count": int(corpus.shape[0]),
                        "model_sha256": args.cache_model_sha256,
                    }
                    for key, expected in expected_identity.items():
                        if identity.get(key) != expected:
                            raise RuntimeError(
                                f"cached endpoint {key} mismatch: "
                                f"{identity.get(key)!r} vs {expected!r}"
                            )
                elif normalized_base_url(
                    str(proxy_observation.get("upstream", ""))
                ) != normalized_base_url(args.llama_url):
                    raise RuntimeError(
                        "real recompute proxy upstream does not match --llama-url"
                    )
                points.append(point)
                print(
                    json.dumps(
                        {
                            "complexity": complexity,
                            "batch_size": batch_size,
                            **{
                                f"recall_at_{value}": point[f"recall_at_{value}"]
                                for value in args.report_k
                            },
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
        "query_embedding_cache_sha256": sha256_file(args.query_cache),
        "corpus_embedding_cache_sha256": sha256_file(args.corpus_cache),
        "query_count": len(query_vectors),
        "query_offset": args.query_offset,
        "query_indices": selected.tolist(),
        "truth": truth_description,
        "timing_scope": (
            "official HNSW backend search plus candidate recompute via ZMQ/OpenAI-compatible "
            "llama.cpp; excludes query embedding, Python passage enrichment, and cold start"
        ),
        "primary_latency_metric": (
            "point-internal raw_per_query milliseconds; outer subprocess wall "
            "time is orchestration telemetry, not a per-point latency metric"
        ),
        "latency_comparability": (
            "cached recompute mode is an algorithmic recall/candidate sweep and "
            "must not be compared with native real-GGUF latency; cross-system "
            "latency requires recompute-mode=real against the attested model"
        ),
        "candidate_recompute_mode": args.recompute_mode,
        "points": points,
    }


def git_output(repo: Path, *command: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *command], text=True
    ).strip()


def official_runtime_provenance() -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": sys.version,
        "executable": str(Path(sys.executable).resolve()),
        "modules": {},
        "distributions": {},
    }
    for name in (
        "leann",
        "leann_backend_hnsw",
        "leann_backend_hnsw.faiss",
        "leann_backend_hnsw._swigfaiss",
        "leann_backend_hnsw.hnsw_backend",
    ):
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        root = path.parent if path.name == "__init__.py" else path
        files = [path]
        if root.is_dir():
            files = sorted(
                item
                for item in root.rglob("*")
                if item.is_file()
                and item.suffix in {".py", ".so", ".dylib", ".pyd"}
            )
        result["modules"][name] = {
            "path": str(path),
            "files": [
                {
                    "path": str(item),
                    "size_bytes": item.stat().st_size,
                    "sha256": sha256_file(item),
                }
                for item in files
            ],
        }
    faiss_module = importlib.import_module("leann_backend_hnsw.faiss")
    faiss_path = Path(faiss_module.__file__).resolve()
    result["backend_faiss"] = {
        "path": str(faiss_path),
        "size_bytes": faiss_path.stat().st_size,
        "sha256": sha256_file(faiss_path),
    }
    packages = importlib.metadata.packages_distributions()
    for import_name in ("leann", "leann_backend_hnsw"):
        for distribution in packages.get(import_name, []):
            try:
                result["distributions"][distribution] = (
                    importlib.metadata.version(distribution)
                )
            except importlib.metadata.PackageNotFoundError:
                pass
    return result


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
        "--ground-truth",
        type=Path,
        help="Precomputed LEANN_GT1 truth shared with the native benchmark",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--report-k",
        type=int,
        nargs="+",
        help="Recall cutoffs; search uses --top-k and truth prefixes",
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
    parser.add_argument(
        "--recompute-mode",
        choices=("cached", "real"),
        default="real",
        help="Label and verify candidate recomputation timing source",
    )
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
    parser.add_argument(
        "--official-repo", type=Path, default=Path("work/reference/LEANN")
    )
    args = parser.parse_args()
    if not args.build and not args.benchmark:
        parser.error("Select --build, --benchmark, or both")
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    if args.report_k is None:
        args.report_k = [args.top_k]
    if any(value <= 0 or value > args.top_k for value in args.report_k):
        parser.error("--report-k values must be positive and <= --top-k")
    args.report_k = sorted(set(args.report_k))
    return args


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.WARNING)
    docs = read_lines(args.documents)
    queries = read_lines(args.queries)
    corpus, cache_meta = read_leann_cache(args.corpus_cache)
    if len(docs) != corpus.shape[0]:
        raise ValueError(f"Documents/cache mismatch: {len(docs)} vs {corpus.shape[0]}")
    if cache_meta["version"] == 2:
        cache_meta["source_validation"] = validate_cache_source(
            cache_meta,
            args.documents.resolve(),
            expected_count=len(docs),
            require_integrity=True,
        )
    cache_meta["sha256"] = sha256_file(args.corpus_cache)
    args.cache_fingerprint = cache_meta["fingerprint"]
    args.corpus_cache_sha256 = cache_meta["sha256"]
    args.cache_source_sha256 = cache_meta.get("source_sha256")
    cache_sidecar = args.corpus_cache.with_name(
        args.corpus_cache.name + ".meta.json"
    )
    args.cache_model_sha256 = None
    if cache_sidecar.exists():
        declared_cache = json.loads(cache_sidecar.read_text(encoding="utf-8"))
        if declared_cache.get("cache", {}).get("sha256") != cache_meta["sha256"]:
            raise ValueError("corpus cache provenance sidecar SHA-256 mismatch")
        args.cache_model_sha256 = declared_cache.get("model", {}).get("sha256")

    official_repo = args.official_repo.resolve()
    report: dict[str, Any] = {
        "schema": "leann.cpp-official-comparison-v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "official": {
            "repository": "https://github.com/StarTrail-org/LEANN",
            "commit": git_output(official_repo, "rev-parse", "HEAD"),
            "commit_date": git_output(official_repo, "show", "-s", "--format=%cI", "HEAD"),
            "core_source": "official pinned commit",
            "backend_hnsw_package": "0.3.7",
            "runtime": official_runtime_provenance(),
            "config": {
                "backend": "hnsw",
                "M": args.m,
                "efConstruction": args.ef_construction,
                "is_compact": True,
                "is_recompute": True,
                "distance_metric": "cosine",
                "check_relative_distance": not (
                    "text-embedding" in args.model_alias.lower()
                    or "openai" in args.model_alias.lower()
                ),
                "embedding_model_alias": args.model_alias,
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
