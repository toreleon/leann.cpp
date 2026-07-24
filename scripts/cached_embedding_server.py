#!/usr/bin/env python3
"""OpenAI-compatible server backed by the exact leann.cpp embedding cache.

This is an algorithmic benchmark aid: it preserves vectors exactly while
removing model inference from official LEANN's candidate-recompute path.
"""

from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from benchmark_cache import (
    iter_nonempty_lines,
    read_cache,
    sha256_file,
    validate_cache_source,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18082)
    parser.add_argument(
        "--documents", type=Path, default=Path("work/datasets/scifact/documents.txt")
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("work/datasets/scifact/nomic-q4-metal.bench.f32"),
    )
    parser.add_argument("--model", default="nomic-embed-text")
    parser.add_argument(
        "--allow-v1",
        action="store_true",
        help="Allow an unbound legacy cache after count validation",
    )
    parser.add_argument("--max-request-bytes", type=int, default=64 * 1024 * 1024)
    args = parser.parse_args()

    documents = list(iter_nonempty_lines(args.documents))
    vectors, cache_meta = read_cache(args.cache)
    if cache_meta["version"] == 2:
        validate_cache_source(
            cache_meta,
            args.documents,
            expected_count=len(documents),
            require_integrity=True,
        )
    elif not args.allow_v1:
        raise ValueError("legacy V1 cache requires explicit --allow-v1")
    if len(documents) != len(vectors):
        raise ValueError(f"Document/cache mismatch: {len(documents)} vs {len(vectors)}")
    cache_sha256 = sha256_file(args.cache)
    sidecar_path = args.cache.with_name(args.cache.name + ".meta.json")
    cache_declared: dict[str, object] | None = None
    if sidecar_path.exists():
        cache_declared = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if (
            cache_declared.get("schema") != "leann-shared-embedding-cache-v1"
            or cache_declared.get("role") != "corpus"
        ):
            raise ValueError("cache provenance sidecar schema/role mismatch")
        if cache_declared.get("cache", {}).get("sha256") != cache_sha256:  # type: ignore[union-attr]
            raise ValueError("cache provenance sidecar SHA-256 mismatch")
        if (
            cache_declared.get("source", {}).get("sha256")  # type: ignore[union-attr]
            != cache_meta.get("source_sha256")
        ):
            raise ValueError("cache provenance sidecar source mismatch")
    elif cache_meta["version"] == 2:
        raise ValueError("integrity-bound cache requires a provenance sidecar")
    cache_identity = {
        "schema": cache_meta["schema"],
        "cache_sha256": cache_sha256,
        "source_sha256": cache_meta.get("source_sha256"),
        "source_size_bytes": cache_meta.get("source_size_bytes"),
        "fingerprint": cache_meta["fingerprint"],
        "dimensions": cache_meta["dimensions"],
        "count": cache_meta["count"],
        "model_sha256": (
            cache_declared.get("model", {}).get("sha256")  # type: ignore[union-attr]
            if cache_declared is not None
            else None
        ),
        "server_script_sha256": sha256_file(Path(__file__)),
    }
    text_to_row: dict[str, int] = {}
    for row, text in enumerate(documents):
        previous = text_to_row.setdefault(text, row)
        if previous != row and not np.array_equal(vectors[previous], vectors[row]):
            raise ValueError(
                "duplicate corpus text maps to different cached vectors; "
                "text-only official lookup would be ambiguous"
            )

    class Metrics:
        def __init__(self) -> None:
            self.lock = threading.Lock()
            self.requests = 0
            self.inputs = 0
            self.http_errors = 0
            self.status_counts: dict[str, int] = {}

        def request(self, inputs: int) -> None:
            with self.lock:
                self.requests += 1
                self.inputs += inputs

        def response(self, status: int) -> None:
            with self.lock:
                key = str(status)
                self.status_counts[key] = self.status_counts.get(key, 0) + 1
                if status >= 400:
                    self.http_errors += 1

        def _snapshot(self) -> dict[str, object]:
            return {
                "mode": "cache",
                "cache_identity": cache_identity,
                "requests": self.requests,
                "inputs": self.inputs,
                "upstream_requests": self.requests,
                "upstream_responses": sum(self.status_counts.values()),
                "http_errors": self.http_errors,
                "network_errors": 0,
                "status_counts": dict(sorted(self.status_counts.items())),
            }

        def snapshot(self) -> dict[str, object]:
            with self.lock:
                return self._snapshot()

        def reset(self) -> dict[str, object]:
            with self.lock:
                previous = self._snapshot()
                self.requests = 0
                self.inputs = 0
                self.http_errors = 0
                self.status_counts.clear()
                return previous

    metrics = Metrics()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, status: int, payload: object, *, record: bool = True) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            if record:
                metrics.response(status)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/metrics":
                self._send(200, metrics.snapshot(), record=False)
                return
            self._send(404, {"error": {"message": "not found"}})

        def do_POST(self) -> None:  # noqa: N802
            if self.path == "/metrics/reset":
                self._send(200, metrics.reset(), record=False)
                return
            if not self.path.endswith("/embeddings"):
                self._send(404, {"error": {"message": "not found"}})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > args.max_request_bytes:
                    self._send(413, {"error": {"message": "invalid request size"}})
                    return
                request = json.loads(self.rfile.read(length))
                if not isinstance(request, dict):
                    raise ValueError("request body must be an object")
                inputs = request.get("input", [])
                if isinstance(inputs, str):
                    inputs = [inputs]
                if not isinstance(inputs, list) or not all(
                    isinstance(text, str) for text in inputs
                ):
                    raise ValueError("input must be a string or list of strings")
                metrics.request(len(inputs))
                data = [
                    {
                        "object": "embedding",
                        "index": idx,
                        "embedding": vectors[text_to_row[text]].tolist(),
                    }
                    for idx, text in enumerate(inputs)
                ]
                payload = {
                    "object": "list",
                    "data": data,
                    "model": request.get("model", args.model),
                    "usage": {"prompt_tokens": 0, "total_tokens": 0},
                }
                self._send(200, payload)
            except KeyError as error:
                self._send(
                    400,
                    {"error": {"message": f"Unknown corpus text: {str(error)[:120]}"}}
                )
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
                self._send(400, {"error": {"message": str(error)[:240]}})

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(
        f"cached embedding server listening on http://{args.host}:{args.port} "
        f"with {len(documents)} vectors ({cache_meta['schema']})",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
