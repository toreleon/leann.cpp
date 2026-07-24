#!/usr/bin/env python3
"""Tiny counting proxy for an OpenAI-compatible embeddings endpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Metrics:
    def __init__(self, upstream: str) -> None:
        self._lock = threading.Lock()
        self.upstream = upstream
        self.script_sha256 = file_sha256(Path(__file__))
        self.requests = 0
        self.inputs = 0
        self.upstream_requests = 0
        self.upstream_responses = 0
        self.http_errors = 0
        self.network_errors = 0
        self.status_counts: dict[str, int] = {}

    def add(self, inputs: int) -> None:
        with self._lock:
            self.requests += 1
            self.inputs += inputs

    def upstream_started(self) -> None:
        with self._lock:
            self.upstream_requests += 1

    def upstream_finished(self, status: int) -> None:
        with self._lock:
            self.upstream_responses += 1
            key = str(status)
            self.status_counts[key] = self.status_counts.get(key, 0) + 1
            if status >= 400:
                self.http_errors += 1

    def network_failed(self) -> None:
        with self._lock:
            self.network_errors += 1

    def reset(self) -> dict[str, int]:
        with self._lock:
            previous = self._snapshot_unlocked()
            self._reset_unlocked()
            return previous

    def _snapshot_unlocked(self) -> dict[str, object]:
        return {
            "mode": "proxy",
            "upstream": self.upstream,
            "proxy_script_sha256": self.script_sha256,
            "requests": self.requests,
            "inputs": self.inputs,
            "upstream_requests": self.upstream_requests,
            "upstream_responses": self.upstream_responses,
            "http_errors": self.http_errors,
            "network_errors": self.network_errors,
            "status_counts": dict(sorted(self.status_counts.items())),
        }

    def _reset_unlocked(self) -> None:
        self.requests = 0
        self.inputs = 0
        self.upstream_requests = 0
        self.upstream_responses = 0
        self.http_errors = 0
        self.network_errors = 0
        self.status_counts.clear()

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return self._snapshot_unlocked()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=18081)
    parser.add_argument("--upstream", default="http://127.0.0.1:18080")
    args = parser.parse_args()

    upstream = args.upstream.rstrip("/")
    metrics = Metrics(upstream)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _json(self, status: int, payload: object) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/metrics":
                self._json(200, metrics.snapshot())
                return
            self._proxy()

        def do_POST(self) -> None:  # noqa: N802
            if self.path == "/metrics/reset":
                self._json(200, metrics.reset())
                return
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            if self.path.endswith("/embeddings"):
                try:
                    request = json.loads(body)
                    items = request.get("input", [])
                    metrics.add(len(items) if isinstance(items, list) else 1)
                except (json.JSONDecodeError, AttributeError):
                    metrics.add(0)
            self._proxy(body)

        def _proxy(self, body: bytes | None = None) -> None:
            headers = {
                key: value
                for key, value in self.headers.items()
                if key.lower() not in {"host", "content-length", "connection"}
            }
            request = urllib.request.Request(
                upstream + self.path,
                data=body,
                headers=headers,
                method=self.command,
            )
            metrics.upstream_started()
            try:
                with urllib.request.urlopen(request, timeout=300) as response:
                    payload = response.read()
                    metrics.upstream_finished(response.status)
                    self.send_response(response.status)
                    for key, value in response.headers.items():
                        if key.lower() not in {
                            "content-length",
                            "transfer-encoding",
                            "connection",
                        }:
                            self.send_header(key, value)
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
            except urllib.error.HTTPError as error:
                payload = error.read()
                metrics.upstream_finished(error.code)
                self.send_response(error.code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except (urllib.error.URLError, TimeoutError) as error:
                metrics.network_failed()
                self._json(
                    502,
                    {
                        "error": {
                            "message": f"embedding upstream unavailable: {error}"
                        }
                    },
                )

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer((args.listen_host, args.listen_port), Handler)
    print(
        f"embedding proxy listening on http://{args.listen_host}:{args.listen_port}, "
        f"upstream={upstream}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
