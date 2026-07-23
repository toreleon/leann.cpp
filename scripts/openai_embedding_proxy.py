#!/usr/bin/env python3
"""Tiny counting proxy for an OpenAI-compatible embeddings endpoint."""

from __future__ import annotations

import argparse
import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests = 0
        self.inputs = 0

    def add(self, inputs: int) -> None:
        with self._lock:
            self.requests += 1
            self.inputs += inputs

    def reset(self) -> dict[str, int]:
        with self._lock:
            previous = {"requests": self.requests, "inputs": self.inputs}
            self.requests = 0
            self.inputs = 0
            return previous

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {"requests": self.requests, "inputs": self.inputs}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=18081)
    parser.add_argument("--upstream", default="http://127.0.0.1:18080")
    args = parser.parse_args()

    metrics = Metrics()
    upstream = args.upstream.rstrip("/")

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
            try:
                with urllib.request.urlopen(request, timeout=300) as response:
                    payload = response.read()
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
                self.send_response(error.code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

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
