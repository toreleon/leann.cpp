#!/usr/bin/env python3
"""OpenAI-compatible server backed by the exact leann.cpp embedding cache.

This is an algorithmic benchmark aid: it preserves vectors exactly while
removing model inference from official LEANN's candidate-recompute path.
"""

from __future__ import annotations

import argparse
import json
import struct
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np


def load_vectors(path: Path) -> np.memmap:
    with path.open("rb") as handle:
        if handle.read(8) != b"LEANNBC1":
            raise ValueError("Not a LEANN benchmark cache")
        dim = struct.unpack("<I", handle.read(4))[0]
        count = struct.unpack("<Q", handle.read(8))[0]
        fingerprint_size = struct.unpack("<I", handle.read(4))[0]
        handle.seek(fingerprint_size, 1)
        offset = handle.tell()
    return np.memmap(
        path, mode="r", dtype="<f4", offset=offset, shape=(count, dim)
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
    args = parser.parse_args()

    documents = args.documents.read_text(encoding="utf-8").splitlines()
    vectors = load_vectors(args.cache)
    if len(documents) != len(vectors):
        raise ValueError(f"Document/cache mismatch: {len(documents)} vs {len(vectors)}")
    text_to_row = {text: row for row, text in enumerate(documents)}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length))
            inputs = request.get("input", [])
            if isinstance(inputs, str):
                inputs = [inputs]
            try:
                data = [
                    {
                        "object": "embedding",
                        "index": idx,
                        "embedding": vectors[text_to_row[text]].tolist(),
                    }
                    for idx, text in enumerate(inputs)
                ]
                payload = json.dumps(
                    {
                        "object": "list",
                        "data": data,
                        "model": request.get("model", "nomic-embed-text"),
                        "usage": {"prompt_tokens": 0, "total_tokens": 0},
                    }
                ).encode("utf-8")
                status = 200
            except KeyError as error:
                payload = json.dumps(
                    {"error": {"message": f"Unknown corpus text: {str(error)[:120]}"}}
                ).encode("utf-8")
                status = 400
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(
        f"cached embedding server listening on http://{args.host}:{args.port} "
        f"with {len(documents)} vectors",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
