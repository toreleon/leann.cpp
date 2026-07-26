#!/usr/bin/env python3
"""Chunk a directory of text or markdown files into a leann.cpp --docs file.

`leann build --docs FILE` reads one document per LF-delimited line with no
escaping, so a chunk must never contain a newline. This script walks a
directory in sorted order, splits each file on blank lines, greedily packs
paragraphs up to a byte budget, and writes one chunk per line.

The budget is in BYTES, not tokens, and that is deliberate. Counting tokens
would mean loading the GGUF vocabulary and pinning this script to one model.
A byte budget instead gives a guarantee that holds for every tokenizer: a
chunk of N bytes can never produce more than N tokens, because no token is
shorter than one byte. So `--ctx >= max_chunk_bytes` always builds, and a
corpus whose measured bytes-per-token ratio is better than 1.0 (English prose
runs near 4) can safely use a much smaller --ctx. The reported
max_chunk_bytes is what to size --ctx against.

The written chunk file is deterministic: the same directory and options produce
byte-identical bytes, with no timestamps, hostnames, or absolute paths in it.
The JSON summary printed to stdout does echo the --output path it was given.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import sys
from collections.abc import Iterator, Sequence


DEFAULT_SUFFIXES = (".md", ".txt")
PARAGRAPH_BREAK = re.compile(r"\n\s*\n")


class ChunkError(RuntimeError):
    """A corpus could not be chunked into valid --docs input."""


def clean_line(value: str) -> str:
    """Collapse all whitespace, matching the other preparation scripts.

    NUL is replaced rather than dropped so byte offsets stay sane, and every
    run of whitespace — including the newlines that would corrupt the
    line-delimited output format — becomes a single space.
    """
    return " ".join(value.replace("\x00", " ").split())


def source_files(
    root: pathlib.Path, suffixes: Sequence[str]
) -> list[pathlib.Path]:
    """Every matching file under root, in a stable sorted order."""
    # `--suffix md` is the natural thing to type, so accept it rather than
    # silently matching nothing and reporting an empty corpus.
    wanted = {
        (suffix if suffix.startswith(".") else "." + suffix).lower()
        for suffix in suffixes
    }
    found: list[pathlib.Path] = []
    for directory, subdirectories, names in os.walk(root):
        # Skip version-control and dependency trees rather than chunking them.
        subdirectories[:] = sorted(
            name
            for name in subdirectories
            if name not in {".git", ".github", "node_modules", "__pycache__"}
        )
        for name in sorted(names):
            path = pathlib.Path(directory) / name
            if path.suffix.lower() in wanted:
                found.append(path)
    return sorted(found)


def split_bytes(text: str, max_bytes: int) -> Iterator[str]:
    """Split an over-long paragraph on a character boundary under max_bytes.

    Slicing the str rather than the encoded bytes keeps every piece valid
    UTF-8; a multi-byte character is never cut in half. The cut point is found
    by bisection rather than by stepping back one character at a time, which
    on multi-byte text would re-encode a near-full slice on every step.
    """
    remaining = text
    while len(remaining.encode("utf-8")) > max_bytes:
        low, high = 0, min(len(remaining), max_bytes)
        while low < high:
            middle = (low + high + 1) // 2
            if len(remaining[:middle].encode("utf-8")) <= max_bytes:
                low = middle
            else:
                high = middle - 1
        if low == 0:
            raise ChunkError(
                f"--max-bytes {max_bytes} is too small for a single character"
            )
        yield remaining[:low]
        remaining = remaining[low:]
    if remaining:
        yield remaining


def chunk_text(text: str, max_bytes: int) -> list[str]:
    """Greedily pack cleaned paragraphs into chunks of at most max_bytes."""
    chunks: list[str] = []
    current = ""
    for raw in PARAGRAPH_BREAK.split(text):
        paragraph = clean_line(raw)
        if not paragraph:
            continue
        candidate = paragraph if not current else f"{current} {paragraph}"
        if len(candidate.encode("utf-8")) <= max_bytes:
            current = candidate
            continue
        if current:
            chunks.append(current)
            current = ""
        pieces = list(split_bytes(paragraph, max_bytes))
        chunks.extend(pieces[:-1])
        current = pieces[-1] if pieces else ""
    if current:
        chunks.append(current)
    return chunks


def chunk_corpus(
    root: pathlib.Path, max_bytes: int, suffixes: Sequence[str]
) -> tuple[list[str], list[str]]:
    """Return (chunks, relative source paths) for a directory tree."""
    files = source_files(root, suffixes)
    if not files:
        raise ChunkError(f"{root}: no files matching {', '.join(suffixes)}")
    chunks: list[str] = []
    relative: list[str] = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as error:
            raise ChunkError(f"{path}: cannot read: {error}") from error
        produced = chunk_text(text, max_bytes)
        if produced:
            # Recorded relative to the root so the output says nothing about
            # where this machine keeps the corpus.
            relative.append(path.relative_to(root).as_posix())
            chunks.extend(produced)
    if not chunks:
        raise ChunkError(f"{root}: every file was empty after cleaning")
    return chunks, relative


def write_chunks(path: pathlib.Path, chunks: Sequence[str]) -> None:
    """Write one chunk per line, refusing anything the format cannot carry."""
    for chunk in chunks:
        if "\n" in chunk or "\r" in chunk:
            raise ChunkError("a chunk contains a line break")
        if not chunk.strip():
            raise ChunkError("a chunk is empty after cleaning")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for chunk in chunks:
            output.write(chunk + "\n")


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=pathlib.Path,
        required=True,
        help="directory to walk",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        required=True,
        help="line-oriented --docs file to write",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=1200,
        help="byte budget per chunk (default 1200)",
    )
    parser.add_argument(
        "--suffix",
        action="append",
        default=None,
        help=(
            "file suffix to include; repeatable "
            f"(default {' '.join(DEFAULT_SUFFIXES)})"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    try:
        if arguments.max_bytes < 16:
            raise ChunkError("--max-bytes must be at least 16")
        suffixes = arguments.suffix or list(DEFAULT_SUFFIXES)
        # Writing into the tree being walked would make a second run ingest
        # the first run's output, silently doubling the corpus and breaking
        # the determinism this script otherwise guarantees.
        resolved_input = arguments.input.resolve()
        resolved_output = arguments.output.resolve()
        if resolved_input in resolved_output.parents:
            raise ChunkError(
                f"--output {arguments.output} is inside --input "
                f"{arguments.input}; a later run would re-ingest it"
            )
        chunks, files = chunk_corpus(
            arguments.input, arguments.max_bytes, suffixes
        )
        write_chunks(arguments.output, chunks)
        encoded = [len(chunk.encode("utf-8")) for chunk in chunks]
        summary = {
            "files": len(files),
            "chunks": len(chunks),
            "total_bytes": sum(encoded),
            "max_chunk_bytes": max(encoded),
            "mean_chunk_bytes": round(sum(encoded) / len(encoded), 1),
            "output": str(arguments.output),
            "sha256": hashlib.sha256(
                arguments.output.read_bytes()
            ).hexdigest(),
            # Restating the guarantee where the operator will actually read
            # it, since picking --ctx wrong is a build failure, not a warning.
            "safe_ctx_tokens": max(encoded),
        }
    except (ChunkError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
