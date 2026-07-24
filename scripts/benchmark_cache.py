#!/usr/bin/env python3
"""Shared readers and writers for leann.cpp benchmark vector caches.

The module deliberately has no dependency beyond NumPy.  It supports the
legacy ``LEANNBC1`` cache and the integrity-bound ``LEANNBC2`` cache.  V2
binds the vectors to the exact bytes of the source line file.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np


MAGIC_V1 = b"LEANNBC1"
MAGIC_V2 = b"LEANNBC2"
MAX_FINGERPRINT_BYTES = 16 * 1024 * 1024
V1_FIXED = struct.Struct("<8sIQI")
V2_FIXED = struct.Struct("<8sIQIQ32s")


def sha256_file(path: Path, *, offset: int = 0, length: int | None = None) -> str:
    digest = hashlib.sha256()
    remaining = length
    with path.open("rb") as source:
        source.seek(offset)
        while remaining is None or remaining:
            size = 1024 * 1024 if remaining is None else min(1024 * 1024, remaining)
            block = source.read(size)
            if not block:
                break
            digest.update(block)
            if remaining is not None:
                remaining -= len(block)
    if remaining not in (None, 0):
        raise ValueError(f"{path}: truncated while hashing")
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as target:
            json.dump(payload, target, indent=2, sort_keys=True)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as target:
            target.write(payload)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def iter_nonempty_lines(path: Path) -> Iterator[str]:
    """Match the native CLI's line reader: strip LF/terminal CR, skip empties."""

    with path.open("r", encoding="utf-8", newline="") as source:
        for line in source:
            if line.endswith("\n"):
                line = line[:-1]
            if line.endswith("\r"):
                line = line[:-1]
            if line:
                yield line


def count_nonempty_lines(path: Path) -> int:
    return sum(1 for _ in iter_nonempty_lines(path))


def source_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
        "nonempty_lines": count_nonempty_lines(resolved),
    }


def _decode_fingerprint(raw: bytes, path: Path) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{path}: cache fingerprint is not UTF-8") from error


def read_cache(
    path: Path,
    *,
    validate_vectors: bool = True,
    require_unit: bool | None = None,
    validation_rows: int = 8192,
) -> tuple[np.memmap, dict[str, Any]]:
    """Open a V1/V2 cache after validating its complete structural envelope."""

    path = path.resolve()
    file_size = path.stat().st_size
    with path.open("rb") as source:
        magic = source.read(8)
        source.seek(0)
        if magic == MAGIC_V1:
            fixed = source.read(V1_FIXED.size)
            if len(fixed) != V1_FIXED.size:
                raise ValueError(f"{path}: truncated V1 cache header")
            _, dimension, count, fingerprint_size = V1_FIXED.unpack(fixed)
            source_size = None
            source_sha256 = None
            version = 1
        elif magic == MAGIC_V2:
            fixed = source.read(V2_FIXED.size)
            if len(fixed) != V2_FIXED.size:
                raise ValueError(f"{path}: truncated V2 cache header")
            (
                _,
                dimension,
                count,
                fingerprint_size,
                source_size,
                source_sha256_raw,
            ) = V2_FIXED.unpack(fixed)
            source_sha256 = source_sha256_raw.hex()
            version = 2
        else:
            raise ValueError(f"{path}: unexpected cache magic {magic!r}")
        if dimension == 0 or count == 0:
            raise ValueError(f"{path}: cache dimension/count must be positive")
        if fingerprint_size > MAX_FINGERPRINT_BYTES:
            raise ValueError(f"{path}: unreasonable fingerprint length")
        fingerprint_raw = source.read(fingerprint_size)
        if len(fingerprint_raw) != fingerprint_size:
            raise ValueError(f"{path}: truncated cache fingerprint")
        fingerprint = _decode_fingerprint(fingerprint_raw, path)
        vector_offset = source.tell()

    vector_bytes = count * dimension * np.dtype("<f4").itemsize
    expected_size = vector_offset + vector_bytes
    if file_size != expected_size:
        raise ValueError(
            f"{path}: cache length mismatch: expected {expected_size}, got {file_size}"
        )
    vectors = np.memmap(
        path,
        mode="r",
        dtype="<f4",
        offset=vector_offset,
        shape=(count, dimension),
    )
    if require_unit is None:
        require_unit = version == 2
    if validate_vectors:
        validate_vector_matrix(
            vectors,
            require_unit=require_unit,
            block_rows=validation_rows,
            label=str(path),
        )
    metadata: dict[str, Any] = {
        "schema": f"LEANNBC{version}",
        "version": version,
        "path": str(path),
        "count": int(count),
        "dimensions": int(dimension),
        "fingerprint": fingerprint,
        "fingerprint_sha256": hashlib.sha256(fingerprint_raw).hexdigest(),
        "vector_offset": vector_offset,
        "vector_bytes": vector_bytes,
        "size_bytes": file_size,
    }
    if version == 2:
        metadata["source_size_bytes"] = int(source_size)
        metadata["source_sha256"] = source_sha256
    return vectors, metadata


def validate_vector_matrix(
    vectors: np.ndarray,
    *,
    require_unit: bool = True,
    block_rows: int = 8192,
    unit_tolerance: float = 2e-3,
    label: str = "vectors",
) -> None:
    if vectors.ndim != 2 or not vectors.shape[0] or not vectors.shape[1]:
        raise ValueError(f"{label}: expected a non-empty rank-2 matrix")
    if block_rows <= 0:
        raise ValueError("block_rows must be positive")
    for begin in range(0, vectors.shape[0], block_rows):
        block = np.asarray(vectors[begin : begin + block_rows], dtype=np.float32)
        if not np.isfinite(block).all():
            raise ValueError(f"{label}: NaN or infinity in rows at/after {begin}")
        norms = np.linalg.norm(block, axis=1)
        if np.any(norms <= np.finfo(np.float32).tiny):
            raise ValueError(f"{label}: zero-length vector at/after row {begin}")
        if require_unit and np.any(np.abs(norms - 1.0) > unit_tolerance):
            worst = float(np.max(np.abs(norms - 1.0)))
            raise ValueError(
                f"{label}: vectors are not normalized (worst norm error {worst:.6g})"
            )


def validate_cache_source(
    metadata: dict[str, Any],
    source_path: Path,
    *,
    expected_count: int | None = None,
    require_integrity: bool = True,
) -> dict[str, Any]:
    identity = source_identity(source_path)
    if expected_count is None:
        expected_count = identity["nonempty_lines"]
    if metadata["count"] != expected_count:
        raise ValueError(
            f"cache/source count mismatch: {metadata['count']} vs {expected_count}"
        )
    if metadata["version"] == 2:
        if metadata["source_size_bytes"] != identity["size_bytes"]:
            raise ValueError("V2 cache/source byte-size mismatch")
        if metadata["source_sha256"] != identity["sha256"]:
            raise ValueError("V2 cache/source SHA-256 mismatch")
    elif require_integrity:
        raise ValueError("legacy V1 cache is not bound to the source file")
    return identity


def cache_v2_header(
    *,
    dimension: int,
    count: int,
    fingerprint: str,
    source_size_bytes: int,
    source_sha256: str,
) -> bytes:
    fingerprint_raw = fingerprint.encode("utf-8")
    if (
        dimension <= 0
        or count <= 0
        or not fingerprint_raw
        or len(fingerprint_raw) > MAX_FINGERPRINT_BYTES
    ):
        raise ValueError("invalid V2 cache header values")
    try:
        source_digest = bytes.fromhex(source_sha256)
    except ValueError as error:
        raise ValueError("source_sha256 is not hexadecimal") from error
    if len(source_digest) != 32 or source_size_bytes < 0:
        raise ValueError("invalid V2 source identity")
    return V2_FIXED.pack(
        MAGIC_V2,
        dimension,
        count,
        len(fingerprint_raw),
        source_size_bytes,
        source_digest,
    ) + fingerprint_raw


def normalize_vectors(values: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or not matrix.shape[0] or not matrix.shape[1]:
        raise ValueError("embedding response must be a non-empty rank-2 matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("embedding response contains NaN or infinity")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= np.finfo(np.float32).tiny):
        raise ValueError("embedding response contains a zero-length vector")
    matrix = np.ascontiguousarray(matrix / norms, dtype="<f4")
    validate_vector_matrix(matrix, require_unit=True)
    return matrix


def write_cache_v2(
    path: Path,
    vectors: np.ndarray,
    *,
    fingerprint: str,
    source_path: Path,
) -> dict[str, Any]:
    normalized = normalize_vectors(vectors)
    identity = source_identity(source_path)
    if normalized.shape[0] != identity["nonempty_lines"]:
        raise ValueError("vector/source count mismatch")
    header = cache_v2_header(
        dimension=normalized.shape[1],
        count=normalized.shape[0],
        fingerprint=fingerprint,
        source_size_bytes=identity["size_bytes"],
        source_sha256=identity["sha256"],
    )
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as target:
            target.write(header)
            target.write(normalized.tobytes(order="C"))
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    _, metadata = read_cache(path)
    metadata["sha256"] = sha256_file(path)
    return metadata


def read_ground_truth(
    path: Path,
    *,
    expected_queries: int | None = None,
    expected_k: int | None = None,
    expected_corpus: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    lines = list(iter_nonempty_lines(path))
    if not lines:
        raise ValueError(f"{path}: empty ground-truth file")
    header = lines[0].split()
    if len(header) != 4 or header[0] != "LEANN_GT1":
        raise ValueError(f"{path}: invalid ground-truth header")
    try:
        query_count, top_k, corpus_count = map(int, header[1:])
    except ValueError as error:
        raise ValueError(f"{path}: non-integer ground-truth header") from error
    if query_count <= 0 or top_k <= 0 or corpus_count < top_k:
        raise ValueError(f"{path}: invalid ground-truth dimensions")
    if len(lines) != query_count + 1:
        raise ValueError(f"{path}: ground-truth row count mismatch")
    truth = np.empty((query_count, top_k), dtype=np.uint32)
    for row_index, line in enumerate(lines[1:]):
        fields = line.split()
        if len(fields) != top_k:
            raise ValueError(f"{path}: row {row_index} does not have k IDs")
        try:
            identifiers = [int(field) for field in fields]
        except ValueError as error:
            raise ValueError(f"{path}: non-integer ID in row {row_index}") from error
        if any(identifier < 0 or identifier >= corpus_count for identifier in identifiers):
            raise ValueError(f"{path}: out-of-range ID in row {row_index}")
        if len(set(identifiers)) != top_k:
            raise ValueError(f"{path}: duplicate ID in row {row_index}")
        truth[row_index] = identifiers
    expected = (
        (expected_queries, query_count, "query count"),
        (expected_k, top_k, "top-k"),
        (expected_corpus, corpus_count, "corpus count"),
    )
    for wanted, observed, label in expected:
        if wanted is not None and wanted != observed:
            raise ValueError(
                f"{path}: ground-truth {label} mismatch: {observed} vs {wanted}"
            )
    return truth, {
        "schema": "LEANN_GT1",
        "path": str(path.resolve()),
        "query_count": query_count,
        "top_k": top_k,
        "corpus_count": corpus_count,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def snapshot_files(paths: Iterable[Path]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for candidate in sorted((path.resolve() for path in paths), key=str):
        if candidate in seen or not candidate.is_file():
            continue
        seen.add(candidate)
        result.append(
            {
                "path": str(candidate),
                "size_bytes": candidate.stat().st_size,
                "sha256": sha256_file(candidate),
            }
        )
    return result
