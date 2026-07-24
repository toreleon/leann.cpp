#!/usr/bin/env python3
"""Prepare deterministic 100K and 1M BEIR NQ benchmark corpora.

The default acquisition path downloads the official BEIR Natural Questions
archive and verifies its published MD5 before extracting it.  ``--archive``
and ``--source-dir`` make the same preparation pipeline usable with an
already-downloaded archive or a small local fixture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import re
import shutil
import sqlite3
import stat
import sys
import tempfile
import urllib.request
import uuid
import zipfile
from dataclasses import dataclass
from typing import BinaryIO, Sequence


SOURCE_URL = (
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/"
    "BEIR/datasets/nq.zip"
)
EXPECTED_MD5 = "d4d3d2e48787a744b6f6e691ff534307"
SCRIPT_VERSION = 3
READ_BLOCK_BYTES = 1024 * 1024
SQL_BATCH_ROWS = 5_000
DEFAULT_MAX_EXTRACTED_BYTES = 64 * 1024**3
DEFAULT_MAX_ARCHIVE_MEMBERS = 100_000
DEFAULT_MAX_DOCUMENT_BYTES = 2_000
MAX_SAFE_DOCUMENT_BYTES = 2_000
DOCUMENT_PREFIX = "search_document: "


@dataclass(frozen=True)
class FileDigest:
    bytes: int
    md5: str
    sha256: str


@dataclass(frozen=True)
class SourceFileStats:
    bytes: int
    lines: int
    sha256: str


@dataclass(frozen=True)
class Qrel:
    query_id: str
    corpus_id: str
    score_text: str
    score: float


@dataclass(frozen=True)
class PreparedDocument:
    text: str
    raw_text_bytes: int
    original_prepared_bytes: int
    truncated: bool


class TrackedBinaryWriter:
    """Write deterministic UTF-8 lines while collecting file metadata."""

    def __init__(self, path: pathlib.Path) -> None:
        self.path = path
        self._handle: BinaryIO = path.open("wb")
        self._sha256 = hashlib.sha256()
        self.bytes = 0
        self.lines = 0

    def write_line(self, value: str) -> None:
        encoded = value.encode("utf-8") + b"\n"
        self._handle.write(encoded)
        self._sha256.update(encoded)
        self.bytes += len(encoded)
        self.lines += 1

    def close(self) -> dict[str, object]:
        self._handle.flush()
        self._handle.close()
        return {
            "path": self.path.name,
            "bytes": self.bytes,
            "lines": self.lines,
            "sha256": self._sha256.hexdigest(),
        }

    def __enter__(self) -> "TrackedBinaryWriter":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if not self._handle.closed:
            self._handle.close()


def file_digest(path: pathlib.Path) -> FileDigest:
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as source:
        for block in iter(lambda: source.read(READ_BLOCK_BYTES), b""):
            byte_count += len(block)
            md5.update(block)
            sha256.update(block)
    return FileDigest(byte_count, md5.hexdigest(), sha256.hexdigest())


def clean_line(value: object, *, field: str, identifier: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise RuntimeError(
            f"{field} for {identifier!r} must be a string, got "
            f"{type(value).__name__}"
        )
    return " ".join(value.replace("\x00", " ").split())


def truncate_utf8(value: str, max_bytes: int) -> tuple[str, bool]:
    """Return the longest valid UTF-8 prefix no larger than ``max_bytes``."""

    encoded = value.encode("utf-8")
    if len(encoded) <= max_bytes:
        return value, False
    truncated = encoded[:max_bytes].decode("utf-8", errors="ignore").rstrip()
    if not truncated:
        raise RuntimeError("document byte budget removed all prepared text")
    if len(truncated.encode("utf-8")) > max_bytes:
        raise AssertionError("UTF-8 truncation exceeded its byte budget")
    return truncated, True


def document_text(
    record: dict[str, object],
    identifier: str,
    *,
    max_document_bytes: int,
) -> PreparedDocument:
    title = clean_line(record.get("title", ""), field="title", identifier=identifier)
    text = clean_line(record.get("text", ""), field="text", identifier=identifier)
    content = f"{title}. {text}" if title and text else title or text
    if not content:
        raise RuntimeError(f"corpus document {identifier!r} has no usable text")
    original = f"{DOCUMENT_PREFIX}{content}"
    prepared, truncated = truncate_utf8(original, max_document_bytes)
    if not prepared.startswith(DOCUMENT_PREFIX):
        raise RuntimeError(
            f"document byte budget is too small to preserve the Nomic prefix: "
            f"{identifier!r}"
        )
    return PreparedDocument(
        text=prepared,
        raw_text_bytes=len(content.encode("utf-8")),
        original_prepared_bytes=len(original.encode("utf-8")),
        truncated=truncated,
    )


def query_text(record: dict[str, object], identifier: str) -> tuple[str, int]:
    text = clean_line(record.get("text"), field="text", identifier=identifier)
    if not text:
        raise RuntimeError(f"query {identifier!r} has no usable text")
    return f"search_query: {text}", len(text.encode("utf-8"))


def identifier(record: dict[str, object], *, kind: str, line_number: int) -> str:
    if "_id" not in record:
        raise RuntimeError(f"{kind} line {line_number} is missing _id")
    result = str(record["_id"])
    if not result:
        raise RuntimeError(f"{kind} line {line_number} has an empty _id")
    if "\t" in result or "\n" in result or "\r" in result:
        raise RuntimeError(f"{kind} ID {result!r} is not TSV-safe")
    return result


def rank_hash(seed: str, value: str) -> bytes:
    return hashlib.sha256((seed + value).encode("utf-8")).digest()


def validate_rate(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be a finite number in [0, 1]")


def validate_expected_md5(value: str) -> str:
    normalized = value.lower()
    if re.fullmatch(r"[0-9a-f]{32}", normalized) is None:
        raise ValueError("--expected-md5 must be exactly 32 hexadecimal characters")
    return normalized


def download_archive(
    destination: pathlib.Path,
    *,
    expected_md5: str,
    timeout_seconds: int,
) -> FileDigest:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        observed = file_digest(destination)
        if observed.md5 == expected_md5:
            return observed

    temporary = destination.with_name(
        f".{destination.name}.download-{uuid.uuid4().hex}"
    )
    request = urllib.request.Request(
        SOURCE_URL,
        headers={"User-Agent": "leann.cpp-nq-scale-preparer/1"},
    )
    try:
        md5 = hashlib.md5()
        sha256 = hashlib.sha256()
        byte_count = 0
        with (
            urllib.request.urlopen(request, timeout=timeout_seconds) as response,
            temporary.open("xb") as output,
        ):
            for block in iter(lambda: response.read(READ_BLOCK_BYTES), b""):
                output.write(block)
                byte_count += len(block)
                md5.update(block)
                sha256.update(block)
        observed = FileDigest(byte_count, md5.hexdigest(), sha256.hexdigest())
        if observed.md5 != expected_md5:
            raise RuntimeError(
                "BEIR NQ archive MD5 mismatch: "
                f"expected {expected_md5}, got {observed.md5}"
            )
        temporary.replace(destination)
        return observed
    finally:
        temporary.unlink(missing_ok=True)


def verify_archive(path: pathlib.Path, expected_md5: str) -> FileDigest:
    if not path.is_file():
        raise RuntimeError(f"archive does not exist or is not a file: {path}")
    observed = file_digest(path)
    if observed.md5 != expected_md5:
        raise RuntimeError(
            "BEIR NQ archive MD5 mismatch: "
            f"expected {expected_md5}, got {observed.md5}"
        )
    return observed


def safe_extract(
    archive: pathlib.Path,
    destination: pathlib.Path,
    *,
    max_members: int,
    max_uncompressed_bytes: int,
) -> None:
    """Extract a ZIP without permitting traversal, links, or archive bombs."""

    destination.mkdir(parents=True, exist_ok=False)
    destination_resolved = destination.resolve()
    seen_targets: set[str] = set()
    with zipfile.ZipFile(archive) as compressed:
        members = compressed.infolist()
        if len(members) > max_members:
            raise RuntimeError(
                f"archive contains {len(members)} members; limit is {max_members}"
            )
        total_size = sum(member.file_size for member in members)
        if total_size > max_uncompressed_bytes:
            raise RuntimeError(
                f"archive expands to {total_size} bytes; "
                f"limit is {max_uncompressed_bytes}"
            )

        for member in members:
            name = member.filename
            if not name or "\x00" in name or "\\" in name:
                raise RuntimeError(f"unsafe zip member name: {name!r}")
            member_path = pathlib.PurePosixPath(name)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise RuntimeError(f"unsafe zip member path: {name!r}")
            if member.flag_bits & 0x1:
                raise RuntimeError(f"encrypted zip member is unsupported: {name!r}")

            unix_mode = (member.external_attr >> 16) & 0xFFFF
            if unix_mode and stat.S_ISLNK(unix_mode):
                raise RuntimeError(f"symbolic link in zip is forbidden: {name!r}")

            target = destination.joinpath(*member_path.parts)
            target_resolved = target.resolve()
            if (
                target_resolved != destination_resolved
                and destination_resolved not in target_resolved.parents
            ):
                raise RuntimeError(f"unsafe zip member path: {name!r}")
            target_key = str(target_resolved).casefold()
            if target_key in seen_targets:
                raise RuntimeError(f"duplicate zip extraction target: {name!r}")
            seen_targets.add(target_key)

            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if unix_mode and not stat.S_ISREG(unix_mode):
                raise RuntimeError(f"non-regular zip member is forbidden: {name!r}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with compressed.open(member) as source, target.open("xb") as output:
                copied = shutil.copyfileobj(source, output, READ_BLOCK_BYTES)
                if copied is not None:
                    raise AssertionError("unexpected copyfileobj return value")
            if target.stat().st_size != member.file_size:
                raise RuntimeError(f"short extraction for zip member: {name!r}")


def find_dataset_root(source: pathlib.Path) -> pathlib.Path:
    source = source.resolve()
    candidates: list[pathlib.Path] = []
    direct = source / "corpus.jsonl"
    if direct.is_file():
        candidates.append(source)
    for corpus in source.glob("*/corpus.jsonl"):
        candidates.append(corpus.parent)
    valid = [
        path
        for path in candidates
        if (path / "queries.jsonl").is_file()
        and (path / "qrels" / "test.tsv").is_file()
    ]
    unique = sorted(set(valid))
    if len(unique) != 1:
        raise RuntimeError(
            f"expected exactly one BEIR dataset under {source}, found {len(unique)}"
        )
    root = unique[0]
    for relative in (
        pathlib.Path("corpus.jsonl"),
        pathlib.Path("queries.jsonl"),
        pathlib.Path("qrels/test.tsv"),
    ):
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"source file must be a regular non-symlink: {path}")
        resolved = path.resolve()
        if resolved != source and source not in resolved.parents:
            raise RuntimeError(f"source file escapes source directory: {path}")
    return root


def parse_queries(
    path: pathlib.Path,
) -> tuple[dict[str, tuple[str, int]], SourceFileStats]:
    queries: dict[str, tuple[str, int]] = {}
    checksum = hashlib.sha256()
    byte_count = 0
    line_count = 0
    with path.open("rb") as source:
        for line_number, raw in enumerate(source, start=1):
            checksum.update(raw)
            byte_count += len(raw)
            line_count += 1
            if not raw.strip():
                raise RuntimeError(f"blank queries.jsonl line {line_number}")
            try:
                record = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise RuntimeError(
                    f"invalid queries.jsonl line {line_number}: {error}"
                ) from error
            if not isinstance(record, dict):
                raise RuntimeError(f"queries.jsonl line {line_number} is not an object")
            query_id = identifier(record, kind="query", line_number=line_number)
            if query_id in queries:
                raise RuntimeError(f"duplicate query ID: {query_id!r}")
            queries[query_id] = query_text(record, query_id)
    if not queries:
        raise RuntimeError("queries.jsonl is empty")
    return queries, SourceFileStats(byte_count, line_count, checksum.hexdigest())


def parse_qrels(
    path: pathlib.Path,
) -> tuple[list[Qrel], dict[str, set[str]], SourceFileStats]:
    qrels: list[Qrel] = []
    positives: dict[str, set[str]] = {}
    seen_pairs: set[tuple[str, str]] = set()
    checksum = hashlib.sha256()
    byte_count = 0
    line_count = 0
    with path.open("rb") as source:
        header_raw = source.readline()
        if not header_raw:
            raise RuntimeError("test qrels file is empty")
        checksum.update(header_raw)
        byte_count += len(header_raw)
        line_count += 1
        try:
            header = header_raw.decode("utf-8").rstrip("\r\n").split("\t")
        except UnicodeDecodeError as error:
            raise RuntimeError("test qrels header is not UTF-8") from error
        if len(header) != 3:
            raise RuntimeError("test qrels header must contain exactly three columns")

        for line_number, raw in enumerate(source, start=2):
            checksum.update(raw)
            byte_count += len(raw)
            line_count += 1
            try:
                fields = raw.decode("utf-8").rstrip("\r\n").split("\t")
            except UnicodeDecodeError as error:
                raise RuntimeError(
                    f"test qrels line {line_number} is not UTF-8"
                ) from error
            if len(fields) != 3 or any(not field for field in fields):
                raise RuntimeError(
                    f"test qrels line {line_number} must have three non-empty fields"
                )
            query_id, corpus_id, score_text = fields
            if any(
                "\t" in value or "\n" in value or "\r" in value
                for value in (query_id, corpus_id)
            ):
                raise RuntimeError(f"invalid qrel ID on line {line_number}")
            try:
                score = float(score_text)
            except ValueError as error:
                raise RuntimeError(
                    f"invalid qrel score on line {line_number}: {score_text!r}"
                ) from error
            if not math.isfinite(score):
                raise RuntimeError(f"non-finite qrel score on line {line_number}")
            pair = (query_id, corpus_id)
            if pair in seen_pairs:
                raise RuntimeError(
                    f"duplicate qrel pair on line {line_number}: {pair!r}"
                )
            seen_pairs.add(pair)
            qrels.append(Qrel(query_id, corpus_id, score_text, score))
            if score > 0.0:
                positives.setdefault(query_id, set()).add(corpus_id)
    if not positives:
        raise RuntimeError("test qrels contain no positive judgments")
    return (
        qrels,
        positives,
        SourceFileStats(byte_count, line_count, checksum.hexdigest()),
    )


def select_queries(
    queries: dict[str, tuple[str, int]],
    positives: dict[str, set[str]],
    *,
    seed: str,
    count: int,
) -> list[str]:
    missing = sorted(set(positives) - set(queries))
    if missing:
        preview = ", ".join(repr(value) for value in missing[:5])
        raise RuntimeError(
            f"{len(missing)} positive-qrel query IDs are absent from queries.jsonl: "
            f"{preview}"
        )
    pool = list(positives)
    if len(pool) < count:
        raise RuntimeError(
            f"only {len(pool)} test queries have positive qrels; requested {count}"
        )
    pool.sort(key=lambda value: (rank_hash(seed, value), value))
    return pool[:count]


def open_database(path: pathlib.Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA cache_size=-65536")
    connection.execute(
        """
        CREATE TABLE corpus (
            corpus_id TEXT PRIMARY KEY NOT NULL,
            rank_hash BLOB NOT NULL,
            text_sha256 BLOB NOT NULL,
            prepared_text TEXT NOT NULL,
            raw_text_bytes INTEGER NOT NULL,
            original_prepared_bytes INTEGER NOT NULL,
            prepared_bytes INTEGER NOT NULL,
            truncated INTEGER NOT NULL CHECK (truncated IN (0, 1)),
            is_positive INTEGER NOT NULL CHECK (is_positive IN (0, 1))
        ) WITHOUT ROWID
        """
    )
    return connection


def flush_corpus_batch(
    connection: sqlite3.Connection,
    batch: list[tuple[object, ...]],
) -> None:
    if not batch:
        return
    try:
        connection.executemany(
            """
            INSERT INTO corpus (
                corpus_id, rank_hash, text_sha256, prepared_text,
                raw_text_bytes, original_prepared_bytes, prepared_bytes,
                truncated, is_positive
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            batch,
        )
    except sqlite3.IntegrityError as error:
        raise RuntimeError(
            "corpus contains a duplicate ID or invalid indexed value"
        ) from error
    batch.clear()


def index_corpus(
    path: pathlib.Path,
    connection: sqlite3.Connection,
    *,
    positive_ids: set[str],
    seed: str,
    max_document_bytes: int,
) -> tuple[SourceFileStats, dict[str, int]]:
    checksum = hashlib.sha256()
    byte_count = 0
    line_count = 0
    prepared_bytes = 0
    original_prepared_bytes = 0
    raw_text_bytes = 0
    truncated_documents = 0
    truncated_bytes_removed = 0
    maximum_prepared_bytes = 0
    seen_positive: set[str] = set()
    batch: list[tuple[object, ...]] = []

    connection.execute("BEGIN")
    try:
        with path.open("rb") as source:
            for line_number, raw in enumerate(source, start=1):
                checksum.update(raw)
                byte_count += len(raw)
                line_count += 1
                if not raw.strip():
                    raise RuntimeError(f"blank corpus.jsonl line {line_number}")
                try:
                    record = json.loads(raw)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise RuntimeError(
                        f"invalid corpus.jsonl line {line_number}: {error}"
                    ) from error
                if not isinstance(record, dict):
                    raise RuntimeError(
                        f"corpus.jsonl line {line_number} is not an object"
                    )
                corpus_id = identifier(
                    record, kind="corpus", line_number=line_number
                )
                document = document_text(
                    record,
                    corpus_id,
                    max_document_bytes=max_document_bytes,
                )
                encoded = document.text.encode("utf-8")
                is_positive = int(corpus_id in positive_ids)
                if is_positive:
                    seen_positive.add(corpus_id)
                batch.append(
                    (
                        corpus_id,
                        rank_hash(seed, corpus_id),
                        hashlib.sha256(encoded).digest(),
                        document.text,
                        document.raw_text_bytes,
                        document.original_prepared_bytes,
                        len(encoded),
                        int(document.truncated),
                        is_positive,
                    )
                )
                prepared_bytes += len(encoded)
                maximum_prepared_bytes = max(maximum_prepared_bytes, len(encoded))
                original_prepared_bytes += document.original_prepared_bytes
                raw_text_bytes += document.raw_text_bytes
                if document.truncated:
                    truncated_documents += 1
                    truncated_bytes_removed += (
                        document.original_prepared_bytes - len(encoded)
                    )
                if len(batch) >= SQL_BATCH_ROWS:
                    flush_corpus_batch(connection, batch)
        flush_corpus_batch(connection, batch)
        connection.commit()
    except BaseException:
        connection.rollback()
        raise

    missing = sorted(positive_ids - seen_positive)
    if missing:
        preview = ", ".join(repr(value) for value in missing[:5])
        raise RuntimeError(
            f"{len(missing)} positive qrel documents are absent from corpus.jsonl: "
            f"{preview}"
        )
    if line_count == 0:
        raise RuntimeError("corpus.jsonl is empty")
    connection.execute("CREATE INDEX corpus_rank_idx ON corpus(rank_hash, corpus_id)")
    return (
        SourceFileStats(byte_count, line_count, checksum.hexdigest()),
        {
            "records": line_count,
            "raw_text_bytes_without_newlines": raw_text_bytes,
            "original_prepared_text_bytes_without_newlines": (
                original_prepared_bytes
            ),
            "prepared_text_bytes_without_newlines": prepared_bytes,
            "truncated_documents": truncated_documents,
            "truncated_bytes_removed": truncated_bytes_removed,
            "maximum_prepared_document_bytes": maximum_prepared_bytes,
        },
    )


def select_documents(
    connection: sqlite3.Connection,
    *,
    positive_count: int,
    small_count: int,
    large_count: int,
) -> dict[str, int]:
    if positive_count > small_count:
        raise RuntimeError(
            f"{positive_count} positive documents do not fit in the "
            f"{small_count}-document nested tier"
        )
    corpus_count = int(connection.execute("SELECT COUNT(*) FROM corpus").fetchone()[0])
    if corpus_count < large_count:
        raise RuntimeError(
            f"corpus has {corpus_count} documents; requested {large_count}"
        )

    connection.execute(
        """
        CREATE TABLE selected (
            corpus_id TEXT PRIMARY KEY NOT NULL,
            tier INTEGER NOT NULL CHECK (tier IN (0, 1)),
            forced INTEGER NOT NULL CHECK (forced IN (0, 1))
        ) WITHOUT ROWID
        """
    )
    used_text_hashes: set[bytes] = set()
    selected_small = 0
    selected_large = 0
    skipped_text_duplicates = 0

    connection.execute("BEGIN")
    try:
        positives = list(
            connection.execute(
                """
                SELECT corpus_id, text_sha256
                FROM corpus
                WHERE is_positive = 1
                ORDER BY rank_hash, corpus_id
                """
            )
        )
        positive_ids_by_text: dict[bytes, list[str]] = {}
        for corpus_id, text_sha256 in positives:
            positive_ids_by_text.setdefault(bytes(text_sha256), []).append(corpus_id)
        duplicate_positive_groups = [
            corpus_ids
            for corpus_ids in positive_ids_by_text.values()
            if len(corpus_ids) > 1
        ]
        if duplicate_positive_groups:
            duplicate_rows = sum(
                len(corpus_ids) - 1 for corpus_ids in duplicate_positive_groups
            )
            preview = "; ".join(
                ", ".join(repr(corpus_id) for corpus_id in corpus_ids[:4])
                for corpus_ids in duplicate_positive_groups[:3]
            )
            raise RuntimeError(
                f"{duplicate_rows} selected positive-qrel document rows reuse "
                "prepared text; published tiers require zero retained text "
                f"duplicates (example ID groups: {preview})"
            )
        positive_rows = [(corpus_id, 0, 1) for corpus_id, _ in positives]
        for _, text_sha256 in positives:
            used_text_hashes.add(bytes(text_sha256))
        connection.executemany(
            "INSERT INTO selected (corpus_id, tier, forced) VALUES (?, ?, ?)",
            positive_rows,
        )
        selected_small = len(positive_rows)
        selected_large = selected_small

        pending: list[tuple[str, int, int]] = []
        candidates = connection.execute(
            """
            SELECT corpus_id, text_sha256
            FROM corpus
            WHERE is_positive = 0
            ORDER BY rank_hash, corpus_id
            """
        )
        for corpus_id, text_sha256 in candidates:
            digest = bytes(text_sha256)
            if digest in used_text_hashes:
                skipped_text_duplicates += 1
                continue
            used_text_hashes.add(digest)
            tier = 0 if selected_small < small_count else 1
            pending.append((corpus_id, tier, 0))
            if tier == 0:
                selected_small += 1
            selected_large += 1
            if len(pending) >= SQL_BATCH_ROWS:
                connection.executemany(
                    """
                    INSERT INTO selected (corpus_id, tier, forced)
                    VALUES (?, ?, ?)
                    """,
                    pending,
                )
                pending.clear()
            if selected_large == large_count:
                break
        if pending:
            connection.executemany(
                "INSERT INTO selected (corpus_id, tier, forced) VALUES (?, ?, ?)",
                pending,
            )
        if selected_small != small_count or selected_large != large_count:
            raise RuntimeError(
                "not enough unique prepared documents to satisfy requested tiers: "
                f"selected {selected_small}/{small_count} and "
                f"{selected_large}/{large_count}"
            )
        if len(used_text_hashes) != selected_large:
            raise RuntimeError(
                "selected corpus contains retained prepared-text duplicates"
            )
        connection.commit()
    except BaseException:
        connection.rollback()
        raise

    connection.execute("CREATE INDEX selected_tier_idx ON selected(tier, corpus_id)")
    return {
        "source_documents": corpus_count,
        "positive_documents": positive_count,
        "small_documents": selected_small,
        "large_documents": selected_large,
        "filler_text_duplicates_skipped": skipped_text_duplicates,
        "unique_selected_texts": len(used_text_hashes),
    }


def duplicate_stats(total: int, unique: int) -> dict[str, object]:
    duplicates = total - unique
    return {
        "rows": total,
        "unique_prepared_texts": unique,
        "duplicate_rows": duplicates,
        "duplicate_rate": duplicates / total if total else 0.0,
    }


def write_documents(
    connection: sqlite3.Connection,
    output: pathlib.Path,
    *,
    small_count: int,
    large_count: int,
) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    paths = {
        "documents_100k": output / "documents-100k.txt",
        "document_ids_100k": output / "document_ids-100k.tsv",
        "documents_1m": output / "documents-1m.txt",
        "document_ids_1m": output / "document_ids-1m.tsv",
    }
    writers = {name: TrackedBinaryWriter(path) for name, path in paths.items()}
    small_raw_bytes = 0
    small_original_prepared_bytes = 0
    small_prepared_bytes = 0
    small_truncated_documents = 0
    small_maximum_prepared_bytes = 0
    large_raw_bytes = 0
    large_original_prepared_bytes = 0
    large_prepared_bytes = 0
    large_truncated_documents = 0
    large_maximum_prepared_bytes = 0
    try:
        rows = connection.execute(
            """
            SELECT c.corpus_id, c.prepared_text, c.raw_text_bytes,
                   c.original_prepared_bytes, c.prepared_bytes, c.truncated
            FROM selected AS s
            JOIN corpus AS c ON c.corpus_id = s.corpus_id
            ORDER BY s.tier, c.rank_hash, c.corpus_id
            """
        )
        for position, (
            corpus_id,
            prepared,
            raw_bytes,
            original_bytes,
            prepared_bytes,
            truncated,
        ) in enumerate(rows):
            if position >= large_count:
                raise RuntimeError("selected corpus unexpectedly exceeds large tier")
            if len(prepared.encode("utf-8")) != int(prepared_bytes):
                raise RuntimeError(
                    f"prepared-byte accounting mismatch for document {corpus_id!r}"
                )
            writers["documents_1m"].write_line(prepared)
            writers["document_ids_1m"].write_line(f"{position}\t{corpus_id}")
            large_raw_bytes += int(raw_bytes)
            large_original_prepared_bytes += int(original_bytes)
            large_prepared_bytes += int(prepared_bytes)
            large_truncated_documents += int(truncated)
            large_maximum_prepared_bytes = max(
                large_maximum_prepared_bytes, int(prepared_bytes)
            )
            if position < small_count:
                writers["documents_100k"].write_line(prepared)
                writers["document_ids_100k"].write_line(f"{position}\t{corpus_id}")
                small_raw_bytes += int(raw_bytes)
                small_original_prepared_bytes += int(original_bytes)
                small_prepared_bytes += int(prepared_bytes)
                small_truncated_documents += int(truncated)
                small_maximum_prepared_bytes = max(
                    small_maximum_prepared_bytes, int(prepared_bytes)
                )
        if writers["documents_1m"].lines != large_count:
            raise RuntimeError(
                f"wrote {writers['documents_1m'].lines} large-tier documents; "
                f"expected {large_count}"
            )
        if writers["documents_100k"].lines != small_count:
            raise RuntimeError(
                f"wrote {writers['documents_100k'].lines} small-tier documents; "
                f"expected {small_count}"
            )
        file_stats = {name: writer.close() for name, writer in writers.items()}
    finally:
        for writer in writers.values():
            if not writer._handle.closed:
                writer._handle.close()

    small_unique = int(
        connection.execute(
            """
            SELECT COUNT(DISTINCT c.text_sha256)
            FROM selected AS s JOIN corpus AS c ON c.corpus_id = s.corpus_id
            WHERE s.tier = 0
            """
        ).fetchone()[0]
    )
    large_unique = int(
        connection.execute(
            """
            SELECT COUNT(DISTINCT c.text_sha256)
            FROM selected AS s JOIN corpus AS c ON c.corpus_id = s.corpus_id
            """
        ).fetchone()[0]
    )
    tier_stats = {
        "100k": {
            "documents": small_count,
            "raw_text_bytes_without_newlines": small_raw_bytes,
            "original_prepared_text_bytes_without_newlines": (
                small_original_prepared_bytes
            ),
            "prepared_text_bytes_without_newlines": small_prepared_bytes,
            "maximum_prepared_document_bytes": small_maximum_prepared_bytes,
            "truncation": {
                "documents": small_truncated_documents,
                "bytes_removed": (
                    small_original_prepared_bytes - small_prepared_bytes
                ),
            },
            "text_duplicates": duplicate_stats(small_count, small_unique),
        },
        "1m": {
            "documents": large_count,
            "raw_text_bytes_without_newlines": large_raw_bytes,
            "original_prepared_text_bytes_without_newlines": (
                large_original_prepared_bytes
            ),
            "prepared_text_bytes_without_newlines": large_prepared_bytes,
            "maximum_prepared_document_bytes": large_maximum_prepared_bytes,
            "truncation": {
                "documents": large_truncated_documents,
                "bytes_removed": (
                    large_original_prepared_bytes - large_prepared_bytes
                ),
            },
            "text_duplicates": duplicate_stats(large_count, large_unique),
        },
    }
    return file_stats, tier_stats


def write_queries_and_qrels(
    output: pathlib.Path,
    *,
    selected_query_ids: Sequence[str],
    queries: dict[str, tuple[str, int]],
    qrels: Sequence[Qrel],
) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    query_position = {
        query_id: position for position, query_id in enumerate(selected_query_ids)
    }
    selected_qrels = [qrel for qrel in qrels if qrel.query_id in query_position]
    selected_qrels.sort(
        key=lambda qrel: (
            query_position[qrel.query_id],
            qrel.corpus_id,
            qrel.score_text,
        )
    )

    paths = {
        "queries": output / "queries-test-300.txt",
        "query_ids": output / "query_ids-test-300.tsv",
        "qrels": output / "qrels-test-300.tsv",
    }
    writers = {name: TrackedBinaryWriter(path) for name, path in paths.items()}
    raw_query_bytes = 0
    prepared_query_bytes = 0
    query_digests: set[bytes] = set()
    positive_judgments = 0
    try:
        for position, query_id in enumerate(selected_query_ids):
            prepared, raw_bytes = queries[query_id]
            writers["queries"].write_line(prepared)
            writers["query_ids"].write_line(f"{position}\t{query_id}")
            raw_query_bytes += raw_bytes
            prepared_query_bytes += len(prepared.encode("utf-8"))
            query_digests.add(hashlib.sha256(prepared.encode("utf-8")).digest())
        writers["qrels"].write_line("query-id\tcorpus-id\tscore")
        for qrel in selected_qrels:
            writers["qrels"].write_line(
                f"{qrel.query_id}\t{qrel.corpus_id}\t{qrel.score_text}"
            )
            if qrel.score > 0.0:
                positive_judgments += 1
        file_stats = {name: writer.close() for name, writer in writers.items()}
    finally:
        for writer in writers.values():
            if not writer._handle.closed:
                writer._handle.close()

    stats = {
        "queries": len(selected_query_ids),
        "raw_query_bytes_without_newlines": raw_query_bytes,
        "prepared_query_bytes_without_newlines": prepared_query_bytes,
        "text_duplicates": duplicate_stats(
            len(selected_query_ids), len(query_digests)
        ),
        "qrel_rows": len(selected_qrels),
        "positive_qrel_rows": positive_judgments,
    }
    return file_stats, stats


def verify_prefix(smaller: pathlib.Path, larger: pathlib.Path) -> None:
    with smaller.open("rb") as small, larger.open("rb") as large:
        line_number = 0
        for line_number, small_line in enumerate(small, start=1):
            large_line = large.readline()
            if small_line != large_line:
                raise RuntimeError(
                    f"{smaller.name} is not a byte-identical prefix of "
                    f"{larger.name} at line {line_number}"
                )
        if line_number == 0:
            raise RuntimeError(f"nested tier file is empty: {smaller}")


def read_id_set(path: pathlib.Path) -> set[str]:
    result: set[str] = set()
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 2 or fields[0] != str(line_number - 1):
                raise RuntimeError(f"invalid ID mapping at {path}:{line_number}")
            if fields[1] in result:
                raise RuntimeError(f"duplicate original ID at {path}:{line_number}")
            result.add(fields[1])
    return result


def source_file_manifest(stats: SourceFileStats) -> dict[str, object]:
    return {
        "bytes": stats.bytes,
        "lines": stats.lines,
        "sha256": stats.sha256,
    }


def publish_directory(staged: pathlib.Path, output: pathlib.Path) -> None:
    if output.is_symlink():
        raise RuntimeError(f"refusing to replace symlink output: {output}")
    if output.exists() and not output.is_dir():
        raise RuntimeError(f"refusing to replace non-directory output: {output}")
    backup = output.with_name(f".{output.name}.backup-{uuid.uuid4().hex}")
    had_previous = output.exists()
    if had_previous:
        output.rename(backup)
    try:
        staged.rename(output)
    except BaseException:
        if had_previous and backup.exists() and not output.exists():
            backup.rename(output)
        raise
    if had_previous:
        shutil.rmtree(backup)


def build_manifest(
    *,
    archive_stats: FileDigest | None,
    expected_md5: str,
    provenance: dict[str, object],
    source_stats: dict[str, SourceFileStats],
    source_corpus_stats: dict[str, int],
    selected_query_ids: Sequence[str],
    selected_positive_ids: set[str],
    selection_stats: dict[str, int],
    tier_stats: dict[str, object],
    query_stats: dict[str, object],
    file_stats: dict[str, dict[str, object]],
    seed: str,
    small_count: int,
    large_count: int,
    max_document_bytes: int,
    max_text_duplicate_rate: float,
    max_query_duplicate_rate: float,
) -> dict[str, object]:
    archive = None
    if archive_stats is not None:
        archive = {
            "bytes": archive_stats.bytes,
            "md5": archive_stats.md5,
            "sha256": archive_stats.sha256,
        }
    return {
        "schema_version": 1,
        "preparer": {
            "name": "prepare_beir_nq_scale.py",
            "version": SCRIPT_VERSION,
        },
        "dataset": {
            "name": "BEIR Natural Questions (NQ)",
            "source_url": SOURCE_URL,
            "official_archive_expected_md5": EXPECTED_MD5,
            "validated_expected_md5": expected_md5,
            "archive": archive,
            "provenance": provenance,
            "source_files": {
                name: source_file_manifest(stats)
                for name, stats in sorted(source_stats.items())
            },
            "source_corpus": source_corpus_stats,
        },
        "normalization": {
            "line_cleaning": (
                "replace NUL with space, split on Unicode whitespace, "
                "join tokens with one ASCII space"
            ),
            "document_format": (
                "search_document: <title>. <text>; a missing title or text "
                "uses the non-empty field without the separator"
            ),
            "document_truncation": {
                "maximum_utf8_bytes_including_prefix": max_document_bytes,
                "algorithm": (
                    "take the longest valid UTF-8 prefix no larger than the "
                    "byte ceiling, then remove trailing Unicode whitespace"
                ),
                "applied_before_text_hashing_deduplication_and_selection": True,
                "token_safety_basis": (
                    "the configured ceiling is no greater than the 2,000-byte "
                    "hard ceiling, which is below the model's "
                    "2,048-token context and leaves room for special tokens; "
                    "a tokenizer cannot emit more byte-fallback tokens than "
                    "the valid UTF-8 input byte count"
                ),
            },
            "query_format": "search_query: <query>",
            "encoding": "UTF-8 with LF line endings",
        },
        "selection": {
            "seed": seed,
            "rank": "SHA-256(UTF-8(seed) || UTF-8(original_id))",
            "queries": (
                "lowest-ranked test query IDs that have at least one positive qrel"
            ),
            "documents": (
                "all selected-query positive documents in the 100K tier; "
                "reject if positive documents reuse prepared text; unique-text "
                "fillers in rank order; each tier ordered by rank"
            ),
            "selected_query_ids_sha256": hashlib.sha256(
                ("\n".join(selected_query_ids) + "\n").encode("utf-8")
            ).hexdigest(),
            "selected_positive_document_ids_sha256": hashlib.sha256(
                ("\n".join(sorted(selected_positive_ids)) + "\n").encode("utf-8")
            ).hexdigest(),
        },
        "counts": {
            **selection_stats,
            "queries": len(selected_query_ids),
        },
        "tiers": tier_stats,
        "queries": query_stats,
        "qrel_coverage": {
            "required_positive_documents": len(selected_positive_ids),
            "covered_positive_documents_100k": len(selected_positive_ids),
            "covered_positive_documents_1m": len(selected_positive_ids),
            "document_coverage_100k": 1.0,
            "document_coverage_1m": 1.0,
            "positive_judgment_coverage_100k": 1.0,
            "positive_judgment_coverage_1m": 1.0,
        },
        "invariants": {
            "100k_is_exact_prefix_of_1m": True,
            "document_ids_are_unique": True,
            "prepared_document_texts_are_unique": True,
            "original_query_document_ids_and_qrel_scores_preserved": True,
            "thresholds": {
                "maximum_document_text_duplicate_rate": max_text_duplicate_rate,
                "maximum_query_text_duplicate_rate": max_query_duplicate_rate,
            },
        },
        "requested_tier_counts": {
            "100k": small_count,
            "1m": large_count,
        },
        "files": file_stats,
    }


def prepare(
    dataset_root: pathlib.Path,
    result_dir: pathlib.Path,
    database_path: pathlib.Path,
    *,
    archive_stats: FileDigest | None,
    expected_md5: str,
    provenance: dict[str, object],
    seed: str,
    query_count: int,
    small_count: int,
    large_count: int,
    max_document_bytes: int,
    max_text_duplicate_rate: float,
    max_query_duplicate_rate: float,
) -> dict[str, object]:
    queries, query_source_stats = parse_queries(dataset_root / "queries.jsonl")
    qrels, positives, qrel_source_stats = parse_qrels(
        dataset_root / "qrels" / "test.tsv"
    )
    selected_query_ids = select_queries(
        queries, positives, seed=seed, count=query_count
    )
    selected_query_set = set(selected_query_ids)
    selected_positive_ids = {
        corpus_id
        for query_id in selected_query_ids
        for corpus_id in positives[query_id]
    }

    connection = open_database(database_path)
    try:
        corpus_source_stats, source_corpus_stats = index_corpus(
            dataset_root / "corpus.jsonl",
            connection,
            positive_ids=selected_positive_ids,
            seed=seed,
            max_document_bytes=max_document_bytes,
        )
        selection_stats = select_documents(
            connection,
            positive_count=len(selected_positive_ids),
            small_count=small_count,
            large_count=large_count,
        )
        result_dir.mkdir(parents=True, exist_ok=False)
        document_files, tier_stats = write_documents(
            connection,
            result_dir,
            small_count=small_count,
            large_count=large_count,
        )
        query_files, query_stats = write_queries_and_qrels(
            result_dir,
            selected_query_ids=selected_query_ids,
            queries=queries,
            qrels=qrels,
        )
    finally:
        connection.close()

    verify_prefix(
        result_dir / "documents-100k.txt", result_dir / "documents-1m.txt"
    )
    verify_prefix(
        result_dir / "document_ids-100k.tsv",
        result_dir / "document_ids-1m.tsv",
    )
    small_ids = read_id_set(result_dir / "document_ids-100k.tsv")
    large_ids = read_id_set(result_dir / "document_ids-1m.tsv")
    if len(small_ids) != small_count or len(large_ids) != large_count:
        raise RuntimeError("output ID cardinality does not match requested tiers")
    if not small_ids <= large_ids:
        raise RuntimeError("100K document IDs are not nested in 1M document IDs")
    if not selected_positive_ids <= small_ids:
        raise RuntimeError("100K tier does not cover every selected positive qrel")

    selected_positive_qrels = [
        qrel
        for qrel in qrels
        if qrel.query_id in selected_query_set and qrel.score > 0.0
    ]
    if any(qrel.corpus_id not in small_ids for qrel in selected_positive_qrels):
        raise RuntimeError("100K tier does not cover every positive qrel judgment")

    for tier_name in ("100k", "1m"):
        maximum_bytes = int(tier_stats[tier_name]["maximum_prepared_document_bytes"])
        if maximum_bytes > max_document_bytes:
            raise RuntimeError(
                f"{tier_name} contains a {maximum_bytes}-byte prepared document; "
                f"limit is {max_document_bytes}"
            )
        duplicate_rows = int(
            tier_stats[tier_name]["text_duplicates"]["duplicate_rows"]
        )
        rate = float(tier_stats[tier_name]["text_duplicates"]["duplicate_rate"])
        if duplicate_rows != 0 or rate != 0.0:
            raise RuntimeError(
                f"{tier_name} retains {duplicate_rows} prepared-text duplicate "
                "rows; published tiers require exactly zero"
            )
    query_duplicate_rate = float(query_stats["text_duplicates"]["duplicate_rate"])
    if query_duplicate_rate > max_query_duplicate_rate:
        raise RuntimeError(
            f"query prepared-text duplicate rate {query_duplicate_rate:.8f} "
            f"exceeds threshold {max_query_duplicate_rate:.8f}"
        )

    files = {**document_files, **query_files}
    manifest = build_manifest(
        archive_stats=archive_stats,
        expected_md5=expected_md5,
        provenance=provenance,
        source_stats={
            "corpus.jsonl": corpus_source_stats,
            "queries.jsonl": query_source_stats,
            "qrels/test.tsv": qrel_source_stats,
        },
        source_corpus_stats=source_corpus_stats,
        selected_query_ids=selected_query_ids,
        selected_positive_ids=selected_positive_ids,
        selection_stats=selection_stats,
        tier_stats=tier_stats,
        query_stats=query_stats,
        file_stats=files,
        seed=seed,
        small_count=small_count,
        large_count=large_count,
        max_document_bytes=max_document_bytes,
        max_text_duplicate_rate=max_text_duplicate_rate,
        max_query_duplicate_rate=max_query_duplicate_rate,
    )
    manifest_path = result_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    json.loads(manifest_path.read_text(encoding="utf-8"))
    return manifest


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare deterministic, nested 100K and 1M BEIR NQ inputs with "
            "Nomic retrieval prefixes and complete provenance."
        )
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("work/datasets/nq-scale"),
        help="destination directory (default: work/datasets/nq-scale)",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--archive",
        type=pathlib.Path,
        help="existing BEIR NQ ZIP; its MD5 is always verified",
    )
    source.add_argument(
        "--source-dir",
        type=pathlib.Path,
        help=(
            "already-extracted directory containing corpus.jsonl, queries.jsonl, "
            "and qrels/test.tsv; intended for trusted local fixtures"
        ),
    )
    parser.add_argument(
        "--cache-archive",
        type=pathlib.Path,
        help=(
            "download destination when neither --archive nor --source-dir is "
            "given (default: <output-parent>/nq.zip)"
        ),
    )
    parser.add_argument(
        "--expected-md5",
        default=EXPECTED_MD5,
        help=(
            f"required archive MD5 (default official NQ hash: {EXPECTED_MD5}); "
            "override explicitly only for a controlled fixture"
        ),
    )
    parser.add_argument(
        "--seed",
        default="leann.cpp-beir-nq-scale-v1",
        help="deterministic selection seed",
    )
    parser.add_argument(
        "--query-count",
        type=int,
        default=300,
        help="number of positive-qrel test queries (default: 300)",
    )
    parser.add_argument(
        "--small-count",
        type=int,
        default=100_000,
        help="nested small corpus size (default: 100000)",
    )
    parser.add_argument(
        "--large-count",
        type=int,
        default=1_000_000,
        help="large corpus size (default: 1000000)",
    )
    parser.add_argument(
        "--max-document-bytes",
        type=int,
        default=DEFAULT_MAX_DOCUMENT_BYTES,
        help=(
            "maximum UTF-8 bytes per prepared document, including the Nomic "
            f"prefix (default and hard maximum: {DEFAULT_MAX_DOCUMENT_BYTES})"
        ),
    )
    parser.add_argument(
        "--max-text-duplicate-rate",
        type=float,
        default=0.0,
        help=(
            "retained prepared-document duplicate rate; must be exactly 0 "
            "for publishable tiers (default: 0)"
        ),
    )
    parser.add_argument(
        "--max-query-duplicate-rate",
        type=float,
        default=0.01,
        help="maximum prepared-query duplicate rate (default: 0.01)",
    )
    parser.add_argument(
        "--max-archive-members",
        type=int,
        default=DEFAULT_MAX_ARCHIVE_MEMBERS,
        help=f"safe-extraction member limit (default: {DEFAULT_MAX_ARCHIVE_MEMBERS})",
    )
    parser.add_argument(
        "--max-extracted-bytes",
        type=int,
        default=DEFAULT_MAX_EXTRACTED_BYTES,
        help=(
            "safe-extraction uncompressed-byte limit "
            f"(default: {DEFAULT_MAX_EXTRACTED_BYTES})"
        ),
    )
    parser.add_argument(
        "--download-timeout",
        type=int,
        default=120,
        help="download socket timeout in seconds (default: 120)",
    )
    return parser.parse_args(argv)


def validate_arguments(arguments: argparse.Namespace) -> None:
    arguments.expected_md5 = validate_expected_md5(arguments.expected_md5)
    if not arguments.seed:
        raise ValueError("--seed must not be empty")
    if arguments.query_count <= 0:
        raise ValueError("--query-count must be positive")
    if arguments.small_count <= 0:
        raise ValueError("--small-count must be positive")
    if arguments.large_count <= arguments.small_count:
        raise ValueError("--large-count must be greater than --small-count")
    minimum_document_bytes = len(DOCUMENT_PREFIX.encode("utf-8")) + 1
    if not (
        minimum_document_bytes
        <= arguments.max_document_bytes
        <= MAX_SAFE_DOCUMENT_BYTES
    ):
        raise ValueError(
            f"--max-document-bytes must be between {minimum_document_bytes} "
            f"and {MAX_SAFE_DOCUMENT_BYTES}, inclusive"
        )
    validate_rate("--max-text-duplicate-rate", arguments.max_text_duplicate_rate)
    if arguments.max_text_duplicate_rate != 0.0:
        raise ValueError(
            "--max-text-duplicate-rate must be exactly 0; published 100K/1M "
            "tiers require zero retained prepared-text duplicates"
        )
    validate_rate("--max-query-duplicate-rate", arguments.max_query_duplicate_rate)
    if arguments.max_archive_members <= 0:
        raise ValueError("--max-archive-members must be positive")
    if arguments.max_extracted_bytes <= 0:
        raise ValueError("--max-extracted-bytes must be positive")
    if arguments.download_timeout <= 0:
        raise ValueError("--download-timeout must be positive")


def run(arguments: argparse.Namespace) -> dict[str, object]:
    validate_arguments(arguments)
    output = arguments.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if arguments.source_dir is not None:
        source_path = arguments.source_dir.resolve()
        if (
            source_path == output
            or output in source_path.parents
            or source_path in output.parents
        ):
            raise RuntimeError("--source-dir and --output must not contain each other")

    staging_root = pathlib.Path(
        tempfile.mkdtemp(prefix=f".{output.name}.prepare-", dir=output.parent)
    )
    archive_stats: FileDigest | None = None
    try:
        if arguments.source_dir is not None:
            supplied_source = arguments.source_dir.resolve()
            dataset_root = find_dataset_root(supplied_source)
            provenance: dict[str, object] = {
                "kind": "source_directory",
                "path": str(supplied_source),
                "archive_validation": "not applicable",
            }
        else:
            if arguments.archive is not None:
                archive = arguments.archive.resolve()
                acquisition = "provided_archive"
            else:
                archive = (
                    arguments.cache_archive.resolve()
                    if arguments.cache_archive is not None
                    else output.parent / "nq.zip"
                )
                archive_stats = download_archive(
                    archive,
                    expected_md5=arguments.expected_md5,
                    timeout_seconds=arguments.download_timeout,
                )
                acquisition = "official_download_or_verified_cache"
            if archive_stats is None:
                archive_stats = verify_archive(archive, arguments.expected_md5)
            extracted = staging_root / "extracted"
            safe_extract(
                archive,
                extracted,
                max_members=arguments.max_archive_members,
                max_uncompressed_bytes=arguments.max_extracted_bytes,
            )
            dataset_root = find_dataset_root(extracted)
            provenance = {
                "kind": acquisition,
                "archive_path": str(archive),
                "archive_validation": "MD5 matched validated_expected_md5",
            }

        result_dir = staging_root / "result"
        manifest = prepare(
            dataset_root,
            result_dir,
            staging_root / "selection.sqlite3",
            archive_stats=archive_stats,
            expected_md5=arguments.expected_md5,
            provenance=provenance,
            seed=arguments.seed,
            query_count=arguments.query_count,
            small_count=arguments.small_count,
            large_count=arguments.large_count,
            max_document_bytes=arguments.max_document_bytes,
            max_text_duplicate_rate=arguments.max_text_duplicate_rate,
            max_query_duplicate_rate=arguments.max_query_duplicate_rate,
        )
        publish_directory(result_dir, output)
        return manifest
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = parse_arguments(argv)
        manifest = run(arguments)
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
