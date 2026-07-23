#!/usr/bin/env python3
"""Download BEIR SciFact and create line-oriented leann.cpp inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import urllib.request
import zipfile


URL = (
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/"
    "BEIR/datasets/scifact.zip"
)
EXPECTED_MD5 = "5f7d1de60b170fc8027bb7898e2efca1"


def digest(path: pathlib.Path, algorithm: str) -> str:
    checksum = hashlib.new(algorithm)
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def clean_line(value: str) -> str:
    return " ".join(value.replace("\x00", " ").split())


def safe_extract(archive: pathlib.Path, destination: pathlib.Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as compressed:
        for member in compressed.infolist():
            target = (destination / member.filename).resolve()
            if destination != target and destination not in target.parents:
                raise RuntimeError(f"unsafe zip member: {member.filename}")
        compressed.extractall(destination)


def download(archive: pathlib.Path) -> None:
    if archive.exists() and digest(archive, "md5") == EXPECTED_MD5:
        return
    temporary = archive.with_suffix(archive.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    archive.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(URL) as response, temporary.open("wb") as out:
            shutil.copyfileobj(response, out)
        observed = digest(temporary, "md5")
        if observed != EXPECTED_MD5:
            raise RuntimeError(
                f"SciFact MD5 mismatch: expected {EXPECTED_MD5}, got {observed}"
            )
        temporary.replace(archive)
    finally:
        temporary.unlink(missing_ok=True)


def read_test_query_ids(qrels_path: pathlib.Path) -> set[str]:
    result: set[str] = set()
    with qrels_path.open(encoding="utf-8") as source:
        header = next(source, None)
        if header is None:
            raise RuntimeError("empty SciFact test qrels")
        for line in source:
            fields = line.rstrip("\n").split("\t")
            if fields:
                result.add(fields[0])
    return result


def prepare(dataset: pathlib.Path, output: pathlib.Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    documents_path = output / "documents.txt"
    document_ids_path = output / "document_ids.tsv"
    queries_path = output / "queries-test.txt"
    query_ids_path = output / "query_ids-test.tsv"

    document_count = 0
    raw_document_bytes = 0
    with (
        (dataset / "corpus.jsonl").open(encoding="utf-8") as source,
        documents_path.open("w", encoding="utf-8") as documents,
        document_ids_path.open("w", encoding="utf-8") as document_ids,
    ):
        for row in source:
            record = json.loads(row)
            title = clean_line(record.get("title", ""))
            text = clean_line(record.get("text", ""))
            document = f"{title}. {text}" if title and text else title or text
            if not document:
                continue
            encoded = document.encode("utf-8")
            raw_document_bytes += len(encoded)
            documents.write(document + "\n")
            document_ids.write(f"{document_count}\t{record['_id']}\n")
            document_count += 1

    test_query_ids = read_test_query_ids(dataset / "qrels" / "test.tsv")
    query_count = 0
    with (
        (dataset / "queries.jsonl").open(encoding="utf-8") as source,
        queries_path.open("w", encoding="utf-8") as queries,
        query_ids_path.open("w", encoding="utf-8") as query_ids,
    ):
        for row in source:
            record = json.loads(row)
            identifier = str(record["_id"])
            if identifier not in test_query_ids:
                continue
            query = clean_line(record["text"])
            if not query:
                continue
            queries.write(query + "\n")
            query_ids.write(f"{query_count}\t{identifier}\n")
            query_count += 1

    if query_count != len(test_query_ids):
        raise RuntimeError(
            f"prepared {query_count} test queries for {len(test_query_ids)} qrels IDs"
        )

    return {
        "dataset": "BEIR SciFact",
        "source_url": URL,
        "archive_md5": EXPECTED_MD5,
        "archive_sha256": digest(output.parent / "scifact.zip", "sha256"),
        "documents": document_count,
        "test_queries": query_count,
        "raw_document_bytes_without_newlines": raw_document_bytes,
        "files": {
            "documents": documents_path.name,
            "document_ids": document_ids_path.name,
            "queries": queries_path.name,
            "query_ids": query_ids_path.name,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("work/datasets/scifact"),
    )
    arguments = parser.parse_args()

    output = arguments.output.resolve()
    archive = output.parent / "scifact.zip"
    extracted_root = output.parent / "source"
    download(archive)
    if not (extracted_root / "scifact" / "corpus.jsonl").exists():
        safe_extract(archive, extracted_root)
    manifest = prepare(extracted_root / "scifact", output)
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
