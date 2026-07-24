#!/usr/bin/env python3
"""Focused synthetic tests for scripts/prepare_beir_nq_scale.py."""

from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest
import zipfile


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_beir_nq_scale.py"


def write_fixture(root: pathlib.Path) -> pathlib.Path:
    dataset = root / "nq"
    (dataset / "qrels").mkdir(parents=True)
    corpus = []
    for index in range(18):
        text = (
            "é" * 1_100 + f" tail {index}"
            if index < 13
            else ("duplicate filler" if index in (14, 15) else f"body {index}")
        )
        corpus.append({"_id": f"d{index}", "title": f"title {index}", "text": text})
    (dataset / "corpus.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in corpus),
        encoding="utf-8",
    )
    queries = [{"_id": f"q{index}", "text": f"question {index}"} for index in range(6)]
    (dataset / "queries.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in queries),
        encoding="utf-8",
    )
    qrel_lines = ["query-id\tcorpus-id\tscore\n"]
    for index in range(6):
        qrel_lines.append(f"q{index}\td{index}\t1\n")
        if index % 2 == 0:
            qrel_lines.append(f"q{index}\td{index + 6}\t2\n")
    (dataset / "qrels" / "test.tsv").write_text(
        "".join(qrel_lines), encoding="utf-8"
    )
    return dataset


def tree_hashes(root: pathlib.Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.iterdir())
        if path.is_file()
    }


class PrepareNqScaleTest(unittest.TestCase):
    def command(self, source: pathlib.Path, output: pathlib.Path) -> list[str]:
        return [
            sys.executable,
            str(SCRIPT),
            "--source-dir",
            str(source),
            "--output",
            str(output),
            "--query-count",
            "3",
            "--small-count",
            "7",
            "--large-count",
            "12",
            "--max-document-bytes",
            "80",
            "--seed",
            "synthetic-seed",
            "--max-text-duplicate-rate",
            "0",
            "--max-query-duplicate-rate",
            "0",
        ]

    def test_source_directory_is_nested_complete_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            dataset = write_fixture(root / "source")
            output = root / "prepared"
            subprocess.run(self.command(dataset, output), check=True, capture_output=True)

            documents_small = (output / "documents-100k.txt").read_text(
                encoding="utf-8"
            ).splitlines()
            documents_large = (output / "documents-1m.txt").read_text(
                encoding="utf-8"
            ).splitlines()
            document_ids_small = (output / "document_ids-100k.tsv").read_text(
                encoding="utf-8"
            ).splitlines()
            document_ids_large = (output / "document_ids-1m.tsv").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(len(documents_small), 7)
            self.assertEqual(len(documents_large), 12)
            self.assertEqual(documents_small, documents_large[:7])
            self.assertEqual(document_ids_small, document_ids_large[:7])
            self.assertEqual(len(set(documents_small)), len(documents_small))
            self.assertEqual(len(set(documents_large)), len(documents_large))
            self.assertEqual(
                len({line.split("\t", 1)[1] for line in document_ids_small}),
                len(document_ids_small),
            )
            self.assertEqual(
                len({line.split("\t", 1)[1] for line in document_ids_large}),
                len(document_ids_large),
            )
            self.assertTrue(
                all(line.startswith("search_document: ") for line in documents_large)
            )
            self.assertTrue(
                all(len(line.encode("utf-8")) <= 80 for line in documents_large)
            )
            self.assertTrue(
                all(
                    line.startswith("search_query: ")
                    for line in (output / "queries-test-300.txt")
                    .read_text(encoding="utf-8")
                    .splitlines()
                )
            )

            manifest = json.loads(
                (output / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertIsNone(manifest["dataset"]["archive"])
            self.assertEqual(manifest["counts"]["small_documents"], 7)
            self.assertEqual(manifest["counts"]["large_documents"], 12)
            self.assertEqual(
                manifest["normalization"]["document_truncation"][
                    "maximum_utf8_bytes_including_prefix"
                ],
                80,
            )
            self.assertGreater(
                manifest["dataset"]["source_corpus"]["truncated_documents"], 0
            )
            self.assertGreater(manifest["tiers"]["100k"]["truncation"]["documents"], 0)
            self.assertLessEqual(
                manifest["tiers"]["1m"]["maximum_prepared_document_bytes"], 80
            )
            self.assertEqual(
                manifest["qrel_coverage"]["positive_judgment_coverage_100k"], 1.0
            )
            self.assertTrue(
                manifest["invariants"]["prepared_document_texts_are_unique"]
            )
            self.assertEqual(
                manifest["invariants"]["thresholds"][
                    "maximum_document_text_duplicate_rate"
                ],
                0.0,
            )
            for tier in ("100k", "1m"):
                self.assertEqual(
                    manifest["tiers"][tier]["text_duplicates"]["duplicate_rows"],
                    0,
                )
                self.assertEqual(
                    manifest["tiers"][tier]["text_duplicates"]["duplicate_rate"],
                    0.0,
                )
            first_hashes = tree_hashes(output)
            subprocess.run(self.command(dataset, output), check=True, capture_output=True)
            self.assertEqual(first_hashes, tree_hashes(output))

    def test_duplicate_positive_prepared_text_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            dataset = write_fixture(root / "source")
            corpus_path = dataset / "corpus.jsonl"
            records = [
                json.loads(line)
                for line in corpus_path.read_text(encoding="utf-8").splitlines()
            ]
            for record in records[:12]:
                record["title"] = "same positive"
                record["text"] = "same prepared text"
            corpus_path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            result = subprocess.run(
                self.command(dataset, root / "prepared"),
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "published tiers require zero retained text duplicates",
                result.stderr,
            )

    def test_duplicate_corpus_id_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            dataset = write_fixture(root / "source")
            corpus_path = dataset / "corpus.jsonl"
            first = corpus_path.read_text(encoding="utf-8").splitlines()[0]
            with corpus_path.open("a", encoding="utf-8") as target:
                target.write(first + "\n")
            result = subprocess.run(
                self.command(dataset, root / "prepared"),
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("duplicate ID", result.stderr)

    def test_nonzero_document_duplicate_threshold_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            dataset = write_fixture(root / "source")
            command = self.command(dataset, root / "prepared")
            threshold_index = command.index("--max-text-duplicate-rate") + 1
            command[threshold_index] = "0.001"
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "--max-text-duplicate-rate must be exactly 0",
                result.stderr,
            )

    def test_archive_hash_and_safe_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            dataset = write_fixture(root / "source")
            archive = root / "fixture.zip"
            with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as compressed:
                for path in sorted(dataset.parent.rglob("*")):
                    if path.is_file():
                        compressed.write(path, path.relative_to(dataset.parent))
            expected_md5 = hashlib.md5(archive.read_bytes()).hexdigest()
            output = root / "prepared"
            command = self.command(dataset, output)
            source_index = command.index("--source-dir")
            del command[source_index : source_index + 2]
            command.extend(["--archive", str(archive), "--expected-md5", expected_md5])
            subprocess.run(command, check=True, capture_output=True)
            manifest = json.loads(
                (output / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["dataset"]["archive"]["md5"], expected_md5)
            self.assertEqual(
                manifest["dataset"]["archive"]["sha256"],
                hashlib.sha256(archive.read_bytes()).hexdigest(),
            )

    def test_default_document_ceiling_is_2000_utf8_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            dataset = write_fixture(root / "source")
            output = root / "prepared"
            command = self.command(dataset, output)
            option_index = command.index("--max-document-bytes")
            del command[option_index : option_index + 2]
            subprocess.run(command, check=True, capture_output=True)
            documents = (output / "documents-1m.txt").read_text(
                encoding="utf-8"
            ).splitlines()
            encoded_lengths = [len(document.encode("utf-8")) for document in documents]
            self.assertTrue(all(length <= 2_000 for length in encoded_lengths))
            self.assertIn(2_000, encoded_lengths)
            manifest = json.loads(
                (output / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["normalization"]["document_truncation"][
                    "maximum_utf8_bytes_including_prefix"
                ],
                2_000,
            )
            self.assertEqual(
                manifest["tiers"]["1m"]["maximum_prepared_document_bytes"], 2_000
            )

    def test_archive_traversal_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            archive = root / "malicious.zip"
            with zipfile.ZipFile(archive, "w") as compressed:
                compressed.writestr("../escape", "no")
            expected_md5 = hashlib.md5(archive.read_bytes()).hexdigest()
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--archive",
                    str(archive),
                    "--expected-md5",
                    expected_md5,
                    "--output",
                    str(root / "prepared"),
                    "--query-count",
                    "1",
                    "--small-count",
                    "1",
                    "--large-count",
                    "2",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("unsafe zip member path", result.stderr)
            self.assertFalse((root / "escape").exists())


if __name__ == "__main__":
    unittest.main()
