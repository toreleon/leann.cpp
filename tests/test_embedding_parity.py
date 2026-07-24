#!/usr/bin/env python3
"""Unit tests for deterministic embedding-parity selection and metrics."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from validate_embedding_parity import (  # noqa: E402
    compare_embeddings,
    evenly_spaced_indices,
    select_document_sample,
    verify_document_origins,
)
from prepare_beir_nq_scale import document_text  # noqa: E402


class EmbeddingParityTest(unittest.TestCase):
    def test_evenly_spaced_indices_are_exact_and_endpoints_inclusive(self) -> None:
        self.assertEqual(evenly_spaced_indices(10, 0), [])
        self.assertEqual(evenly_spaced_indices(10, 1), [0])
        self.assertEqual(evenly_spaced_indices(10, 4), [0, 3, 6, 9])
        self.assertEqual(evenly_spaced_indices(3, 8), [0, 1, 2])

    def test_selection_unions_spaced_longest_and_declared_truncated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            documents = Path(directory) / "documents.txt"
            documents.write_text(
                "\n".join(
                    [
                        "a",
                        "bb",
                        "longest-low-index",
                        "dddd",
                        "eeeee",
                        "longest-high-idx",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            rows, metadata = select_document_sample(
                documents,
                expected_count=6,
                evenly_spaced=3,
                longest=1,
                truncated_indices=[3],
            )
            by_index = {row["source_index"]: row for row in rows}
            self.assertEqual(sorted(by_index), [0, 2, 3, 5])
            self.assertEqual(
                by_index[2]["categories"], ["evenly_spaced", "longest"]
            )
            self.assertEqual(
                by_index[3]["categories"],
                ["explicit_truncation_candidate"],
            )
            self.assertEqual(
                metadata["category_counts"][
                    "explicit_truncation_candidate"
                ],
                1,
            )
            rows_without_explicit, _ = select_document_sample(
                documents,
                expected_count=6,
                evenly_spaced=1,
                longest=1,
                truncated_indices=[],
            )
            self.assertEqual(
                [row["source_index"] for row in rows_without_explicit],
                [0, 2],
            )

    def test_source_archive_proves_truncation_and_exact_prepared_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = [
                {"_id": "doc-a", "title": "", "text": "short"},
                {
                    "_id": "doc-b",
                    "title": "long",
                    "text": "x" * 200,
                },
            ]
            ceiling = 48
            prepared = [
                document_text(
                    record,
                    str(record["_id"]),
                    max_document_bytes=ceiling,
                )
                for record in records
            ]
            document_ids = root / "document_ids.tsv"
            document_ids.write_text(
                "0\tdoc-a\n1\tdoc-b\n", encoding="utf-8"
            )
            archive = root / "fixture.zip"
            with zipfile.ZipFile(archive, "w") as compressed:
                compressed.writestr(
                    "nq/corpus.jsonl",
                    "".join(json.dumps(record) + "\n" for record in records),
                )
            rows = [
                {
                    "kind": "document",
                    "source_index": index,
                    "text": value.text,
                    "utf8_bytes": len(value.text.encode("utf-8")),
                    "categories": ["longest"] if index else ["evenly_spaced"],
                }
                for index, value in enumerate(prepared)
            ]
            proof = verify_document_origins(
                archive,
                document_ids,
                rows,
                expected_count=2,
                max_document_bytes=ceiling,
            )
        self.assertEqual(proof["source_verified_truncated_indices"], [1])
        self.assertTrue(rows[1]["source_verified_truncated"])
        self.assertIn("source_verified_truncated", rows[1]["categories"])

    def test_metrics_report_worst_rows_and_threshold_input(self) -> None:
        reference = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
        )
        native = np.array(
            [[1.0, 0.0, 0.0], [0.01, 0.99995, 0.0]], dtype=np.float32
        )
        native /= np.linalg.norm(native, axis=1, keepdims=True)
        rows = [
            {
                "kind": "query",
                "source_index": 0,
                "utf8_bytes": 1,
                "categories": ["all_queries"],
            },
            {
                "kind": "document",
                "source_index": 7,
                "utf8_bytes": 2000,
                "categories": ["longest", "source_verified_truncated"],
            },
        ]
        metrics = compare_embeddings(reference, native, rows)
        self.assertEqual(metrics["rows"], 2)
        self.assertEqual(metrics["worst_cosine_row"]["source_index"], 7)
        self.assertEqual(
            metrics["worst_absolute_component"]["source_index"], 7
        )
        self.assertAlmostEqual(
            metrics["minimum_cosine_similarity"], 0.99995, places=5
        )
        self.assertGreater(metrics["maximum_absolute_difference"], 0.009)
        self.assertEqual(metrics["by_kind"]["query"]["rows"], 1)
        self.assertEqual(metrics["by_kind"]["document"]["rows"], 1)


if __name__ == "__main__":
    unittest.main()
