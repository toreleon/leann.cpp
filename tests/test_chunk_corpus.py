#!/usr/bin/env python3
"""Tests for the directory-to---docs chunker used to prepare a corpus."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import chunk_corpus as chunker  # noqa: E402
from chunk_corpus import ChunkError  # noqa: E402


SCRIPT = ROOT / "scripts" / "chunk_corpus.py"


def write_tree(root: pathlib.Path, files: dict[str, str]) -> None:
    """Create files under root, in the given (deliberate) order."""
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def read_docs(path: pathlib.Path) -> list[str]:
    """Read a --docs file exactly the way `leann build` does."""
    data = path.read_bytes()
    if not data:
        return []
    if not data.endswith(b"\n"):
        raise AssertionError("--docs file does not end with a newline")
    if b"\r" in data:
        raise AssertionError("--docs file contains a carriage return")
    return data.decode("utf-8").split("\n")[:-1]


def run_main(argv: list[str]) -> tuple[int, str, str]:
    """Call main() in-process, capturing stdout and stderr."""
    out = io.StringIO()
    err = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = chunker.main(argv)
    return code, out.getvalue(), err.getvalue()


class ChunkBudgetTest(unittest.TestCase):
    def test_long_paragraph_is_split_under_the_byte_budget(self) -> None:
        # One paragraph far longer than the budget: the load-bearing case,
        # because a chunk of N bytes can never tokenize to more than N tokens
        # and --ctx is sized against the reported maximum.
        paragraph = "alpha bravo charlie delta echo foxtrot " * 400
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source = root / "corpus"
            write_tree(source, {"long.md": paragraph})
            output = root / "docs.txt"
            code, stdout, stderr = run_main(
                [
                    "--input",
                    str(source),
                    "--output",
                    str(output),
                    "--max-bytes",
                    "64",
                ]
            )
            self.assertEqual(code, 0, stderr)
            chunks = read_docs(output)
            self.assertGreater(len(chunks), 100)
            for chunk in chunks:
                self.assertLessEqual(len(chunk.encode("utf-8")), 64)
            summary = json.loads(stdout)
            self.assertLessEqual(summary["max_chunk_bytes"], 64)
            self.assertEqual(
                summary["safe_ctx_tokens"], summary["max_chunk_bytes"]
            )

    def test_budget_holds_for_every_size_and_content_mix(self) -> None:
        text = (
            "short\n\n"
            + "word " * 500
            + "\n\n"
            + "unbroken" * 300
            + "\n\n"
            + "tail paragraph\n"
        )
        for budget in (16, 17, 23, 64, 199, 1200):
            with self.subTest(budget=budget):
                chunks = chunker.chunk_text(text, budget)
                self.assertTrue(chunks)
                widest = max(len(chunk.encode("utf-8")) for chunk in chunks)
                self.assertLessEqual(widest, budget)
                for chunk in chunks:
                    self.assertNotIn("\n", chunk)
                    self.assertTrue(chunk.strip())

    def test_chunk_text_rejects_a_budget_below_one_character(self) -> None:
        with self.assertRaisesRegex(ChunkError, "too small for a single"):
            chunker.chunk_text("\U0001f600" * 4, 2)


class LineFormatTest(unittest.TestCase):
    def test_no_chunk_holds_a_line_break_or_is_empty(self) -> None:
        messy = (
            "first line\nsecond line of the same paragraph\r\nthird\n"
            "\n"
            "   \n"
            "\n"
            "\tpadded paragraph with a NUL \x00 inside\n"
            "\n\n\n"
            "final paragraph\n"
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source = root / "corpus"
            write_tree(source, {"messy.txt": messy})
            output = root / "docs.txt"
            code, _, stderr = run_main(
                [
                    "--input",
                    str(source),
                    "--output",
                    str(output),
                    "--max-bytes",
                    "32",
                ]
            )
            self.assertEqual(code, 0, stderr)
            chunks = read_docs(output)
            self.assertTrue(chunks)
            for chunk in chunks:
                self.assertNotIn("\n", chunk)
                self.assertNotIn("\r", chunk)
                self.assertNotEqual(chunk, "")
                self.assertTrue(chunk.strip())
                self.assertNotIn("\x00", chunk)

    def test_write_chunks_refuses_a_chunk_the_format_cannot_carry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary) / "docs.txt"
            with self.assertRaisesRegex(ChunkError, "line break"):
                chunker.write_chunks(output, ["fine", "bad\nchunk"])
            with self.assertRaisesRegex(ChunkError, "empty after cleaning"):
                chunker.write_chunks(output, ["fine", "   "])

    def test_a_chunk_never_spans_two_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source = root / "corpus"
            write_tree(
                source,
                {"a.md": "alpha text\n", "b.md": "bravo text\n"},
            )
            output = root / "docs.txt"
            code, stdout, stderr = run_main(
                [
                    "--input",
                    str(source),
                    "--output",
                    str(output),
                    "--max-bytes",
                    "1200",
                ]
            )
            self.assertEqual(code, 0, stderr)
            self.assertEqual(read_docs(output), ["alpha text", "bravo text"])
            self.assertEqual(json.loads(stdout)["files"], 2)


class MultiByteTest(unittest.TestCase):
    def test_cjk_is_never_split_mid_character(self) -> None:
        source_text = "漢字仮名交じり文字列" * 120
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source = root / "corpus"
            write_tree(source, {"cjk.txt": source_text})
            output = root / "docs.txt"
            code, stdout, stderr = run_main(
                [
                    "--input",
                    str(source),
                    "--output",
                    str(output),
                    "--max-bytes",
                    "32",
                ]
            )
            self.assertEqual(code, 0, stderr)
            chunks = read_docs(output)
            self.assertGreater(len(chunks), 10)
            for chunk in chunks:
                encoded = chunk.encode("utf-8")
                self.assertLessEqual(len(encoded), 32)
                # A valid chunk never begins with a UTF-8 continuation byte
                # and survives an encode/decode round trip unchanged.
                self.assertNotEqual(encoded[0] & 0xC0, 0x80)
                self.assertEqual(encoded.decode("utf-8"), chunk)
            # The characters come back in source order with nothing lost or
            # duplicated: this text has no whitespace, so the chunker only
            # ever slices it.
            self.assertEqual("".join(chunks), source_text)
            self.assertEqual(
                json.loads(stdout)["total_bytes"],
                len(source_text.encode("utf-8")),
            )

    def test_accented_latin_and_astral_characters_round_trip(self) -> None:
        for source_text in (
            "áéíóúñüàèìòùçãõ" * 80,
            "\U0001f600\U0001f680\U0001f9ea" * 80,
        ):
            with self.subTest(sample=source_text[:2]):
                chunks = chunker.chunk_text(source_text, 17)
                self.assertGreater(len(chunks), 5)
                for chunk in chunks:
                    encoded = chunk.encode("utf-8")
                    self.assertLessEqual(len(encoded), 17)
                    self.assertEqual(encoded.decode("utf-8"), chunk)
                self.assertEqual("".join(chunks), source_text)

    def test_multi_byte_split_survives_a_write_read_round_trip(self) -> None:
        source_text = "日本語テキスト" * 60
        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary) / "docs.txt"
            chunks = chunker.chunk_text(source_text, 20)
            chunker.write_chunks(output, chunks)
            self.assertEqual(read_docs(output), chunks)
            self.assertEqual("".join(read_docs(output)), source_text)


class DeterminismTest(unittest.TestCase):
    def test_the_same_tree_twice_is_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source = root / "corpus"
            write_tree(
                source,
                {
                    "zulu.md": "zulu paragraph one\n\nzulu paragraph two\n",
                    "alpha.md": "alpha " * 300,
                    "nested/inner.txt": "nested text\n\nmore nested text\n",
                },
            )
            first = root / "first.txt"
            second = root / "second.txt"
            code_a, out_a, err_a = run_main(
                [
                    "--input",
                    str(source),
                    "--output",
                    str(first),
                    "--max-bytes",
                    "48",
                ]
            )
            code_b, out_b, err_b = run_main(
                [
                    "--input",
                    str(source),
                    "--output",
                    str(second),
                    "--max-bytes",
                    "48",
                ]
            )
            self.assertEqual(code_a, 0, err_a)
            self.assertEqual(code_b, 0, err_b)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            summary_a = json.loads(out_a)
            summary_b = json.loads(out_b)
            self.assertEqual(summary_a["sha256"], summary_b["sha256"])
            del summary_a["output"], summary_b["output"]
            self.assertEqual(summary_a, summary_b)
            # No absolute machine path leaks into the docs file itself.
            self.assertNotIn(
                str(root).encode("utf-8"), first.read_bytes()
            )

    def test_file_order_is_sorted_not_creation_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source = root / "corpus"
            # Deliberately non-alphabetical creation order.
            write_tree(
                source,
                {
                    "zulu.md": "ZULU body\n",
                    "sub/aaa.md": "SUBAAA body\n",
                    "alpha.md": "ALPHA body\n",
                    "mike.md": "MIKE body\n",
                },
            )
            output = root / "docs.txt"
            code, _, stderr = run_main(
                [
                    "--input",
                    str(source),
                    "--output",
                    str(output),
                    "--max-bytes",
                    "1200",
                ]
            )
            self.assertEqual(code, 0, stderr)
            chunks = read_docs(output)
            self.assertEqual(
                [chunk.split()[0] for chunk in chunks],
                ["ALPHA", "MIKE", "SUBAAA", "ZULU"],
            )

    def test_source_files_returns_a_sorted_list(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = pathlib.Path(temporary) / "corpus"
            write_tree(
                source,
                {
                    "zulu.md": "z\n",
                    "b/deep.md": "d\n",
                    "alpha.md": "a\n",
                },
            )
            files = chunker.source_files(source, [".md"])
            self.assertEqual(files, sorted(files))
            self.assertEqual(
                [path.relative_to(source).as_posix() for path in files],
                ["alpha.md", "b/deep.md", "zulu.md"],
            )


class DirectoryFilterTest(unittest.TestCase):
    def test_git_and_node_modules_trees_are_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source = root / "corpus"
            write_tree(
                source,
                {
                    "keep.md": "KEEP body\n",
                    ".git/objects/skip.md": "GITSKIP body\n",
                    "node_modules/pkg/skip.md": "NODESKIP body\n",
                    ".github/workflows/skip.md": "GHSKIP body\n",
                    "__pycache__/skip.md": "PYCSKIP body\n",
                    "deep/node_modules/skip.md": "DEEPNODESKIP body\n",
                },
            )
            output = root / "docs.txt"
            code, stdout, stderr = run_main(
                [
                    "--input",
                    str(source),
                    "--output",
                    str(output),
                    "--max-bytes",
                    "1200",
                ]
            )
            self.assertEqual(code, 0, stderr)
            self.assertEqual(read_docs(output), ["KEEP body"])
            self.assertEqual(json.loads(stdout)["files"], 1)


class SuffixFilterTest(unittest.TestCase):
    def test_suffix_is_repeatable_and_filters(self) -> None:
        tree = {
            "a.md": "AMD body\n",
            "b.txt": "BTXT body\n",
            "c.rst": "CRST body\n",
            "d.MD": "DUPPER body\n",
            "e.log": "ELOG body\n",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source = root / "corpus"
            write_tree(source, tree)

            def markers(argv_tail: list[str], name: str) -> list[str]:
                output = root / name
                code, _, stderr = run_main(
                    [
                        "--input",
                        str(source),
                        "--output",
                        str(output),
                        "--max-bytes",
                        "1200",
                    ]
                    + argv_tail
                )
                self.assertEqual(code, 0, stderr)
                return [chunk.split()[0] for chunk in read_docs(output)]

            # Default suffixes.
            self.assertEqual(
                markers([], "default.txt"), ["AMD", "BTXT", "DUPPER"]
            )
            # Single explicit suffix replaces the default entirely.
            self.assertEqual(
                markers(["--suffix", ".rst"], "rst.txt"), ["CRST"]
            )
            # Repeated --suffix accumulates.
            self.assertEqual(
                markers(
                    ["--suffix", ".rst", "--suffix", ".log"], "both.txt"
                ),
                ["CRST", "ELOG"],
            )
            self.assertEqual(
                markers(
                    ["--suffix", ".md", "--suffix", ".rst"], "mdrst.txt"
                ),
                ["AMD", "CRST", "DUPPER"],
            )


class ErrorCaseTest(unittest.TestCase):
    def test_empty_directory_exits_nonzero_and_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source = root / "empty"
            source.mkdir()
            output = root / "docs.txt"
            code, stdout, stderr = run_main(
                ["--input", str(source), "--output", str(output)]
            )
            self.assertEqual(code, 1)
            self.assertEqual(stdout, "")
            self.assertIn("no files matching", stderr)
            self.assertFalse(output.exists())

    def test_no_matching_suffix_exits_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source = root / "corpus"
            write_tree(source, {"only.rst": "body\n", "also.bin": "body\n"})
            output = root / "docs.txt"
            code, _, stderr = run_main(
                ["--input", str(source), "--output", str(output)]
            )
            self.assertEqual(code, 1)
            self.assertIn("no files matching", stderr)
            self.assertIn(".md", stderr)
            self.assertFalse(output.exists())

    def test_max_bytes_below_the_minimum_exits_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source = root / "corpus"
            write_tree(source, {"a.md": "body\n"})
            output = root / "docs.txt"
            for budget in ("15", "0", "-1"):
                with self.subTest(budget=budget):
                    code, stdout, stderr = run_main(
                        [
                            "--input",
                            str(source),
                            "--output",
                            str(output),
                            "--max-bytes",
                            budget,
                        ]
                    )
                    self.assertEqual(code, 1)
                    self.assertEqual(stdout, "")
                    self.assertIn("--max-bytes must be at least 16", stderr)
                    self.assertFalse(output.exists())

    def test_missing_input_directory_exits_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            code, _, stderr = run_main(
                [
                    "--input",
                    str(root / "absent"),
                    "--output",
                    str(root / "docs.txt"),
                ]
            )
            self.assertEqual(code, 1)
            self.assertIn("no files matching", stderr)

    def test_whitespace_only_corpus_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = pathlib.Path(temporary) / "corpus"
            write_tree(source, {"blank.md": "\n\n   \n\t\n", "b.txt": ""})
            with self.assertRaisesRegex(ChunkError, "empty after cleaning"):
                chunker.chunk_corpus(source, 1200, [".md", ".txt"])

    def test_chunk_corpus_reports_a_missing_match(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = pathlib.Path(temporary)
            with self.assertRaises(ChunkError) as context:
                chunker.chunk_corpus(source, 1200, [".md"])
            self.assertIn("no files matching .md", str(context.exception))


class SummaryTest(unittest.TestCase):
    def test_summary_matches_the_written_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source = root / "corpus"
            write_tree(
                source,
                {
                    "a.md": "alpha " * 200 + "\n\nsecond alpha paragraph\n",
                    "b.txt": "bravo body\n\n" + "bravo " * 90,
                    "sub/c.md": "日本語の段落 " * 30,
                },
            )
            output = root / "nested" / "docs.txt"
            code, stdout, stderr = run_main(
                [
                    "--input",
                    str(source),
                    "--output",
                    str(output),
                    "--max-bytes",
                    "96",
                ]
            )
            self.assertEqual(code, 0, stderr)
            summary = json.loads(stdout)
            self.assertEqual(
                sorted(summary),
                [
                    "chunks",
                    "files",
                    "max_chunk_bytes",
                    "mean_chunk_bytes",
                    "output",
                    "safe_ctx_tokens",
                    "sha256",
                    "total_bytes",
                ],
            )
            chunks = read_docs(output)
            encoded = [len(chunk.encode("utf-8")) for chunk in chunks]
            self.assertEqual(summary["chunks"], len(chunks))
            self.assertEqual(summary["files"], 3)
            self.assertEqual(summary["max_chunk_bytes"], max(encoded))
            self.assertEqual(
                summary["safe_ctx_tokens"], summary["max_chunk_bytes"]
            )
            self.assertEqual(summary["total_bytes"], sum(encoded))
            self.assertEqual(
                summary["mean_chunk_bytes"],
                round(sum(encoded) / len(encoded), 1),
            )
            self.assertEqual(summary["output"], str(output))
            self.assertEqual(
                summary["sha256"],
                hashlib.sha256(output.read_bytes()).hexdigest(),
            )
            self.assertLessEqual(summary["max_chunk_bytes"], 96)
            # The chunk count is exactly the line count of the written file.
            self.assertEqual(
                summary["chunks"], output.read_bytes().count(b"\n")
            )

    def test_empty_files_do_not_count_toward_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source = root / "corpus"
            write_tree(
                source,
                {"real.md": "content\n", "blank.md": "\n \n", "zero.txt": ""},
            )
            output = root / "docs.txt"
            code, stdout, stderr = run_main(
                ["--input", str(source), "--output", str(output)]
            )
            self.assertEqual(code, 0, stderr)
            summary = json.loads(stdout)
            self.assertEqual(summary["files"], 1)
            self.assertEqual(summary["chunks"], 1)
            self.assertEqual(read_docs(output), ["content"])


class EndToEndTest(unittest.TestCase):
    def test_subprocess_run_exits_zero_with_parseable_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source = root / "corpus"
            write_tree(
                source,
                {
                    "doc.md": "first paragraph\n\n" + "filler word " * 200,
                    "notes/other.txt": "ünïcödé paragraph\n",
                    ".git/hidden.md": "must not appear\n",
                },
            )
            output = root / "docs.txt"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--input",
                    str(source),
                    "--output",
                    str(output),
                    "--max-bytes",
                    "128",
                ],
                capture_output=True,
                text=True,
                cwd=str(ROOT),
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            summary = json.loads(completed.stdout)
            chunks = read_docs(output)
            self.assertEqual(summary["chunks"], len(chunks))
            self.assertEqual(summary["files"], 2)
            self.assertLessEqual(summary["max_chunk_bytes"], 128)
            self.assertEqual(
                summary["sha256"],
                hashlib.sha256(output.read_bytes()).hexdigest(),
            )
            self.assertNotIn("must not appear", output.read_text("utf-8"))

    def test_subprocess_run_exits_nonzero_on_a_bad_budget(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source = root / "corpus"
            write_tree(source, {"doc.md": "body\n"})
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--input",
                    str(source),
                    "--output",
                    str(root / "docs.txt"),
                    "--max-bytes",
                    "4",
                ],
                capture_output=True,
                text=True,
                cwd=str(ROOT),
            )
            self.assertEqual(completed.returncode, 1)
            self.assertEqual(completed.stdout, "")
            self.assertIn("--max-bytes must be at least 16", completed.stderr)


if __name__ == "__main__":
    unittest.main()
