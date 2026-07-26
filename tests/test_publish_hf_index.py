#!/usr/bin/env python3
"""Tests for packing a leann artifact pair into a Hugging Face repository."""

from __future__ import annotations

import base64
import contextlib
import datetime as dt
import hashlib
import io
import json
import os
import pathlib
import re
import socket
import sys
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import publish_hf_index  # noqa: E402
from publish_hf_index import (  # noqa: E402
    MANIFEST_MAGIC,
    MANIFEST_NAME,
    PublishError,
    commit_ndjson,
    lfs_batch_payload,
    preupload_payload,
    render_manifest,
    resolve_url,
    validate_manifest_name,
    validate_repo_id,
)


# Fixed artifact bodies: 527 and 442 bytes, both three-digit sizes, so no size
# rendered into the manifest or the card can ever look like a year.
INDEX_BYTES = b"LEANNC03 synthetic index bytes\n" * 17
DOCS_BYTES = b"LEANDC02 synthetic document store\n" * 13

# What the fake `leann stats --format json` reports. The keys are the ones the
# real CLI emits; the numbers are deliberately comma-free of four-digit runs.
STATS = {
    "nodes": 4321,
    "dimension": 768,
    "approximation": "pq-m16-b8",
    "index_bytes": 98765,
    "raw_document_bytes": 654321,
    "pair_identity": "5b" * 32,
    "embedder": "llama:nomic-embed-text-v1.5:gpu0",
    "model_source": "hf:nomic-ai/nomic-embed-text-v1.5-GGUF/model.Q4_K_M.gguf",
    "model_sha256": "9f" * 32,
    "model_bytes": 87654321,
    "context_tokens": 512,
    "document_prefix": "search_document: ",
    "query_prefix": "search_query: ",
    "card": {"corpus": "beir-nq-mini", "builder": "leann.cpp"},
}

DIGEST_FIELD = re.compile(r"^sha256:[0-9a-f]{64}$")
HEX_DIGEST = re.compile(r"[0-9a-f]{64}")


def _boom(*args: object, **kwargs: object) -> None:
    raise AssertionError("network")


@contextlib.contextmanager
def no_network():
    """Make any attempt to reach the network an immediate failure.

    `pack` is specified to open no socket at all, the same rule the C++
    `leann pull` follows, so the proof is that packing still succeeds with
    both the high-level and the low-level entry points removed.
    """
    with mock.patch("socket.socket", new=_boom):
        with mock.patch("urllib.request.urlopen", new=_boom):
            yield


class PackFixture:
    """A pair of dummy artifacts plus a fake leann binary that reports stats.

    The fake is a tiny executable script rather than a patched subprocess:
    `pack` shells out, and the point is to exercise that boundary without
    needing a built binary or a real index.
    """

    def __init__(
        self, root: pathlib.Path, stats: dict[str, object] | None = None
    ) -> None:
        self.root = root
        self.prefix = root / "corpus"
        self.index = root / "corpus.leann"
        self.docs = root / "corpus.docs"
        self.index.write_bytes(INDEX_BYTES)
        self.docs.write_bytes(DOCS_BYTES)
        self.stats = json.loads(json.dumps(STATS if stats is None else stats))

        self.leann = root / "fake-leann"
        self.stats_path = root / "fake-leann.stats.json"
        self.argv_path = root / "fake-leann.argv"
        self.stats_path.write_text(
            json.dumps(self.stats), encoding="utf-8"
        )
        interpreter = sys.executable
        shebang = (
            f"#!{interpreter}"
            if interpreter and " " not in interpreter
            else "#!/usr/bin/env python3"
        )
        self.leann.write_text(
            "\n".join(
                [
                    shebang,
                    "import json, pathlib, sys",
                    "here = pathlib.Path(__file__)",
                    "here.with_name(here.name + '.argv').write_text(",
                    "    json.dumps(sys.argv[1:]) + '\\n', encoding='utf-8')",
                    "sys.stdout.write(",
                    "    here.with_name(here.name + '.stats.json')",
                    "    .read_text(encoding='utf-8'))",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        self.leann.chmod(0o755)

    def pack_argv(
        self, output: pathlib.Path, *extra: str, repo: str = "owner/name"
    ) -> list[str]:
        return [
            "pack",
            "--index",
            str(self.prefix),
            "--output",
            str(output),
            "--repo",
            repo,
            "--leann",
            str(self.leann),
            *extra,
        ]


def run_cli(argv: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = publish_hf_index.main(argv)
    return code, out.getvalue(), err.getvalue()


def manifest_records(text: str) -> list[list[str]]:
    """Split a manifest the way the C++ reader does: LF lines, tab fields."""
    lines = text.split("\n")
    assert lines[-1] == ""
    return [line.split("\t") for line in lines[1:-1]]


class PackTest(unittest.TestCase):
    def test_pack_writes_only_the_four_expected_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            fixture = PackFixture(root)
            output = root / "out"
            code, stdout, stderr = run_cli(fixture.pack_argv(output))
            self.assertEqual(code, 0, stderr)

            self.assertEqual(
                sorted(path.name for path in output.iterdir()),
                [
                    ".gitattributes",
                    "README.md",
                    "corpus.docs",
                    "corpus.leann",
                    "leann.manifest",
                ],
            )
            self.assertEqual(
                (output / "corpus.leann").read_bytes(), INDEX_BYTES
            )
            self.assertEqual((output / "corpus.docs").read_bytes(), DOCS_BYTES)

            summary = json.loads(stdout)
            self.assertIs(summary["pushed"], False)
            self.assertEqual(summary["prefix"], "corpus")
            self.assertEqual(
                [entry["name"] for entry in summary["files"]],
                ["corpus.docs", "corpus.leann"],
            )
            self.assertEqual(
                summary["model"]["source"], STATS["model_source"]
            )

            # The stats really came from the binary, invoked as documented.
            self.assertEqual(
                json.loads(fixture.argv_path.read_text(encoding="utf-8")),
                ["stats", "--index", str(fixture.prefix), "--format", "json"],
            )

    def test_manifest_grammar_matches_the_cpp_reader(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            fixture = PackFixture(root)
            output = root / "out"
            code, _, stderr = run_cli(
                fixture.pack_argv(output, "--revision", "v1")
            )
            self.assertEqual(code, 0, stderr)

            raw = (output / MANIFEST_NAME).read_bytes()
            self.assertNotIn(b"\r", raw)
            self.assertTrue(raw.endswith(b"\n"))
            self.assertFalse(raw.endswith(b"\n\n"))
            text = raw.decode("utf-8")
            self.assertEqual(text.split("\n")[0], MANIFEST_MAGIC)
            self.assertEqual(text.split("\n")[0], "LEANNMF1")

            records = manifest_records(text)
            self.assertEqual(
                [record[0] for record in records],
                [
                    "repo",
                    "type",
                    "revision",
                    "prefix",
                    "file",
                    "file",
                    "model",
                ],
            )
            self.assertEqual(records[0], ["repo", "owner/name"])
            self.assertEqual(records[1], ["type", "model"])
            self.assertEqual(records[2], ["revision", "v1"])
            self.assertEqual(records[3], ["prefix", "corpus"])

            files = [record for record in records if record[0] == "file"]
            self.assertEqual(
                [record[1] for record in files],
                sorted(record[1] for record in files),
            )
            self.assertEqual(
                [record[1] for record in files],
                ["corpus.docs", "corpus.leann"],
            )
            for record in files:
                self.assertEqual(len(record), 4)
                name, size, digest = record[1], record[2], record[3]
                self.assertRegex(size, r"^[0-9]+$")
                self.assertEqual(
                    int(size), (output / name).stat().st_size
                )
                self.assertRegex(digest, DIGEST_FIELD)
                self.assertEqual(digest, digest.lower())

            model = records[-1]
            self.assertEqual(len(model), 4)
            self.assertEqual(model[1], STATS["model_source"])
            self.assertEqual(int(model[2]), STATS["model_bytes"])
            self.assertEqual(model[3], "sha256:" + str(STATS["model_sha256"]))

            for record in records:
                self.assertTrue(all(field != "" for field in record))
            self.assertNotIn("", text.split("\n")[1:-1])

    def test_manifest_digests_are_the_real_sha256_of_the_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            fixture = PackFixture(root)
            output = root / "out"
            self.assertEqual(run_cli(fixture.pack_argv(output))[0], 0)

            text = (output / MANIFEST_NAME).read_text(encoding="utf-8")
            files = [
                record
                for record in manifest_records(text)
                if record[0] == "file"
            ]
            self.assertEqual(len(files), 2)
            for _, name, size, digest in files:
                packed = output / name
                expected = hashlib.sha256(packed.read_bytes()).hexdigest()
                self.assertEqual(digest, f"sha256:{expected}")
                self.assertEqual(int(size), len(packed.read_bytes()))
                # And the packed copy is the source artifact, byte for byte.
                self.assertEqual(
                    packed.read_bytes(), (root / name).read_bytes()
                )

    def test_manifest_omits_the_model_record_when_none_is_recorded(
        self,
    ) -> None:
        stats = dict(STATS)
        stats["model_source"] = ""
        stats["model_sha256"] = "0" * 64
        stats["model_bytes"] = 0
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            fixture = PackFixture(root, stats=stats)
            output = root / "out"
            code, stdout, stderr = run_cli(fixture.pack_argv(output))
            self.assertEqual(code, 0, stderr)
            self.assertIsNone(json.loads(stdout)["model"])
            text = (output / MANIFEST_NAME).read_text(encoding="utf-8")
            self.assertEqual(
                [record[0] for record in manifest_records(text)],
                ["repo", "type", "revision", "prefix", "file", "file"],
            )

    def test_dataset_repo_type_is_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            fixture = PackFixture(root)
            output = root / "out"
            code, _, stderr = run_cli(
                fixture.pack_argv(output, "--repo-type", "dataset")
            )
            self.assertEqual(code, 0, stderr)
            text = (output / MANIFEST_NAME).read_text(encoding="utf-8")
            self.assertIn("type\tdataset", text)
            card = (output / "README.md").read_text(encoding="utf-8")
            self.assertIn("pretty_name: name", card)

    def test_pack_opens_no_socket(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            fixture = PackFixture(root)
            output = root / "out"
            with no_network():
                code, _, stderr = run_cli(fixture.pack_argv(output))
            self.assertEqual(code, 0, stderr)
            self.assertTrue((output / MANIFEST_NAME).is_file())

    def test_pack_leaves_no_temporary_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            fixture = PackFixture(root)
            output = root / "out"
            self.assertEqual(run_cli(fixture.pack_argv(output))[0], 0)
            # Written twice: the second pass replaces existing files.
            self.assertEqual(run_cli(fixture.pack_argv(output))[0], 0)
            leftovers = [
                path.name
                for path in output.iterdir()
                if path.name.endswith(".tmp") or ".tmp." in path.name
            ]
            self.assertEqual(leftovers, [])

    def test_pack_rejects_a_half_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            fixture = PackFixture(root)
            fixture.docs.unlink()
            code, _, stderr = run_cli(fixture.pack_argv(root / "out"))
            self.assertEqual(code, 1)
            self.assertIn("missing .docs artifact", stderr)

    def test_readme_frontmatter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            fixture = PackFixture(root)
            output = root / "out"
            code, _, stderr = run_cli(
                fixture.pack_argv(
                    output, "--tag", "beir", "--language", "en"
                )
            )
            self.assertEqual(code, 0, stderr)

            card = (output / "README.md").read_text(encoding="utf-8")
            self.assertTrue(card.startswith("---\n"))
            lines = card.split("\n")
            closing = lines.index("---", 1)
            self.assertGreater(closing, 1)
            frontmatter = lines[1:closing]
            self.assertIn("license: apache-2.0", frontmatter)
            self.assertIn("tags:", frontmatter)
            self.assertIn("language:", frontmatter)
            self.assertIn("  - en", frontmatter)
            for tag in ("leann", "leann.cpp", "retrieval", "rag", "beir"):
                self.assertIn(f"  - {tag}", frontmatter)
            self.assertIn("# owner/name", lines)
            self.assertIn(
                f"leann verify --index corpus --manifest {MANIFEST_NAME}",
                card,
            )

    def test_gitattributes_marks_both_artifacts_as_lfs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            fixture = PackFixture(root)
            output = root / "out"
            self.assertEqual(run_cli(fixture.pack_argv(output))[0], 0)
            text = (output / ".gitattributes").read_text(encoding="utf-8")
            self.assertIn("*.leann filter=lfs", text)
            self.assertIn("*.docs filter=lfs", text)

    def test_pack_is_byte_identical_across_runs_and_leaks_no_environment(
        self,
    ) -> None:
        hostname = socket.gethostname()
        year = str(dt.date.today().year)
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            fixture = PackFixture(root)
            first, second = root / "a", root / "b"
            self.assertEqual(run_cli(fixture.pack_argv(first))[0], 0)
            self.assertEqual(run_cli(fixture.pack_argv(second))[0], 0)

            for name in (MANIFEST_NAME, "README.md", ".gitattributes"):
                left = (first / name).read_bytes()
                right = (second / name).read_bytes()
                self.assertEqual(left, right, f"{name} is not deterministic")

                text = left.decode("utf-8")
                self.assertNotIn("/Users/", text)
                self.assertNotIn(str(root), text)
                self.assertNotIn(str(first), text)
                if len(hostname) >= 5:
                    self.assertNotIn(hostname, text)
                # SHA-256 hex is content-derived and may contain any four
                # digits by chance, so the timestamp check ignores digests.
                self.assertNotIn(year, HEX_DIGEST.sub("<digest>", text))


class PushRefusalTest(unittest.TestCase):
    def packed(self, root: pathlib.Path) -> pathlib.Path:
        fixture = PackFixture(root)
        output = root / "out"
        code, _, stderr = run_cli(fixture.pack_argv(output))
        self.assertEqual(code, 0, stderr)
        return output

    def push_argv(self, directory: pathlib.Path, *extra: str) -> list[str]:
        return [
            "push",
            "--directory",
            str(directory),
            "--repo",
            "owner/name",
            *extra,
        ]

    def test_push_without_yes_refuses_before_the_network(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = self.packed(pathlib.Path(temporary))
            with mock.patch.dict(
                os.environ, {"HF_TOKEN": "not-a-real-token"}
            ):
                with no_network():
                    code, _, stderr = run_cli(self.push_argv(directory))
            self.assertEqual(code, 1)
            self.assertIn("refusing to upload without --yes", stderr)

    def test_push_without_a_token_refuses_before_the_network(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = self.packed(pathlib.Path(temporary))
            with mock.patch.dict(os.environ, {"HF_TOKEN": ""}):
                with no_network():
                    code, _, stderr = run_cli(
                        self.push_argv(directory, "--yes")
                    )
            self.assertEqual(code, 1)
            self.assertIn("no Hugging Face token", stderr)

    def test_push_rejects_a_bad_repository_before_anything_else(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = self.packed(pathlib.Path(temporary))
            with mock.patch.dict(os.environ, {"HF_TOKEN": "token"}):
                with no_network():
                    code, _, stderr = run_cli(
                        [
                            "push",
                            "--directory",
                            str(directory),
                            "--repo",
                            "no-slash",
                            "--yes",
                        ]
                    )
            self.assertEqual(code, 1)
            self.assertIn("repository must be OWNER/NAME", stderr)


class PayloadTest(unittest.TestCase):
    def test_resolve_url_for_model_and_dataset_repositories(self) -> None:
        self.assertEqual(
            resolve_url("model", "owner/name", "main", "corpus.leann"),
            "https://huggingface.co/owner/name/resolve/main/corpus.leann",
        )
        dataset = resolve_url("dataset", "owner/name", "v2", "corpus.docs")
        self.assertEqual(
            dataset,
            "https://huggingface.co/datasets/owner/name/resolve/v2/"
            "corpus.docs",
        )
        self.assertIn("/datasets/", dataset)
        self.assertNotIn(
            "/datasets/",
            resolve_url("model", "owner/name", "main", "corpus.docs"),
        )

    def test_resolve_url_rejects_traversal_and_absolute_paths(self) -> None:
        for name in ("../secret", "a/../../b", "/etc/passwd"):
            with self.assertRaises(PublishError) as context:
                resolve_url("model", "owner/name", "main", name)
            self.assertIn("unsafe manifest file name", str(context.exception))
        with self.assertRaisesRegex(PublishError, "OWNER/NAME"):
            resolve_url("model", "owner", "main", "corpus.leann")

    def test_preupload_payload_shape(self) -> None:
        payload = preupload_payload(
            [("corpus.leann", 527, b"\x00\x01head"), ("README.md", 12, b"# x")]
        )
        self.assertEqual(list(payload), ["files"])
        self.assertEqual(len(payload["files"]), 2)
        first = payload["files"][0]
        self.assertEqual(sorted(first), ["path", "sample", "size"])
        self.assertEqual(first["path"], "corpus.leann")
        self.assertEqual(first["size"], 527)
        self.assertEqual(base64.b64decode(first["sample"]), b"\x00\x01head")
        json.dumps(payload)

    def test_lfs_batch_payload_shape(self) -> None:
        oid = hashlib.sha256(INDEX_BYTES).hexdigest()
        payload = lfs_batch_payload([(oid, 527)])
        self.assertEqual(payload["operation"], "upload")
        self.assertEqual(payload["transfers"], ["basic"])
        self.assertEqual(payload["hash_algo"], "sha_256")
        self.assertEqual(payload["objects"], [{"oid": oid, "size": 527}])
        json.dumps(payload)

    def test_commit_ndjson_is_one_object_per_line(self) -> None:
        oid = hashlib.sha256(DOCS_BYTES).hexdigest()
        text = commit_ndjson(
            "Publish leann.cpp index",
            [("README.md", b"# card\n")],
            [("corpus.docs", oid, 442)],
        )
        self.assertTrue(text.endswith("\n"))
        self.assertFalse(text.endswith("\n\n"))
        lines = text[:-1].split("\n")
        self.assertEqual(len(lines), 3)
        records = [json.loads(line) for line in lines]

        self.assertEqual(records[0]["key"], "header")
        self.assertTrue(records[0]["value"]["summary"])
        self.assertEqual(
            records[0]["value"]["summary"], "Publish leann.cpp index"
        )

        self.assertEqual(records[1]["key"], "file")
        self.assertEqual(records[1]["value"]["path"], "README.md")
        self.assertEqual(records[1]["value"]["encoding"], "base64")
        self.assertEqual(
            base64.b64decode(records[1]["value"]["content"]), b"# card\n"
        )

        self.assertEqual(records[2]["key"], "lfsFile")
        self.assertEqual(records[2]["value"]["path"], "corpus.docs")
        self.assertEqual(records[2]["value"]["algo"], "sha256")
        self.assertRegex(records[2]["value"]["oid"], r"^[0-9a-f]{64}$")
        self.assertEqual(records[2]["value"]["size"], 442)


class ValidationTest(unittest.TestCase):
    def test_validate_repo_id_accepts_owner_slash_name(self) -> None:
        self.assertEqual(validate_repo_id("owner/name"), "owner/name")

    def test_validate_repo_id_rejects_bad_cases(self) -> None:
        for repo in ("", "name", "/name", "owner/", "owner//name", "a/b/c"):
            with self.assertRaises(PublishError) as context:
                validate_repo_id(repo)
            self.assertIn("OWNER/NAME", str(context.exception))

    def test_validate_manifest_name_accepts_plain_names(self) -> None:
        for name in ("corpus.leann", MANIFEST_NAME, "sub/corpus.docs"):
            self.assertEqual(validate_manifest_name(name), name)

    def test_validate_manifest_name_rejects_bad_cases(self) -> None:
        with self.assertRaisesRegex(PublishError, "empty"):
            validate_manifest_name("")
        for name in (
            "/absolute",
            "../escape",
            "a/../b",
            "..",
            "back\\slash",
            "tab\there",
            "null\x00byte",
            "carriage\rreturn",
            "line\nfeed",
            "bell\x07",
            "delete\x7f",
        ):
            with self.assertRaises(PublishError) as context:
                validate_manifest_name(name)
            self.assertIn("unsafe manifest file name", str(context.exception))

    def test_no_field_can_forge_an_extra_manifest_record(self) -> None:
        """A newline in any field injects a record rather than corrupting one.

        An injected `file` record is a file `leann pull` would go on to print
        a download command for, so every field is held to the rule, not just
        the ones that name a path.
        """
        files = [("corpus.leann", 1, "a" * 64), ("corpus.docs", 2, "b" * 64)]
        injection = "main\nfile\tevil.bin\t10\tsha256:" + "c" * 64
        with self.assertRaises(PublishError) as context:
            render_manifest(
                "owner/name", "model", injection, "corpus", files, None
            )
        self.assertIn("revision", str(context.exception))

        for bad in ("origin\r\nx", "origin\nfile\tevil.bin\t1\tsha256:x"):
            with self.assertRaises(PublishError) as context:
                render_manifest(
                    "owner/name",
                    "model",
                    "main",
                    "corpus",
                    files,
                    (bad, 1, "c" * 64),
                )
            self.assertIn("model source", str(context.exception))

        # `repo` used to be shape-checked only, so a newline in it passed the
        # OWNER/NAME test and injected a record the C++ reader then accepted.
        with self.assertRaises(PublishError) as context:
            render_manifest(
                "owner/n\nfile\tevil.bin\t10\tsha256:" + "c" * 64,
                "model",
                "main",
                "corpus",
                files,
                None,
            )
        self.assertIn("repository", str(context.exception))

        # A dot segment is inert in a name but not in a URL path.
        for field, value in (
            ("revision", "../../other/repo/resolve/main"),
            ("repo", "owner/.."),
        ):
            with self.assertRaises(PublishError):
                render_manifest(
                    value if field == "repo" else "owner/name",
                    "model",
                    value if field == "revision" else "main",
                    "corpus",
                    files,
                    None,
                )

        # The good path still renders exactly one record per input.
        rendered = render_manifest(
            "owner/name", "model", "main", "corpus", files, None
        )
        self.assertEqual(
            sum(1 for line in rendered.splitlines() if line.startswith("file\t")),
            2,
        )


if __name__ == "__main__":
    unittest.main()
