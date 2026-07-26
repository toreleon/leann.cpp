#!/usr/bin/env python3
"""Pack a built leann.cpp artifact pair into a Hugging Face repository.

Two subcommands. `pack` reads a published `.leann`/`.docs` pair, digests it,
and writes an upload-ready directory: the artifacts, a `leann.manifest` in the
LEANNMF1 format that `leann pull` and `leann verify` read, a `.gitattributes`
marking both artifacts as LFS, and a README model card. `pack` opens no socket
and its output is byte-identical across runs — no timestamps, no hostnames, no
absolute local paths.

`push` uploads that directory over the Hugging Face HTTP API using only the
standard library. It is gated on an explicit --yes and a token from --token-file
or HF_TOKEN.

What this does not do: the manifest is not signed and there is no trust root.
A matching digest proves the bytes are the ones the publisher recorded, not
that the publisher is trustworthy, and not that the index is any good.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import pathlib
import shutil
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Sequence
from typing import Any


MANIFEST_MAGIC = "LEANNMF1"
MANIFEST_NAME = "leann.manifest"
HF_ENDPOINT = "https://huggingface.co"
READ_BLOCK_BYTES = 1024 * 1024
# Anything at or above this is uploaded through git-lfs rather than committed
# inline. Both leann artifacts are expected to exceed it.
LFS_THRESHOLD_BYTES = 10 * 1024 * 1024
# The preupload endpoint wants a base64 sample of the head of each file so it
# can classify binary content.
PREUPLOAD_SAMPLE_BYTES = 512


class PublishError(RuntimeError):
    """A pair could not be packed or a repository could not be updated."""


def hash_file(path: pathlib.Path) -> tuple[str, int]:
    """Return (sha256 hex, size) for a file, streamed in 1 MiB blocks."""
    checksum = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as source:
            for block in iter(lambda: source.read(READ_BLOCK_BYTES), b""):
                checksum.update(block)
                size += len(block)
    except OSError as error:
        raise PublishError(f"{path}: cannot read: {error}") from error
    return checksum.hexdigest(), size


def atomic_write(path: pathlib.Path, payload: str) -> None:
    """Write text through a temporary file so a failure leaves no partial."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


# The same allowlist app/main.cpp applies, for two reasons. An embedded tab or
# newline injects a whole extra manifest record rather than corrupting one, and
# `leann pull` interpolates these values into a single-quoted shell command
# that a person pastes into a terminal, so an apostrophe would end the quoting.
# Repository IDs, git revisions, and file names never legitimately need
# anything outside this set, so refusing beats escaping.
MANIFEST_TOKEN_CHARACTERS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-/"
)


def validate_manifest_text(value: str, field: str) -> str:
    """The weaker rule, for fields that are only ever printed as prose.

    A tab or a newline here would inject a whole extra manifest record rather
    than corrupt one, and any control character would break the line-oriented
    output that reproduces it.
    """
    if not value:
        raise PublishError(f"manifest {field} is empty")
    if "\t" in value or any(
        ord(character) < 0x20 or ord(character) == 0x7F for character in value
    ):
        raise PublishError(f"unsafe manifest {field}: {value!r}")
    return value


def validate_manifest_field(value: str, field: str) -> str:
    """The stricter rule, for every field that reaches a printed command."""
    validate_manifest_text(value, field)
    if not MANIFEST_TOKEN_CHARACTERS.issuperset(value):
        raise PublishError(f"unsafe manifest {field}: {value!r}")
    # A dot segment is inert in a name but not in a URL path: a revision of
    # "../../other/repo/resolve/main" would silently redirect every printed
    # download to a different repository.
    if ".." in value:
        raise PublishError(f"unsafe manifest {field}: {value!r}")
    return value


def validate_manifest_name(name: str) -> str:
    """Match the C++ reader's rules exactly, so pack cannot emit a manifest
    the reader would refuse."""
    validate_manifest_field(name, "file name")
    if name.startswith("/") or ".." in name:
        raise PublishError(f"unsafe manifest file name: {name!r}")
    return name


def validate_repo_id(repo_id: str) -> str:
    parts = repo_id.split("/")
    if len(parts) != 2 or not all(parts):
        raise PublishError(f"repository must be OWNER/NAME, got: {repo_id!r}")
    # Held to the same allowlist as every other manifest field. Checking only
    # the shape would let a newline through, and a newline in a manifest field
    # injects a whole extra record rather than corrupting one.
    validate_manifest_field(repo_id, "repository")
    return repo_id


def render_manifest(
    repo_id: str,
    repo_type: str,
    revision: str,
    prefix: str,
    files: Sequence[tuple[str, int, str]],
    model: tuple[str, int, str] | None,
) -> str:
    """Build LEANNMF1 text. Tab-separated, fixed record order, LF endings."""
    lines = [MANIFEST_MAGIC]
    lines.append(f"repo\t{validate_repo_id(repo_id)}")
    if repo_type not in ("model", "dataset"):
        raise PublishError(f"repo type must be model or dataset: {repo_type!r}")
    lines.append(f"type\t{repo_type}")
    lines.append(f"revision\t{validate_manifest_field(revision, 'revision')}")
    lines.append(f"prefix\t{validate_manifest_name(prefix)}")
    for name, size, digest in files:
        validate_manifest_name(name)
        lines.append(f"file\t{name}\t{size}\tsha256:{digest}")
    if model is not None:
        source, size, digest = model
        # A model origin string legitimately contains characters the token
        # allowlist excludes — "hf:owner/repo/file.gguf" has a colon — and it
        # is only ever printed as prose, never inside a command, so it takes
        # the weaker rule.
        validate_manifest_text(source, "model source")
        lines.append(f"model\t{source}\t{size}\tsha256:{digest}")
    return "\n".join(lines) + "\n"


def gitattributes_text() -> str:
    """HF's default .gitattributes covers .gguf and .safetensors but not
    these, so both artifacts must be marked explicitly or the push commits
    hundreds of megabytes into git proper."""
    return (
        "*.leann filter=lfs diff=lfs merge=lfs -text\n"
        "*.docs filter=lfs diff=lfs merge=lfs -text\n"
    )


def render_card(
    repo_id: str,
    repo_type: str,
    prefix: str,
    stats: dict[str, Any],
    files: Sequence[tuple[str, int, str]],
    license_id: str,
    tags: Sequence[str],
    languages: Sequence[str],
) -> str:
    """Render README.md with YAML frontmatter, built by string join.

    PyYAML is not an allowed dependency, and the frontmatter this needs is a
    handful of scalars and string lists.
    """
    frontmatter = ["---", f"license: {license_id}"]
    for language in languages:
        if language:
            frontmatter.append("language:")
            break
    for language in languages:
        if language:
            frontmatter.append(f"  - {language}")
    frontmatter.append("tags:")
    for tag in sorted({*tags, "leann", "leann.cpp", "retrieval", "rag"}):
        frontmatter.append(f"  - {tag}")
    if repo_type == "dataset":
        frontmatter.append(f"pretty_name: {repo_id.split('/')[-1]}")
    frontmatter.append("---")

    model_source = stats.get("model_source", "")
    body = [
        "",
        f"# {repo_id}",
        "",
        "A [leann.cpp](https://github.com/toreleon/leann.cpp) index: a pruned",
        "HNSW graph plus PQ codes, with the dense document vectors discarded.",
        "Document embeddings are recomputed at query time, which is why the",
        f"index is {int(stats.get('index_bytes', 0)):,} bytes for",
        f"{int(stats.get('raw_document_bytes', 0)):,} bytes of source text.",
        "",
        "## Use",
        "",
        "```bash",
        f"leann pull hf:{repo_id}",
        "# follow the printed commands, then:",
        f"leann verify --index {prefix} --manifest {MANIFEST_NAME}",
        f"leann search --index {prefix} --query 'your question' \\",
        "  --embedder llama --model <the GGUF named below> \\",
        f"  --ctx {int(stats.get('context_tokens', 512))}",
        "```",
        "",
        "`leann pull` prints commands and opens no socket; nothing is fetched",
        "or verified by the binary until you run those commands and `verify`.",
        "",
        "## Embedder",
        "",
        "This index can only be searched with the model it was built from.",
        "The fingerprint is checked on every query and a mismatch is refused.",
        "",
        f"- Source: `{model_source}`" if model_source else "- Source: unrecorded",
        f"- SHA-256: `{stats.get('model_sha256', '')}`",
        f"- Size: {int(stats.get('model_bytes', 0)):,} bytes",
        f"- Fingerprint: `{stats.get('embedder', '')}`",
        f"- Document prefix: `{stats.get('document_prefix', '')}`",
        f"- Query prefix: `{stats.get('query_prefix', '')}`",
        "",
        "## Index",
        "",
        f"- Nodes: {int(stats.get('nodes', 0)):,}",
        f"- Dimension: {int(stats.get('dimension', 0))}",
        f"- Approximation: {stats.get('approximation', '')}",
        f"- Pair identity: `{stats.get('pair_identity', '')}`",
        "",
        "## Files",
        "",
        "| File | Bytes | SHA-256 |",
        "| --- | --- | --- |",
    ]
    for name, size, digest in files:
        body.append(f"| `{name}` | {size:,} | `{digest}` |")

    card = stats.get("card") or {}
    if card:
        body.extend(["", "## Card", "", "| Key | Value |", "| --- | --- |"])
        for key in sorted(card):
            body.append(f"| `{key}` | {card[key]} |")

    body.extend(
        [
            "",
            "## What the digests do and do not establish",
            "",
            "The manifest is not signed and there is no trust root. A matching",
            "SHA-256 proves the bytes you downloaded are the bytes recorded",
            "here; it does not establish that the publisher is trustworthy, and",
            "it says nothing about retrieval quality.",
            "",
            "The digests also pin one specific build. Embeddings are not",
            "bit-identical across backends or batch shapes, so rebuilding this",
            "corpus on other hardware, or with different `--batch-tokens` or",
            "`--parallel`, produces a valid index with different bytes.",
            "",
        ]
    )
    return "\n".join(frontmatter + body)


def read_stats(leann_cli: str, prefix: pathlib.Path) -> dict[str, Any]:
    """Ask the leann binary to describe the pair, as JSON.

    The binary is the only thing that can read the header, and going through
    it means the card can never disagree with the artifact.
    """
    import subprocess

    command = [leann_cli, "stats", "--index", str(prefix), "--format", "json"]
    try:
        completed = subprocess.run(
            command, check=True, capture_output=True, text=True
        )
    except FileNotFoundError as error:
        raise PublishError(f"cannot run {leann_cli}: {error}") from error
    except subprocess.CalledProcessError as error:
        raise PublishError(
            f"leann stats failed for {prefix}: {error.stderr.strip()}"
        ) from error
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise PublishError(f"leann stats did not emit JSON: {error}") from error


def place_artifact(source: pathlib.Path, destination: pathlib.Path) -> None:
    """Hard-link the artifact into the output directory, copying only when
    the link cannot be made. An index can be gigabytes; copying it by default
    would double the disk cost of packing."""
    destination.unlink(missing_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def pack(arguments: argparse.Namespace) -> dict[str, Any]:
    prefix = arguments.index
    pair = {
        ".leann": prefix.with_name(prefix.name + ".leann"),
        ".docs": prefix.with_name(prefix.name + ".docs"),
    }
    for suffix, path in pair.items():
        if not path.is_file():
            raise PublishError(f"{path}: missing {suffix} artifact")

    stats = read_stats(arguments.leann, prefix)
    output = arguments.output
    output.mkdir(parents=True, exist_ok=True)

    files: list[tuple[str, int, str]] = []
    for path in pair.values():
        place_artifact(path, output / path.name)
        digest, size = hash_file(output / path.name)
        files.append((path.name, size, digest))
    files.sort()

    model: tuple[str, int, str] | None = None
    model_source = arguments.model_source or stats.get("model_source", "")
    model_digest = stats.get("model_sha256", "")
    model_bytes = int(stats.get("model_bytes", 0))
    if model_source and model_bytes and set(model_digest) != {"0"}:
        model = (model_source, model_bytes, model_digest)

    manifest = render_manifest(
        arguments.repo,
        arguments.repo_type,
        arguments.revision,
        prefix.name,
        files,
        model,
    )
    atomic_write(output / MANIFEST_NAME, manifest)
    atomic_write(output / ".gitattributes", gitattributes_text())
    atomic_write(
        output / "README.md",
        render_card(
            arguments.repo,
            arguments.repo_type,
            prefix.name,
            stats,
            files,
            arguments.license,
            arguments.tag or [],
            arguments.language or [],
        ),
    )
    return {
        "output": str(output),
        "repo": arguments.repo,
        "repo_type": arguments.repo_type,
        "prefix": prefix.name,
        "manifest": str(output / MANIFEST_NAME),
        "files": [
            {"name": name, "bytes": size, "sha256": digest}
            for name, size, digest in files
        ],
        "model": (
            {"source": model[0], "bytes": model[1], "sha256": model[2]}
            if model
            else None
        ),
        "pushed": False,
    }


def api_base(repo_type: str, repo_id: str) -> str:
    kind = "datasets" if repo_type == "dataset" else "models"
    return f"{HF_ENDPOINT}/api/{kind}/{repo_id}"


def lfs_base(repo_type: str, repo_id: str) -> str:
    if repo_type == "dataset":
        return f"{HF_ENDPOINT}/datasets/{repo_id}.git/info/lfs/objects/batch"
    return f"{HF_ENDPOINT}/{repo_id}.git/info/lfs/objects/batch"


def resolve_url(
    repo_type: str, repo_id: str, revision: str, path_in_repo: str
) -> str:
    """The URL a client downloads one file from. Verified shape; `-L` is
    required because this answers with a redirect to the CDN."""
    validate_repo_id(repo_id)
    validate_manifest_name(path_in_repo)
    quoted = urllib.parse.quote(path_in_repo)
    if repo_type == "dataset":
        return f"{HF_ENDPOINT}/datasets/{repo_id}/resolve/{revision}/{quoted}"
    return f"{HF_ENDPOINT}/{repo_id}/resolve/{revision}/{quoted}"


def preupload_payload(
    entries: Sequence[tuple[str, int, bytes]],
) -> dict[str, Any]:
    """Body for the preupload endpoint. All three per-file keys are required."""
    return {
        "files": [
            {
                "path": path,
                "size": size,
                "sample": base64.b64encode(sample).decode("ascii"),
            }
            for path, size, sample in entries
        ]
    }


def lfs_batch_payload(
    objects: Sequence[tuple[str, int]],
) -> dict[str, Any]:
    return {
        "operation": "upload",
        "transfers": ["basic"],
        "hash_algo": "sha_256",
        "objects": [{"oid": oid, "size": size} for oid, size in objects],
    }


def commit_ndjson(
    summary: str,
    regular: Sequence[tuple[str, bytes]],
    lfs: Sequence[tuple[str, str, int]],
) -> str:
    """Newline-delimited JSON commit body, header first."""
    lines = [json.dumps({"key": "header", "value": {"summary": summary}})]
    for path, content in regular:
        lines.append(
            json.dumps(
                {
                    "key": "file",
                    "value": {
                        "path": path,
                        "content": base64.b64encode(content).decode("ascii"),
                        "encoding": "base64",
                    },
                }
            )
        )
    for path, oid, size in lfs:
        lines.append(
            json.dumps(
                {
                    "key": "lfsFile",
                    "value": {
                        "path": path,
                        "oid": oid,
                        "algo": "sha256",
                        "size": size,
                    },
                }
            )
        )
    return "\n".join(lines) + "\n"


def request_json(
    url: str,
    token: str,
    payload: Any = None,
    content_type: str = "application/json",
    method: str = "POST",
    raw: bytes | None = None,
) -> Any:
    body = raw
    if body is None and payload is not None:
        body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, method=method)
    request.add_header("Authorization", f"Bearer {token}")
    if body is not None:
        request.add_header("Content-Type", content_type)
    try:
        with urllib.request.urlopen(request) as response:
            text = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:500]
        raise PublishError(
            f"{method} {url} failed with HTTP {error.code}: {detail}"
        ) from error
    except urllib.error.URLError as error:
        raise PublishError(f"{method} {url} failed: {error.reason}") from error
    return json.loads(text) if text.strip() else {}


def read_token(token_file: pathlib.Path | None) -> str:
    if token_file is not None:
        try:
            return token_file.read_text(encoding="utf-8").strip()
        except OSError as error:
            raise PublishError(f"{token_file}: cannot read: {error}") from error
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise PublishError(
            "no Hugging Face token: set HF_TOKEN or pass --token-file"
        )
    return token


def push(arguments: argparse.Namespace) -> dict[str, Any]:
    # Both gates are checked before anything opens a socket, so a mistaken
    # invocation cannot reach the network at all.
    if not arguments.yes:
        raise PublishError("refusing to upload without --yes")
    token = read_token(arguments.token_file)

    directory = arguments.directory
    manifest_path = directory / MANIFEST_NAME
    if not manifest_path.is_file():
        raise PublishError(f"{manifest_path}: missing; run pack first")

    # Upload exactly what pack produced. Iterating the directory instead would
    # commit whatever else happens to be sitting in it — a stray note, a
    # half-finished download, a credentials file.
    manifest_text = manifest_path.read_text(encoding="utf-8")
    allowed = {MANIFEST_NAME, "README.md", ".gitattributes"}
    recorded: dict[str, str] = {}
    for line in manifest_text.splitlines()[1:]:
        fields = line.split("\t")
        if not fields:
            continue
        if fields[0] == "file" and len(fields) == 4:
            allowed.add(fields[1])
        elif fields[0] in ("repo", "type", "revision") and len(fields) == 2:
            recorded[fields[0]] = fields[1]
    # The manifest is what a downloader reads to build its URLs, so publishing
    # it under a different repository, type, or revision than it names would
    # produce an artifact whose own instructions point somewhere else.
    for key, value in (
        ("repo", arguments.repo),
        ("type", arguments.repo_type),
        ("revision", arguments.revision),
    ):
        if key in recorded and recorded[key] != value:
            raise PublishError(
                f"{manifest_path} records {key} {recorded[key]!r} but push "
                f"was asked for {value!r}; re-run pack with matching options"
            )
    uploads = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.name in allowed
    )
    missing = allowed - {path.name for path in uploads}
    if missing:
        raise PublishError(
            f"{directory}: missing {', '.join(sorted(missing))}; run pack first"
        )
    entries: list[tuple[str, int, bytes]] = []
    for path in uploads:
        with path.open("rb") as source:
            sample = source.read(PREUPLOAD_SAMPLE_BYTES)
        entries.append((path.name, path.stat().st_size, sample))

    identity = request_json(
        f"{HF_ENDPOINT}/api/whoami-v2", token, method="GET"
    )
    if arguments.create_repo:
        owner, name = arguments.repo.split("/")
        request_json(
            f"{HF_ENDPOINT}/api/repos/create",
            token,
            {
                "name": name,
                "organization": owner,
                "type": arguments.repo_type,
                "private": bool(arguments.private),
            },
        )

    base = api_base(arguments.repo_type, arguments.repo)
    decisions = request_json(
        f"{base}/preupload/{arguments.revision}",
        token,
        preupload_payload(entries),
    )
    modes = {
        item["path"]: item.get("uploadMode", "regular")
        for item in decisions.get("files", [])
    }

    regular: list[tuple[str, bytes]] = []
    lfs: list[tuple[str, str, int]] = []
    lfs_objects: list[tuple[str, int]] = []
    oids: dict[str, str] = {}
    for path in uploads:
        if modes.get(path.name, "regular") == "lfs":
            digest, size = hash_file(path)
            oids[path.name] = digest
            lfs_objects.append((digest, size))
            lfs.append((path.name, digest, size))
        else:
            regular.append((path.name, path.read_bytes()))

    if lfs_objects:
        batch = request_json(
            lfs_base(arguments.repo_type, arguments.repo),
            token,
            lfs_batch_payload(lfs_objects),
            content_type="application/vnd.git-lfs+json",
        )
        by_oid = {item["oid"]: item for item in batch.get("objects", [])}
        for path in uploads:
            oid = oids.get(path.name)
            if oid is None:
                continue
            action = (by_oid.get(oid, {}).get("actions") or {}).get("upload")
            if action is None:
                # Already present server-side, or an action shape this script
                # does not implement. Either way, do not silently skip bytes.
                if by_oid.get(oid, {}).get("actions"):
                    raise PublishError(
                        f"{path.name}: unsupported LFS upload action; upload "
                        "this file with git-lfs or the huggingface_hub client"
                    )
                continue
            upload = urllib.request.Request(
                action["href"], data=path.read_bytes(), method="PUT"
            )
            for key, value in (action.get("header") or {}).items():
                upload.add_header(key, value)
            try:
                with urllib.request.urlopen(upload) as response:
                    response.read()
            except urllib.error.HTTPError as error:
                raise PublishError(
                    f"{path.name}: LFS upload failed with HTTP {error.code}"
                ) from error

    result = request_json(
        f"{base}/commit/{arguments.revision}",
        token,
        content_type="application/x-ndjson",
        raw=commit_ndjson(
            arguments.message, regular, lfs
        ).encode("utf-8"),
    )
    return {
        "pushed": True,
        "repo": arguments.repo,
        "repo_type": arguments.repo_type,
        "revision": arguments.revision,
        "user": identity.get("name", ""),
        "commit": result.get("commitOid", ""),
        "commit_url": result.get("commitUrl", ""),
        "files": [path.name for path in uploads],
    }


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    packer = subparsers.add_parser(
        "pack", help="write an upload-ready directory; opens no socket"
    )
    packer.add_argument("--index", type=pathlib.Path, required=True)
    packer.add_argument("--output", type=pathlib.Path, required=True)
    packer.add_argument("--repo", required=True, help="OWNER/NAME")
    packer.add_argument(
        "--repo-type", choices=("model", "dataset"), default="model"
    )
    packer.add_argument("--revision", default="main")
    packer.add_argument("--license", default="apache-2.0")
    packer.add_argument("--tag", action="append")
    packer.add_argument("--language", action="append")
    packer.add_argument(
        "--model-source",
        default="",
        help="override the model origin recorded in the index",
    )
    packer.add_argument(
        "--leann",
        default="leann",
        help="path to the leann binary used to read the index (default leann)",
    )

    pusher = subparsers.add_parser(
        "push", help="upload a packed directory to Hugging Face"
    )
    pusher.add_argument("--directory", type=pathlib.Path, required=True)
    pusher.add_argument("--repo", required=True, help="OWNER/NAME")
    pusher.add_argument(
        "--repo-type", choices=("model", "dataset"), default="model"
    )
    pusher.add_argument("--revision", default="main")
    pusher.add_argument("--token-file", type=pathlib.Path)
    pusher.add_argument("--create-repo", action="store_true")
    pusher.add_argument("--private", action="store_true")
    pusher.add_argument("--message", default="Publish leann.cpp index")
    pusher.add_argument(
        "--yes", action="store_true", help="required to upload anything"
    )

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    try:
        validate_repo_id(arguments.repo)
        if arguments.command == "pack":
            summary = pack(arguments)
        else:
            summary = push(arguments)
    except (PublishError, FileNotFoundError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
