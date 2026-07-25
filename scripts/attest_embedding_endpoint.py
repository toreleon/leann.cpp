#!/usr/bin/env python3
"""Create fail-closed evidence that a cache checkpoint used a live llama-server.

The capture phase binds an integrity-checked cache checkpoint to:

* two successful ``/health`` and ``/props`` observations;
* the exact GGUF bytes named by ``/props`` and by the checkpoint;
* an expected llama.cpp ``build_info`` value;
* the checkpoint's exact source bytes, model descriptor, endpoint, and input hash;
* on macOS (``ps``/``lsof``) and on Linux (``/proc``), the unchanged listening
  PID, process start time, command, working directory, executable bytes, model
  argument, host, and port observed before and after the file/HTTP checks; and
* a required post-run native parity binding.

The finalize phase consumes the immutable live capture and a later
``validate_embedding_parity.py`` report. It does not need the server to remain
alive, but it rejects parity evidence that is not bound to the same source,
model, fingerprint, dimension, and tier.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import re
import shlex
import socket
import stat
import struct
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from typing import Any


SCHEMA = "leann-embedding-endpoint-attestation-v1"
PARITY_SCHEMA = "leann-embedding-parity-v1"
MAX_HTTP_RESPONSE_BYTES = 4 * 1024 * 1024
TIER_COUNTS = {"100k": 100_000, "1m": 1_000_000}


class AttestationError(RuntimeError):
    """The requested evidence could not be proven."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_mapping(value: object, description: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AttestationError(f"{description} must be a JSON object")
    return value


def _require_string(value: object, description: str) -> str:
    if not isinstance(value, str) or not value:
        raise AttestationError(f"{description} must be a non-empty string")
    return value


def _require_integer(
    value: object, description: str, *, minimum: int = 0
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise AttestationError(
            f"{description} must be an integer greater than or equal to {minimum}"
        )
    return value


def _parse_iso8601(value: object, description: str) -> dt.datetime:
    raw = _require_string(value, description)
    try:
        parsed = dt.datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as error:
        raise AttestationError(f"{description} is not ISO-8601") from error
    if parsed.tzinfo is None:
        raise AttestationError(f"{description} must include a timezone")
    return parsed.astimezone(dt.timezone.utc)


def _stat_fields(metadata: os.stat_result) -> dict[str, Any]:
    birth = getattr(metadata, "st_birthtime", None)
    return {
        "device": int(metadata.st_dev),
        "inode": int(metadata.st_ino),
        "size_bytes": int(metadata.st_size),
        "mtime_ns": int(metadata.st_mtime_ns),
        "ctime_ns": int(metadata.st_ctime_ns),
        "birth_time_epoch": float(birth) if birth is not None else None,
        "birth_time": (
            dt.datetime.fromtimestamp(float(birth), dt.timezone.utc).isoformat()
            if birth is not None
            else None
        ),
    }


def hash_regular_file(path: pathlib.Path) -> dict[str, Any]:
    """Hash one stable regular file through a single descriptor."""

    resolved = path.resolve(strict=True)
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    descriptor = os.open(resolved, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise AttestationError(f"{resolved} is not a regular file")
        digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in identity_fields):
        raise AttestationError(f"{resolved} changed while it was being hashed")
    return {
        "path": str(resolved),
        **_stat_fields(after),
        "sha256": digest.hexdigest(),
    }


def read_json_artifact(path: pathlib.Path) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = path.resolve(strict=True)
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    descriptor = os.open(resolved, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise AttestationError(f"{resolved} is not a regular file")
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
            digest.update(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in identity_fields):
        raise AttestationError(f"{resolved} changed while it was being read")
    try:
        payload = json.loads(b"".join(chunks))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AttestationError(f"{resolved} is not valid UTF-8 JSON") from error
    return _require_mapping(payload, str(resolved)), {
        "path": str(resolved),
        **_stat_fields(after),
        "sha256": digest.hexdigest(),
    }


def hash_source_file(path: pathlib.Path) -> dict[str, Any]:
    """Hash source bytes and count lines with the benchmark's non-empty semantics."""

    resolved = path.resolve(strict=True)
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    descriptor = os.open(resolved, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise AttestationError(f"{resolved} is not a regular file")
        digest = hashlib.sha256()
        nonempty_lines = 0
        pending = b""
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            pieces = (pending + block).split(b"\n")
            pending = pieces.pop()
            for line in pieces:
                if line.endswith(b"\r"):
                    line = line[:-1]
                if line:
                    nonempty_lines += 1
        if pending.endswith(b"\r"):
            pending = pending[:-1]
        if pending:
            nonempty_lines += 1
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in identity_fields):
        raise AttestationError(f"{resolved} changed while it was being hashed")
    return {
        "path": str(resolved),
        **_stat_fields(after),
        "sha256": digest.hexdigest(),
        "nonempty_lines": nonempty_lines,
    }


def hash_file_prefix(path: pathlib.Path, length: int) -> str:
    if length <= 0:
        raise AttestationError("prefix length must be positive")
    resolved = path.resolve(strict=True)
    descriptor = os.open(resolved, os.O_RDONLY)
    try:
        before = os.fstat(descriptor)
        digest = hashlib.sha256()
        remaining = length
        while remaining:
            block = os.read(descriptor, min(1024 * 1024, remaining))
            if not block:
                raise AttestationError(f"{resolved} is shorter than its prefix")
            digest.update(block)
            remaining -= len(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in identity_fields):
        raise AttestationError(f"{resolved} changed while its prefix was hashed")
    return digest.hexdigest()


def require_same_file_identity(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    description: str,
) -> None:
    for key in (
        "path",
        "device",
        "inode",
        "size_bytes",
        "mtime_ns",
        "ctime_ns",
        "sha256",
    ):
        if before.get(key) != after.get(key):
            raise AttestationError(f"{description} changed during capture")


def atomic_write_json(path: pathlib.Path, payload: object) -> None:
    """Durably replace one JSON file and fsync its parent directory."""

    resolved = path.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{resolved.name}.", suffix=".tmp", dir=resolved.parent
    )
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as target:
            json.dump(payload, target, indent=2, sort_keys=True, allow_nan=False)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, resolved)
        directory_descriptor = os.open(resolved.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def endpoint_urls(value: str) -> dict[str, Any]:
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme not in ("http", "https"):
        raise AttestationError("--endpoint must use http or https")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise AttestationError("--endpoint must not contain credentials/query/fragment")
    if not parsed.hostname:
        raise AttestationError("--endpoint must contain a host")
    path = parsed.path.rstrip("/")
    if path not in ("", "/v1", "/v1/embeddings"):
        raise AttestationError(
            "--endpoint path must be empty, /v1, or /v1/embeddings"
        )
    try:
        port = parsed.port
    except ValueError as error:
        raise AttestationError("--endpoint contains an invalid port") from error
    if port is None:
        port = 443 if parsed.scheme == "https" else 80
    host_for_url = (
        f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
    )
    default_port = 443 if parsed.scheme == "https" else 80
    authority = host_for_url if port == default_port else f"{host_for_url}:{port}"
    origin = f"{parsed.scheme}://{authority}"
    return {
        "origin": origin,
        "health": origin + "/health",
        "props": origin + "/props",
        "embeddings": origin + "/v1/embeddings",
        "scheme": parsed.scheme,
        "host": parsed.hostname,
        "port": port,
    }


def query_json(url: str, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status_code = int(response.status)
            raw = response.read(MAX_HTTP_RESPONSE_BYTES + 1)
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise AttestationError(f"endpoint request failed for {url}: {error}") from error
    if status_code != 200:
        raise AttestationError(f"{url} returned HTTP {status_code}")
    if len(raw) > MAX_HTTP_RESPONSE_BYTES:
        raise AttestationError(f"{url} response exceeds the safety limit")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AttestationError(f"{url} did not return valid UTF-8 JSON") from error
    return {
        "url": url,
        "status": status_code,
        "payload_sha256": hashlib.sha256(raw).hexdigest(),
        "payload": payload,
    }


def _run_read_only(argv: Sequence[str]) -> str:
    result = subprocess.run(
        list(argv),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise AttestationError(
            f"{' '.join(argv)} failed with exit {result.returncode}: {detail}"
        )
    return result.stdout


def _lsof_records(raw: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in raw.splitlines():
        if not line:
            continue
        kind, value = line[0], line[1:]
        if kind == "p":
            if current is not None:
                records.append(current)
            current = {"pid": int(value), "names": []}
        elif current is not None and kind == "c":
            current["command_name"] = value
        elif current is not None and kind == "n":
            current["names"].append(value)
    if current is not None:
        records.append(current)
    return records


def _listener_host_and_port(name: str) -> tuple[str, int]:
    if name.startswith("["):
        closing = name.rfind("]:")
        if closing < 0:
            raise AttestationError(f"cannot parse listener address {name!r}")
        host = name[1:closing]
        port_text = name[closing + 2 :]
    else:
        host, separator, port_text = name.rpartition(":")
        if not separator:
            raise AttestationError(f"cannot parse listener address {name!r}")
    try:
        return host, int(port_text)
    except ValueError as error:
        raise AttestationError(f"cannot parse listener port {name!r}") from error


def _resolved_host_addresses(host: str) -> set[str]:
    addresses: set[str] = set()
    try:
        for item in socket.getaddrinfo(host, None, type=socket.SOCK_STREAM):
            addresses.add(item[4][0])
    except socket.gaierror as error:
        raise AttestationError(f"cannot resolve endpoint host {host!r}") from error
    return addresses


def _option_value(argv: Sequence[str], names: set[str]) -> str | None:
    found: str | None = None
    for index, token in enumerate(argv):
        if token in names:
            if index + 1 >= len(argv):
                raise AttestationError(f"process command has no value after {token}")
            found = argv[index + 1]
        else:
            for name in names:
                prefix = name + "="
                if token.startswith(prefix):
                    found = token[len(prefix) :]
    return found


def _process_start(raw: str) -> tuple[float, str]:
    match = re.fullmatch(
        r"\s*([A-Za-z]{3}\s+[A-Za-z]{3}\s+\d{1,2}\s+"
        r"\d{2}:\d{2}:\d{2}\s+\d{4})\s+(.+?)\s*",
        raw,
        flags=re.DOTALL,
    )
    if match is None:
        raise AttestationError("cannot parse ps start time and command")
    try:
        naive = dt.datetime.strptime(match.group(1), "%a %b %d %H:%M:%S %Y")
    except ValueError as error:
        raise AttestationError("cannot parse ps process start time") from error
    epoch = time.mktime(naive.timetuple())
    return epoch, match.group(2)


def darwin_process_snapshot(
    *,
    host: str,
    port: int,
    expected_pid: int | None,
    expected_artifact: pathlib.Path,
) -> dict[str, Any]:
    """Bind one Darwin listener to ps/lsof identity and immutable files."""

    if sys.platform != "darwin":
        raise AttestationError("Darwin ps/lsof process proof is unavailable")
    lsof_argv = ["lsof", "-nP"]
    if expected_pid is not None:
        lsof_argv.extend(["-a", "-p", str(expected_pid)])
    lsof_argv.extend([f"-iTCP:{port}", "-sTCP:LISTEN", "-Fpcn"])
    records = _lsof_records(_run_read_only(lsof_argv))
    if expected_pid is not None:
        records = [record for record in records if record["pid"] == expected_pid]
    if len(records) != 1:
        raise AttestationError(
            f"expected exactly one listening process on {host}:{port}, "
            f"found {len(records)}"
        )
    record = records[0]
    endpoint_addresses = _resolved_host_addresses(host)
    matching_names: list[str] = []
    for name in record["names"]:
        listener_host, listener_port = _listener_host_and_port(name)
        if listener_port == port and (
            listener_host == host or listener_host in endpoint_addresses
        ):
            matching_names.append(name)
    if len(matching_names) != 1:
        raise AttestationError(
            f"listener PID {record['pid']} is not bound exactly to {host}:{port}"
        )

    pid = int(record["pid"])
    ps_raw = _run_read_only(
        ["ps", "-ww", "-p", str(pid), "-o", "lstart=", "-o", "command="]
    )
    start_epoch, command = _process_start(ps_raw)
    try:
        argv = shlex.split(command)
    except ValueError as error:
        raise AttestationError("cannot parse process command line") from error
    if not argv:
        raise AttestationError("process command line is empty")

    cwd_raw = _run_read_only(["lsof", "-a", "-p", str(pid), "-d", "cwd", "-Fn"])
    cwd_values = [
        line[1:] for line in cwd_raw.splitlines() if line.startswith("n")
    ]
    if len(cwd_values) != 1:
        raise AttestationError(f"cannot prove working directory for PID {pid}")
    cwd = pathlib.Path(cwd_values[0]).resolve(strict=True)

    executable_path = pathlib.Path(argv[0])
    if not executable_path.is_absolute():
        executable_path = cwd / executable_path
    executable = hash_regular_file(executable_path)

    command_port = _option_value(argv, {"--port"})
    command_host = _option_value(argv, {"--host"})
    command_model = _option_value(argv, {"--model", "-m"})
    if command_port is None or command_host is None or command_model is None:
        raise AttestationError(
            "process command must explicitly declare --host, --port, and --model"
        )
    try:
        parsed_command_port = int(command_port)
    except ValueError as error:
        raise AttestationError("process --port is not an integer") from error
    if parsed_command_port != port or command_host != host:
        raise AttestationError("process command host/port differs from the listener")
    command_model_path = pathlib.Path(command_model)
    if not command_model_path.is_absolute():
        command_model_path = cwd / command_model_path
    command_model_path = command_model_path.resolve(strict=True)
    if command_model_path != expected_artifact.resolve(strict=True):
        raise AttestationError(
            "process --model path differs from the expected GGUF artifact"
        )

    stable = {
        "provider": "darwin-lsof-ps-v1",
        "pid": pid,
        "process_start_epoch": start_epoch,
        "process_start_time": dt.datetime.fromtimestamp(
            start_epoch, dt.timezone.utc
        ).isoformat(),
        "command": command,
        "command_sha256": hashlib.sha256(command.encode("utf-8")).hexdigest(),
        "working_directory": str(cwd),
        "listener": {
            "host": host,
            "port": port,
            "lsof_name": matching_names[0],
        },
        "executable": executable,
        "command_model_path": str(command_model_path),
    }
    return {**stable, "identity_sha256": canonical_hash(stable)}


def _linux_boot_time_epoch() -> float:
    with open("/proc/stat", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("btime "):
                try:
                    return float(line.split()[1])
                except (IndexError, ValueError) as error:
                    raise AttestationError("cannot parse /proc/stat btime") from error
    raise AttestationError("/proc/stat does not report btime")


def _linux_hex_address(raw: str) -> str:
    """Decode a /proc/net/tcp local_address column into a printable host."""

    if len(raw) == 8:
        packed = struct.pack("<I", int(raw, 16))
        return socket.inet_ntop(socket.AF_INET, packed)
    if len(raw) == 32:
        packed = b"".join(
            struct.pack("<I", int(raw[offset : offset + 8], 16))
            for offset in range(0, 32, 8)
        )
        return socket.inet_ntop(socket.AF_INET6, packed)
    raise AttestationError(f"cannot parse /proc/net address {raw!r}")


def _linux_listening_socket_inodes(host: str, port: int) -> dict[int, str]:
    """Map listening socket inodes bound to host:port to their printed address."""

    endpoint_addresses = _resolved_host_addresses(host)
    inodes: dict[int, str] = {}
    for table in ("/proc/net/tcp", "/proc/net/tcp6"):
        path = pathlib.Path(table)
        if not path.exists():
            continue
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            if index == 0:
                continue
            fields = line.split()
            if len(fields) < 10 or fields[3] != "0A":
                continue
            address, _, port_text = fields[1].partition(":")
            try:
                listener_port = int(port_text, 16)
            except ValueError as error:
                raise AttestationError(
                    f"cannot parse {table} listener port {fields[1]!r}"
                ) from error
            if listener_port != port:
                continue
            listener_host = _linux_hex_address(address)
            if listener_host != host and listener_host not in endpoint_addresses:
                continue
            try:
                inodes[int(fields[9])] = f"{listener_host}:{listener_port}"
            except ValueError as error:
                raise AttestationError(
                    f"cannot parse {table} socket inode {fields[9]!r}"
                ) from error
    return inodes


def _linux_pids_holding_inodes(inodes: Mapping[int, str]) -> dict[int, str]:
    """Find every PID whose open descriptors include one of ``inodes``."""

    wanted = {f"socket:[{inode}]": name for inode, name in inodes.items()}
    owners: dict[int, str] = {}
    for entry in pathlib.Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        descriptors = entry / "fd"
        try:
            handles = list(descriptors.iterdir())
        except (PermissionError, FileNotFoundError, NotADirectoryError):
            continue
        for handle in handles:
            try:
                target = os.readlink(handle)
            except OSError:
                continue
            if target in wanted:
                owners[int(entry.name)] = wanted[target]
                break
    return owners


def _linux_process_start_epoch(pid: int) -> float:
    raw = pathlib.Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    closing = raw.rfind(")")
    if closing < 0:
        raise AttestationError(f"cannot parse /proc/{pid}/stat")
    fields = raw[closing + 2 :].split()
    # Fields after comm are numbered from 3; starttime is field 22.
    if len(fields) < 20:
        raise AttestationError(f"/proc/{pid}/stat is truncated")
    try:
        ticks = float(fields[19])
    except ValueError as error:
        raise AttestationError(f"cannot parse /proc/{pid}/stat starttime") from error
    hertz = os.sysconf("SC_CLK_TCK")
    if not hertz or hertz <= 0:
        raise AttestationError("cannot resolve SC_CLK_TCK")
    return _linux_boot_time_epoch() + ticks / float(hertz)


def linux_process_snapshot(
    *,
    host: str,
    port: int,
    expected_pid: int | None,
    expected_artifact: pathlib.Path,
) -> dict[str, Any]:
    """Bind one Linux listener to /proc identity and immutable files."""

    if not sys.platform.startswith("linux"):
        raise AttestationError("Linux /proc process proof is unavailable")
    inodes = _linux_listening_socket_inodes(host, port)
    if not inodes:
        raise AttestationError(f"no listening socket is bound to {host}:{port}")
    owners = _linux_pids_holding_inodes(inodes)
    if expected_pid is not None:
        owners = {
            pid: name for pid, name in owners.items() if pid == expected_pid
        }
    if len(owners) != 1:
        raise AttestationError(
            f"expected exactly one listening process on {host}:{port}, "
            f"found {len(owners)}"
        )
    pid, listener_name = next(iter(owners.items()))

    start_epoch = _linux_process_start_epoch(pid)
    raw_cmdline = pathlib.Path(f"/proc/{pid}/cmdline").read_bytes()
    argv = [token for token in raw_cmdline.decode("utf-8").split("\0") if token]
    if not argv:
        raise AttestationError("process command line is empty")
    command = shlex.join(argv)

    try:
        cwd = pathlib.Path(os.readlink(f"/proc/{pid}/cwd")).resolve(strict=True)
    except OSError as error:
        raise AttestationError(
            f"cannot prove working directory for PID {pid}"
        ) from error
    try:
        executable_path = pathlib.Path(os.readlink(f"/proc/{pid}/exe"))
    except OSError as error:
        raise AttestationError(f"cannot prove executable for PID {pid}") from error
    executable = hash_regular_file(executable_path)

    command_port = _option_value(argv, {"--port"})
    command_host = _option_value(argv, {"--host"})
    command_model = _option_value(argv, {"--model", "-m"})
    if command_port is None or command_host is None or command_model is None:
        raise AttestationError(
            "process command must explicitly declare --host, --port, and --model"
        )
    try:
        parsed_command_port = int(command_port)
    except ValueError as error:
        raise AttestationError("process --port is not an integer") from error
    if parsed_command_port != port or command_host != host:
        raise AttestationError("process command host/port differs from the listener")
    command_model_path = pathlib.Path(command_model)
    if not command_model_path.is_absolute():
        command_model_path = cwd / command_model_path
    command_model_path = command_model_path.resolve(strict=True)
    if command_model_path != expected_artifact.resolve(strict=True):
        raise AttestationError(
            "process --model path differs from the expected GGUF artifact"
        )

    stable = {
        "provider": "linux-proc-v1",
        "pid": pid,
        "process_start_epoch": start_epoch,
        "process_start_time": dt.datetime.fromtimestamp(
            start_epoch, dt.timezone.utc
        ).isoformat(),
        "command": command,
        "command_sha256": hashlib.sha256(command.encode("utf-8")).hexdigest(),
        "working_directory": str(cwd),
        "listener": {
            "host": host,
            "port": port,
            "proc_net_name": listener_name,
        },
        "executable": executable,
        "command_model_path": str(command_model_path),
    }
    return {**stable, "identity_sha256": canonical_hash(stable)}


def platform_process_snapshot(
    *,
    host: str,
    port: int,
    expected_pid: int | None,
    expected_artifact: pathlib.Path,
) -> dict[str, Any]:
    """Dispatch to the process-proof provider for the running platform."""

    if sys.platform == "darwin":
        provider = darwin_process_snapshot
    elif sys.platform.startswith("linux"):
        provider = linux_process_snapshot
    else:
        raise AttestationError(
            f"process proof is unavailable on platform {sys.platform!r}"
        )
    return provider(
        host=host,
        port=port,
        expected_pid=expected_pid,
        expected_artifact=expected_artifact,
    )


def _checkpoint_stable(payload: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "schema",
        "source",
        "cache_path",
        "fingerprint",
        "model",
        "role",
        "bindings",
        "embedding_endpoint",
        "batch_size",
        "requested_dimension",
        "prefix_cache_sha256",
        "input_hash",
        "dimension",
        "header_size",
        "prefix_seed",
    )
    return {key: payload.get(key) for key in keys}


def _resolve_declared_path(value: object, base: pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(_require_string(value, "declared path"))
    if not path.is_absolute():
        path = base / path
    return path.resolve(strict=True)


def _checkpoint_start_artifact(
    checkpoint: Mapping[str, Any],
    checkpoint_path: pathlib.Path,
) -> dict[str, Any]:
    cache_path_value = _require_string(
        checkpoint.get("cache_path"), "checkpoint cache_path"
    )
    cache_path = pathlib.Path(cache_path_value)
    if not cache_path.is_absolute():
        cache_path = checkpoint_path.parent / cache_path
    cache_path = cache_path.resolve()
    candidates = [
        cache_path,
        cache_path.with_name(cache_path.name + ".partial"),
    ]
    existing = [candidate for candidate in candidates if candidate.is_file()]
    if len(existing) != 1:
        raise AttestationError(
            "checkpoint must have exactly one cache or partial-cache start artifact"
        )
    resolved = existing[0].resolve(strict=True)
    metadata = resolved.stat()
    if not stat.S_ISREG(metadata.st_mode):
        raise AttestationError("checkpoint start artifact is not a regular file")
    result = {"path": str(resolved), **_stat_fields(metadata)}
    if result["birth_time_epoch"] is None:
        if checkpoint.get("created_at") is not None:
            created_at = _parse_iso8601(
                checkpoint.get("created_at"), "checkpoint created_at"
            )
            result["birth_time_epoch"] = created_at.timestamp()
            result["birth_time"] = created_at.isoformat()
            result["start_time_source"] = "checkpoint.created_at"
        else:
            result["start_time_source"] = (
                "unavailable: filesystem has no birth time and the checkpoint "
                "predates created_at support"
            )
    else:
        result["start_time_source"] = (
            "filesystem birth time of the append-only cache/partial artifact"
        )
    return result


def _prefix_seed_evidence(
    *,
    checkpoint: Mapping[str, Any],
    checkpoint_path: pathlib.Path,
    source_path: pathlib.Path,
    artifact: Mapping[str, Any],
    endpoint: Mapping[str, Any],
) -> dict[str, Any] | None:
    raw_seed = checkpoint.get("prefix_seed")
    if raw_seed is None:
        return None
    seed = _require_mapping(raw_seed, "checkpoint prefix_seed")
    prefix_path = _resolve_declared_path(
        seed.get("cache"), checkpoint_path.parent
    )
    prefix_cache = hash_regular_file(prefix_path)
    expected_cache_sha = _require_string(
        seed.get("cache_sha256"), "prefix cache SHA-256"
    )
    if (
        prefix_cache["sha256"] != expected_cache_sha
        or checkpoint.get("prefix_cache_sha256") != expected_cache_sha
    ):
        raise AttestationError("prefix cache SHA-256 differs from checkpoint")
    rows = _require_integer(seed.get("rows"), "prefix cache rows", minimum=1)
    prefix_size = _require_integer(
        seed.get("source_prefix_size_bytes"),
        "prefix source byte size",
        minimum=1,
    )
    prefix_source_sha = _require_string(
        seed.get("source_prefix_sha256"), "prefix source SHA-256"
    )
    if hash_file_prefix(source_path, prefix_size) != prefix_source_sha:
        raise AttestationError("target source bytes do not match prefix-cache source")
    if seed.get("vectors_copied_bitwise") is not True:
        raise AttestationError("checkpoint does not prove bitwise prefix vector reuse")

    sidecar_path = prefix_path.with_name(prefix_path.name + ".meta.json")
    sidecar, sidecar_file = read_json_artifact(sidecar_path)
    if (
        sidecar.get("schema") != "leann-shared-embedding-cache-v1"
        or sidecar.get("role") != "corpus"
    ):
        raise AttestationError("prefix cache has invalid provenance sidecar")
    sidecar_cache = _require_mapping(sidecar.get("cache"), "prefix sidecar cache")
    if (
        sidecar_cache.get("sha256") != prefix_cache["sha256"]
        or sidecar_cache.get("count") != rows
        or sidecar_cache.get("source_size_bytes") != prefix_size
        or sidecar_cache.get("source_sha256") != prefix_source_sha
        or sidecar_cache.get("dimensions") != checkpoint.get("dimension")
        or sidecar_cache.get("fingerprint") != checkpoint.get("fingerprint")
    ):
        raise AttestationError("prefix cache sidecar binding differs")
    sidecar_source = _require_mapping(
        sidecar.get("source"), "prefix sidecar source"
    )
    prefix_source_path = _resolve_declared_path(
        sidecar_source.get("path"), sidecar_path.parent
    )
    prefix_source = hash_source_file(prefix_source_path)
    for key, expected in (
        ("size_bytes", prefix_size),
        ("sha256", prefix_source_sha),
        ("nonempty_lines", rows),
    ):
        if (
            sidecar_source.get(key) != expected
            or prefix_source.get(key) != expected
        ):
            raise AttestationError(f"prefix cache source {key} differs")
    sidecar_model = _require_mapping(sidecar.get("model"), "prefix sidecar model")
    sidecar_artifact = _require_mapping(
        sidecar_model.get("artifact"), "prefix sidecar model artifact"
    )
    checkpoint_model = _require_mapping(checkpoint.get("model"), "checkpoint model")
    if (
        sidecar_model.get("sha256") != checkpoint_model.get("sha256")
        or sidecar_artifact.get("sha256") != artifact.get("sha256")
    ):
        raise AttestationError("prefix cache model differs from checkpoint model")
    generation = _require_mapping(
        sidecar.get("generation"), "prefix sidecar generation"
    )
    if generation.get("endpoint") != endpoint["embeddings"]:
        raise AttestationError("prefix cache generation endpoint differs")

    start_epoch = prefix_cache.get("birth_time_epoch")
    start_source = "filesystem birth time of completed prefix cache"
    if start_epoch is None:
        created_at = _parse_iso8601(
            sidecar.get("created_at"), "prefix sidecar created_at"
        )
        start_epoch = created_at.timestamp()
        start_source = "prefix sidecar created_at"
    return {
        "cache": prefix_cache,
        "cache_metadata": dict(sidecar_cache),
        "sidecar": sidecar_file,
        "source": prefix_source,
        "model": dict(sidecar_model),
        "rows": rows,
        "source_prefix_size_bytes": prefix_size,
        "source_prefix_sha256": prefix_source_sha,
        "vectors_copied_bitwise": True,
        "generation_endpoint": generation["endpoint"],
        "start_time_epoch": start_epoch,
        "start_time": dt.datetime.fromtimestamp(
            float(start_epoch), dt.timezone.utc
        ).isoformat(),
        "start_time_source": start_source,
    }


def _validate_checkpoint(
    *,
    checkpoint: Mapping[str, Any],
    checkpoint_path: pathlib.Path,
    source: Mapping[str, Any],
    artifact: Mapping[str, Any],
    endpoint: Mapping[str, Any],
    tier: str,
    expected_source_count: int,
) -> dict[str, Any]:
    if checkpoint.get("schema") != "leann-cache-generation-checkpoint-v1":
        raise AttestationError("unsupported cache checkpoint schema")
    if checkpoint.get("role") != "corpus":
        raise AttestationError("endpoint attestation requires a corpus checkpoint")
    if checkpoint.get("status") not in ("running", "ready-to-publish"):
        raise AttestationError("checkpoint is not running or ready-to-publish")
    if checkpoint.get("embedding_endpoint") != endpoint["embeddings"]:
        raise AttestationError("checkpoint embedding endpoint differs")

    declared_source = _require_mapping(
        checkpoint.get("source"), "checkpoint source"
    )
    declared_source_path = _resolve_declared_path(
        declared_source.get("path"), checkpoint_path.parent
    )
    if declared_source_path != pathlib.Path(str(source["path"])):
        raise AttestationError("checkpoint source path differs from the hashed source")
    for key in ("size_bytes", "sha256", "nonempty_lines"):
        if declared_source.get(key) != source[key]:
            raise AttestationError(f"checkpoint source {key} differs")
    if source["nonempty_lines"] != expected_source_count:
        raise AttestationError(
            f"tier {tier} requires {expected_source_count} source rows, "
            f"got {source['nonempty_lines']}"
        )

    model = _require_mapping(checkpoint.get("model"), "checkpoint model")
    declared_artifact = _require_mapping(
        model.get("artifact"), "checkpoint model artifact"
    )
    declared_artifact_path = _resolve_declared_path(
        declared_artifact.get("path"), checkpoint_path.parent
    )
    if declared_artifact_path != pathlib.Path(str(artifact["path"])):
        raise AttestationError("checkpoint model artifact path differs")
    for key in ("size_bytes", "sha256"):
        if declared_artifact.get(key) != artifact[key]:
            raise AttestationError(f"checkpoint model artifact {key} differs")
    descriptor = {
        "embedding_model": model.get("embedding_model"),
        "native_fingerprint": model.get("native_fingerprint"),
        "declared_identity": model.get("declared_identity"),
        "artifact": declared_artifact,
    }
    if model.get("sha256") != canonical_hash(descriptor):
        raise AttestationError("checkpoint model descriptor SHA-256 differs")

    dimension = _require_integer(
        checkpoint.get("dimension"), "checkpoint dimension", minimum=1
    )
    requested_dimension = checkpoint.get("requested_dimension")
    if requested_dimension is not None and requested_dimension != dimension:
        raise AttestationError("checkpoint requested/observed dimensions differ")
    completed_rows = _require_integer(
        checkpoint.get("completed_rows"), "checkpoint completed_rows"
    )
    if completed_rows > expected_source_count:
        raise AttestationError("checkpoint completed_rows exceeds the tier")
    _parse_iso8601(checkpoint.get("updated_at"), "checkpoint updated_at")
    _require_string(checkpoint.get("input_hash"), "checkpoint input_hash")
    _require_string(checkpoint.get("fingerprint"), "checkpoint fingerprint")
    return {
        "tier": tier,
        "expected_source_count": expected_source_count,
        "source": dict(source),
        "model_descriptor_sha256": model["sha256"],
        "model_artifact_sha256": artifact["sha256"],
        "fingerprint": checkpoint["fingerprint"],
        "dimension": dimension,
        "checkpoint_input_hash": checkpoint["input_hash"],
    }


def _validate_props(
    *,
    health: Mapping[str, Any],
    props: Mapping[str, Any],
    expected_build_info: str,
    expected_artifact: pathlib.Path,
    server_cwd: pathlib.Path | None,
) -> dict[str, Any]:
    health_payload = _require_mapping(health.get("payload"), "health payload")
    if health_payload.get("status") != "ok":
        raise AttestationError("llama-server /health status is not ok")
    props_payload = _require_mapping(props.get("payload"), "props payload")
    build_info = _require_string(
        props_payload.get("build_info"), "props build_info"
    )
    if build_info != expected_build_info:
        raise AttestationError(
            f"props build_info {build_info!r} differs from "
            f"{expected_build_info!r}"
        )
    model_path = pathlib.Path(
        _require_string(props_payload.get("model_path"), "props model_path")
    )
    if not model_path.is_absolute():
        if server_cwd is None:
            raise AttestationError(
                "relative props model_path requires a verified server working directory"
            )
        model_path = server_cwd / model_path
    model_path = model_path.resolve(strict=True)
    if model_path != expected_artifact.resolve(strict=True):
        raise AttestationError("props model_path differs from expected artifact")
    return {
        "model_path": str(model_path),
        "build_info": build_info,
        "model_alias": props_payload.get("model_alias"),
        "model_ftype": props_payload.get("model_ftype"),
    }


ProcessProbe = Callable[..., dict[str, Any]]
JsonQuery = Callable[[str, float], dict[str, Any]]


def capture_attestation(
    *,
    endpoint_value: str,
    expected_artifact_path: pathlib.Path,
    expected_build_info: str,
    expected_pid: int | None,
    checkpoint_path: pathlib.Path,
    tier: str,
    expected_source_count: int,
    timeout: float,
    require_process_proof: bool,
    process_probe: ProcessProbe = platform_process_snapshot,
    json_query: JsonQuery = query_json,
) -> dict[str, Any]:
    """Capture a live, race-checked endpoint/checkpoint attestation."""

    endpoint = endpoint_urls(endpoint_value)
    if endpoint["scheme"] != "http":
        raise AttestationError("process-bound llama-server attestation requires HTTP")
    tool = hash_regular_file(pathlib.Path(__file__))
    artifact = hash_regular_file(expected_artifact_path)
    checkpoint_before, checkpoint_file_before = read_json_artifact(checkpoint_path)
    source_path = _resolve_declared_path(
        _require_mapping(
            checkpoint_before.get("source"), "checkpoint source"
        ).get("path"),
        checkpoint_path.resolve().parent,
    )
    source = hash_source_file(source_path)
    start_before = _checkpoint_start_artifact(
        checkpoint_before, checkpoint_path.resolve()
    )
    prefix_seed = _prefix_seed_evidence(
        checkpoint=checkpoint_before,
        checkpoint_path=checkpoint_path.resolve(),
        source_path=source_path,
        artifact=artifact,
        endpoint=endpoint,
    )

    proof_requested = require_process_proof or expected_pid is not None
    process_before: dict[str, Any] | None = None
    process_after: dict[str, Any] | None = None
    if proof_requested or sys.platform == "darwin" or sys.platform.startswith("linux"):
        try:
            process_before = process_probe(
                host=endpoint["host"],
                port=endpoint["port"],
                expected_pid=expected_pid,
                expected_artifact=expected_artifact_path,
            )
        except AttestationError:
            if proof_requested:
                raise
    if require_process_proof and process_before is None:
        raise AttestationError("required process proof is unavailable")
    if expected_pid is not None and (
        process_before is None or process_before.get("pid") != expected_pid
    ):
        raise AttestationError("listener PID differs from --expected-pid")

    health_before = json_query(str(endpoint["health"]), timeout)
    props_before = json_query(str(endpoint["props"]), timeout)
    server_cwd = (
        pathlib.Path(str(process_before["working_directory"]))
        if process_before is not None
        else None
    )
    props_identity_before = _validate_props(
        health=health_before,
        props=props_before,
        expected_build_info=expected_build_info,
        expected_artifact=expected_artifact_path,
        server_cwd=server_cwd,
    )
    parity_binding = _validate_checkpoint(
        checkpoint=checkpoint_before,
        checkpoint_path=checkpoint_path.resolve(),
        source=source,
        artifact=artifact,
        endpoint=endpoint,
        tier=tier,
        expected_source_count=expected_source_count,
    )

    health_after = json_query(str(endpoint["health"]), timeout)
    props_after = json_query(str(endpoint["props"]), timeout)
    props_identity_after = _validate_props(
        health=health_after,
        props=props_after,
        expected_build_info=expected_build_info,
        expected_artifact=expected_artifact_path,
        server_cwd=server_cwd,
    )
    if props_identity_after != props_identity_before:
        raise AttestationError("llama-server props identity changed during capture")
    artifact_after = hash_regular_file(expected_artifact_path)
    require_same_file_identity(
        artifact, artifact_after, "expected GGUF artifact"
    )
    source_after = source_path.resolve(strict=True).stat()
    for key, observed in (
        ("device", source_after.st_dev),
        ("inode", source_after.st_ino),
        ("size_bytes", source_after.st_size),
        ("mtime_ns", source_after.st_mtime_ns),
        ("ctime_ns", source_after.st_ctime_ns),
    ):
        if source.get(key) != observed:
            raise AttestationError("checkpoint source changed during capture")

    if process_before is not None:
        process_after = process_probe(
            host=endpoint["host"],
            port=endpoint["port"],
            expected_pid=expected_pid,
            expected_artifact=expected_artifact_path,
        )
        if (
            process_after.get("identity_sha256")
            != process_before.get("identity_sha256")
        ):
            raise AttestationError("listener process identity changed during capture")
    checkpoint_after, checkpoint_file_after = read_json_artifact(checkpoint_path)
    if _checkpoint_stable(checkpoint_after) != _checkpoint_stable(checkpoint_before):
        raise AttestationError("checkpoint stable identity changed during capture")
    before_rows = _require_integer(
        checkpoint_before.get("completed_rows"),
        "checkpoint completed_rows before capture",
    )
    after_rows = _require_integer(
        checkpoint_after.get("completed_rows"),
        "checkpoint completed_rows after capture",
    )
    if after_rows < before_rows:
        raise AttestationError("checkpoint progress moved backwards during capture")
    _validate_checkpoint(
        checkpoint=checkpoint_after,
        checkpoint_path=checkpoint_path.resolve(),
        source=source,
        artifact=artifact,
        endpoint=endpoint,
        tier=tier,
        expected_source_count=expected_source_count,
    )
    start_after = _checkpoint_start_artifact(
        checkpoint_after, checkpoint_path.resolve()
    )
    for key in ("path", "device", "inode", "birth_time_epoch"):
        if start_after.get(key) != start_before.get(key):
            raise AttestationError("checkpoint start artifact changed during capture")

    process_proof: dict[str, Any]
    if process_before is None or process_after is None:
        process_proof = {
            "status": "unavailable",
            "required": require_process_proof,
            "platform": sys.platform,
            "server_started_before_checkpoint": None,
        }
    else:
        process_start = float(process_before["process_start_epoch"])
        start_candidates: list[tuple[float, str]] = []
        if start_before["birth_time_epoch"] is not None:
            start_candidates.append(
                (
                    float(start_before["birth_time_epoch"]),
                    str(start_before["start_time_source"]),
                )
            )
        if prefix_seed is not None:
            start_candidates.append(
                (
                    float(prefix_seed["start_time_epoch"]),
                    str(prefix_seed["start_time_source"]),
                )
            )
        if not start_candidates:
            raise AttestationError(
                "cannot prove cache generation start: filesystem birth time and "
                "checkpoint.created_at are unavailable, and there is no "
                "identity-bound prefix seed"
            )
        checkpoint_start, checkpoint_start_source = min(
            start_candidates, key=lambda value: value[0]
        )
        if process_start >= checkpoint_start:
            raise AttestationError(
                "server process did not start before cache generation"
            )
        if artifact["mtime_ns"] / 1_000_000_000 > process_start:
            raise AttestationError(
                "GGUF artifact was modified after the server process started"
            )
        checkpoint_updated = _parse_iso8601(
            checkpoint_before.get("updated_at"), "checkpoint updated_at"
        ).timestamp()
        if process_start >= checkpoint_updated:
            raise AttestationError(
                "server process did not start before checkpoint update"
            )
        process_proof = {
            "status": "verified",
            "required": require_process_proof,
            "platform": sys.platform,
            "before": process_before,
            "after": process_after,
            "unchanged": True,
            "server_started_before_checkpoint": True,
            "checkpoint_start_source": checkpoint_start_source,
            "earliest_cache_generation_start_time": dt.datetime.fromtimestamp(
                checkpoint_start, dt.timezone.utc
            ).isoformat(),
        }

    endpoint_evidence = {
        "origin": endpoint["origin"],
        "embedding_url": endpoint["embeddings"],
        "host": endpoint["host"],
        "port": endpoint["port"],
        "health_before": health_before,
        "props_before": props_before,
        "health_after": health_after,
        "props_after": props_after,
        "props_identity": props_identity_before,
    }
    evidence = {
        "tool": tool,
        "endpoint": endpoint_evidence,
        "model_artifact": artifact,
        "model_artifact_after": artifact_after,
        "model_artifact_unchanged": True,
        "expected_build_info": expected_build_info,
        "checkpoint": {
            "before": checkpoint_file_before,
            "after": checkpoint_file_after,
            "stable_identity": _checkpoint_stable(checkpoint_before),
            "completed_rows_before": before_rows,
            "completed_rows_after": after_rows,
            "start_artifact_before": start_before,
            "start_artifact_after": start_after,
        },
        "source": source,
        "source_unchanged": True,
        "prefix_seed": prefix_seed,
        "process_proof": process_proof,
        "required_post_run_native_parity": parity_binding,
    }
    return {
        "schema": SCHEMA,
        "phase": "live-capture",
        "created_at": utc_now(),
        "attestation_id": canonical_hash(evidence),
        "evidence": evidence,
        "post_run_native_parity": {
            "status": "required-pending",
            "finalization": (
                "Run this script's finalize command with the immutable live "
                "capture and a passing validate_embedding_parity.py report."
            ),
        },
    }


def _validate_live_capture(report: Mapping[str, Any]) -> dict[str, Any]:
    if report.get("schema") != SCHEMA or report.get("phase") != "live-capture":
        raise AttestationError("input is not a live endpoint capture")
    evidence = _require_mapping(report.get("evidence"), "capture evidence")
    if report.get("attestation_id") != canonical_hash(evidence):
        raise AttestationError("live capture attestation_id differs")
    _parse_iso8601(report.get("created_at"), "capture created_at")
    process_proof = _require_mapping(
        evidence.get("process_proof"), "capture process proof"
    )
    if process_proof.get("required") is True and process_proof.get("status") != "verified":
        raise AttestationError("required live process proof was not verified")
    return evidence


def finalize_attestation(
    *,
    capture_path: pathlib.Path,
    parity_path: pathlib.Path,
) -> dict[str, Any]:
    """Bind a later native parity report to an immutable live capture."""

    capture, capture_file = read_json_artifact(capture_path)
    evidence = _validate_live_capture(capture)
    binding = _require_mapping(
        evidence.get("required_post_run_native_parity"),
        "required post-run parity binding",
    )
    parity, parity_file = read_json_artifact(parity_path)
    if parity.get("schema") != PARITY_SCHEMA or parity.get("passed") is not True:
        raise AttestationError("native parity report is not a passing v1 report")
    capture_time = _parse_iso8601(capture.get("created_at"), "capture created_at")
    parity_time = _parse_iso8601(parity.get("created_at"), "parity created_at")
    if parity_time <= capture_time:
        raise AttestationError("native parity report was not created after capture")

    dataset = _require_mapping(parity.get("dataset"), "parity dataset")
    parity_tier = dataset.get("tier")
    final_binding = dict(binding)
    prefix_cache_sha256: str | None = None
    if parity_tier != binding.get("tier"):
        prefix = _require_mapping(
            evidence.get("prefix_seed"), "live capture prefix seed"
        )
        if binding.get("tier") != "1m" or parity_tier != "100k":
            raise AttestationError("native parity tier differs from live capture")
        prefix_source = _require_mapping(
            prefix.get("source"), "live capture prefix source"
        )
        prefix_cache = _require_mapping(
            prefix.get("cache_metadata"), "live capture prefix cache metadata"
        )
        prefix_model = _require_mapping(
            prefix.get("model"), "live capture prefix model"
        )
        if (
            prefix_model.get("sha256")
            != binding.get("model_descriptor_sha256")
        ):
            raise AttestationError("prefix parity model differs from live capture")
        prefix_cache_sha256 = _require_string(
            _require_mapping(
                prefix.get("cache"), "live capture prefix cache"
            ).get("sha256"),
            "live capture prefix cache SHA-256",
        )
        final_binding = {
            "tier": parity_tier,
            "expected_source_count": prefix.get("rows"),
            "source": prefix_source,
            "model_descriptor_sha256": binding[
                "model_descriptor_sha256"
            ],
            "model_artifact_sha256": binding["model_artifact_sha256"],
            "fingerprint": prefix_cache.get("fingerprint"),
            "dimension": prefix_cache.get("dimensions"),
            "checkpoint_input_hash": binding["checkpoint_input_hash"],
            "scope": "prefix-seed",
            "parent_tier": binding["tier"],
            "prefix_cache_sha256": prefix_cache_sha256,
        }
    parity_documents = _require_mapping(
        dataset.get("documents"), "parity dataset documents"
    )
    expected_source = _require_mapping(
        final_binding.get("source"), "capture source binding"
    )
    for key in ("sha256", "size_bytes", "nonempty_lines"):
        if parity_documents.get(key) != expected_source.get(key):
            raise AttestationError(f"native parity source {key} differs")

    official = _require_mapping(
        parity.get("official_comparison_reference"),
        "parity official comparison reference",
    )
    corpus_cache = _require_mapping(
        official.get("corpus_cache"), "parity corpus cache"
    )
    expected_cache_fields = {
        "source_sha256": expected_source["sha256"],
        "source_size_bytes": expected_source["size_bytes"],
        "count": final_binding["expected_source_count"],
        "dimensions": final_binding["dimension"],
        "fingerprint": final_binding["fingerprint"],
    }
    for key, expected in expected_cache_fields.items():
        if corpus_cache.get(key) != expected:
            raise AttestationError(f"native parity corpus cache {key} differs")
    if (
        prefix_cache_sha256 is not None
        and corpus_cache.get("sha256") != prefix_cache_sha256
    ):
        raise AttestationError("native parity prefix cache SHA-256 differs")

    official_model = _require_mapping(official.get("model"), "parity model")
    official_artifact = _require_mapping(
        official_model.get("artifact"), "parity model artifact"
    )
    if (
        official_model.get("sha256")
        != final_binding["model_descriptor_sha256"]
        or official_artifact.get("sha256")
        != final_binding["model_artifact_sha256"]
    ):
        raise AttestationError("official-path parity model differs")
    native = _require_mapping(parity.get("native"), "parity native evidence")
    native_model = _require_mapping(native.get("model"), "parity native model")
    if native_model.get("sha256") != final_binding["model_artifact_sha256"]:
        raise AttestationError("native parity GGUF differs")
    native_cache = _require_mapping(native.get("cache"), "parity native cache")
    if (
        native_cache.get("dimensions") != final_binding["dimension"]
        or native_cache.get("fingerprint") != final_binding["fingerprint"]
    ):
        raise AttestationError("native parity cache model metadata differs")

    acceptance = _require_mapping(parity.get("acceptance"), "parity acceptance")
    metrics = _require_mapping(parity.get("metrics"), "parity metrics")
    threshold = acceptance.get("minimum_cosine_similarity")
    minimum = metrics.get("minimum_cosine_similarity")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or isinstance(minimum, bool)
        or not isinstance(minimum, (int, float))
        or minimum < threshold
    ):
        raise AttestationError("native parity cosine acceptance is invalid")

    final_evidence = {
        "live_capture": {
            "artifact": capture_file,
            "attestation_id": capture["attestation_id"],
            "report": capture,
        },
        "native_parity": {
            "artifact": parity_file,
            "input_hash": parity.get("input_hash"),
            "minimum_cosine_similarity": minimum,
            "required_minimum_cosine_similarity": threshold,
            "native_binary": native.get("binary"),
            "native_model": native_model,
            "corpus_cache_sha256": corpus_cache.get("sha256"),
        },
        "binding": final_binding,
    }
    return {
        "schema": SCHEMA,
        "phase": "finalized",
        "created_at": utc_now(),
        "attestation_id": canonical_hash(final_evidence),
        "evidence": final_evidence,
        "post_run_native_parity": {"status": "verified"},
    }


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture a process-bound llama-server/cache attestation or finalize "
            "it with post-run native embedding parity."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser("capture")
    capture.add_argument("--endpoint", required=True)
    capture.add_argument("--expected-artifact", required=True, type=pathlib.Path)
    capture.add_argument("--expected-build-info", required=True)
    capture.add_argument("--expected-pid", type=int)
    capture.add_argument("--checkpoint", required=True, type=pathlib.Path)
    capture.add_argument("--tier", required=True, choices=tuple(TIER_COUNTS))
    capture.add_argument("--output", required=True, type=pathlib.Path)
    capture.add_argument("--timeout", type=float, default=10.0)
    capture.add_argument("--require-process-proof", action="store_true")

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--capture", required=True, type=pathlib.Path)
    finalize.add_argument("--parity-report", required=True, type=pathlib.Path)
    finalize.add_argument("--output", required=True, type=pathlib.Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = parse_arguments(argv)
        if arguments.command == "capture":
            if arguments.expected_pid is not None and arguments.expected_pid <= 0:
                raise AttestationError("--expected-pid must be positive")
            if arguments.timeout <= 0:
                raise AttestationError("--timeout must be positive")
            report = capture_attestation(
                endpoint_value=arguments.endpoint,
                expected_artifact_path=arguments.expected_artifact,
                expected_build_info=arguments.expected_build_info,
                expected_pid=arguments.expected_pid,
                checkpoint_path=arguments.checkpoint,
                tier=arguments.tier,
                expected_source_count=TIER_COUNTS[arguments.tier],
                timeout=arguments.timeout,
                require_process_proof=arguments.require_process_proof,
            )
        else:
            report = finalize_attestation(
                capture_path=arguments.capture,
                parity_path=arguments.parity_report,
            )
        atomic_write_json(arguments.output, report)
    except (
        AttestationError,
        FileNotFoundError,
        OSError,
        ValueError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "output": str(arguments.output.resolve()),
                "phase": report["phase"],
                "attestation_id": report["attestation_id"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
