from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json_atomic(path: Path, value: Any) -> str:
    """Write, fsync, replace, then hash an immutable JSON artifact."""

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.incomplete")
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    ).encode("utf-8")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return sha256_file(path)


def verify_manifest(root: Path, manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    root = root.resolve()
    for name, record in (manifest.get("files") or {}).items():
        if not isinstance(record, dict):
            errors.append(f"{name}: invalid record")
            continue
        path = (root / str(record.get("path") or "")).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            errors.append(f"{name}: missing")
            continue
        expected = str(record.get("sha256") or "")
        actual = sha256_file(path)
        if expected != actual:
            errors.append(f"{name}: sha256 mismatch")
    return errors


def quarantine_incomplete(root: Path) -> list[str]:
    """Move only interrupted R1 temp files into a recoverable quarantine."""

    root = root.resolve()
    quarantine = root / "_incomplete"
    moved: list[str] = []
    for path in root.rglob("*.incomplete"):
        if not path.is_file() or quarantine in path.parents:
            continue
        relative = path.relative_to(root)
        destination = quarantine / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(path, destination)
        moved.append(relative.as_posix())
    return moved

