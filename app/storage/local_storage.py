from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from app.core.paths import get_project_root, get_storage_root


@dataclass(frozen=True)
class StoredFile:
    absolute_path: Path
    relative_path: str
    filename: str
    size_bytes: int
    sha256: str | None = None


class LocalStorage:
    """Project-local storage with path traversal protection."""

    def __init__(self) -> None:
        self.project_root = get_project_root().resolve()
        self.storage_root = get_storage_root().resolve()
        self.storage_root.mkdir(parents=True, exist_ok=True)

    def resolve_path(self, value: str | Path) -> Path:
        path = Path(value)
        resolved = path.resolve() if path.is_absolute() else (self.project_root / path).resolve()
        if not resolved.is_relative_to(self.storage_root):
            raise ValueError(f"Storage path escapes STORAGE_ROOT: {value}")
        return resolved

    def save_upload_file(self, *, upload_file: UploadFile, subdir: str | Path) -> StoredFile:
        directory = (self.storage_root / Path(subdir)).resolve()
        if not directory.is_relative_to(self.storage_root):
            raise ValueError(f"Upload subdirectory escapes STORAGE_ROOT: {subdir}")
        directory.mkdir(parents=True, exist_ok=True)

        source_name = Path(upload_file.filename or "upload.bin").name
        filename = f"{uuid4().hex}_{source_name}"
        output_path = directory / filename

        upload_file.file.seek(0)
        digest = hashlib.sha256()
        with output_path.open("wb") as output:
            while True:
                chunk = upload_file.file.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                output.write(chunk)

        return StoredFile(
            absolute_path=output_path,
            relative_path=output_path.relative_to(self.project_root).as_posix(),
            filename=filename,
            size_bytes=output_path.stat().st_size,
            sha256=digest.hexdigest(),
        )

    def delete_file_if_exists(self, value: str | Path) -> bool:
        path = self.resolve_path(value)
        if not path.exists():
            return False
        if not path.is_file():
            raise ValueError(f"Refusing to delete non-file storage path: {path}")
        path.unlink()
        return True
