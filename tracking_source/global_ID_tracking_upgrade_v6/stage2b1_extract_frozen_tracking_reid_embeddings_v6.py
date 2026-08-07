#!/usr/bin/env python
"""
Stage 2-B1: materialize the frozen Tracking-domain crops and cache Sports OSNet
embeddings.

This stage is deliberately representation-only. It:
  * verifies the Stage 2-B0 protocol, candidate, manifest, and cache contracts;
  * re-hashes the frozen Stage 2-A6 checkpoint before model construction;
  * reads only train images named by the frozen frame-sampling manifest;
  * applies the frozen exact-GT crop and RGB 256x128 preprocessing contract;
  * L2-normalizes each 512-d crop embedding immediately after model output;
  * writes one atomic, exact-hash, resume-safe shard per sequence.

It does NOT read pair labels, identity labels, team/role/jersey metadata, official
test data, or challenge data. It does not aggregate segment prototypes, compute
cosine similarities, evaluate candidates, select a threshold, create edges, or
perform global linking.
"""

from __future__ import annotations

import argparse
import _codecs
import csv
import hashlib
import importlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import tempfile
import types
from collections import Counter, OrderedDict
from dataclasses import dataclass
from datetime import datetime
from itertools import groupby
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


STAGE = "stage2b1_extract_frozen_tracking_reid_embeddings_v6"
SCRIPT_VERSION = "6.0.0-stage2b1"

EXPECTED_CHAMPION_ID = "frozen_deep_eiou_sports_osnet_x1_0"
EXPECTED_ARCHITECTURE = "osnet_x1_0"
EXPECTED_CHECKPOINT_SHA256 = (
    "8d5b2fd8763db34c2aad69810466adf413f0426d9f8119d322227e0e639c5fbd"
)
EXPECTED_CHECKPOINT_BYTES = 30_393_613
EXPECTED_SOURCE_COMMIT = "2fc7d270eec8290fda075b976b5e06683c788e1e"
EXPECTED_CHAMPION_CONTRACT_SHA256 = (
    "4407294bee738965b8c0aedc23e5163c22de8a9a57c7b966e28b4d7866500dc6"
)
EXPECTED_PROTOCOL_CONTRACT_SHA256 = (
    "968f76882f7a4260ab8e94c47327be959c946c4ed50070f027b0b7e318f04e5a"
)
EXPECTED_CANDIDATE_REGISTRY_SHA256 = (
    "1d3f5c2c92f0bdd09f499811d4d9edc5266f691fa9462643237b33164bbaf76e"
)
EXPECTED_EMBEDDING_CACHE_CONTRACT_SHA256 = (
    "2adc329161ec682d949e91f51de34fdf6c83accd55f786e3ee252d2b58f3e382"
)
EXPECTED_CACHE_NAMESPACE_SHA256 = (
    "f3b67f85002cf9e6f18522e15fe32713a073ef9a451e67c71b1d802121cf12b7"
)
EXPECTED_FRAME_MANIFEST_SHA256 = (
    "cd706c72d8717ecb7e3c11532df29ae349ad9e63d3f7dc7f886bc4ae3240236a"
)
EXPECTED_ROWS = 288_272
EXPECTED_SEQUENCES = 57
EXPECTED_SEGMENTS = 20_245
EXPECTED_SAMPLER_SELECTIONS = 161_960
EXPECTED_SHARED_SELECTIONS = 35_648
EXPECTED_EMBEDDING_DIMENSION = 512
EXPECTED_CROP_CONTRACT_ID = "exact_gt_bbox_floor_ceil_clamp_v1"
EXPECTED_GAMES = {"gameID=4", "gameID=6", "gameID=9"}
EXPECTED_PREPROCESSING = {
    "channel_order": "RGB",
    "image_size_hw": [256, 128],
    "normalization_mean": [0.485, 0.456, 0.406],
    "normalization_std": [0.229, 0.224, 0.225],
    "pixel_scale": "uint8 to float [0,1] via ToTensor contract",
}

FRAME_MANIFEST_COLUMNS = (
    "manifest_row_id",
    "segment_id",
    "source_game_key",
    "sequence_id",
    "frame_id",
    "image_relative_path",
    "bbox_left",
    "bbox_top",
    "bbox_width",
    "bbox_height",
    "bbox_area",
    "uniform_8_selected",
    "quality_stratified_8_selected",
    "sampler_membership",
    "crop_contract_id",
)

KNOWN_OUTPUTS = (
    "summary.json",
    "report.md",
    "resolved_config.json",
    "input_hashes.json",
    "manifest_audit.json",
    "model_load_contract.json",
    "execution_contract.json",
    "embedding_cache_index.json",
    "leakage_audit.json",
    "shard_manifest.csv",
    "failure_manifest.csv",
    "warnings.csv",
    "failure.json",
)
INCOMPLETE_MARKER = ".stage2b1_incomplete.json"


class AuditError(RuntimeError):
    """Raised when a frozen contract or execution invariant is violated."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def verify_embedded_hash(
    value: Mapping[str, Any],
    hash_field: str,
    expected: str,
    context: str,
) -> None:
    require(value.get(hash_field) == expected, f"{context}: embedded {hash_field} changed")
    payload = dict(value)
    payload.pop(hash_field, None)
    actual = canonical_sha256(payload)
    require(actual == expected, f"{context}: canonical payload hash changed: {actual}")


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"Cannot parse JSON {path}: {exc}") from exc


def dump_json_text(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, dump_json_text(value))


def atomic_write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def csv_rows(path: Path) -> Iterable[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames is not None, f"{path}: missing CSV header")
        require(
            tuple(reader.fieldnames) == FRAME_MANIFEST_COLUMNS,
            f"{path}: frame manifest schema changed\n"
            f"Expected: {FRAME_MANIFEST_COLUMNS}\nActual: {tuple(reader.fieldnames)}",
        )
        try:
            for row in reader:
                require(None not in row, f"{path}: row contains extra columns")
                yield row
        except csv.Error as exc:
            raise AuditError(f"{path}: CSV parse error: {exc}") from exc


def parse_bool(value: str, context: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise AuditError(f"{context}: expected true/false, got {value!r}")


def parse_int(value: str, context: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise AuditError(f"{context}: expected integer, got {value!r}") from exc


def parse_float(value: str, context: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AuditError(f"{context}: expected float, got {value!r}") from exc
    require(math.isfinite(result), f"{context}: non-finite float")
    return result


def safe_relative_train_image_path(relative: str, sequence_id: str) -> PurePosixPath:
    path = PurePosixPath(relative)
    require(not path.is_absolute(), f"Absolute image path is forbidden: {relative}")
    require(".." not in path.parts, f"Parent traversal is forbidden: {relative}")
    require(
        len(path.parts) == 4
        and path.parts[0] == "train"
        and path.parts[1] == sequence_id
        and path.parts[2] == "img1",
        f"Only train/<sequence>/img1 images are allowed: {relative}",
    )
    require(path.suffix.lower() == ".jpg", f"Unexpected image extension: {relative}")
    return path


def row_contract_hash_update(digest: Any, row: Mapping[str, str]) -> None:
    digest.update(canonical_json_bytes({column: row[column] for column in FRAME_MANIFEST_COLUMNS}))
    digest.update(b"\n")


@dataclass(frozen=True)
class SequenceManifestInfo:
    sequence_id: str
    source_game_key: str
    row_count: int
    sequence_manifest_sha256: str
    manifest_row_ids_sha256: str


def audit_frame_manifest(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"Missing frame sampling manifest: {path}")
    raw_hash = sha256_file(path)
    require(
        raw_hash == EXPECTED_FRAME_MANIFEST_SHA256,
        f"frame_sampling_manifest.csv SHA-256 changed: {raw_hash}",
    )

    row_count = 0
    segment_ids: set[str] = set()
    manifest_row_ids: set[str] = set()
    segment_uniform: Counter[str] = Counter()
    segment_quality: Counter[str] = Counter()
    segment_union: Counter[str] = Counter()
    sequence_infos: list[SequenceManifestInfo] = []
    completed_sequences: set[str] = set()
    current_sequence: str | None = None
    current_game: str | None = None
    current_count = 0
    current_digest = hashlib.sha256()
    current_ids_digest = hashlib.sha256()
    uniform_total = 0
    quality_total = 0
    shared_total = 0

    def finish_sequence() -> None:
        nonlocal current_sequence, current_game, current_count
        nonlocal current_digest, current_ids_digest
        if current_sequence is None or current_game is None:
            return
        sequence_infos.append(
            SequenceManifestInfo(
                sequence_id=current_sequence,
                source_game_key=current_game,
                row_count=current_count,
                sequence_manifest_sha256=current_digest.hexdigest(),
                manifest_row_ids_sha256=current_ids_digest.hexdigest(),
            )
        )
        completed_sequences.add(current_sequence)
        current_sequence = None
        current_game = None
        current_count = 0
        current_digest = hashlib.sha256()
        current_ids_digest = hashlib.sha256()

    for row_number, row in enumerate(csv_rows(path), start=2):
        context = f"{path.name}:{row_number}"
        sequence_id = row["sequence_id"]
        source_game = row["source_game_key"]
        require(sequence_id.startswith("SNMOT-"), f"{context}: unexpected sequence_id")
        require(source_game in EXPECTED_GAMES, f"{context}: unexpected source game")
        safe_relative_train_image_path(row["image_relative_path"], sequence_id)

        if current_sequence is None:
            require(sequence_id not in completed_sequences, f"{context}: repeated sequence group")
            current_sequence = sequence_id
            current_game = source_game
        elif sequence_id != current_sequence:
            finish_sequence()
            require(sequence_id not in completed_sequences, f"{context}: non-contiguous sequence")
            current_sequence = sequence_id
            current_game = source_game
        require(source_game == current_game, f"{context}: sequence spans source games")

        segment_id = row["segment_id"]
        frame_id = parse_int(row["frame_id"], context)
        left = parse_float(row["bbox_left"], context)
        top = parse_float(row["bbox_top"], context)
        width = parse_float(row["bbox_width"], context)
        height = parse_float(row["bbox_height"], context)
        area = parse_float(row["bbox_area"], context)
        require(segment_id.startswith("seg_"), f"{context}: invalid segment_id")
        require(frame_id > 0, f"{context}: non-positive frame_id")
        require(width > 0 and height > 0 and area > 0, f"{context}: non-positive bbox")
        require(math.isclose(width * height, area, abs_tol=1e-9), f"{context}: area mismatch")
        require(math.isfinite(left) and math.isfinite(top), f"{context}: non-finite bbox origin")
        require(
            row["crop_contract_id"] == EXPECTED_CROP_CONTRACT_ID,
            f"{context}: crop contract changed",
        )

        uniform = parse_bool(row["uniform_8_selected"], context)
        quality = parse_bool(row["quality_stratified_8_selected"], context)
        require(uniform or quality, f"{context}: row belongs to no sampler")
        expected_memberships = []
        if uniform:
            expected_memberships.append("uniform_8")
        if quality:
            expected_memberships.append("quality_stratified_8")
        require(
            row["sampler_membership"] == "|".join(expected_memberships),
            f"{context}: sampler_membership mismatch",
        )
        expected_row_key = f"{segment_id}|{frame_id}|{EXPECTED_CROP_CONTRACT_ID}"
        expected_row_id = (
            "frm_" + hashlib.sha256(expected_row_key.encode("utf-8")).hexdigest()[:20]
        )
        require(row["manifest_row_id"] == expected_row_id, f"{context}: manifest_row_id mismatch")
        require(expected_row_id not in manifest_row_ids, f"{context}: duplicate manifest_row_id")
        manifest_row_ids.add(expected_row_id)

        segment_ids.add(segment_id)
        segment_union[segment_id] += 1
        if uniform:
            segment_uniform[segment_id] += 1
            uniform_total += 1
        if quality:
            segment_quality[segment_id] += 1
            quality_total += 1
        if uniform and quality:
            shared_total += 1

        row_contract_hash_update(current_digest, row)
        current_ids_digest.update(expected_row_id.encode("utf-8"))
        current_ids_digest.update(b"\n")
        current_count += 1
        row_count += 1

    finish_sequence()

    require(row_count == EXPECTED_ROWS, f"Frame manifest row count changed: {row_count}")
    require(len(sequence_infos) == EXPECTED_SEQUENCES, "Sequence count changed")
    require(len(segment_ids) == EXPECTED_SEGMENTS, "Segment count changed")
    require(uniform_total == EXPECTED_SAMPLER_SELECTIONS, "uniform_8 selection count changed")
    require(quality_total == EXPECTED_SAMPLER_SELECTIONS, "quality sampler count changed")
    require(shared_total == EXPECTED_SHARED_SELECTIONS, "shared sampler count changed")
    require(
        all(segment_uniform[segment] == 8 for segment in segment_ids),
        "At least one segment does not have exactly eight uniform selections",
    )
    require(
        all(segment_quality[segment] == 8 for segment in segment_ids),
        "At least one segment does not have exactly eight quality selections",
    )
    require(
        all(8 <= segment_union[segment] <= 16 for segment in segment_ids),
        "At least one segment has an invalid union size",
    )
    return {
        "status": "PASS",
        "frame_sampling_manifest_sha256": raw_hash,
        "rows": row_count,
        "sequences": len(sequence_infos),
        "segments": len(segment_ids),
        "uniform_8_selections": uniform_total,
        "quality_stratified_8_selections": quality_total,
        "shared_selections": shared_total,
        "sequence_infos": [
            {
                "sequence_id": info.sequence_id,
                "source_game_key": info.source_game_key,
                "row_count": info.row_count,
                "sequence_manifest_sha256": info.sequence_manifest_sha256,
                "manifest_row_ids_sha256": info.manifest_row_ids_sha256,
            }
            for info in sequence_infos
        ],
    }


def verify_stage2b0(stage2b0_dir: Path) -> dict[str, Any]:
    paths = {
        "summary": stage2b0_dir / "summary.json",
        "protocol_contract": stage2b0_dir / "protocol_contract.json",
        "candidate_registry": stage2b0_dir / "candidate_registry.json",
        "embedding_cache_contract": stage2b0_dir / "embedding_cache_contract.json",
        "frame_sampling_manifest": stage2b0_dir / "frame_sampling_manifest.csv",
    }
    for name, path in paths.items():
        require(path.is_file(), f"Missing Stage 2-B0 {name}: {path}")

    summary = load_json(paths["summary"])
    protocol = load_json(paths["protocol_contract"])
    registry = load_json(paths["candidate_registry"])
    cache_contract = load_json(paths["embedding_cache_contract"])

    require(summary["status"] == "PASS", "Stage 2-B0 status is not PASS")
    require(summary["leakage_audit_status"] == "PASS", "Stage 2-B0 leakage audit failed")
    require(summary["official_test_status"] == "LOCKED_NOT_READ", "Official test was read")
    require(summary["challenge_status"] == "BLIND_NOT_READ", "Challenge was read")
    for field in (
        "training_performed",
        "image_inference_performed",
        "embedding_extraction_performed",
        "cosine_similarity_computed",
        "threshold_selection_performed",
        "threshold_sweep_performed",
        "edge_creation_performed",
        "global_linking_performed",
    ):
        require(summary[field] is False, f"Stage 2-B0 unexpectedly reports {field}=true")
    require(
        summary["protocol_contract_sha256"] == EXPECTED_PROTOCOL_CONTRACT_SHA256,
        "Stage 2-B0 protocol hash changed",
    )
    require(
        summary["candidate_registry_sha256"] == EXPECTED_CANDIDATE_REGISTRY_SHA256,
        "Stage 2-B0 candidate registry hash changed",
    )
    require(
        summary["embedding_cache_contract_sha256"]
        == EXPECTED_EMBEDDING_CACHE_CONTRACT_SHA256,
        "Stage 2-B0 cache contract hash changed",
    )

    verify_embedded_hash(
        protocol,
        "protocol_contract_sha256",
        EXPECTED_PROTOCOL_CONTRACT_SHA256,
        "protocol_contract.json",
    )
    verify_embedded_hash(
        registry,
        "candidate_registry_sha256",
        EXPECTED_CANDIDATE_REGISTRY_SHA256,
        "candidate_registry.json",
    )
    verify_embedded_hash(
        cache_contract,
        "embedding_cache_contract_sha256",
        EXPECTED_EMBEDDING_CACHE_CONTRACT_SHA256,
        "embedding_cache_contract.json",
    )
    require(protocol["status"] == "FROZEN", "Protocol is not frozen")
    require(registry["status"] == "FROZEN", "Candidate registry is not frozen")
    require(registry["candidate_count"] == 6, "Candidate registry count changed")
    require(
        cache_contract["status"] == "FROZEN_FOR_STAGE2B1",
        "Cache contract is not frozen for Stage 2-B1",
    )
    require(
        cache_contract["cache_namespace_sha256"] == EXPECTED_CACHE_NAMESPACE_SHA256,
        "Cache namespace changed",
    )
    require(
        canonical_sha256(cache_contract["namespace_payload"])
        == EXPECTED_CACHE_NAMESPACE_SHA256,
        "Cache namespace payload changed",
    )
    require(
        cache_contract["namespace_payload"]["frame_sampling_manifest_sha256"]
        == EXPECTED_FRAME_MANIFEST_SHA256,
        "Cache namespace manifest hash changed",
    )
    require(
        protocol["crop_contract"]["preprocessing"] == EXPECTED_PREPROCESSING,
        "Preprocessing contract changed",
    )
    require(
        protocol["crop_contract"]["crop_contract_id"] == EXPECTED_CROP_CONTRACT_ID,
        "Crop contract changed",
    )
    return {
        "paths": paths,
        "summary": summary,
        "protocol": protocol,
        "registry": registry,
        "cache_contract": cache_contract,
    }


def verify_champion(stage2a6_dir: Path) -> tuple[Path, dict[str, Any]]:
    path = stage2a6_dir / "champion_contract.json"
    require(path.is_file(), f"Missing Stage 2-A6 champion contract: {path}")
    actual_hash = sha256_file(path)
    require(
        actual_hash == EXPECTED_CHAMPION_CONTRACT_SHA256,
        f"champion_contract.json SHA-256 changed: {actual_hash}",
    )
    champion = load_json(path)
    require(champion["status"] == "FROZEN", "Champion contract is not frozen")
    require(champion["candidate_id"] == EXPECTED_CHAMPION_ID, "Champion ID changed")
    require(champion["architecture"] == EXPECTED_ARCHITECTURE, "Architecture changed")
    require(
        champion["checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA256,
        "Champion checkpoint hash changed",
    )
    require(champion["checkpoint_bytes"] == EXPECTED_CHECKPOINT_BYTES, "Checkpoint size changed")
    require(champion["source_commit"] == EXPECTED_SOURCE_COMMIT, "Source commit changed")
    require(
        champion["embedding_dimension"] == EXPECTED_EMBEDDING_DIMENSION,
        "Embedding dimension changed",
    )
    require(champion["preprocessing"] == EXPECTED_PREPROCESSING, "Champion preprocessing changed")
    require(champion["official_test_status"] == "LOCKED_NOT_READ", "Champion read official test")
    require(champion["challenge_status"] == "BLIND_NOT_READ", "Champion read challenge")
    require(champion["threshold_status"] == "NOT_SELECTED", "Champion selected a threshold")
    return path, champion


def git_output(root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise AuditError(f"Cannot inspect frozen Deep-EIoU git source at {root}: {exc}") from exc
    return completed.stdout.strip()


def verify_deep_eiou_source(root: Path) -> dict[str, Any]:
    require(root.is_dir(), f"Missing Deep-EIoU source root: {root}")
    commit = git_output(root, "rev-parse", "HEAD")
    require(commit == EXPECTED_SOURCE_COMMIT, f"Deep-EIoU commit changed: {commit}")
    tracked_status = git_output(root, "status", "--porcelain", "--untracked-files=no")
    require(not tracked_status, "Deep-EIoU tracked source tree is dirty")
    return {
        "root": str(root.resolve()),
        "commit": commit,
        "tracked_tree_clean": True,
    }


def import_vendored_osnet(deep_eiou_root: Path) -> tuple[Any, Path]:
    candidates = (
        deep_eiou_root / "Deep-EIoU" / "reid",
        deep_eiou_root / "reid",
    )
    reid_root = next((path for path in candidates if (path / "torchreid").is_dir()), None)
    require(reid_root is not None, f"Cannot find vendored Deep-EIoU torchreid under {deep_eiou_root}")
    package_root = reid_root / "torchreid"
    require(
        (package_root / "models" / "osnet.py").is_file(),
        f"Missing vendored OSNet source: {package_root / 'models' / 'osnet.py'}",
    )

    for module_name in list(sys.modules):
        if module_name == "torchreid" or module_name.startswith("torchreid."):
            del sys.modules[module_name]
    package = types.ModuleType("torchreid")
    package.__path__ = [str(package_root)]
    package.__package__ = "torchreid"
    package.__file__ = str(package_root / "__init__.py")
    sys.modules["torchreid"] = package
    models = importlib.import_module("torchreid.models")
    require(hasattr(models, EXPECTED_ARCHITECTURE), "Vendored torchreid lacks osnet_x1_0")
    return models, reid_root


def safe_load_checkpoint(torch: Any, checkpoint_path: Path) -> tuple[Any, dict[str, Any]]:
    serialization = torch.serialization
    get_unsafe = getattr(serialization, "get_unsafe_globals_in_checkpoint", None)
    unsafe_names: list[str] = []
    inspection_status = "API_UNAVAILABLE"
    if get_unsafe is not None:
        try:
            unsafe_names = sorted(set(get_unsafe(str(checkpoint_path))))
            inspection_status = "INSPECTED"
        except RuntimeError as exc:
            message = str(exc)
            require(
                "Expected hasRecord" in message,
                "Unsafe-global inspection failed for a reason other than the expected "
                f"trusted legacy container: {message}",
            )
            inspection_status = "TRUSTED_LEGACY_CONTAINER_NOT_ZIP_INSPECTABLE"

    import numpy as np

    safe_name_to_object: dict[str, Any] = {
        "numpy.core.multiarray.scalar": np.core.multiarray.scalar,
        "numpy._core.multiarray.scalar": np.core.multiarray.scalar,
        "numpy.dtype": np.dtype,
        "_codecs.encode": _codecs.encode,
    }
    float32_dtype_class = type(np.dtype(np.float32))
    safe_name_to_object["numpy.dtypes.Float32DType"] = float32_dtype_class
    allowed_names = set(safe_name_to_object)
    unknown_names = sorted(set(unsafe_names) - allowed_names)
    require(
        not unknown_names,
        "Checkpoint requests non-allowlisted globals; unsafe pickle loading is forbidden: "
        + ", ".join(unknown_names),
    )
    safe_objects: list[Any] = []
    seen_object_ids: set[int] = set()
    for value in safe_name_to_object.values():
        if id(value) not in seen_object_ids:
            safe_objects.append(value)
            seen_object_ids.add(id(value))

    safe_globals = getattr(serialization, "safe_globals", None)
    try:
        if safe_globals is not None:
            with safe_globals(safe_objects):
                payload = torch.load(
                    str(checkpoint_path),
                    map_location="cpu",
                    weights_only=True,
                )
        else:
            require(
                not unsafe_names,
                "torch.serialization.safe_globals is unavailable for this legacy checkpoint",
            )
            payload = torch.load(
                str(checkpoint_path),
                map_location="cpu",
                weights_only=True,
            )
    except Exception as exc:
        raise AuditError(
            "Safe weights-only checkpoint load failed; weights_only=False is forbidden: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    return payload, {
        "checkpoint_loaded_with_weights_only": True,
        "unsafe_global_inspection_status": inspection_status,
        "unsafe_globals_reported_before_allowlist": unsafe_names,
        "safe_legacy_globals_allowlisted": sorted(allowed_names),
        "unsafe_pickle_loading_used": False,
    }


def checkpoint_state_dict(payload: Any) -> Mapping[str, Any]:
    if isinstance(payload, Mapping):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            candidate = payload.get(key)
            if isinstance(candidate, Mapping) and candidate:
                if any(hasattr(value, "shape") for value in candidate.values()):
                    return candidate
        if payload and all(hasattr(value, "shape") for value in payload.values()):
            return payload
    raise AuditError("Checkpoint does not contain a recognizable tensor state_dict")


def strip_one_model_prefix(key: str) -> str:
    for prefix in ("module.", "model."):
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def build_model(
    torch: Any,
    models: Any,
    checkpoint_path: Path,
    device: Any,
) -> tuple[Any, dict[str, Any]]:
    constructor = getattr(models, EXPECTED_ARCHITECTURE)
    model = constructor(
        num_classes=1,
        loss="softmax",
        pretrained=False,
        use_gpu=device.type == "cuda",
    )
    payload, safe_loader_contract = safe_load_checkpoint(torch, checkpoint_path)
    source_state = checkpoint_state_dict(payload)
    target_state = model.state_dict()
    matched: dict[str, Any] = {}
    ignored_unexpected: list[str] = []
    ignored_shape_mismatch: list[dict[str, Any]] = []
    duplicate_normalized_keys: list[str] = []

    for original_key, tensor in source_state.items():
        normalized_key = strip_one_model_prefix(str(original_key))
        if normalized_key in matched:
            duplicate_normalized_keys.append(normalized_key)
            continue
        if normalized_key not in target_state:
            ignored_unexpected.append(normalized_key)
            continue
        if tuple(tensor.shape) != tuple(target_state[normalized_key].shape):
            ignored_shape_mismatch.append(
                {
                    "key": normalized_key,
                    "checkpoint_shape": list(tensor.shape),
                    "model_shape": list(target_state[normalized_key].shape),
                }
            )
            continue
        matched[normalized_key] = tensor

    require(not duplicate_normalized_keys, "Checkpoint has duplicate normalized state keys")
    missing_non_classifier = sorted(
        key
        for key in target_state
        if key not in matched and not key.startswith("classifier.")
    )
    require(
        not missing_non_classifier,
        "Checkpoint is missing non-classifier backbone tensors: "
        + ", ".join(missing_non_classifier[:20]),
    )
    merged_state = dict(target_state)
    merged_state.update(matched)
    model.load_state_dict(merged_state, strict=True)
    model.eval()
    model.to(device)
    del payload
    return model, {
        "status": "PASS",
        "architecture": EXPECTED_ARCHITECTURE,
        "model_constructor": (
            'osnet_x1_0(num_classes=1, loss="softmax", pretrained=False, '
            f"use_gpu={device.type == 'cuda'})"
        ),
        "matched_tensor_count": len(matched),
        "target_tensor_count": len(target_state),
        "ignored_unexpected_tensor_count": len(ignored_unexpected),
        "ignored_unexpected_tensors": sorted(ignored_unexpected),
        "ignored_shape_mismatch_count": len(ignored_shape_mismatch),
        "ignored_shape_mismatches": ignored_shape_mismatch,
        "missing_non_classifier_tensors": missing_non_classifier,
        "classifier_mismatch_only_allowed": True,
        **safe_loader_contract,
    }


def resolve_device(torch: Any, requested: str) -> Any:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda":
        require(torch.cuda.is_available(), "CUDA was requested but is unavailable")
        return torch.device("cuda")
    require(requested == "cpu", f"Unsupported device {requested!r}")
    return torch.device("cpu")


def configure_determinism(torch: Any) -> None:
    random.seed(0)
    try:
        import numpy as np

        np.random.seed(0)
    except ImportError:
        pass
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def package_version(module: Any) -> str:
    return str(getattr(module, "__version__", "UNKNOWN"))


def build_execution_contract(
    torch: Any,
    torchvision: Any,
    pillow_version: str,
    numpy_version: str,
    device: Any,
    batch_size: int,
) -> dict[str, Any]:
    device_name = (
        torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else platform.processor() or "CPU"
    )
    payload = {
        "contract_version": 1,
        "stage": STAGE,
        "cache_namespace_sha256": EXPECTED_CACHE_NAMESPACE_SHA256,
        "device_type": device.type,
        "device_name": device_name,
        "batch_size": batch_size,
        "inference_precision": "float32_no_amp",
        "model_eval_mode": True,
        "torch_inference_mode": True,
        "determinism": {
            "seed": 0,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "shuffle": False,
        },
        "preprocessing_implementation": (
            "torchvision.transforms.Compose([Resize((256,128)), ToTensor(), "
            "Normalize(ImageNet mean/std)])"
        ),
        "versions": {
            "python": platform.python_version(),
            "torch": package_version(torch),
            "torchvision": package_version(torchvision),
            "pillow": pillow_version,
            "numpy": numpy_version,
        },
    }
    result = dict(payload)
    result["execution_contract_sha256"] = canonical_sha256(payload)
    return result


@dataclass(frozen=True)
class CropRecord:
    original_index: int
    manifest_row_id: str
    segment_id: str
    frame_id: int
    image_relative_path: str
    bbox_left: float
    bbox_top: float
    bbox_width: float
    bbox_height: float
    uniform_selected: bool
    quality_selected: bool


def manifest_sequence_groups(path: Path) -> Iterable[tuple[str, list[CropRecord], dict[str, Any]]]:
    for sequence_id, raw_group in groupby(csv_rows(path), key=lambda row: row["sequence_id"]):
        rows = list(raw_group)
        digest = hashlib.sha256()
        ids_digest = hashlib.sha256()
        records: list[CropRecord] = []
        source_games = {row["source_game_key"] for row in rows}
        require(len(source_games) == 1, f"{sequence_id}: multiple source games")
        for original_index, row in enumerate(rows):
            row_contract_hash_update(digest, row)
            ids_digest.update(row["manifest_row_id"].encode("utf-8"))
            ids_digest.update(b"\n")
            records.append(
                CropRecord(
                    original_index=original_index,
                    manifest_row_id=row["manifest_row_id"],
                    segment_id=row["segment_id"],
                    frame_id=int(row["frame_id"]),
                    image_relative_path=row["image_relative_path"],
                    bbox_left=float(row["bbox_left"]),
                    bbox_top=float(row["bbox_top"]),
                    bbox_width=float(row["bbox_width"]),
                    bbox_height=float(row["bbox_height"]),
                    uniform_selected=row["uniform_8_selected"] == "true",
                    quality_selected=row["quality_stratified_8_selected"] == "true",
                )
            )
        yield sequence_id, records, {
            "sequence_id": sequence_id,
            "source_game_key": next(iter(source_games)),
            "row_count": len(records),
            "sequence_manifest_sha256": digest.hexdigest(),
            "manifest_row_ids_sha256": ids_digest.hexdigest(),
        }


class CropDataset:
    """Pixel-only dataset. It never receives identity, pair, team, role, or jersey data."""

    def __init__(
        self,
        tracking_root: Path,
        records: Sequence[CropRecord],
        transform: Any,
        image_cache_size: int = 8,
    ) -> None:
        self.tracking_root = tracking_root
        self.records = records
        self.transform = transform
        self.image_cache_size = image_cache_size
        self.image_cache: OrderedDict[str, Any] = OrderedDict()
        self.load_order = sorted(
            range(len(records)),
            key=lambda index: (
                records[index].image_relative_path,
                records[index].segment_id,
                records[index].manifest_row_id,
            ),
        )

    def __len__(self) -> int:
        return len(self.load_order)

    def _load_rgb(self, relative_path: str) -> Any:
        if relative_path in self.image_cache:
            image = self.image_cache.pop(relative_path)
            self.image_cache[relative_path] = image
            return image
        from PIL import Image

        absolute_path = self.tracking_root.joinpath(*PurePosixPath(relative_path).parts)
        if not absolute_path.is_file():
            raise FileNotFoundError(str(absolute_path))
        with Image.open(absolute_path) as source:
            image = source.convert("RGB")
            image.load()
        self.image_cache[relative_path] = image
        while len(self.image_cache) > self.image_cache_size:
            self.image_cache.popitem(last=False)
        return image

    def __getitem__(self, load_index: int) -> dict[str, Any]:
        record = self.records[self.load_order[load_index]]
        try:
            image = self._load_rgb(record.image_relative_path)
        except FileNotFoundError as exc:
            return {
                "error": {
                    "sequence_id": PurePosixPath(record.image_relative_path).parts[1],
                    "manifest_row_id": record.manifest_row_id,
                    "segment_id": record.segment_id,
                    "frame_id": record.frame_id,
                    "image_relative_path": record.image_relative_path,
                    "reason_code": "IMAGE_NOT_FOUND",
                    "message": str(exc),
                }
            }
        except Exception as exc:
            return {
                "error": {
                    "sequence_id": PurePosixPath(record.image_relative_path).parts[1],
                    "manifest_row_id": record.manifest_row_id,
                    "segment_id": record.segment_id,
                    "frame_id": record.frame_id,
                    "image_relative_path": record.image_relative_path,
                    "reason_code": "IMAGE_OPEN_ERROR",
                    "message": f"{type(exc).__name__}: {exc}",
                }
            }

        image_width, image_height = image.size
        x1 = max(0, min(image_width, math.floor(record.bbox_left)))
        y1 = max(0, min(image_height, math.floor(record.bbox_top)))
        x2 = max(0, min(image_width, math.ceil(record.bbox_left + record.bbox_width)))
        y2 = max(0, min(image_height, math.ceil(record.bbox_top + record.bbox_height)))
        if x2 <= x1 or y2 <= y1:
            return {
                "error": {
                    "sequence_id": PurePosixPath(record.image_relative_path).parts[1],
                    "manifest_row_id": record.manifest_row_id,
                    "segment_id": record.segment_id,
                    "frame_id": record.frame_id,
                    "image_relative_path": record.image_relative_path,
                    "reason_code": "INVALID_CROP_AFTER_CLAMP",
                    "message": (
                        f"image={image_width}x{image_height}, "
                        f"crop=({x1},{y1},{x2},{y2})"
                    ),
                }
            }
        try:
            tensor = self.transform(image.crop((x1, y1, x2, y2)))
        except Exception as exc:
            return {
                "error": {
                    "sequence_id": PurePosixPath(record.image_relative_path).parts[1],
                    "manifest_row_id": record.manifest_row_id,
                    "segment_id": record.segment_id,
                    "frame_id": record.frame_id,
                    "image_relative_path": record.image_relative_path,
                    "reason_code": "TRANSFORM_ERROR",
                    "message": f"{type(exc).__name__}: {exc}",
                }
            }
        return {
            "error": None,
            "tensor": tensor,
            "original_index": record.original_index,
            "crop_width": x2 - x1,
            "crop_height": y2 - y1,
        }


def collate_crop_items(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    errors = [item["error"] for item in items if item.get("error") is not None]
    valid = [item for item in items if item.get("error") is None]
    return {
        "errors": errors,
        "tensors": [item["tensor"] for item in valid],
        "original_indices": [item["original_index"] for item in valid],
        "crop_widths": [item["crop_width"] for item in valid],
        "crop_heights": [item["crop_height"] for item in valid],
    }


def atomic_save_npz(path: Path, arrays: Mapping[str, Any], numpy: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as stream:
            numpy.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def shard_paths(output_dir: Path, sequence_id: str) -> tuple[Path, Path]:
    shard_dir = output_dir / "shards"
    return (
        shard_dir / f"{sequence_id}.embeddings.npz",
        shard_dir / f"{sequence_id}.meta.json",
    )


def cached_shard_status(
    output_dir: Path,
    sequence_id: str,
    records: Sequence[CropRecord],
    sequence_contract: Mapping[str, Any],
    execution_contract_sha256: str,
    numpy: Any,
    resume: bool,
) -> dict[str, Any] | None:
    shard_path, meta_path = shard_paths(output_dir, sequence_id)
    shard_exists = shard_path.exists()
    meta_exists = meta_path.exists()
    if not shard_exists and not meta_exists:
        return None
    require(
        resume,
        f"Existing shard requires --resume or --overwrite: {shard_path}",
    )
    require(
        shard_exists and meta_exists,
        f"{sequence_id}: partial cache found; exact resume forbids silent reuse/rebuild",
    )
    meta = load_json(meta_path)
    for field, expected in (
        ("status", "COMPLETE"),
        ("sequence_id", sequence_id),
        ("source_game_key", sequence_contract["source_game_key"]),
        ("cache_namespace_sha256", EXPECTED_CACHE_NAMESPACE_SHA256),
        ("execution_contract_sha256", execution_contract_sha256),
        ("sequence_manifest_sha256", sequence_contract["sequence_manifest_sha256"]),
        ("manifest_row_ids_sha256", sequence_contract["manifest_row_ids_sha256"]),
        ("row_count", len(records)),
        ("embedding_dimension", EXPECTED_EMBEDDING_DIMENSION),
        ("embedding_dtype", "float32"),
        ("per_crop_l2_normalized", True),
        ("forbidden_metadata_cached", False),
    ):
        require(meta.get(field) == expected, f"{sequence_id}: cached meta mismatch for {field}")
    require(
        meta.get("embedding_shard_bytes") == shard_path.stat().st_size,
        f"{sequence_id}: cached shard byte size changed",
    )
    actual_shard_hash = sha256_file(shard_path)
    require(
        actual_shard_hash == meta["embedding_shard_sha256"],
        f"{sequence_id}: embedding shard SHA-256 mismatch",
    )
    try:
        with numpy.load(shard_path, allow_pickle=False) as arrays:
            expected_keys = {
                "manifest_row_id",
                "segment_id",
                "frame_id",
                "uniform_8_selected",
                "quality_stratified_8_selected",
                "embedding",
            }
            require(set(arrays.files) == expected_keys, f"{sequence_id}: cached array keys changed")
            actual_ids = arrays["manifest_row_id"].tolist()
            actual_segments = arrays["segment_id"].tolist()
            actual_frames = arrays["frame_id"].tolist()
            actual_uniform = arrays["uniform_8_selected"].astype(bool).tolist()
            actual_quality = arrays["quality_stratified_8_selected"].astype(bool).tolist()
            expected_ids = [record.manifest_row_id for record in records]
            expected_segments = [record.segment_id for record in records]
            expected_frames = [record.frame_id for record in records]
            expected_uniform = [record.uniform_selected for record in records]
            expected_quality = [record.quality_selected for record in records]
            require(actual_ids == expected_ids, f"{sequence_id}: cached manifest row IDs changed")
            require(actual_segments == expected_segments, f"{sequence_id}: cached segment IDs changed")
            require(actual_frames == expected_frames, f"{sequence_id}: cached frame IDs changed")
            require(actual_uniform == expected_uniform, f"{sequence_id}: cached uniform flags changed")
            require(actual_quality == expected_quality, f"{sequence_id}: cached quality flags changed")
            embeddings = arrays["embedding"]
            require(
                embeddings.shape == (len(records), EXPECTED_EMBEDDING_DIMENSION),
                f"{sequence_id}: cached embedding shape changed",
            )
            require(embeddings.dtype == numpy.float32, f"{sequence_id}: cached dtype changed")
            require(numpy.isfinite(embeddings).all(), f"{sequence_id}: non-finite cached embedding")
            norms = numpy.linalg.norm(embeddings, axis=1)
            require(
                numpy.allclose(norms, 1.0, rtol=1e-5, atol=1e-5),
                f"{sequence_id}: cached embeddings are not L2-normalized",
            )
            norm_min = float(norms.min())
            norm_max = float(norms.max())
    except AuditError:
        raise
    except Exception as exc:
        raise AuditError(f"{sequence_id}: cannot validate cached shard: {exc}") from exc
    return {
        **meta,
        "embedding_shard_path": str(shard_path.resolve()),
        "embedding_meta_path": str(meta_path.resolve()),
        "embedding_norm_min": norm_min,
        "embedding_norm_max": norm_max,
        "cache_action": "REUSED",
    }


def extract_sequence(
    output_dir: Path,
    tracking_root: Path,
    sequence_id: str,
    records: Sequence[CropRecord],
    sequence_contract: Mapping[str, Any],
    model: Any,
    transform: Any,
    torch: Any,
    numpy: Any,
    device: Any,
    batch_size: int,
    num_workers: int,
    execution_contract_sha256: str,
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    dataset = CropDataset(tracking_root, records, transform)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        collate_fn=collate_crop_items,
        persistent_workers=num_workers > 0,
    )
    embeddings = numpy.empty(
        (len(records), EXPECTED_EMBEDDING_DIMENSION),
        dtype=numpy.float32,
    )
    completed = numpy.zeros(len(records), dtype=numpy.bool_)
    crop_width_min: int | None = None
    crop_width_max: int | None = None
    crop_height_min: int | None = None
    crop_height_max: int | None = None

    with torch.inference_mode():
        for batch in loader:
            if batch["errors"]:
                failures.extend(batch["errors"])
                raise AuditError(
                    f"{sequence_id}: {len(batch['errors'])} invalid/missing crop(s); "
                    "the frozen contract forbids silent skipping"
                )
            require(batch["tensors"], f"{sequence_id}: empty inference batch")
            images = torch.stack(batch["tensors"], dim=0).to(
                device=device,
                dtype=torch.float32,
                non_blocking=device.type == "cuda",
            )
            output = model(images)
            require(torch.is_tensor(output), f"{sequence_id}: model output is not a tensor")
            require(
                output.ndim == 2 and output.shape[1] == EXPECTED_EMBEDDING_DIMENSION,
                f"{sequence_id}: model output shape changed: {tuple(output.shape)}",
            )
            output = output.float()
            raw_norms = torch.linalg.vector_norm(output, ord=2, dim=1)
            require(torch.isfinite(output).all().item(), f"{sequence_id}: non-finite model output")
            require((raw_norms > 0).all().item(), f"{sequence_id}: zero-norm model output")
            normalized = torch.nn.functional.normalize(output, p=2, dim=1, eps=1e-12)
            normalized_cpu = normalized.cpu().numpy().astype(numpy.float32, copy=False)
            indices = numpy.asarray(batch["original_indices"], dtype=numpy.int64)
            require(not completed[indices].any(), f"{sequence_id}: duplicate inference row")
            embeddings[indices] = normalized_cpu
            completed[indices] = True

            width_batch_min = min(batch["crop_widths"])
            width_batch_max = max(batch["crop_widths"])
            height_batch_min = min(batch["crop_heights"])
            height_batch_max = max(batch["crop_heights"])
            crop_width_min = (
                width_batch_min if crop_width_min is None else min(crop_width_min, width_batch_min)
            )
            crop_width_max = (
                width_batch_max if crop_width_max is None else max(crop_width_max, width_batch_max)
            )
            crop_height_min = (
                height_batch_min
                if crop_height_min is None
                else min(crop_height_min, height_batch_min)
            )
            crop_height_max = (
                height_batch_max
                if crop_height_max is None
                else max(crop_height_max, height_batch_max)
            )

    require(completed.all(), f"{sequence_id}: incomplete embedding rows")
    require(numpy.isfinite(embeddings).all(), f"{sequence_id}: non-finite embedding cache")
    norms = numpy.linalg.norm(embeddings, axis=1)
    require(
        numpy.allclose(norms, 1.0, rtol=1e-5, atol=1e-5),
        f"{sequence_id}: embeddings are not L2-normalized",
    )

    shard_path, meta_path = shard_paths(output_dir, sequence_id)
    arrays = {
        "manifest_row_id": numpy.asarray(
            [record.manifest_row_id for record in records],
            dtype="<U24",
        ),
        "segment_id": numpy.asarray(
            [record.segment_id for record in records],
            dtype="<U24",
        ),
        "frame_id": numpy.asarray([record.frame_id for record in records], dtype=numpy.int32),
        "uniform_8_selected": numpy.asarray(
            [record.uniform_selected for record in records],
            dtype=numpy.uint8,
        ),
        "quality_stratified_8_selected": numpy.asarray(
            [record.quality_selected for record in records],
            dtype=numpy.uint8,
        ),
        "embedding": embeddings,
    }
    atomic_save_npz(shard_path, arrays, numpy)
    shard_hash = sha256_file(shard_path)
    metadata = {
        "status": "COMPLETE",
        "sequence_id": sequence_id,
        "source_game_key": sequence_contract["source_game_key"],
        "cache_namespace_sha256": EXPECTED_CACHE_NAMESPACE_SHA256,
        "execution_contract_sha256": execution_contract_sha256,
        "sequence_manifest_sha256": sequence_contract["sequence_manifest_sha256"],
        "manifest_row_ids_sha256": sequence_contract["manifest_row_ids_sha256"],
        "row_count": len(records),
        "embedding_dimension": EXPECTED_EMBEDDING_DIMENSION,
        "embedding_dtype": "float32",
        "per_crop_l2_normalized": True,
        "embedding_shard_sha256": shard_hash,
        "embedding_shard_bytes": shard_path.stat().st_size,
        "embedding_norm_min": float(norms.min()),
        "embedding_norm_max": float(norms.max()),
        "crop_width_min_after_clamp": crop_width_min,
        "crop_width_max_after_clamp": crop_width_max,
        "crop_height_min_after_clamp": crop_height_min,
        "crop_height_max_after_clamp": crop_height_max,
        "forbidden_metadata_cached": False,
        "created_at": now_iso(),
    }
    atomic_write_json(meta_path, metadata)
    return {
        **metadata,
        "embedding_shard_path": str(shard_path.resolve()),
        "embedding_meta_path": str(meta_path.resolve()),
        "cache_action": "EXTRACTED",
    }


def prepare_output(
    output_dir: Path,
    sequence_ids: Sequence[str],
    overwrite: bool,
    resume: bool,
) -> None:
    if output_dir.exists() and not overwrite and not resume:
        require(
            not any(output_dir.iterdir()),
            f"Output directory is not empty; use --resume or --overwrite: {output_dir}",
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "shards").mkdir(parents=True, exist_ok=True)
    if overwrite:
        for filename in KNOWN_OUTPUTS:
            (output_dir / filename).unlink(missing_ok=True)
        (output_dir / INCOMPLETE_MARKER).unlink(missing_ok=True)
        for sequence_id in sequence_ids:
            shard_path, meta_path = shard_paths(output_dir, sequence_id)
            shard_path.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage2b0-dir",
        type=Path,
        required=True,
        help="Stage 2-B0 frozen protocol output directory.",
    )
    parser.add_argument(
        "--stage2a6-dir",
        type=Path,
        required=True,
        help="Stage 2-A6 frozen backbone decision output directory.",
    )
    parser.add_argument(
        "--tracking-root",
        type=Path,
        required=True,
        help="SoccerNet Tracking dataset root containing train/SNMOT-*/img1.",
    )
    parser.add_argument(
        "--deep-eiou-root",
        type=Path,
        required=True,
        help="Frozen official Deep-EIoU git checkout root.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Frozen sports_model.pth.tar-60. Defaults to champion_contract.json path.",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def render_report(summary: Mapping[str, Any]) -> str:
    counts = summary["counts"]
    return f"""# SoccerNet Tracking Stage 2-B1 frozen crop embedding cache

- Status: **{summary['status']}**
- Generated: `{summary['generated_at']}`
- Stage: `{STAGE}`
- Script version: `{SCRIPT_VERSION}`
- Frozen champion: `{EXPECTED_CHAMPION_ID}`
- Checkpoint SHA-256: `{EXPECTED_CHECKPOINT_SHA256}`
- Cache namespace: `{EXPECTED_CACHE_NAMESPACE_SHA256}`

## Result

- Manifest rows: {counts['manifest_rows']:,}
- Sequences: {counts['sequences']:,}
- Segments represented: {counts['segments']:,}
- Embeddings extracted this run: {counts['embeddings_extracted_this_run']:,}
- Embeddings reused this run: {counts['embeddings_reused_this_run']:,}
- Sequence shards extracted / reused: {counts['shards_extracted_this_run']} / {counts['shards_reused_this_run']}
- Failed or skipped crops: {counts['failed_crops']} / {counts['skipped_crops']}
- Embedding dimension / dtype: 512 / float32
- Per-crop L2 normalization: immediately after model output

## Leakage and scope

- Images read: train split paths from the exact frozen manifest only
- Official test / challenge: **LOCKED_NOT_READ / BLIND_NOT_READ**
- Pair/identity/team/role/jersey/manual labels read for inference: **NONE**
- Training: **NONE**
- Prototype aggregation / cosine evaluation: **NONE / NONE**
- Threshold selection / edge creation / global linking: **NONE / NONE / NONE**

## PASS meaning

PASS means every one of the {counts['manifest_rows']:,} frozen crop rows has one
finite, 512-d, float32, L2-normalized embedding in an atomically committed
sequence shard whose manifest, checkpoint, source, execution, and cache hashes
match exactly. It does not demonstrate Tracking-domain ReID transfer quality,
select a prototype candidate or threshold, create a link, or prove the
one-minute 22-player product goal.

## Next stage

Stage 2-B2 may now construct the six pre-registered segment prototypes from
this cache and perform threshold-free, game-disjoint representation transfer
evaluation. It must not add candidates or calibrate a linking threshold.
"""


def main() -> None:
    args = parse_args()
    require(args.batch_size > 0, "--batch-size must be positive")
    require(args.num_workers >= 0, "--num-workers must be non-negative")

    stage2b0_dir = args.stage2b0_dir.resolve()
    stage2a6_dir = args.stage2a6_dir.resolve()
    tracking_root = args.tracking_root.resolve()
    deep_eiou_root = args.deep_eiou_root.resolve()
    output_dir = args.out.resolve()
    require(tracking_root.is_dir(), f"Missing Tracking dataset root: {tracking_root}")

    b0 = verify_stage2b0(stage2b0_dir)
    manifest_path = b0["paths"]["frame_sampling_manifest"]
    manifest_audit = audit_frame_manifest(manifest_path)
    champion_path, champion = verify_champion(stage2a6_dir)
    source_contract = verify_deep_eiou_source(deep_eiou_root)

    checkpoint_path = (
        args.checkpoint.resolve()
        if args.checkpoint is not None
        else Path(champion["checkpoint_path"]).resolve()
    )
    require(checkpoint_path.is_file(), f"Missing frozen checkpoint: {checkpoint_path}")
    checkpoint_bytes = checkpoint_path.stat().st_size
    checkpoint_hash = sha256_file(checkpoint_path)
    require(checkpoint_bytes == EXPECTED_CHECKPOINT_BYTES, "Checkpoint byte size changed")
    require(checkpoint_hash == EXPECTED_CHECKPOINT_SHA256, "Checkpoint SHA-256 changed")

    sequence_ids = [item["sequence_id"] for item in manifest_audit["sequence_infos"]]
    prepare_output(output_dir, sequence_ids, args.overwrite, args.resume)
    atomic_write_json(
        output_dir / INCOMPLETE_MARKER,
        {
            "stage": STAGE,
            "status": "INCOMPLETE",
            "started_at": now_iso(),
            "cache_namespace_sha256": EXPECTED_CACHE_NAMESPACE_SHA256,
        },
    )

    failures: list[dict[str, Any]] = []
    failure_fields = (
        "sequence_id",
        "manifest_row_id",
        "segment_id",
        "frame_id",
        "image_relative_path",
        "reason_code",
        "message",
    )
    atomic_write_csv(output_dir / "failure_manifest.csv", failure_fields, failures)

    try:
        import numpy as np
        import PIL
        import torch
        import torchvision
        from torchvision import transforms

        configure_determinism(torch)
        device = resolve_device(torch, args.device)
        execution_contract = build_execution_contract(
            torch=torch,
            torchvision=torchvision,
            pillow_version=package_version(PIL),
            numpy_version=package_version(np),
            device=device,
            batch_size=args.batch_size,
        )
        execution_hash = execution_contract["execution_contract_sha256"]
        atomic_write_json(output_dir / "execution_contract.json", execution_contract)

        transform = transforms.Compose(
            [
                transforms.Resize((256, 128)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        model: Any | None = None
        model_load_contract: dict[str, Any] | None = None
        shard_entries: list[dict[str, Any]] = []
        extracted_rows = 0
        reused_rows = 0
        extracted_shards = 0
        reused_shards = 0

        expected_info_by_sequence = {
            item["sequence_id"]: item for item in manifest_audit["sequence_infos"]
        }
        seen_sequences: list[str] = []
        for sequence_id, records, sequence_contract in manifest_sequence_groups(manifest_path):
            seen_sequences.append(sequence_id)
            require(sequence_id in expected_info_by_sequence, f"Unexpected sequence {sequence_id}")
            require(
                sequence_contract == expected_info_by_sequence[sequence_id],
                f"{sequence_id}: second-pass manifest contract changed",
            )
            cached = cached_shard_status(
                output_dir=output_dir,
                sequence_id=sequence_id,
                records=records,
                sequence_contract=sequence_contract,
                execution_contract_sha256=execution_hash,
                numpy=np,
                resume=args.resume,
            )
            if cached is not None:
                shard_entries.append(cached)
                reused_rows += len(records)
                reused_shards += 1
                print(
                    f"[REUSED] {sequence_id}: {len(records)} embeddings",
                    flush=True,
                )
                continue

            if model is None:
                models, reid_root = import_vendored_osnet(deep_eiou_root)
                model, model_load_contract = build_model(
                    torch=torch,
                    models=models,
                    checkpoint_path=checkpoint_path,
                    device=device,
                )
                model_load_contract.update(
                    {
                        "checkpoint_path": str(checkpoint_path),
                        "checkpoint_sha256": checkpoint_hash,
                        "checkpoint_bytes": checkpoint_bytes,
                        "source_commit": source_contract["commit"],
                        "vendored_reid_root": str(reid_root.resolve()),
                        "device": str(device),
                    }
                )
                atomic_write_json(
                    output_dir / "model_load_contract.json",
                    model_load_contract,
                )

            extracted = extract_sequence(
                output_dir=output_dir,
                tracking_root=tracking_root,
                sequence_id=sequence_id,
                records=records,
                sequence_contract=sequence_contract,
                model=model,
                transform=transform,
                torch=torch,
                numpy=np,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                execution_contract_sha256=execution_hash,
                failures=failures,
            )
            shard_entries.append(extracted)
            extracted_rows += len(records)
            extracted_shards += 1
            print(
                f"[EXTRACTED] {sequence_id}: {len(records)} embeddings",
                flush=True,
            )

        require(seen_sequences == sequence_ids, "Manifest sequence order changed")
        require(len(shard_entries) == EXPECTED_SEQUENCES, "Incomplete shard count")
        require(
            extracted_rows + reused_rows == EXPECTED_ROWS,
            "Incomplete embedding cache row count",
        )
        require(not failures, "Failed crop records exist")

        if model_load_contract is None:
            existing_model_contract = output_dir / "model_load_contract.json"
            require(
                existing_model_contract.is_file(),
                "All shards were reused but model_load_contract.json is missing",
            )
            model_load_contract = load_json(existing_model_contract)
            require(
                model_load_contract["checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA256,
                "Reused model load contract checkpoint changed",
            )

        stable_shards = [
            {
                "sequence_id": entry["sequence_id"],
                "source_game_key": entry["source_game_key"],
                "row_count": entry["row_count"],
                "sequence_manifest_sha256": entry["sequence_manifest_sha256"],
                "manifest_row_ids_sha256": entry["manifest_row_ids_sha256"],
                "embedding_shard_sha256": entry["embedding_shard_sha256"],
                "embedding_shard_bytes": entry["embedding_shard_bytes"],
                "embedding_dimension": entry["embedding_dimension"],
                "embedding_dtype": entry["embedding_dtype"],
                "per_crop_l2_normalized": entry["per_crop_l2_normalized"],
            }
            for entry in shard_entries
        ]
        index_contract = {
            "contract_version": 1,
            "stage": STAGE,
            "status": "COMPLETE",
            "cache_namespace_sha256": EXPECTED_CACHE_NAMESPACE_SHA256,
            "execution_contract_sha256": execution_hash,
            "frame_sampling_manifest_sha256": EXPECTED_FRAME_MANIFEST_SHA256,
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "source_commit": EXPECTED_SOURCE_COMMIT,
            "embedding_dimension": EXPECTED_EMBEDDING_DIMENSION,
            "embedding_dtype": "float32",
            "per_crop_l2_normalized": True,
            "total_rows": EXPECTED_ROWS,
            "total_sequences": EXPECTED_SEQUENCES,
            "shards": stable_shards,
        }
        cache_index = {
            **index_contract,
            "embedding_cache_index_sha256": canonical_sha256(index_contract),
            "generated_at": now_iso(),
        }
        atomic_write_json(output_dir / "embedding_cache_index.json", cache_index)

        shard_manifest_fields = (
            "sequence_id",
            "source_game_key",
            "row_count",
            "cache_action",
            "sequence_manifest_sha256",
            "manifest_row_ids_sha256",
            "embedding_shard_sha256",
            "embedding_shard_bytes",
            "embedding_norm_min",
            "embedding_norm_max",
            "embedding_shard_path",
            "embedding_meta_path",
        )
        atomic_write_csv(
            output_dir / "shard_manifest.csv",
            shard_manifest_fields,
            (
                {field: entry[field] for field in shard_manifest_fields}
                for entry in shard_entries
            ),
        )
        atomic_write_csv(output_dir / "failure_manifest.csv", failure_fields, failures)

        input_paths = {
            "stage2b0_summary": b0["paths"]["summary"],
            "protocol_contract": b0["paths"]["protocol_contract"],
            "candidate_registry": b0["paths"]["candidate_registry"],
            "embedding_cache_contract": b0["paths"]["embedding_cache_contract"],
            "frame_sampling_manifest": manifest_path,
            "champion_contract": champion_path,
            "checkpoint": checkpoint_path,
        }
        input_hashes = {
            "stage": STAGE,
            "status": "PASS",
            "inputs": {
                name: {
                    "path": str(path.resolve()),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for name, path in input_paths.items()
            },
            "source": source_contract,
        }
        atomic_write_json(output_dir / "input_hashes.json", input_hashes)
        atomic_write_json(output_dir / "manifest_audit.json", manifest_audit)

        leakage_audit = {
            "status": "PASS",
            "stage": STAGE,
            "official_test_status": "LOCKED_NOT_READ",
            "challenge_status": "BLIND_NOT_READ",
            "allowed_image_path_prefix": "train/",
            "manifest_paths_audited_before_image_reads": True,
            "image_rows_read": extracted_rows,
            "image_rows_reused_without_read_this_run": reused_rows,
            "pair_files_read": [],
            "identity_inventory_read": False,
            "team_role_jersey_metadata_read_for_inference": False,
            "manual_labels_read": False,
            "forbidden_metadata_cached": False,
            "training_performed": False,
            "prototype_aggregation_performed": False,
            "cosine_similarity_computed": False,
            "threshold_selection_performed": False,
            "threshold_sweep_performed": False,
            "edge_creation_performed": False,
            "global_linking_performed": False,
        }
        atomic_write_json(output_dir / "leakage_audit.json", leakage_audit)

        resolved_config = {
            "stage": STAGE,
            "script_version": SCRIPT_VERSION,
            "stage2b0_dir": str(stage2b0_dir),
            "stage2a6_dir": str(stage2a6_dir),
            "tracking_root": str(tracking_root),
            "deep_eiou_root": str(deep_eiou_root),
            "checkpoint": str(checkpoint_path),
            "device_requested": args.device,
            "device_resolved": str(device),
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "out": str(output_dir),
            "resume": args.resume,
            "overwrite": args.overwrite,
        }
        atomic_write_json(output_dir / "resolved_config.json", resolved_config)

        warnings = [
            {
                "severity": "WARNING",
                "category": "provenance",
                "code": "SPORTS_RECIPE_UNDISCLOSED",
                "count": 1,
                "message": (
                    "The official Deep-EIoU sports checkpoint training recipe is "
                    "undisclosed; checkpoint bytes, source commit, preprocessing, and "
                    "execution/cache contracts are hash-frozen."
                ),
            },
            {
                "severity": "INFO",
                "category": "scope",
                "code": "EMBEDDINGS_ONLY_NO_EVALUATION_OR_THRESHOLD",
                "count": 1,
                "message": (
                    "Stage 2-B1 creates crop embeddings only. It does not aggregate "
                    "prototypes, evaluate ReID transfer, or select a linking threshold."
                ),
            },
        ]
        atomic_write_csv(
            output_dir / "warnings.csv",
            ("severity", "category", "code", "count", "message"),
            warnings,
        )

        summary = {
            "status": "PASS",
            "stage": STAGE,
            "script_version": SCRIPT_VERSION,
            "generated_at": now_iso(),
            "champion_candidate_id": EXPECTED_CHAMPION_ID,
            "architecture": EXPECTED_ARCHITECTURE,
            "checkpoint_sha256": checkpoint_hash,
            "source_commit": source_contract["commit"],
            "protocol_contract_sha256": EXPECTED_PROTOCOL_CONTRACT_SHA256,
            "candidate_registry_sha256": EXPECTED_CANDIDATE_REGISTRY_SHA256,
            "embedding_cache_contract_sha256": (
                EXPECTED_EMBEDDING_CACHE_CONTRACT_SHA256
            ),
            "cache_namespace_sha256": EXPECTED_CACHE_NAMESPACE_SHA256,
            "execution_contract_sha256": execution_hash,
            "embedding_cache_index_sha256": cache_index[
                "embedding_cache_index_sha256"
            ],
            "frame_sampling_manifest_sha256": EXPECTED_FRAME_MANIFEST_SHA256,
            "counts": {
                "manifest_rows": EXPECTED_ROWS,
                "sequences": EXPECTED_SEQUENCES,
                "segments": EXPECTED_SEGMENTS,
                "embeddings_extracted_this_run": extracted_rows,
                "embeddings_reused_this_run": reused_rows,
                "shards_extracted_this_run": extracted_shards,
                "shards_reused_this_run": reused_shards,
                "failed_crops": 0,
                "skipped_crops": 0,
            },
            "embedding_dimension": EXPECTED_EMBEDDING_DIMENSION,
            "embedding_dtype": "float32",
            "per_crop_l2_normalized": True,
            "training_performed": False,
            "prototype_aggregation_performed": False,
            "cosine_similarity_computed": False,
            "threshold_selection_performed": False,
            "threshold_sweep_performed": False,
            "edge_creation_performed": False,
            "global_linking_performed": False,
            "official_test_status": "LOCKED_NOT_READ",
            "challenge_status": "BLIND_NOT_READ",
            "leakage_audit_status": "PASS",
            "next_stage": (
                "Stage 2-B2 threshold-free representation transfer evaluation of all "
                "and only the six frozen candidates. Do not select a threshold in B2."
            ),
        }
        atomic_write_json(output_dir / "summary.json", summary)
        atomic_write_text(output_dir / "report.md", render_report(summary))
        (output_dir / INCOMPLETE_MARKER).unlink(missing_ok=True)

        print("SoccerNet Tracking Stage 2-B1 frozen embedding cache complete")
        print("Status              : PASS")
        print(f"Crop embeddings     : {EXPECTED_ROWS}")
        print(f"Sequence shards     : {EXPECTED_SEQUENCES}")
        print(f"Extracted / reused  : {extracted_rows} / {reused_rows}")
        print("Embedding           : 512-d float32, per-crop L2 normalized")
        print("Threshold/linking   : NONE / NONE")
        print("Test/challenge      : LOCKED_NOT_READ / BLIND_NOT_READ")
        print(f"Cache namespace     : {EXPECTED_CACHE_NAMESPACE_SHA256}")
        print(f"Cache index SHA-256 : {cache_index['embedding_cache_index_sha256']}")
        print(f"Output              : {output_dir}")
    except Exception as exc:
        if failures:
            atomic_write_csv(output_dir / "failure_manifest.csv", failure_fields, failures)
        failure_payload = {
            "status": "FAIL",
            "stage": STAGE,
            "failed_at": now_iso(),
            "error_type": type(exc).__name__,
            "message": str(exc),
            "failed_crop_count": len(failures),
        }
        atomic_write_json(output_dir / "failure.json", failure_payload)
        raise


def cli() -> None:
    try:
        main()
    except AuditError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    cli()
