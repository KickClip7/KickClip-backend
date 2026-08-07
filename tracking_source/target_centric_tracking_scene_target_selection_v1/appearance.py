from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from common import atomic_json, load_module, portable


def frozen_sports_osnet_score_function(
    *,
    project_root: Path,
    output_root: Path,
    reference_set: dict[str, Any],
    requested_device: str,
):
    """Return a blind earlier-candidate ranker using frozen V2 B1 metrics.

    It does not apply a new identity threshold and cannot confirm an identity.
    The ranked output is always routed to user confirmation.
    """

    import cv2
    import numpy as np
    import torch

    loader = load_module(
        "scene_selection_r2_safe_osnet",
        project_root
        / "target_centric_tracking_v2_production_r2"
        / "safe_sports_osnet_loader.py",
    )
    stage2b = load_module(
        "scene_selection_frozen_v1_stage2b",
        project_root
        / "target_centric_tracking_v1"
        / "stage2b_run_same_shot_reentry_reacquisition.py",
    )
    frozen_b1 = load_module(
        "scene_selection_frozen_v2_stage3b1",
        project_root
        / "target_centric_tracking_v2"
        / "stage3b1_rank_postcut_candidates_with_frozen_reid.py",
    )
    model, device, load_contract = loader.load_sports_osnet(
        project_root,
        device_name=requested_device,
    )
    portable_load_contract = deepcopy(load_contract)
    for key in ("checkpoint", "vendored_reid_root"):
        raw = portable_load_contract.get(key)
        if raw:
            path = Path(str(raw)).resolve()
            try:
                portable_load_contract[key] = path.relative_to(
                    project_root.resolve()
                ).as_posix()
            except ValueError:
                portable_load_contract[key] = path.name
    transform = stage2b.build_transform()

    reference_crops: dict[str, Any] = {}
    for reference in reference_set.get("references", []):
        path = output_root / reference["crop_artifact"]
        image = cv2.imread(str(path))
        if image is not None:
            reference_crops[str(reference["reference_id"])] = image
    if not reference_crops:
        raise RuntimeError("Target reference set has no decodable crops.")
    target_embedded = stage2b.embed_crops(
        reference_crops,
        model,
        transform,
        torch,
        device,
        32,
    )
    embedding_root = (
        output_root
        / "target_references"
        / str(reference_set["selection_id"])
        / "embeddings"
    )
    embedding_root.mkdir(parents=True, exist_ok=True)
    reference_by_id = {
        str(row["reference_id"]): row
        for row in reference_set.get("references", [])
    }
    for reference_id, embedding in target_embedded.items():
        artifact = embedding_root / f"{reference_id}.npy"
        np.save(artifact, embedding.astype(np.float32), allow_pickle=False)
        reference_by_id[reference_id]["embedding_artifact"] = portable(
            artifact, output_root
        )
    atomic_json(output_root / "target_reference_set.json", reference_set)
    target_gallery = np.stack(
        [target_embedded[key] for key in reference_crops]
    ).astype(np.float32)
    target_gallery = frozen_b1.l2_normalize_rows(target_gallery)
    target_prototype = frozen_b1.l2_normalize_vector(
        np.mean(target_gallery, axis=0).astype(np.float32)
    )
    negative_gallery = np.empty((0, 512), dtype=np.float32)

    def score(
        selected: dict[str, Any],
        earlier: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        del selected
        rows = []
        for candidate in earlier:
            crops: dict[str, Any] = {}
            for index, relative in enumerate(
                candidate.get("artifacts", {}).get("crops", [])
            ):
                image = cv2.imread(str(output_root / relative))
                if image is not None:
                    crops[f"{candidate['candidate_id']}:{index}"] = image
            if not crops:
                continue
            embedded = stage2b.embed_crops(
                crops,
                model,
                transform,
                torch,
                device,
                32,
            )
            matrix = np.stack([embedded[key] for key in crops]).astype(
                np.float32
            )
            metrics = frozen_b1.compute_candidate_metrics(
                matrix,
                target_gallery,
                target_prototype,
                negative_gallery,
            )
            rows.append(
                {
                    "candidate": candidate,
                    "retrieval_score": float(metrics["retrieval_score"]),
                    "prototype_similarity": float(
                        metrics["prototype_target_similarity"]
                    ),
                    "frozen_metrics": {
                        key: float(value)
                        for key, value in metrics.items()
                        if key != "prototype"
                    },
                    "provenance": {
                        "ranking": (
                            "FROZEN_V2_STAGE3B1_COMPUTE_CANDIDATE_METRICS"
                        ),
                        "sports_osnet_load_contract": portable_load_contract,
                        "evaluation_target_ids_used": False,
                    },
                }
            )
        rows.sort(
            key=lambda row: (
                row["retrieval_score"],
                row["prototype_similarity"],
                row["frozen_metrics"]["crop_margin_median"],
                row["frozen_metrics"]["positive_margin_support_ratio"],
            ),
            reverse=True,
        )
        return rows

    return score
