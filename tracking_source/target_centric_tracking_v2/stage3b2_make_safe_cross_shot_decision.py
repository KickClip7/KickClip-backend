#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""KickClip Target-Centric Tracking V2 - Stage 3-B2.

Convert Stage 3-B1 retrieval rankings into a conservative cross-shot decision.

Possible operational states:
- AUTO_REACQUIRED_CANDIDATE:
    Automatic evidence is sufficiently strong, but visual review is still
    required in this development protocol. No link is created here.
- AMBIGUOUS:
    Plausible candidates exist, but the top candidate is not safe enough for
    automatic selection. User confirmation fallback is required.
- SAFE_REJECTED:
    No candidate has enough evidence. Remain SEARCHING.

Manual target labels are read only after the automatic decision has been
computed, and are used only for evaluation. They never influence the decision.

No target is linked and no memory is updated in this stage.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


STAGE = "stage3b2_make_safe_cross_shot_decision"
VERSION = "target-centric-v2-stage3b2-1.0.0"

OUTPUT_FILES = (
    "stage3b2_decision.json",
    "stage3b2_gate_audit.csv",
    "stage3b2_review_candidates.jpg",
    "stage3b2_summary.json",
    "stage3b2_report.md",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
    )

    parser.add_argument(
        "--test-name",
        required=True,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "runs/target_centric_tracking_v2"
        ),
    )

    # These are conservative development gates, not validated operating
    # thresholds. They must remain frozen for later generalization tests.
    parser.add_argument(
        "--minimum-retrieval-score",
        type=float,
        default=0.65,
    )

    parser.add_argument(
        "--minimum-prototype-similarity",
        type=float,
        default=0.60,
    )

    parser.add_argument(
        "--minimum-median-margin",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--minimum-positive-margin-support",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--minimum-top1-top2-gap",
        type=float,
        default=0.08,
    )

    parser.add_argument(
        "--minimum-plausible-score",
        type=float,
        default=0.45,
    )

    parser.add_argument(
        "--minimum-plausible-prototype",
        type=float,
        default=0.45,
    )

    parser.add_argument(
        "--review-candidate-count",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def validate_test_name(
    value: str,
) -> str:
    allowed = set(
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789._-"
    )

    if (
        not value
        or value in {".", ".."}
        or any(
            character not in allowed
            for character in value
        )
    ):
        raise ValueError(
            "Invalid --test-name"
        )

    return value


def resolve(
    root: Path,
    value: Path,
) -> Path:
    value = value.expanduser()

    if value.is_absolute():
        return value.resolve()

    return (
        root
        / value
    ).resolve()


def read_json(
    path: Path,
) -> dict[str, Any]:
    value = json.loads(
        path.read_text(
            encoding="utf-8-sig"
        )
    )

    if not isinstance(
        value,
        dict,
    ):
        raise TypeError(
            f"Expected JSON object: {path}"
        )

    return value


def read_csv(
    path: Path,
) -> list[dict[str, str]]:
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        return list(
            csv.DictReader(
                stream
            )
        )


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as stream:
        for chunk in iter(
            lambda: stream.read(
                8 * 1024 * 1024
            ),
            b"",
        ):
            digest.update(
                chunk
            )

    return digest.hexdigest()


def atomic_text(
    path: Path,
    text: str,
) -> None:
    temporary = path.with_name(
        path.name + ".tmp"
    )

    temporary.write_text(
        text,
        encoding="utf-8",
        newline="\n",
    )

    os.replace(
        temporary,
        path,
    )


def atomic_json(
    path: Path,
    value: Any,
) -> None:
    atomic_text(
        path,
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
    )


def write_csv(
    path: Path,
    rows: Sequence[
        Mapping[str, Any]
    ],
    fieldnames: Sequence[str],
) -> None:
    temporary = path.with_name(
        path.name + ".tmp"
    )

    with temporary.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(
                fieldnames
            ),
            extrasaction="raise",
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

        stream.flush()
        os.fsync(
            stream.fileno()
        )

    os.replace(
        temporary,
        path,
    )


def prepare_outputs(
    output_dir: Path,
    overwrite: bool,
) -> None:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing = [
        output_dir / name
        for name in OUTPUT_FILES
        if (
            output_dir / name
        ).exists()
    ]

    if existing and not overwrite:
        raise FileExistsError(
            "Stage 3-B2 outputs already exist. "
            "Use --overwrite:\n"
            + "\n".join(
                str(path)
                for path in existing
            )
        )

    for path in existing:
        if not path.is_file():
            raise IsADirectoryError(
                path
            )

        path.unlink()


def float_value(
    row: Mapping[str, Any],
    key: str,
) -> float:
    value = float(
        row[key]
    )

    if not math.isfinite(
        value
    ):
        raise ValueError(
            f"Non-finite value for {key}: {value}"
        )

    return value


def bool_text(
    value: bool,
) -> str:
    return (
        "PASS"
        if value
        else "FAIL"
    )


def fit_image(
    image: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    canvas = np.zeros(
        (
            height,
            width,
            3,
        ),
        dtype=np.uint8,
    )

    source_height, source_width = (
        image.shape[:2]
    )

    scale = min(
        width / source_width,
        height / source_height,
    )

    resized_width = max(
        1,
        int(
            round(
                source_width * scale
            )
        ),
    )

    resized_height = max(
        1,
        int(
            round(
                source_height * scale
            )
        ),
    )

    resized = cv2.resize(
        image,
        (
            resized_width,
            resized_height,
        ),
        interpolation=cv2.INTER_AREA,
    )

    offset_x = (
        width - resized_width
    ) // 2

    offset_y = (
        height - resized_height
    ) // 2

    canvas[
        offset_y:
        offset_y + resized_height,
        offset_x:
        offset_x + resized_width,
    ] = resized

    return canvas


def make_review_sheet(
    ranked_rows: Sequence[
        Mapping[str, Any]
    ],
    strip_dir: Path,
    output_path: Path,
    candidate_count: int,
    operational_state: str,
) -> None:
    selected_rows = list(
        ranked_rows[
            :candidate_count
        ]
    )

    if not selected_rows:
        raise RuntimeError(
            "No candidates available for review"
        )

    card_width = 640
    card_height = 350
    header_height = 90

    cards: list[np.ndarray] = []

    for row in selected_rows:
        candidate_id = str(
            row["candidate_id"]
        )

        strip_path = (
            strip_dir
            / f"{candidate_id}.jpg"
        )

        if not strip_path.is_file():
            raise FileNotFoundError(
                strip_path
            )

        strip = cv2.imread(
            str(
                strip_path
            )
        )

        if strip is None:
            raise RuntimeError(
                f"Cannot read strip: {strip_path}"
            )

        card = np.zeros(
            (
                card_height,
                card_width,
                3,
            ),
            dtype=np.uint8,
        )

        fitted = fit_image(
            strip,
            card_width,
            card_height - header_height,
        )

        card[
            header_height:,
            :,
        ] = fitted

        rank = int(
            row["retrieval_rank"]
        )

        score = float_value(
            row,
            "retrieval_score",
        )

        prototype = float_value(
            row,
            "prototype_target_similarity",
        )

        margin = float_value(
            row,
            "crop_margin_median",
        )

        support = float_value(
            row,
            "positive_margin_support_ratio",
        )

        cv2.putText(
            card,
            (
                f"RANK #{rank} {candidate_id} "
                f"| state={operational_state}"
            ),
            (
                8,
                25,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.53,
            (
                255,
                255,
                255,
            ),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            card,
            (
                f"score={score:.6f} "
                f"prototype={prototype:.6f}"
            ),
            (
                8,
                52,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (
                255,
                255,
                255,
            ),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            card,
            (
                f"median target-negative margin={margin:.6f} "
                f"positive support={support:.3f}"
            ),
            (
                8,
                77,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (
                255,
                255,
                255,
            ),
            1,
            cv2.LINE_AA,
        )

        cards.append(
            card
        )

    sheet = np.vstack(
        cards
    )

    if not cv2.imwrite(
        str(
            output_path
        ),
        sheet,
    ):
        raise RuntimeError(
            f"Cannot write review sheet: {output_path}"
        )


def build_report(
    summary: Mapping[str, Any],
) -> str:
    top = summary[
        "automatic_decision"
    ][
        "top_candidate"
    ]

    gates = summary[
        "automatic_decision"
    ][
        "gates"
    ]

    lines = [
        (
            "# KickClip Target-Centric Tracking V2 "
            "— Stage 3-B2"
        ),
        "",
        f"- Status: `{summary['status']}`",
        f"- Decision: `{summary['decision']}`",
        (
            "- Operational state: "
            f"`{summary['operational_state']}`"
        ),
        (
            "- Top automatic candidate: "
            f"`{top['candidate_id']}`"
        ),
        (
            "- Top retrieval score: "
            f"`{top['retrieval_score']:.6f}`"
        ),
        (
            "- Top-1 / top-2 gap: "
            f"`{top['top1_top2_gap']:.6f}`"
        ),
        (
            "- Evaluation-only top candidate correct: "
            f"`{summary['evaluation']['top_candidate_is_target']}`"
        ),
        "",
        "## Automatic gates",
        "",
    ]

    for name, gate in gates.items():
        lines.append(
            f"- {name}: `{gate['passed']}` "
            f"(value={gate['value']}, "
            f"required={gate['required']})"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "The automatic decision was computed before "
                "the evaluation-only target labels were read."
            ),
            (
                "No candidate was linked and target memory "
                "was not updated."
            ),
            (
                "For AMBIGUOUS, the next authorized stage is "
                "explicit user confirmation among the top "
                "review candidates."
            ),
        ]
    )

    return (
        "\n".join(
            lines
        )
        + "\n"
    )


def main() -> int:
    arguments = parse_args()

    if arguments.review_candidate_count < 1:
        raise ValueError(
            "--review-candidate-count must be positive"
        )

    threshold_values = (
        arguments.minimum_retrieval_score,
        arguments.minimum_prototype_similarity,
        arguments.minimum_median_margin,
        arguments.minimum_positive_margin_support,
        arguments.minimum_top1_top2_gap,
        arguments.minimum_plausible_score,
        arguments.minimum_plausible_prototype,
    )

    if any(
        not math.isfinite(value)
        for value in threshold_values
    ):
        raise ValueError(
            "All gate values must be finite"
        )

    if not (
        0.0
        <= arguments.minimum_positive_margin_support
        <= 1.0
    ):
        raise ValueError(
            "Invalid positive-margin support threshold"
        )

    root = (
        arguments.project_root
        .expanduser()
        .resolve()
    )

    if not root.is_dir():
        raise FileNotFoundError(
            root
        )

    test_name = validate_test_name(
        arguments.test_name
    )

    output_dir = (
        resolve(
            root,
            arguments.output_root,
        )
        / test_name
    )

    if not output_dir.is_dir():
        raise FileNotFoundError(
            output_dir
        )

    prepare_outputs(
        output_dir,
        arguments.overwrite,
    )

    stage3b0_path = (
        output_dir
        / "stage3b0_summary.json"
    )

    stage3b1_path = (
        output_dir
        / "stage3b1_summary.json"
    )

    ranked_path = (
        output_dir
        / "stage3b1_ranked_candidates.csv"
    )

    evaluation_path = (
        output_dir
        / "stage3b1_evaluation.json"
    )

    for path in (
        stage3b0_path,
        stage3b1_path,
        ranked_path,
        evaluation_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(
                path
            )

    stage3b0 = read_json(
        stage3b0_path
    )

    stage3b1 = read_json(
        stage3b1_path
    )

    ranked_rows = read_csv(
        ranked_path
    )

    if (
        stage3b0.get(
            "status"
        )
        != "PASS"
    ):
        raise RuntimeError(
            "Stage 3-B0 must be PASS"
        )

    if (
        stage3b1.get(
            "status"
        )
        != "PASS"
        or stage3b1.get(
            "decision"
        )
        != (
            "AUTHORIZE_MANDATORY_"
            "STAGE3B1_RETRIEVAL_"
            "RANKING_REVIEW"
        )
    ):
        raise RuntimeError(
            "Stage 3-B1 must authorize ranking review"
        )

    if len(
        ranked_rows
    ) < 2:
        raise RuntimeError(
            "Stage 3-B2 requires at least two candidates"
        )

    ranked_rows.sort(
        key=lambda row: int(
            row["retrieval_rank"]
        )
    )

    expected_ranks = list(
        range(
            1,
            len(
                ranked_rows
            )
            + 1,
        )
    )

    actual_ranks = [
        int(
            row["retrieval_rank"]
        )
        for row in ranked_rows
    ]

    if actual_ranks != expected_ranks:
        raise RuntimeError(
            f"Invalid retrieval ranks: {actual_ranks}"
        )

    top1 = ranked_rows[0]
    top2 = ranked_rows[1]

    top1_score = float_value(
        top1,
        "retrieval_score",
    )

    top2_score = float_value(
        top2,
        "retrieval_score",
    )

    top1_prototype = float_value(
        top1,
        "prototype_target_similarity",
    )

    top1_margin = float_value(
        top1,
        "crop_margin_median",
    )

    top1_support = float_value(
        top1,
        "positive_margin_support_ratio",
    )

    top1_top2_gap = (
        top1_score
        - top2_score
    )

    # The automatic decision is fully computed here,
    # before evaluation-only target IDs are read.
    gates = {
        "retrieval_score": {
            "value": top1_score,
            "required": (
                f">={arguments.minimum_retrieval_score}"
            ),
            "passed": (
                top1_score
                >= arguments.minimum_retrieval_score
            ),
        },
        "prototype_similarity": {
            "value": top1_prototype,
            "required": (
                f">={arguments.minimum_prototype_similarity}"
            ),
            "passed": (
                top1_prototype
                >= arguments.minimum_prototype_similarity
            ),
        },
        "median_target_negative_margin": {
            "value": top1_margin,
            "required": (
                f">{arguments.minimum_median_margin}"
            ),
            "passed": (
                top1_margin
                > arguments.minimum_median_margin
            ),
        },
        "positive_margin_support": {
            "value": top1_support,
            "required": (
                f">={arguments.minimum_positive_margin_support}"
            ),
            "passed": (
                top1_support
                >= arguments.minimum_positive_margin_support
            ),
        },
        "top1_top2_score_gap": {
            "value": top1_top2_gap,
            "required": (
                f">={arguments.minimum_top1_top2_gap}"
            ),
            "passed": (
                top1_top2_gap
                >= arguments.minimum_top1_top2_gap
            ),
        },
    }

    all_auto_gates_passed = all(
        bool(
            gate["passed"]
        )
        for gate in gates.values()
    )

    plausible_candidate_exists = (
        top1_score
        >= arguments.minimum_plausible_score
        and top1_prototype
        >= arguments.minimum_plausible_prototype
    )

    if all_auto_gates_passed:
        operational_state = (
            "AUTO_REACQUIRED_CANDIDATE"
        )

        decision = (
            "AUTHORIZE_MANDATORY_"
            "STAGE3B2_AUTO_CANDIDATE_"
            "VISUAL_REVIEW"
        )

        automatically_proposed_candidate = str(
            top1["candidate_id"]
        )

    elif plausible_candidate_exists:
        operational_state = (
            "AMBIGUOUS"
        )

        decision = (
            "AUTHORIZE_STAGE3B3_"
            "USER_CONFIRMATION_FALLBACK"
        )

        automatically_proposed_candidate = None

    else:
        operational_state = (
            "SAFE_REJECTED_SEARCHING"
        )

        decision = (
            "RETAIN_SEARCHING_NO_"
            "CROSS_SHOT_TARGET_SELECTED"
        )

        automatically_proposed_candidate = None

    # Evaluation-only labels are deliberately loaded only after
    # the automatic operational state has been finalized.
    evaluation = read_json(
        evaluation_path
    )

    evaluation_target_ids = set(
        str(candidate_id)
        for candidate_id in evaluation.get(
            "target_candidate_ids",
            [],
        )
    )

    top_candidate_id = str(
        top1["candidate_id"]
    )

    top_candidate_is_target = (
        top_candidate_id
        in evaluation_target_ids
    )

    automatic_wrong_link_prevented = bool(
        not top_candidate_is_target
        and operational_state
        != "AUTO_REACQUIRED_CANDIDATE"
    )

    automatic_correct_candidate_proposed = bool(
        top_candidate_is_target
        and operational_state
        == "AUTO_REACQUIRED_CANDIDATE"
    )

    gate_rows = [
        {
            "gate_name": name,
            "value": gate["value"],
            "required": gate["required"],
            "passed": gate["passed"],
        }
        for name, gate in gates.items()
    ]

    gate_audit_path = (
        output_dir
        / "stage3b2_gate_audit.csv"
    )

    write_csv(
        gate_audit_path,
        gate_rows,
        list(
            gate_rows[0]
        ),
    )

    strip_dir = Path(
        str(
            stage3b0[
                "outputs"
            ][
                "tracklet_strips"
            ]
        )
    ).resolve()

    if not strip_dir.is_dir():
        raise FileNotFoundError(
            strip_dir
        )

    review_sheet_path = (
        output_dir
        / "stage3b2_review_candidates.jpg"
    )

    make_review_sheet(
        ranked_rows,
        strip_dir,
        review_sheet_path,
        min(
            arguments.review_candidate_count,
            len(
                ranked_rows
            ),
        ),
        operational_state,
    )

    automatic_decision = {
        "operational_state": (
            operational_state
        ),
        "decision": (
            decision
        ),
        "automatically_proposed_candidate": (
            automatically_proposed_candidate
        ),
        "all_auto_gates_passed": (
            all_auto_gates_passed
        ),
        "plausible_candidate_exists": (
            plausible_candidate_exists
        ),
        "top_candidate": {
            "candidate_id": (
                top_candidate_id
            ),
            "retrieval_rank": int(
                top1["retrieval_rank"]
            ),
            "retrieval_score": (
                top1_score
            ),
            "prototype_target_similarity": (
                top1_prototype
            ),
            "crop_margin_median": (
                top1_margin
            ),
            "positive_margin_support_ratio": (
                top1_support
            ),
            "top1_top2_gap": (
                top1_top2_gap
            ),
        },
        "second_candidate": {
            "candidate_id": str(
                top2["candidate_id"]
            ),
            "retrieval_rank": int(
                top2["retrieval_rank"]
            ),
            "retrieval_score": (
                top2_score
            ),
        },
        "gates": (
            gates
        ),
    }

    decision_payload = {
        "stage": (
            STAGE
        ),
        "version": (
            VERSION
        ),
        "generated_at": (
            datetime.now(
                timezone.utc
            )
            .astimezone()
            .isoformat(
                timespec="seconds"
            )
        ),
        "test_name": (
            test_name
        ),
        "automatic_decision": (
            automatic_decision
        ),
        "policy": {
            "minimum_retrieval_score": (
                arguments.minimum_retrieval_score
            ),
            "minimum_prototype_similarity": (
                arguments.minimum_prototype_similarity
            ),
            "minimum_median_margin": (
                arguments.minimum_median_margin
            ),
            "minimum_positive_margin_support": (
                arguments.minimum_positive_margin_support
            ),
            "minimum_top1_top2_gap": (
                arguments.minimum_top1_top2_gap
            ),
            "minimum_plausible_score": (
                arguments.minimum_plausible_score
            ),
            "minimum_plausible_prototype": (
                arguments.minimum_plausible_prototype
            ),
            "threshold_origin": (
                "CONSERVATIVE_DEVELOPMENT_"
                "POLICY_NOT_GENERALIZATION_VALIDATED"
            ),
        },
        "authorization": {
            "cross_shot_link_created": (
                False
            ),
            "target_memory_updated": (
                False
            ),
            "user_confirmation_required": (
                operational_state
                == "AMBIGUOUS"
            ),
        },
    }

    decision_path = (
        output_dir
        / "stage3b2_decision.json"
    )

    atomic_json(
        decision_path,
        decision_payload,
    )

    summary = {
        "stage": (
            STAGE
        ),
        "version": (
            VERSION
        ),
        "generated_at": (
            decision_payload[
                "generated_at"
            ]
        ),
        "status": (
            "PASS"
        ),
        "decision": (
            decision
        ),
        "operational_state": (
            operational_state
        ),
        "test_name": (
            test_name
        ),
        "automatic_decision": (
            automatic_decision
        ),
        "evaluation": {
            "label_source": (
                "STAGE3B1_USER_VISUAL_"
                "REVIEW_EVALUATION_ONLY"
            ),
            "manual_labels_read_after_"
            "automatic_decision": (
                True
            ),
            "target_candidate_ids": sorted(
                evaluation_target_ids
            ),
            "top_candidate_is_target": (
                top_candidate_is_target
            ),
            "automatic_wrong_link_prevented": (
                automatic_wrong_link_prevented
            ),
            "automatic_correct_candidate_proposed": (
                automatic_correct_candidate_proposed
            ),
        },
        "counts": {
            "ranked_candidate_count": (
                len(
                    ranked_rows
                )
            ),
            "passed_auto_gate_count": sum(
                bool(
                    gate["passed"]
                )
                for gate in gates.values()
            ),
            "total_auto_gate_count": (
                len(
                    gates
                )
            ),
            "review_candidate_count": min(
                arguments.review_candidate_count,
                len(
                    ranked_rows
                ),
            ),
        },
        "inputs": {
            "stage3b0_summary": str(
                stage3b0_path
            ),
            "stage3b1_summary": str(
                stage3b1_path
            ),
            "ranked_candidates": str(
                ranked_path
            ),
            "ranked_candidates_sha256": (
                sha256_file(
                    ranked_path
                )
            ),
            "evaluation": str(
                evaluation_path
            ),
        },
        "outputs": {
            "decision": str(
                decision_path
            ),
            "gate_audit": str(
                gate_audit_path
            ),
            "review_candidates": str(
                review_sheet_path
            ),
        },
        "safety_invariants": {
            "manual_target_labels_used_for_decision": (
                False
            ),
            "manual_target_labels_read_only_after_decision": (
                True
            ),
            "target_candidate_selected": (
                operational_state
                == "AUTO_REACQUIRED_CANDIDATE"
            ),
            "cross_shot_link_created": (
                False
            ),
            "target_memory_updated": (
                False
            ),
            "tracklets_merged": (
                False
            ),
            "frozen_phase1_modified": (
                False
            ),
            "threshold_search_performed": (
                False
            ),
            "policy_generalization_validated": (
                False
            ),
        },
    }

    summary_path = (
        output_dir
        / "stage3b2_summary.json"
    )

    atomic_json(
        summary_path,
        summary,
    )

    atomic_text(
        output_dir
        / "stage3b2_report.md",
        build_report(
            summary
        ),
    )

    print(
        "KickClip Target-Centric Tracking V2 "
        "Stage 3-B2 complete"
    )

    print(
        "Status                         : PASS"
    )

    print(
        f"Decision                       : "
        f"{decision}"
    )

    print(
        f"Operational state              : "
        f"{operational_state}"
    )

    print(
        f"Top automatic candidate        : "
        f"{top_candidate_id}"
    )

    print(
        f"Top retrieval score            : "
        f"{top1_score:.6f}"
    )

    print(
        f"Top prototype similarity       : "
        f"{top1_prototype:.6f}"
    )

    print(
        f"Top median margin              : "
        f"{top1_margin:.6f}"
    )

    print(
        f"Top positive support           : "
        f"{top1_support:.3f}"
    )

    print(
        f"Top1 / Top2 score gap          : "
        f"{top1_top2_gap:.6f}"
    )

    print(
        "Automatic gates:"
    )

    for name, gate in gates.items():
        print(
            f"  {name:<34}: "
            f"{bool_text(bool(gate['passed']))} "
            f"(value={gate['value']}, "
            f"required={gate['required']})"
        )

    print(
        f"Evaluation top candidate target: "
        f"{top_candidate_is_target}"
    )

    print(
        f"Wrong auto-link prevented      : "
        f"{automatic_wrong_link_prevented}"
    )

    print(
        f"Automatically proposed target  : "
        f"{automatically_proposed_candidate}"
    )

    print(
        "Cross-shot link created        : NONE"
    )

    print(
        "Target memory update           : NONE"
    )

    print(
        "Policy status                  : "
        "DEVELOPMENT_ONLY_NOT_"
        "GENERALIZATION_VALIDATED"
    )

    print(
        f"Review candidates              : "
        f"{review_sheet_path}"
    )

    print(
        f"Output                         : "
        f"{output_dir}"
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(
            main()
        )

    except KeyboardInterrupt:
        print(
            "Stage 3-B2 interrupted",
            file=sys.stderr,
        )

        raise SystemExit(
            130
        )

    except Exception as exc:
        print(
            "Stage 3-B2 fatal error: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )

        raise SystemExit(
            2
        )