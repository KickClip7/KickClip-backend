from typing import Any

from app.ai.tasks.highlight_spotting.label_map import (
    SUPPORTED_EVENT_LABELS,
    get_display_info,
    normalize_label,
)


class HighlightPostprocessor:
    """Postprocess raw highlight predictions.

    7회차에서는 실제 모델 raw output이 없을 수 있으므로,
    predictor가 만든 후보 dict들을 서비스 표준 event candidate로 정규화한다.
    이후 실제 모델 output이 연결되면 이곳에서 threshold, class threshold, NMS를 적용한다.
    """

    def __init__(
        self,
        default_threshold: float = 0.35,
        max_candidates: int = 20,
        nms_window_sec: float = 8.0,
        merge_overlapping_scenes: bool = True,
        min_duration_sec: float = 4.0,
        max_duration_sec: float = 12.0,
        pre_event_sec: float = 4.0,
        post_event_sec: float = 4.0,
        class_thresholds: dict[str, float] | None = None,
        class_priority: dict[str, int] | None = None,
    ):
        self.default_threshold = default_threshold
        self.max_candidates = max_candidates
        self.nms_window_sec = nms_window_sec
        self.merge_overlapping_scenes = merge_overlapping_scenes
        self.min_duration_sec = min_duration_sec
        self.max_duration_sec = max_duration_sec
        self.pre_event_sec = pre_event_sec
        self.post_event_sec = post_event_sec
        self.class_thresholds = class_thresholds or {}
        self.class_priority = class_priority or {}

    def process(
        self,
        raw_predictions: list[dict[str, Any]],
        match_duration_sec: float | None,
    ) -> list[dict[str, Any]]:
        normalized = [
            self._normalize_candidate(candidate, match_duration_sec)
            for candidate in raw_predictions
        ]

        filtered = [
            candidate
            for candidate in normalized
            if candidate["confidence"] >= self._threshold_for(candidate["label"])
        ]

        filtered.sort(
            key=lambda item: (
                item["confidence"],
                item["highlight_score"],
            ),
            reverse=True,
        )

        nms_applied = self._apply_time_nms(filtered)
        scenes = (
            self._merge_overlapping(nms_applied, match_duration_sec)
            if self.merge_overlapping_scenes
            else nms_applied
        )

        scenes.sort(key=lambda item: item["timestamp_sec"])

        return scenes[: self.max_candidates]

    def _normalize_candidate(
        self,
        candidate: dict[str, Any],
        match_duration_sec: float | None,
    ) -> dict[str, Any]:
        raw_label = candidate.get("label")
        if not isinstance(raw_label, str) or not raw_label.strip():
            raise ValueError("Champion prediction is missing an event label")
        label = normalize_label(raw_label)
        if label not in SUPPORTED_EVENT_LABELS:
            raise ValueError(f"unsupported Champion event label: {raw_label!r}")
        confidence = float(candidate.get("confidence") or 0.0)

        raw_timestamp = candidate.get("timestamp_sec")
        if raw_timestamp is None:
            raw_timestamp = candidate.get("time_sec")
        timestamp_sec = float(raw_timestamp or 0.0)
        half = candidate.get("half")

        start_sec = candidate.get("start_sec")
        end_sec = candidate.get("end_sec")

        if start_sec is None:
            start_sec = timestamp_sec - self.pre_event_sec

        if end_sec is None:
            end_sec = timestamp_sec + self.post_event_sec

        start_sec = max(float(start_sec), 0.0)
        end_sec = max(float(end_sec), start_sec + self.min_duration_sec)

        max_end_sec = match_duration_sec if match_duration_sec else None
        if max_end_sec is not None:
            end_sec = min(end_sec, max_end_sec)

        duration_sec = max(end_sec - start_sec, 0.1)

        if duration_sec > self.max_duration_sec:
            end_sec = start_sec + self.max_duration_sec
            if max_end_sec is not None:
                end_sec = min(end_sec, max_end_sec)
            duration_sec = max(end_sec - start_sec, 0.1)

        display = get_display_info(label)
        highlight_score = candidate.get("highlight_score")
        if highlight_score is None:
            highlight_score = round(confidence * 10, 2)

        title = candidate.get("title") or self._build_title(
            display["title"],
            timestamp_sec,
        )

        source_prediction = {
            "time_sec": round(timestamp_sec, 3),
            "label": str(candidate.get("label") or label),
            "confidence": round(confidence, 6),
        }

        return {
            "event_type": label,
            "label": label,
            "half": int(half) if half is not None else None,
            "timestamp_sec": round(timestamp_sec, 3),
            "start_sec": round(start_sec, 3),
            "end_sec": round(end_sec, 3),
            "duration_sec": round(duration_sec, 3),
            "confidence": round(confidence, 4),
            "highlight_score": round(float(highlight_score), 2),
            "title": title,
            "description": candidate.get("description") or display["description"],
            "team_name": candidate.get("team_name"),
            "player_ids": candidate.get("player_ids") or [],
            "metadata": {
                **(candidate.get("metadata") or {}),
                "tag": display["tag"],
                "source": candidate.get("source", "highlight_spotting"),
                "source_predictions": [source_prediction],
                "scene_provenance": {
                    "policy": "highlight_scene_normalization_v1",
                    "default_threshold": self.default_threshold,
                    "class_threshold": self._threshold_for(label),
                    "pre_event_sec": self.pre_event_sec,
                    "post_event_sec": self.post_event_sec,
                    "nms_window_sec": self.nms_window_sec,
                    "max_duration_sec": self.max_duration_sec,
                },
            },
        }

    def _threshold_for(self, label: str) -> float:
        return float(self.class_thresholds.get(label, self.default_threshold))

    def _apply_time_nms(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []

        for candidate in candidates:
            overlaps = False

            for existing in selected:
                same_label = existing["label"] == candidate["label"]
                time_close = (
                    abs(existing["timestamp_sec"] - candidate["timestamp_sec"])
                    <= self.nms_window_sec
                )

                if same_label and time_close:
                    self._append_source_predictions(existing, candidate)
                    overlaps = True
                    break

            if not overlaps:
                selected.append(candidate)

        return selected

    def _merge_overlapping(
        self,
        candidates: list[dict[str, Any]],
        match_duration_sec: float | None,
    ) -> list[dict[str, Any]]:
        """Merge only genuinely overlapping scene windows.

        Same-label temporal NMS runs first. Cross-label events are then merged
        when their existing pre/post-roll windows overlap and the resulting
        interval still obeys the configured clip length limit. This preserves
        Shot+Goal provenance without turning a chain of nearby point
        predictions into one long clip.
        """

        ordered = sorted(candidates, key=lambda item: item["start_sec"])
        scenes: list[dict[str, Any]] = []

        for candidate in ordered:
            if not scenes:
                scenes.append(candidate)
                continue

            previous = scenes[-1]
            merged_start = min(previous["start_sec"], candidate["start_sec"])
            merged_end = max(previous["end_sec"], candidate["end_sec"])
            overlaps = candidate["start_sec"] <= previous["end_sec"]
            within_limit = merged_end - merged_start <= self.max_duration_sec

            if overlaps and not within_limit:
                # Keep both source events but remove duplicate render time.
                # The midpoint between representative timestamps is stable and
                # does not invent a longer clip than the configured limit.
                boundary = (
                    float(previous["timestamp_sec"])
                    + float(candidate["timestamp_sec"])
                ) / 2
                boundary = max(
                    float(candidate["start_sec"]),
                    min(float(previous["end_sec"]), boundary),
                )
                previous["end_sec"] = round(boundary, 3)
                previous["duration_sec"] = round(
                    previous["end_sec"] - previous["start_sec"],
                    3,
                )
                candidate["start_sec"] = round(boundary, 3)
                candidate["duration_sec"] = round(
                    candidate["end_sec"] - candidate["start_sec"],
                    3,
                )
                for item in (previous, candidate):
                    metadata = dict(item.get("metadata") or {})
                    provenance = dict(metadata.get("scene_provenance") or {})
                    provenance.update(
                        {
                            "overlap_resolved": True,
                            "overlap_rule": "split_at_representative_midpoint",
                        }
                    )
                    metadata["scene_provenance"] = provenance
                    item["metadata"] = metadata
                scenes.append(candidate)
                continue

            if not overlaps:
                scenes.append(candidate)
                continue

            primary = max(
                (previous, candidate),
                key=lambda item: (
                    self.class_priority.get(item["label"], 0),
                    item["confidence"],
                ),
            )
            source_predictions = self._combined_source_predictions(
                previous,
                candidate,
            )
            previous.update(
                {
                    "event_type": primary["event_type"],
                    "label": primary["label"],
                    "timestamp_sec": primary["timestamp_sec"],
                    "start_sec": max(0.0, round(merged_start, 3)),
                    "end_sec": round(
                        min(merged_end, match_duration_sec)
                        if match_duration_sec is not None
                        else merged_end,
                        3,
                    ),
                    "confidence": primary["confidence"],
                    "highlight_score": max(
                        previous["highlight_score"],
                        candidate["highlight_score"],
                    ),
                    "title": primary["title"],
                    "description": primary["description"],
                    "team_name": primary.get("team_name"),
                    "player_ids": sorted(
                        set(previous.get("player_ids") or [])
                        | set(candidate.get("player_ids") or [])
                    ),
                }
            )
            previous["duration_sec"] = round(
                previous["end_sec"] - previous["start_sec"],
                3,
            )
            previous_metadata = dict(previous.get("metadata") or {})
            provenance = dict(previous_metadata.get("scene_provenance") or {})
            provenance.update(
                {
                    "merged": True,
                    "merge_rule": "overlapping_pre_post_roll_within_max_duration",
                    "source_prediction_count": len(source_predictions),
                }
            )
            previous_metadata.update(
                {
                    "tag": (primary.get("metadata") or {}).get("tag"),
                    "source_predictions": source_predictions,
                    "scene_provenance": provenance,
                }
            )
            previous["metadata"] = previous_metadata

        return scenes

    @staticmethod
    def _combined_source_predictions(
        *candidates: dict[str, Any],
    ) -> list[dict[str, Any]]:
        rows: dict[tuple[float, str, float], dict[str, Any]] = {}
        for candidate in candidates:
            for prediction in (candidate.get("metadata") or {}).get(
                "source_predictions",
                [],
            ):
                key = (
                    float(prediction["time_sec"]),
                    str(prediction["label"]),
                    float(prediction["confidence"]),
                )
                rows[key] = dict(prediction)
        return sorted(rows.values(), key=lambda item: item["time_sec"])

    def _append_source_predictions(
        self,
        target: dict[str, Any],
        duplicate: dict[str, Any],
    ) -> None:
        metadata = dict(target.get("metadata") or {})
        predictions = self._combined_source_predictions(target, duplicate)
        metadata["source_predictions"] = predictions
        provenance = dict(metadata.get("scene_provenance") or {})
        provenance.update(
            {
                "nms_deduplicated": True,
                "source_prediction_count": len(predictions),
            }
        )
        metadata["scene_provenance"] = provenance
        target["metadata"] = metadata

    @staticmethod
    def _build_title(prefix: str, timestamp_sec: float) -> str:
        minute = int(timestamp_sec // 60)
        second = int(timestamp_sec % 60)
        return f"{prefix} {minute:02d}:{second:02d}"
