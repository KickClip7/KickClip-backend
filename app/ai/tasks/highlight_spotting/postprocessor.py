from typing import Any

from app.ai.tasks.highlight_spotting.label_map import (
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
        min_duration_sec: float = 4.0,
        max_duration_sec: float = 12.0,
        pre_event_sec: float = 4.0,
        post_event_sec: float = 4.0,
        class_thresholds: dict[str, float] | None = None,
    ):
        self.default_threshold = default_threshold
        self.max_candidates = max_candidates
        self.nms_window_sec = nms_window_sec
        self.min_duration_sec = min_duration_sec
        self.max_duration_sec = max_duration_sec
        self.pre_event_sec = pre_event_sec
        self.post_event_sec = post_event_sec
        self.class_thresholds = class_thresholds or {}

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

        nms_applied.sort(key=lambda item: item["timestamp_sec"])

        return nms_applied[: self.max_candidates]

    def _normalize_candidate(
        self,
        candidate: dict[str, Any],
        match_duration_sec: float | None,
    ) -> dict[str, Any]:
        label = normalize_label(str(candidate.get("label") or "shot"))
        confidence = float(candidate.get("confidence") or 0.0)

        timestamp_sec = float(candidate.get("timestamp_sec") or 0.0)
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
                    overlaps = True
                    break

            if not overlaps:
                selected.append(candidate)

        return selected

    @staticmethod
    def _build_title(prefix: str, timestamp_sec: float) -> str:
        minute = int(timestamp_sec // 60)
        second = int(timestamp_sec % 60)
        return f"{prefix} {minute:02d}:{second:02d}"