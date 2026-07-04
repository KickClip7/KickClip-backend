from __future__ import annotations

from app.ai.tasks.player_tracking.types import DetectedPlayerSeed, PlayerTrackingInput


class DummyPlayerDetector:
    """Dummy player detector.

    실제 detector가 없으므로 TimelineEvent.player_ids에서 선수 후보를 추출한다.
    후보가 부족해도 프론트 선수별 모드 테스트를 위해 기본 선수 몇 명을 생성한다.
    """

    name = "dummy_player_detector_v2"

    def detect(self, task_input: PlayerTrackingInput) -> list[DetectedPlayerSeed]:
        raw_ids: set[str] = set()

        for event in task_input.events:
            for player_id in event.player_ids or []:
                raw_ids.add(str(player_id))

        defaults = {
            "player_007",
            "player_010",
            "player_009",
            "player_005",
        }
        raw_ids.update(defaults)

        return [
            self._build_seed(
                raw_player_id=raw_id,
                home_team=task_input.home_team,
                away_team=task_input.away_team,
            )
            for raw_id in sorted(raw_ids)
        ]

    def _build_seed(
        self,
        raw_player_id: str,
        home_team: str | None,
        away_team: str | None,
    ) -> DetectedPlayerSeed:
        number = self._extract_number(raw_player_id)

        role_map = {
            5: "DF",
            7: "FW",
            9: "FW",
            10: "MF",
        }

        team_hint = away_team if number == 5 else home_team
        display_name = f"{number}번 선수" if number is not None else "미확인 선수"

        return DetectedPlayerSeed(
            raw_player_id=raw_player_id,
            number=number,
            display_name=display_name,
            team_hint=team_hint,
            role=role_map.get(number),
            confidence=1.0,
            metadata={"source": self.name},
        )

    @staticmethod
    def _extract_number(raw_player_id: str) -> int | None:
        digits = "".join(ch for ch in raw_player_id if ch.isdigit())
        if not digits:
            return None

        try:
            return int(digits[-3:])
        except ValueError:
            return None


class UnimplementedRealPlayerDetector:
    """Placeholder for model-backed detector.

    15회차의 목적은 dummy를 실제 모델로 즉시 교체하는 것이 아니라,
    실제 detector가 들어갈 인터페이스를 고정하는 것이다.
    """

    name = "real_player_detector_placeholder"

    def detect(self, task_input: PlayerTrackingInput) -> list[DetectedPlayerSeed]:
        raise NotImplementedError(
            "Real player detector is not implemented yet. "
            "Set player_tracking.mode=dummy until a detector model is connected."
        )


def create_player_detector(mode: str = "dummy"):
    normalized = (mode or "dummy").lower()
    if normalized == "dummy":
        return DummyPlayerDetector()
    if normalized in {"real", "model"}:
        return UnimplementedRealPlayerDetector()
    raise ValueError(f"Unsupported player detector mode: {mode}")
