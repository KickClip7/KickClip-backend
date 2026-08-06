from types import SimpleNamespace

from app.domains.highlight.candidate_discovery import (
    PlayerCandidateDiscoveryService,
)


def detection(class_name: str, bbox: list[float]):
    return SimpleNamespace(
        class_name=class_name,
        bbox_xyxy=bbox,
        confidence=0.9,
        class_id=0,
    )


def test_candidate_role_filter_allows_only_player_and_goalkeeper() -> None:
    detections = [
        detection("player", [0.0, 0.0, 20.0, 100.0]),
        detection(" GoalKeeper ", [30.0, 0.0, 50.0, 100.0]),
        detection("referee", [60.0, 0.0, 80.0, 100.0]),
        detection("staff", [90.0, 0.0, 110.0, 100.0]),
        detection("ball", [120.0, 0.0, 140.0, 100.0]),
        detection("", [150.0, 0.0, 170.0, 100.0]),
    ]

    eligible, metrics = (
        PlayerCandidateDiscoveryService._filter_candidate_detections(
            detections,
            frame_height=1000,
        )
    )

    assert [item.class_name.strip().lower() for item in eligible] == [
        "player",
        "goalkeeper",
    ]
    assert metrics["policy_version"] == "PLAYER_GOALKEEPER_ONLY_R1"
    assert metrics["allowed_class_names"] == ["goalkeeper", "player"]
    assert metrics["raw_detection_count"] == 6
    assert metrics["allowed_role_detection_count"] == 2
    assert metrics["eligible_candidate_detection_count"] == 2
    assert metrics["excluded_non_target_role_count"] == 4
    assert metrics["excluded_too_small_count"] == 0
    assert metrics["excluded_role_counts"] == {
        "ball": 1,
        "referee": 1,
        "staff": 1,
        "unknown": 1,
    }


def test_candidate_role_filter_keeps_raw_cache_but_rejects_small_boxes() -> None:
    detections = [
        detection("player", [0.0, 0.0, 10.0, 34.0]),
        detection("goalkeeper", [20.0, 0.0, 30.0, 35.0]),
        detection("staff", [40.0, 0.0, 50.0, 200.0]),
    ]

    eligible, metrics = (
        PlayerCandidateDiscoveryService._filter_candidate_detections(
            detections,
            frame_height=1000,
        )
    )

    assert len(eligible) == 1
    assert eligible[0].class_name == "goalkeeper"
    assert metrics["raw_detection_count"] == 3
    assert metrics["allowed_role_detection_count"] == 2
    assert metrics["eligible_candidate_detection_count"] == 1
    assert metrics["excluded_non_target_role_count"] == 1
    assert metrics["excluded_too_small_count"] == 1
    assert metrics["excluded_role_counts"] == {"staff": 1}
