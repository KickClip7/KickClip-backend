from app.domains.analysis.job_status import COMPLETED, QUEUED


DEFAULT_FULL_MATCH_STEPS = [
    {
        "step_key": "upload",
        "label": "영상 업로드",
        "status": COMPLETED,
        "progress": 100,
    },
    {
        "step_key": "feature_extraction",
        "label": "영상 피처 추출",
        "status": QUEUED,
        "progress": 0,
    },
    {
        "step_key": "event_classification",
        "label": "하이라이트 이벤트 추론",
        "status": QUEUED,
        "progress": 0,
    },
    {
        "step_key": "player_tracking",
        "label": "선수 인식",
        "status": QUEUED,
        "progress": 0,
    },
    {
        "step_key": "scoring",
        "label": "스코어링",
        "status": QUEUED,
        "progress": 0,
    },
]


def get_default_steps_for_job_type(job_type: str) -> list[dict]:
    if job_type == "FULL_MATCH_ANALYSIS":
        return [step.copy() for step in DEFAULT_FULL_MATCH_STEPS]

    if job_type == "HIGHLIGHT_SPOTTING":
        return [
            {
                "step_key": "event_classification",
                "label": "이벤트 분류",
                "status": QUEUED,
                "progress": 0,
            },
            {
                "step_key": "scoring",
                "label": "스코어링",
                "status": QUEUED,
                "progress": 0,
            },
        ]

    if job_type == "PLAYER_TRACKING":
        return [
            {
                "step_key": "player_tracking",
                "label": "선수 인식",
                "status": QUEUED,
                "progress": 0,
            },
        ]

    if job_type == "BALL_TRACKING":
        return [
            {
                "step_key": "ball_tracking",
                "label": "공 추적",
                "status": QUEUED,
                "progress": 0,
            },
        ]

    if job_type == "TIMELINE_FUSION":
        return [
            {
                "step_key": "timeline_fusion",
                "label": "타임라인 통합",
                "status": QUEUED,
                "progress": 0,
            },
        ]

    return []


def get_first_pending_step_key(steps: list[dict]) -> str | None:
    for step in steps:
        if step.get("status") != COMPLETED:
            return step.get("step_key")
    return None
