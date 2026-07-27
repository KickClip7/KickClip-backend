SUPPORTED_EVENT_LABELS = [
    "goal",
    "shot",
    "foul",
    "card",
    "free_kick",
    "corner",
]


LABEL_TO_DISPLAY = {
    "goal": {
        "tag": "GOAL",
        "title": "GOAL",
        "description": "A goal event detected by Action Spotting.",
    },
    "shot": {
        "tag": "SHOT",
        "title": "SHOT",
        "description": "A shot event detected by Action Spotting.",
    },
    "foul": {
        "tag": "FOUL",
        "title": "FOUL",
        "description": "A foul event detected by Action Spotting.",
    },
    "card": {
        "tag": "CARD",
        "title": "CARD",
        "description": "A card event detected by Action Spotting.",
    },
    "free_kick": {
        "tag": "FREE KICK",
        "title": "FREE KICK",
        "description": "A free-kick event detected by Action Spotting.",
    },
    "corner": {
        "tag": "CORNER",
        "title": "CORNER",
        "description": "A corner event detected by Action Spotting.",
    },
}


def normalize_label(label: str) -> str:
    normalized = label.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized == "freekick":
        normalized = "free_kick"
    if normalized in {"corner_kick", "cornerkick"}:
        normalized = "corner"
    return normalized


def get_display_info(label: str) -> dict:
    normalized = normalize_label(label)
    return LABEL_TO_DISPLAY.get(
        normalized,
        {
            "tag": normalized.upper().replace("_", " "),
            "title": normalized.upper().replace("_", " "),
            "description": "An event detected by the configured model.",
        },
    )
