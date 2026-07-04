from app.ai.tasks.highlight_spotting.adapters.base import (
    ChampionAdapterPreflightReport,
    ChampionArtifactPaths,
    ChampionCheckpointSummary,
    ChampionModelSpec,
    HighlightModelAdapter,
    HighlightRawPrediction,
)
from app.ai.tasks.highlight_spotting.adapters.soccer_highlight_former import (
    DEFAULT_CHAMPION_MODEL_DIR,
    SoccerHighlightFormerAdapter,
    build_sliding_windows,
    inspect_champion_checkpoint,
    load_champion_model_spec,
    resolve_champion_artifact_paths,
)

__all__ = [
    "ChampionAdapterPreflightReport",
    "ChampionArtifactPaths",
    "ChampionCheckpointSummary",
    "ChampionModelSpec",
    "DEFAULT_CHAMPION_MODEL_DIR",
    "HighlightModelAdapter",
    "HighlightRawPrediction",
    "SoccerHighlightFormerAdapter",
    "build_sliding_windows",
    "inspect_champion_checkpoint",
    "load_champion_model_spec",
    "resolve_champion_artifact_paths",
]
