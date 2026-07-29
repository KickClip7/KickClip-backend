from app.ai.tasks.highlight_spotting.adapters.base import (
    ChampionAdapterPreflightReport,
    ChampionArtifactPaths,
    ChampionCheckpointSummary,
    ChampionModelSpec,
    HighlightModelAdapter,
    HighlightRawPrediction,
)
from app.ai.tasks.highlight_spotting.adapters.soccer_spotter_v9 import (
    DEFAULT_CHAMPION_MODEL_DIR,
    SoccerSpotterV9Adapter,
    inspect_v9_checkpoint,
    load_v9_model_spec,
    resolve_v9_artifact_paths,
)

__all__ = [
    "ChampionAdapterPreflightReport",
    "ChampionArtifactPaths",
    "ChampionCheckpointSummary",
    "ChampionModelSpec",
    "DEFAULT_CHAMPION_MODEL_DIR",
    "HighlightModelAdapter",
    "HighlightRawPrediction",
    "SoccerSpotterV9Adapter",
    "inspect_v9_checkpoint",
    "load_v9_model_spec",
    "resolve_v9_artifact_paths",
]
