from pathlib import Path
from typing import Any

import yaml

from app.ai.registry.model_card import ModelCard
from app.core.paths import get_project_root


class ModelRegistry:
    """Load AI model registry config from YAML.

    실제 모델 checkpoint path를 코드에 하드코딩하지 않기 위해 사용한다.
    6회차에서는 dummy task 검증용으로 사용하고,
    7회차부터 highlight_spotting champion 모델 정보를 여기서 읽는다.
    """

    def __init__(self, registry_path: str | Path = "configs/model_registry.yaml"):
        self.project_root = get_project_root()
        self.registry_path = self._resolve_path(registry_path)
        self._raw: dict[str, Any] = {}
        self.reload()

    def reload(self) -> None:
        if not self.registry_path.exists():
            self._raw = {"models": {}}
            return

        with self.registry_path.open("r", encoding="utf-8") as file:
            loaded = yaml.safe_load(file) or {}

        if not isinstance(loaded, dict):
            raise ValueError("model_registry.yaml must contain a YAML object.")

        self._raw = loaded

    def get_model_card(
        self,
        task_type: str,
        alias: str = "champion",
    ) -> ModelCard:
        models = self._raw.get("models") or {}
        task_models = models.get(task_type) or {}
        entry = task_models.get(alias)

        if entry is None:
            raise KeyError(
                f"Model registry entry not found: task_type={task_type}, alias={alias}"
            )

        if not isinstance(entry, dict):
            raise ValueError(
                f"Model registry entry must be a dict: task_type={task_type}, alias={alias}"
            )

        known_fields = {
            "id",
            "model_name",
            "model_version",
            "description",
            "checkpoint_path",
            "config_path",
            "enabled",
        }
        extra = {key: value for key, value in entry.items() if key not in known_fields}

        return ModelCard(
            id=entry["id"],
            model_name=entry["model_name"],
            model_version=entry.get("model_version"),
            description=entry.get("description"),
            checkpoint_path=entry.get("checkpoint_path"),
            config_path=entry.get("config_path"),
            enabled=entry.get("enabled", True),
            extra=extra,
        )

    def get_raw(self) -> dict[str, Any]:
        return self._raw

    def _resolve_path(self, path: str | Path) -> Path:
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate
        return self.project_root / candidate