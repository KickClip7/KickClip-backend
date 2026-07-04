from typing import Any

from pydantic import BaseModel, Field


class ModelCard(BaseModel):
    id: str
    model_name: str
    model_version: str | None = None
    description: str | None = None
    checkpoint_path: str | None = None
    config_path: str | None = None
    enabled: bool = True

    # highlight_spotting 같은 task에서 추가로 사용할 수 있는 값들
    extra: dict[str, Any] = Field(default_factory=dict)