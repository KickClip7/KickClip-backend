from pathlib import Path

from app.core.config import get_settings


def get_project_root() -> Path:
    """Return project root directory.

    app/core/paths.py 기준으로:
    app/core/paths.py -> app/core -> app -> project root
    """
    return Path(__file__).resolve().parents[2]


def get_storage_root() -> Path:
    settings = get_settings()
    storage_root = Path(settings.STORAGE_ROOT)

    if storage_root.is_absolute():
        return storage_root

    return get_project_root() / storage_root