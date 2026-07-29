from app.core.config import Settings, get_settings
from app.domains.auth.security import create_artifact_token


def build_signed_artifact_url(
    artifact_id: str,
    user_id: str,
    *,
    settings: Settings | None = None,
) -> tuple[str, int]:
    resolved_settings = settings or get_settings()
    token, expires_in = create_artifact_token(
        artifact_id=artifact_id,
        user_id=user_id,
        secret_key=resolved_settings.AUTH_SECRET_KEY,
        expires_minutes=resolved_settings.MEDIA_SIGNED_URL_MINUTES,
    )
    return (
        f"/api/v1/artifacts/{artifact_id}/download?token={token}",
        expires_in,
    )
