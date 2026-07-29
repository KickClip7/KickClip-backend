from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any


PASSWORD_ITERATIONS = 600_000


class InvalidTokenError(ValueError):
    pass


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PASSWORD_ITERATIONS,
    )
    return "pbkdf2_sha256${}${}${}".format(
        PASSWORD_ITERATIONS,
        _b64encode(salt),
        _b64encode(digest),
    )


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, iterations_text, salt_text, digest_text = encoded.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        expected = _b64decode(digest_text)
        actual = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            _b64decode(salt_text),
            int(iterations_text),
        )
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return False


def create_access_token(
    *,
    user_id: str,
    secret_key: str,
    expires_minutes: int,
) -> tuple[str, int]:
    now = datetime.now(timezone.utc)
    expires = now + timedelta(minutes=expires_minutes)
    payload = {
        "sub": user_id,
        "type": "access",
        "iat": int(now.timestamp()),
        "exp": int(expires.timestamp()),
        "jti": secrets.token_hex(16),
    }
    return encode_jwt(payload, secret_key), expires_minutes * 60


def decode_access_token(token: str, secret_key: str) -> dict[str, Any]:
    payload = decode_jwt(token, secret_key)
    if payload.get("type") != "access" or not payload.get("sub"):
        raise InvalidTokenError("invalid access token payload")
    return payload


def create_media_token(
    *,
    asset_id: str,
    user_id: str,
    secret_key: str,
    expires_minutes: int,
) -> tuple[str, int]:
    now = datetime.now(timezone.utc)
    expires = now + timedelta(minutes=expires_minutes)
    payload = {
        "sub": user_id,
        "asset_id": asset_id,
        "type": "media",
        "iat": int(now.timestamp()),
        "exp": int(expires.timestamp()),
        "jti": secrets.token_hex(16),
    }
    return encode_jwt(payload, secret_key), expires_minutes * 60


def decode_media_token(
    token: str,
    *,
    asset_id: str,
    secret_key: str,
) -> dict[str, Any]:
    payload = decode_jwt(token, secret_key)
    if payload.get("type") != "media" or payload.get("asset_id") != asset_id:
        raise InvalidTokenError("invalid media token scope")
    return payload


def create_artifact_token(
    *,
    artifact_id: str,
    user_id: str,
    secret_key: str,
    expires_minutes: int,
) -> tuple[str, int]:
    now = datetime.now(timezone.utc)
    expires = now + timedelta(minutes=expires_minutes)
    payload = {
        "sub": user_id,
        "artifact_id": artifact_id,
        "type": "artifact",
        "iat": int(now.timestamp()),
        "exp": int(expires.timestamp()),
        "jti": secrets.token_hex(16),
    }
    return encode_jwt(payload, secret_key), expires_minutes * 60


def decode_artifact_token(
    token: str,
    *,
    artifact_id: str,
    secret_key: str,
) -> dict[str, Any]:
    payload = decode_jwt(token, secret_key)
    if (
        payload.get("type") != "artifact"
        or payload.get("artifact_id") != artifact_id
    ):
        raise InvalidTokenError("invalid artifact token scope")
    return payload


def encode_jwt(payload: dict[str, Any], secret_key: str) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    signing_input = "{}.{}".format(
        _b64encode_json(header),
        _b64encode_json(payload),
    )
    signature = hmac.new(
        secret_key.encode("utf-8"),
        signing_input.encode("ascii"),
        hashlib.sha256,
    ).digest()
    return f"{signing_input}.{_b64encode(signature)}"


def decode_jwt(token: str, secret_key: str) -> dict[str, Any]:
    try:
        header_text, payload_text, signature_text = token.split(".", 2)
        signing_input = f"{header_text}.{payload_text}"
        expected = hmac.new(
            secret_key.encode("utf-8"),
            signing_input.encode("ascii"),
            hashlib.sha256,
        ).digest()
        if not hmac.compare_digest(expected, _b64decode(signature_text)):
            raise InvalidTokenError("invalid token signature")
        header = json.loads(_b64decode(header_text))
        payload = json.loads(_b64decode(payload_text))
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise InvalidTokenError("malformed token") from exc

    if header.get("alg") != "HS256":
        raise InvalidTokenError("unsupported token algorithm")
    now = int(datetime.now(timezone.utc).timestamp())
    if not isinstance(payload.get("exp"), int) or payload["exp"] <= now:
        raise InvalidTokenError("token expired")
    return payload


def create_refresh_token() -> str:
    return secrets.token_urlsafe(48)


def hash_refresh_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def refresh_expiry(days: int) -> datetime:
    return datetime.now(timezone.utc) + timedelta(days=days)


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _b64decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


def _b64encode_json(value: dict[str, Any]) -> str:
    return _b64encode(
        json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )
