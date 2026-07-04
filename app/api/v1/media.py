from collections.abc import Iterator
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.domains.media.model import MediaAsset
from app.domains.media.schema import MediaAssetRead, MediaAssetResponse
from app.domains.media.service import MediaAssetService
from app.storage.local_storage import LocalStorage


router = APIRouter()

CHUNK_SIZE = 1024 * 1024


@router.get(
    "/{asset_id}",
    response_model=MediaAssetResponse,
    summary="미디어 에셋 단건 조회",
)
def get_media_asset(
    asset_id: str,
    db: Session = Depends(get_db),
) -> MediaAssetResponse:
    service = MediaAssetService(db)
    asset = service.get_media_asset(asset_id)

    if asset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Media asset not found",
        )

    base = MediaAssetRead.model_validate(asset)
    return MediaAssetResponse(
        **base.model_dump(),
        stream_url=f"/api/v1/media/{asset.asset_id}/stream",
        download_url=f"/api/v1/media/{asset.asset_id}/download",
    )


@router.get(
    "/{asset_id}/stream",
    summary="미디어 파일 스트리밍",
)
def stream_media_asset(
    asset_id: str,
    request: Request,
    db: Session = Depends(get_db),
) -> StreamingResponse:
    service = MediaAssetService(db)
    asset = _get_asset_or_404(service, asset_id)
    file_path = _resolve_asset_file_path_or_404(asset)

    return _build_range_response(
        request=request,
        file_path=file_path,
        media_type=asset.mime_type or "application/octet-stream",
    )


@router.head(
    "/{asset_id}/stream",
    summary="미디어 파일 스트리밍 HEAD 확인",
)
def head_stream_media_asset(
    asset_id: str,
    db: Session = Depends(get_db),
) -> Response:
    service = MediaAssetService(db)
    asset = _get_asset_or_404(service, asset_id)
    file_path = _resolve_asset_file_path_or_404(asset)

    return Response(
        status_code=status.HTTP_200_OK,
        headers={
            "Accept-Ranges": "bytes",
            "Content-Length": str(file_path.stat().st_size),
            "Content-Type": asset.mime_type or "application/octet-stream",
        },
    )


@router.get(
    "/{asset_id}/download",
    summary="미디어 파일 다운로드",
)
def download_media_asset(
    asset_id: str,
    db: Session = Depends(get_db),
) -> FileResponse:
    service = MediaAssetService(db)
    asset = _get_asset_or_404(service, asset_id)
    file_path = _resolve_asset_file_path_or_404(asset)

    filename = asset.original_filename or file_path.name

    return FileResponse(
        path=file_path,
        media_type=asset.mime_type or "application/octet-stream",
        filename=filename,
    )


def _get_asset_or_404(
    service: MediaAssetService,
    asset_id: str,
) -> MediaAsset:
    asset = service.get_media_asset(asset_id)

    if asset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Media asset not found",
        )

    return asset


def _resolve_asset_file_path_or_404(asset: MediaAsset) -> Path:
    storage = LocalStorage()
    file_path = storage.resolve_path(asset.file_path)

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Media file not found on storage.",
        )

    return file_path


def _build_range_response(
    request: Request,
    file_path: Path,
    media_type: str,
) -> StreamingResponse:
    file_size = file_path.stat().st_size
    range_header = request.headers.get("range")

    if file_size <= 0:
        return StreamingResponse(
            iter(()),
            status_code=status.HTTP_200_OK,
            media_type=media_type,
            headers={
                "Accept-Ranges": "bytes",
                "Content-Length": "0",
            },
        )

    if range_header:
        start, end = _parse_range_header(range_header, file_size)
        content_length = end - start + 1

        headers = {
            "Accept-Ranges": "bytes",
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Content-Length": str(content_length),
            "Content-Type": media_type,
        }

        return StreamingResponse(
            _iter_file_range(file_path, start, end),
            status_code=status.HTTP_206_PARTIAL_CONTENT,
            media_type=media_type,
            headers=headers,
        )

    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(file_size),
        "Content-Type": media_type,
    }

    return StreamingResponse(
        _iter_file_range(file_path, 0, file_size - 1),
        status_code=status.HTTP_200_OK,
        media_type=media_type,
        headers=headers,
    )


def _parse_range_header(range_header: str, file_size: int) -> tuple[int, int]:
    if not range_header.startswith("bytes="):
        _raise_range_not_satisfiable(file_size, "Invalid Range header.")

    range_value = range_header.replace("bytes=", "", 1).strip()

    if "," in range_value:
        _raise_range_not_satisfiable(
            file_size,
            "Multiple ranges are not supported.",
        )

    parts = range_value.split("-", maxsplit=1)
    if len(parts) != 2:
        _raise_range_not_satisfiable(file_size, "Invalid Range format.")

    start_text, end_text = parts

    try:
        if start_text == "":
            if end_text == "":
                _raise_range_not_satisfiable(file_size, "Invalid suffix range.")

            suffix_length = int(end_text)
            if suffix_length <= 0:
                _raise_range_not_satisfiable(file_size, "Invalid suffix range.")

            start = max(file_size - suffix_length, 0)
            end = file_size - 1
            return start, end

        start = int(start_text)
        end = int(end_text) if end_text else file_size - 1

    except ValueError:
        _raise_range_not_satisfiable(file_size, "Invalid Range number.")

    if start < 0 or start >= file_size or end < start:
        _raise_range_not_satisfiable(
            file_size,
            "Requested range is not satisfiable.",
        )

    end = min(end, file_size - 1)

    return start, end


def _raise_range_not_satisfiable(file_size: int, detail: str) -> None:
    raise HTTPException(
        status_code=status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE,
        detail=detail,
        headers={"Content-Range": f"bytes */{file_size}"},
    )


def _iter_file_range(
    file_path: Path,
    start: int,
    end: int,
) -> Iterator[bytes]:
    with file_path.open("rb") as file:
        file.seek(start)
        remaining = end - start + 1

        while remaining > 0:
            chunk_size = min(CHUNK_SIZE, remaining)
            data = file.read(chunk_size)

            if not data:
                break

            remaining -= len(data)
            yield data