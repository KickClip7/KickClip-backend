from __future__ import annotations

import base64
import json
import re
import subprocess
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.domains.agent.schema import (
    ExportAssistantRecommendation,
    ExportMetadataRecommendResponse,
    ThumbnailRecommendation,
)
from app.domains.clip_plan.model import ClipPlan
from app.domains.clip_plan.repository import ClipPlanRepository
from app.domains.media.repository import MediaAssetRepository
from app.storage.local_storage import LocalStorage


@dataclass(frozen=True)
class SampledFrame:
    index: int
    edited_timestamp_sec: float
    source_timestamp_sec: float
    path: Path


class QwenExportAssistant:
    """Create publishing suggestions from sampled frames of the edited clip."""

    def __init__(self, db: Session):
        self.clip_plans = ClipPlanRepository(db)
        self.media_assets = MediaAssetRepository(db)
        self.storage = LocalStorage()
        self.settings = get_settings()

    def recommend(
        self,
        *,
        clip_plan_id: str | None = None,
        match_id: str | None = None,
        language: str = "ko",
    ) -> ExportMetadataRecommendResponse:
        payload = self._recommend_payload(
            clip_plan_id=clip_plan_id,
            match_id=match_id,
            language=language,
            targets={"title", "hashtags", "thumbnail"},
        )
        return ExportMetadataRecommendResponse.model_validate(payload)

    def recommend_selected(
        self,
        *,
        targets: list[str],
        clip_plan_id: str | None = None,
        match_id: str | None = None,
        language: str = "ko",
    ) -> ExportAssistantRecommendation:
        normalized_targets = list(dict.fromkeys(targets))
        allowed_targets = {"title", "hashtags", "thumbnail"}
        if not normalized_targets or any(target not in allowed_targets for target in normalized_targets):
            raise ValueError("추천 항목은 title, hashtags, thumbnail 중 하나 이상이어야 합니다.")
        payload = self._recommend_payload(
            clip_plan_id=clip_plan_id,
            match_id=match_id,
            language=language,
            targets=set(normalized_targets),
        )
        payload["requested_fields"] = normalized_targets
        return ExportAssistantRecommendation.model_validate(payload)

    def _recommend_payload(
        self,
        *,
        clip_plan_id: str | None,
        match_id: str | None,
        language: str,
        targets: set[str],
    ) -> dict[str, Any]:
        clip_plan = self.clip_plans.get_by_id(clip_plan_id) if clip_plan_id else None
        if clip_plan_id and clip_plan is None:
            raise ValueError("ClipPlan not found")
        if clip_plan is not None and not clip_plan.items:
            raise ValueError("추천할 편집 장면이 없습니다.")

        source_match_id = (
            clip_plan.project.match_id if clip_plan is not None else match_id
        )
        if not source_match_id:
            raise ValueError("분석할 경기 영상 정보가 없습니다.")
        source_path, source_duration = self._source_video(source_match_id)
        with tempfile.TemporaryDirectory(prefix="kickclip-qwen-") as temp_dir:
            if clip_plan is not None:
                points = self._sample_points(clip_plan, max(3, self.settings.QWEN_EXPORT_SAMPLE_FRAMES))
                summary = clip_plan.summary or "편집된 축구 하이라이트"
                timeline_name = "편집본"
            else:
                points = self._full_video_sample_points(
                    source_duration,
                    max(3, self.settings.QWEN_EXPORT_SAMPLE_FRAMES),
                )
                summary = "업로드된 축구 경기 원본 영상"
                timeline_name = "원본"

            frames = self._extract_frames(points, source_path, Path(temp_dir))
            result = self._run_qwen(frames, summary, timeline_name, language, targets)
            payload: dict[str, Any] = {
                "model_id": self.settings.QWEN_MODEL_ID,
                "sampled_frame_count": len(frames),
            }
            if "title" in targets:
                payload["title"] = str(result.get("title") or summary or "KickClip Highlight").strip()
            if "hashtags" in targets:
                payload["hashtags"] = self._normalize_hashtags(result.get("hashtags"))
            if "thumbnail" in targets:
                selected = self._select_frame(frames, result.get("thumbnail_candidate"))
                payload["thumbnail"] = ThumbnailRecommendation(
                    timestamp_sec=round(selected.edited_timestamp_sec, 2),
                    source_timestamp_sec=round(selected.source_timestamp_sec, 2),
                    timestamp_label=self._format_timestamp(selected.edited_timestamp_sec),
                    reason=str(result.get("thumbnail_reason") or "경기 흐름이 가장 잘 드러나는 장면입니다.").strip(),
                    image_data_url=self._as_data_url(selected.path),
                ).model_dump(mode="json")
            return payload

    def _extract_frames(
        self,
        points: list[tuple[float, float]],
        source_path: Path,
        output_dir: Path,
    ) -> list[SampledFrame]:
        frames: list[SampledFrame] = []
        for index, (edited_sec, source_sec) in enumerate(points, start=1):
            frame_path = output_dir / f"candidate_{index:02d}.jpg"
            command = [
                "ffmpeg", "-y", "-ss", f"{source_sec:.3f}", "-i", source_path.as_posix(),
                "-frames:v", "1", "-vf", "scale=896:-2", "-q:v", "3", frame_path.as_posix(),
            ]
            completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if completed.returncode == 0 and frame_path.exists():
                frames.append(SampledFrame(index, edited_sec, source_sec, frame_path))
        if not frames:
            raise RuntimeError("영상에서 Qwen 분석용 프레임을 추출하지 못했습니다. FFmpeg와 원본 영상을 확인해주세요.")
        return frames

    @staticmethod
    def _full_video_sample_points(duration_sec: float, count: int) -> list[tuple[float, float]]:
        if duration_sec <= 0:
            raise ValueError("업로드 영상의 길이를 확인할 수 없습니다.")
        margin = min(1.0, duration_sec / 10)
        points = [
            margin + (duration_sec - margin * 2) * ((index + 0.5) / count)
            for index in range(count)
        ]
        return [(point, point) for point in points]

    @staticmethod
    def _sample_points(clip_plan: ClipPlan, count: int) -> list[tuple[float, float]]:
        spans: list[tuple[float, float, float]] = []
        cursor = 0.0
        for item in sorted(clip_plan.items, key=lambda row: row.order_index):
            duration = max(0.0, float(item.end_sec) - float(item.start_sec))
            if duration > 0:
                spans.append((cursor, float(item.start_sec), duration))
                cursor += duration
        if cursor <= 0:
            raise ValueError("ClipPlan에 유효한 영상 구간이 없습니다.")

        margin = min(0.5, cursor / 10)
        targets = [margin + (cursor - margin * 2) * ((index + 0.5) / count) for index in range(count)]
        points: list[tuple[float, float]] = []
        for edited_sec in targets:
            for span_start, source_start, duration in spans:
                if edited_sec <= span_start + duration:
                    points.append((edited_sec, source_start + max(0.0, edited_sec - span_start)))
                    break
        return points

    def _run_qwen(
        self,
        frames: list[SampledFrame],
        summary: str,
        timeline_name: str,
        language: str,
        targets: set[str],
    ) -> dict[str, Any]:
        model, processor, device = _load_qwen_runtime(self.settings.QWEN_MODEL_ID)
        target_names = {
            "title": "제목",
            "hashtags": "해시태그",
            "thumbnail": "썸네일 후보",
        }
        requested = ", ".join(target_names[target] for target in ("title", "hashtags", "thumbnail") if target in targets)
        response_fields: list[str] = []
        if "title" in targets:
            response_fields.append('"title":"..."')
        if "hashtags" in targets:
            response_fields.append('"hashtags":["#축구"]')
        if "thumbnail" in targets:
            response_fields.extend(['"thumbnail_candidate":1', '"thumbnail_reason":"..."'])
        response_example = "{" + ",".join(response_fields) + "}"
        content: list[dict[str, Any]] = [{
            "type": "text",
            "text": (
                "당신은 축구 하이라이트 숏폼 콘텐츠 에디터입니다. 이어지는 이미지는 편집본에서 시간순으로 뽑은 후보 프레임입니다. "
                f"요청된 항목({requested})만 생성하고 요청되지 않은 항목은 분석하거나 응답에 포함하지 마세요. "
                "제목은 시청을 유도하는 한국어 한 문장, 해시태그는 검색에 유용한 6~10개, 썸네일은 가장 강한 후보 번호와 이유를 뜻합니다. "
                f"반드시 {response_example} 형식의 JSON 객체만 출력하세요. "
                f"영상 범위: {summary}, 응답 언어: {language}."
            ),
        }]
        for frame in frames:
            content.extend([
                {"type": "image", "image": frame.path.as_uri()},
                {"type": "text", "text": f"후보 {frame.index}: {timeline_name} {self._format_timestamp(frame.edited_timestamp_sec)}"},
            ])

        messages = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        from qwen_vl_utils import process_vision_info

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        ).to(device)
        generated = model.generate(**inputs, max_new_tokens=self.settings.QWEN_MAX_NEW_TOKENS)
        trimmed = [out[len(source):] for source, out in zip(inputs.input_ids, generated)]
        output = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return self._parse_json(output)

    def _source_video(self, match_id: str) -> tuple[Path, float]:
        assets = self.media_assets.list_by_match(match_id)
        for asset_type in ("RAW_VIDEO", "RAW_VIDEO_HALF1", "WEB_PREVIEW_VIDEO"):
            asset = next((row for row in assets if row.asset_type == asset_type), None)
            if asset is not None:
                path = self.storage.resolve_path(asset.file_path)
                if path.exists():
                    duration = float(asset.duration_sec or 0)
                    return path, duration or self._probe_duration(path)
        raise ValueError("Qwen이 분석할 원본 영상을 찾지 못했습니다.")

    @staticmethod
    def _probe_duration(path: Path) -> float:
        completed = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", path.as_posix(),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            return float(completed.stdout.strip())
        except (TypeError, ValueError) as exc:
            raise RuntimeError("업로드 영상의 재생 시간을 확인하지 못했습니다.") from exc

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.IGNORECASE).strip()
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise RuntimeError(f"Qwen 응답을 JSON으로 해석하지 못했습니다: {text[:240]}")
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Qwen이 유효하지 않은 JSON을 반환했습니다: {text[:240]}") from exc

    @staticmethod
    def _select_frame(frames: list[SampledFrame], candidate: Any) -> SampledFrame:
        try:
            index = int(candidate)
        except (TypeError, ValueError):
            index = max(1, len(frames) // 2)
        return next((frame for frame in frames if frame.index == index), frames[len(frames) // 2])

    @staticmethod
    def _normalize_hashtags(value: Any) -> list[str]:
        values = value if isinstance(value, list) else re.split(r"[,\s]+", str(value or ""))
        normalized: list[str] = []
        for item in values:
            tag = re.sub(r"\s+", "", str(item).strip())
            if not tag:
                continue
            tag = tag if tag.startswith("#") else f"#{tag}"
            if tag not in normalized:
                normalized.append(tag)
        return normalized[:10] or ["#축구", "#하이라이트", "#KickClip"]

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        total = max(0, int(round(seconds)))
        return f"{total // 60:02d}:{total % 60:02d}"

    @staticmethod
    def _as_data_url(path: Path) -> str:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"


@lru_cache(maxsize=1)
def _load_qwen_runtime(model_id: str):
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(
        model_id,
        min_pixels=128 * 28 * 28,
        max_pixels=512 * 28 * 28,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype="auto", low_cpu_mem_usage=True
    ).to(device)
    model.eval()
    return model, processor, device
