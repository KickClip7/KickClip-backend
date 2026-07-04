from sqlalchemy.orm import Session

from app.domains.clip_plan.model import ClipPlan
from app.domains.clip_plan.repository import ClipPlanRepository
from app.domains.clip_plan.schema import (
    ClipPlanCreate,
    ClipPlanItemCreate,
    ClipPlanUpdateRequest,
    ExportOptionsUpdateRequest,
    ManualClipPlanCreateRequest,
)
from app.domains.match.repository import MatchRepository
from app.domains.timeline.model import TimelineEvent
from app.domains.timeline.repository import TimelineEventRepository


class ClipPlanService:
    """ClipPlan 공통 저장/조회/수정 서비스."""

    def __init__(self, db: Session):
        self.db = db
        self.repository = ClipPlanRepository(db)
        self.match_repository = MatchRepository(db)
        self.timeline_event_repository = TimelineEventRepository(db)

    def create_clip_plan(self, data: ClipPlanCreate) -> ClipPlan:
        """Agent 등 내부 코드에서 재사용 가능한 ClipPlan 생성 메서드."""
        self._ensure_match_exists(data.match_id)
        validated_items = self._validate_items(data.match_id, data.items)
        actual_duration_sec = self._sum_item_duration(data.items)

        clip_plan = self.repository.create_plan(
            match_id=data.match_id,
            mode=data.mode,
            summary=data.summary,
            target_duration_sec=data.target_duration_sec,
            actual_duration_sec=data.actual_duration_sec or actual_duration_sec,
            created_by=data.created_by,
            options=data.options,
        )
        self._create_items(
            clip_plan_id=clip_plan.clip_plan_id,
            item_data_list=data.items,
            timeline_events_by_id=validated_items,
        )
        self.db.flush()

        loaded = self.repository.get_by_id(clip_plan.clip_plan_id)
        if loaded is None:
            raise RuntimeError("Created clip plan could not be loaded.")
        return loaded

    def create_manual_clip_plan(self, data: ManualClipPlanCreateRequest) -> ClipPlan:
        self._ensure_match_exists(data.match_id)
        validated_items = self._validate_items(data.match_id, data.items)
        actual_duration_sec = self._sum_item_duration(data.items)

        try:
            clip_plan = self.repository.create_plan(
                match_id=data.match_id,
                mode=data.mode,
                summary=data.summary,
                target_duration_sec=data.target_duration_sec,
                actual_duration_sec=actual_duration_sec,
                created_by="user",
                options={
                    **data.options,
                    "source": "manual",
                },
            )
            self._create_items(
                clip_plan_id=clip_plan.clip_plan_id,
                item_data_list=data.items,
                timeline_events_by_id=validated_items,
            )
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise

        loaded = self.repository.get_by_id(clip_plan.clip_plan_id)
        if loaded is None:
            raise RuntimeError("Created clip plan could not be loaded.")
        return loaded

    def get_clip_plan(self, clip_plan_id: str) -> ClipPlan | None:
        return self.repository.get_by_id(clip_plan_id)

    def update_clip_plan(
        self,
        clip_plan_id: str,
        data: ClipPlanUpdateRequest,
    ) -> ClipPlan | None:
        clip_plan = self.repository.get_by_id(clip_plan_id)
        if clip_plan is None:
            return None

        fields_set = data.model_fields_set
        actual_duration_sec = clip_plan.actual_duration_sec

        try:
            if "items" in fields_set:
                items = data.items or []
                validated_items = self._validate_items(clip_plan.match_id, items)
                self.repository.delete_items_by_plan(clip_plan.clip_plan_id)
                self._create_items(
                    clip_plan_id=clip_plan.clip_plan_id,
                    item_data_list=items,
                    timeline_events_by_id=validated_items,
                )
                actual_duration_sec = self._sum_item_duration(items)

            if "mode" in fields_set and data.mode is not None:
                clip_plan.mode = data.mode
            if "summary" in fields_set:
                clip_plan.summary = data.summary
            if "target_duration_sec" in fields_set:
                clip_plan.target_duration_sec = data.target_duration_sec
            if "items" in fields_set:
                clip_plan.actual_duration_sec = actual_duration_sec
            if "options" in fields_set:
                # PATCH이므로 기존 options를 보존하면서 전달된 key만 덮어쓴다.
                clip_plan.options = {
                    **(clip_plan.options or {}),
                    **(data.options or {}),
                }

            self.db.flush()
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise

        return self.repository.get_by_id(clip_plan_id)

    def update_export_options(
        self,
        clip_plan_id: str,
        data: ExportOptionsUpdateRequest,
    ) -> ClipPlan | None:
        """프론트 미리보기/내보내기 설정을 ClipPlan options에 저장한다.

        RenderJob은 다음 회차에서 clip_plan.options의 title, ratio,
        captions_enabled, music, quality 값을 그대로 사용할 수 있다.
        """
        clip_plan = self.repository.get_by_id(clip_plan_id)
        if clip_plan is None:
            return None

        patch = data.model_dump(exclude_unset=True)
        if not patch:
            return clip_plan

        try:
            current_options = dict(clip_plan.options or {})
            export_options = dict(current_options.get("export_options") or {})

            # RenderJob에서 바로 쓰기 쉽도록 options 최상위에도 저장하고,
            # export_options에도 같은 값을 보관한다.
            for key, value in patch.items():
                current_options[key] = value
                export_options[key] = value

            current_options["export_options"] = export_options
            current_options["export_options_source"] = "preview_export_settings"

            clip_plan.options = current_options
            self.db.flush()
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise

        return self.repository.get_by_id(clip_plan_id)

    def _ensure_match_exists(self, match_id: str) -> None:
        if self.match_repository.get_by_id(match_id) is None:
            raise ValueError("Match not found")

    def _validate_items(
        self,
        match_id: str,
        items: list[ClipPlanItemCreate],
    ) -> dict[str, TimelineEvent]:
        events_by_id: dict[str, TimelineEvent] = {}

        for item in items:
            event = self.timeline_event_repository.get_by_id(item.timeline_event_id)
            if event is None:
                raise ValueError(f"TimelineEvent not found: {item.timeline_event_id}")
            if event.match_id != match_id:
                raise ValueError(
                    f"TimelineEvent does not belong to match: {item.timeline_event_id}"
                )
            if item.end_sec <= item.start_sec:
                raise ValueError("end_sec must be greater than start_sec")
            events_by_id[item.timeline_event_id] = event

        return events_by_id

    def _create_items(
        self,
        *,
        clip_plan_id: str,
        item_data_list: list[ClipPlanItemCreate],
        timeline_events_by_id: dict[str, TimelineEvent],
    ) -> None:
        for fallback_index, item_data in enumerate(item_data_list):
            event = timeline_events_by_id[item_data.timeline_event_id]
            duration_sec = max(0.0, item_data.end_sec - item_data.start_sec)
            reason = item_data.reason or self._build_default_reason(event)

            self.repository.create_item(
                clip_plan_id=clip_plan_id,
                timeline_event_id=item_data.timeline_event_id,
                start_sec=item_data.start_sec,
                end_sec=item_data.end_sec,
                duration_sec=duration_sec,
                order_index=item_data.order_index if item_data.order_index is not None else fallback_index,
                reason=reason,
                metadata_={
                    **item_data.metadata,
                    "source": item_data.metadata.get("source", "manual"),
                },
            )

    @staticmethod
    def _sum_item_duration(items: list[ClipPlanItemCreate]) -> float:
        return round(sum(max(0.0, item.end_sec - item.start_sec) for item in items), 2)

    @staticmethod
    def _build_default_reason(event: TimelineEvent) -> str:
        return f"manually selected {event.label} event"
