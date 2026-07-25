from sqlalchemy import delete, select
from sqlalchemy.orm import Session, selectinload

from app.domains.clip_plan.model import ClipPlan, ClipPlanItem
from app.domains.project.model import Project


class ClipPlanRepository:
    def __init__(self, db: Session):
        self.db = db

    def create_plan(
        self,
        *,
        project_id: str,
        mode: str,
        summary: str | None,
        target_duration_sec: float | None,
        actual_duration_sec: float | None,
        created_by: str,
        options: dict,
    ) -> ClipPlan:
        clip_plan = ClipPlan(
            project_id=project_id,
            mode=mode,
            summary=summary,
            target_duration_sec=target_duration_sec,
            actual_duration_sec=actual_duration_sec,
            created_by=created_by,
            options=options,
        )
        self.db.add(clip_plan)
        self.db.flush()
        return clip_plan

    def create_item(
        self,
        *,
        clip_plan_id: str,
        timeline_event_id: str,
        start_sec: float,
        end_sec: float,
        duration_sec: float,
        order_index: int,
        reason: str | None = None,
        metadata_: dict | None = None,
    ) -> ClipPlanItem:
        item = ClipPlanItem(
            clip_plan_id=clip_plan_id,
            timeline_event_id=timeline_event_id,
            start_sec=start_sec,
            end_sec=end_sec,
            duration_sec=duration_sec,
            order_index=order_index,
            reason=reason,
            metadata_=metadata_ or {},
        )
        self.db.add(item)
        self.db.flush()
        return item

    def get_by_id(self, clip_plan_id: str) -> ClipPlan | None:
        stmt = (
            select(ClipPlan)
            .options(
                selectinload(ClipPlan.items),
                selectinload(ClipPlan.project).selectinload(Project.match),
            )
            .where(ClipPlan.clip_plan_id == clip_plan_id)
        )
        return self.db.scalar(stmt)

    def list_by_project(self, project_id: str) -> list[ClipPlan]:
        stmt = (
            select(ClipPlan)
            .options(
                selectinload(ClipPlan.items),
                selectinload(ClipPlan.project).selectinload(Project.match),
            )
            .where(ClipPlan.project_id == project_id)
            .order_by(ClipPlan.created_at.desc())
        )
        return list(self.db.scalars(stmt).all())

    def update_plan(
        self,
        *,
        clip_plan: ClipPlan,
        mode: str | None = None,
        summary: str | None = None,
        target_duration_sec: float | None = None,
        actual_duration_sec: float | None = None,
        options: dict | None = None,
    ) -> ClipPlan:
        if mode is not None:
            clip_plan.mode = mode
        if summary is not None:
            clip_plan.summary = summary
        if target_duration_sec is not None:
            clip_plan.target_duration_sec = target_duration_sec
        if actual_duration_sec is not None:
            clip_plan.actual_duration_sec = actual_duration_sec
        if options is not None:
            clip_plan.options = options

        self.db.flush()
        return clip_plan

    def delete_items_by_plan(self, clip_plan_id: str) -> int:
        stmt = delete(ClipPlanItem).where(ClipPlanItem.clip_plan_id == clip_plan_id)
        result = self.db.execute(stmt)
        self.db.flush()
        return int(result.rowcount or 0)
