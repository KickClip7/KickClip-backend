from __future__ import annotations

from sqlalchemy import (
    JSON,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin
from app.utils.id_generator import generate_prefixed_id


class PlayerFocusSubject(Base, TimestampMixin):
    """A user-defined edit subject, not a match-wide player identity."""

    __tablename__ = "player_focus_subjects"

    focus_subject_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        default=lambda: generate_prefixed_id("focus"),
    )
    project_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("projects.project_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    display_name: Mapped[str] = mapped_column(String(100), nullable=False)
    identity_source: Mapped[str] = mapped_column(
        String(32),
        default="USER_DEFINED",
        nullable=False,
    )
    anchor_scene_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("timeline_events.timeline_event_id", ondelete="RESTRICT"),
        nullable=False,
    )
    # Kept as a validated identifier rather than an FK to avoid a schema cycle:
    # candidates belong to a revision, while the revision references this subject.
    anchor_candidate_id: Mapped[str] = mapped_column(String(64), nullable=False)
    appearance_memory_ref: Mapped[str | None] = mapped_column(
        String(1024),
        nullable=True,
    )
    metadata_: Mapped[dict] = mapped_column(
        "metadata",
        JSON,
        default=dict,
        nullable=False,
    )

    project = relationship("Project")
    anchor_scene = relationship("TimelineEvent")


class HighlightRevision(Base, TimestampMixin):
    """Immutable editing intent and selection state within one Project."""

    __tablename__ = "highlight_revisions"
    __table_args__ = (
        UniqueConstraint(
            "project_id",
            "revision_number",
            name="uq_highlight_revisions_project_number",
        ),
    )

    revision_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        default=lambda: generate_prefixed_id("hrev"),
    )
    project_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("projects.project_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    revision_number: Mapped[int] = mapped_column(Integer, nullable=False)
    parent_revision_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("highlight_revisions.revision_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    action_spotting_job_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("analysis_jobs.analysis_job_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    user_request: Mapped[str] = mapped_column(Text, nullable=False)
    structured_request: Mapped[dict] = mapped_column(
        JSON,
        default=dict,
        nullable=False,
    )
    selected_scene_ids: Mapped[list[str]] = mapped_column(
        JSON,
        default=list,
        nullable=False,
    )
    # Every source scene keeps an explicit selection state and render strategy.
    scene_selection: Mapped[list[dict]] = mapped_column(
        JSON,
        default=list,
        nullable=False,
    )
    focus_mode: Mapped[str] = mapped_column(
        String(16),
        default="NONE",
        nullable=False,
        index=True,
    )
    focus_subject_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("player_focus_subjects.focus_subject_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    clip_plan_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("clip_plans.clip_plan_id", ondelete="SET NULL"),
        nullable=True,
    )
    render_job_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("render_jobs.render_job_id", ondelete="SET NULL"),
        nullable=True,
    )
    status: Mapped[str] = mapped_column(
        String(64),
        default="ACTION_SPOTTING_RUNNING",
        nullable=False,
        index=True,
    )
    pending_action: Mapped[str | None] = mapped_column(String(64), nullable=True)
    options: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    project = relationship("Project")
    parent_revision = relationship(
        "HighlightRevision",
        remote_side=[revision_id],
    )
    action_spotting_job = relationship("AnalysisJob")
    focus_subject = relationship("PlayerFocusSubject")
    clip_plan = relationship("ClipPlan", foreign_keys=[clip_plan_id])
    render_job = relationship("RenderJob", foreign_keys=[render_job_id])


class HighlightDraft(Base, TimestampMixin):
    """Mutable, recoverable editor snapshot for one Project."""

    __tablename__ = "highlight_drafts"
    __table_args__ = (
        UniqueConstraint(
            "project_id",
            name="uq_highlight_drafts_project_id",
        ),
    )

    draft_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        default=lambda: generate_prefixed_id("hdraft"),
    )
    project_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("projects.project_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    revision_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("highlight_revisions.revision_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    clip_plan_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("clip_plans.clip_plan_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    saved_by_user_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("users.user_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    state: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)

    project = relationship("Project")
    revision = relationship("HighlightRevision")
    clip_plan = relationship("ClipPlan")
    saved_by = relationship("User")


class ScenePlayerCandidate(Base, TimestampMixin):
    """Scene-local selectable track seed with no inferred real-world identity."""

    __tablename__ = "scene_player_candidates"
    __table_args__ = (
        UniqueConstraint(
            "revision_id",
            "candidate_id",
            name="uq_scene_player_candidates_revision_candidate",
        ),
    )

    candidate_row_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        default=lambda: generate_prefixed_id("pcand"),
    )
    candidate_id: Mapped[str] = mapped_column(String(255), nullable=False)
    revision_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("highlight_revisions.revision_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    scene_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("timeline_events.timeline_event_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    anchor_time_sec: Mapped[float] = mapped_column(Float, nullable=False)
    anchor_source_time_sec: Mapped[float] = mapped_column(Float, nullable=False)
    anchor_frame_index: Mapped[int] = mapped_column(Integer, nullable=False)
    bbox_xyxy: Mapped[list[float]] = mapped_column(JSON, nullable=False)
    thumbnail_artifact_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("artifacts.artifact_id", ondelete="SET NULL"),
        nullable=True,
    )
    track_length_frames: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    trackability_score: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[str] = mapped_column(
        String(32),
        default="AVAILABLE",
        nullable=False,
        index=True,
    )
    metadata_: Mapped[dict] = mapped_column(
        "metadata",
        JSON,
        default=dict,
        nullable=False,
    )

    revision = relationship("HighlightRevision")
    scene = relationship("TimelineEvent")
    thumbnail_artifact = relationship("Artifact")


class SceneTrackingBinding(Base, TimestampMixin):
    """Maps one revision scene and user-confirmed candidate to a tracking job."""

    __tablename__ = "scene_tracking_bindings"
    __table_args__ = (
        UniqueConstraint(
            "revision_id",
            "scene_id",
            name="uq_scene_tracking_bindings_revision_scene",
        ),
    )

    binding_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        default=lambda: generate_prefixed_id("sbind"),
    )
    revision_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("highlight_revisions.revision_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    scene_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("timeline_events.timeline_event_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    focus_subject_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("player_focus_subjects.focus_subject_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    selected_candidate_id: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )
    tracking_job_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("tracking_jobs.tracking_job_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    scene_clip_asset_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("media_assets.asset_id", ondelete="SET NULL"),
        nullable=True,
    )
    source_start_time_sec: Mapped[float | None] = mapped_column(Float, nullable=True)
    source_end_time_sec: Mapped[float | None] = mapped_column(Float, nullable=True)
    source_start_frame: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source_end_frame: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source_fps: Mapped[float | None] = mapped_column(Float, nullable=True)
    clip_fps: Mapped[float | None] = mapped_column(Float, nullable=True)
    clip_frame_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    confirmation_source: Mapped[str] = mapped_column(
        String(32),
        default="USER",
        nullable=False,
    )
    target_presence_status: Mapped[str] = mapped_column(
        String(32),
        default="USER_CONFIRMATION_REQUIRED",
        nullable=False,
        index=True,
    )
    render_strategy: Mapped[str] = mapped_column(
        String(32),
        default="TARGET_CENTERED",
        nullable=False,
    )
    status: Mapped[str] = mapped_column(
        String(64),
        default="QUEUED_FOR_EXTRACTION",
        nullable=False,
        index=True,
    )
    timeline_summary: Mapped[dict] = mapped_column(
        JSON,
        default=dict,
        nullable=False,
    )
    metadata_: Mapped[dict] = mapped_column(
        "metadata",
        JSON,
        default=dict,
        nullable=False,
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    revision = relationship("HighlightRevision")
    scene = relationship("TimelineEvent")
    focus_subject = relationship("PlayerFocusSubject")
    tracking_job = relationship("TrackingJob")
    scene_clip_asset = relationship("MediaAsset")


class SceneTargetSelection(Base, TimestampMixin):
    """Immutable user choice of a scene-wide local candidate revision."""

    __tablename__ = "scene_target_selections"
    __table_args__ = (
        UniqueConstraint(
            "revision_id",
            "scene_id",
            "selection_revision",
            name="uq_scene_target_selection_revision",
        ),
    )

    selection_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        default=lambda: generate_prefixed_id("tsel"),
    )
    owner_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("users.user_id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    match_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("matches.match_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    project_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("projects.project_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    revision_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("highlight_revisions.revision_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    scene_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("timeline_events.timeline_event_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    selection_revision: Mapped[int] = mapped_column(Integer, nullable=False)
    selected_candidate_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
    )
    status: Mapped[str] = mapped_column(
        String(64),
        default="WAITING_EARLIER_ANCHOR_CONFIRMATION",
        nullable=False,
        index=True,
    )
    artifact_root: Mapped[str] = mapped_column(String(2048), nullable=False)
    selection_artifact: Mapped[dict] = mapped_column(
        JSON, default=dict, nullable=False
    )
    reference_set_artifact: Mapped[dict] = mapped_column(
        JSON, default=dict, nullable=False
    )
    earlier_proposals_artifact: Mapped[dict] = mapped_column(
        JSON, default=dict, nullable=False
    )
    earlier_decision_artifact: Mapped[dict] = mapped_column(
        JSON, default=dict, nullable=False
    )
    candidate_cache_key: Mapped[str] = mapped_column(
        String(64), nullable=False, index=True
    )
    tracking_cache_key: Mapped[str | None] = mapped_column(
        String(64), nullable=True, index=True
    )
    tracking_job_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("tracking_jobs.tracking_job_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    metadata_: Mapped[dict] = mapped_column(
        "metadata", JSON, default=dict, nullable=False
    )

    owner = relationship("User")
    revision = relationship("HighlightRevision")
    scene = relationship("TimelineEvent")
    tracking_job = relationship("TrackingJob")


class SceneTargetSelectionReference(Base, TimestampMixin):
    """Portable reference crop belonging to one immutable target selection."""

    __tablename__ = "scene_target_selection_references"
    __table_args__ = (
        UniqueConstraint(
            "selection_id",
            "reference_id",
            name="uq_scene_target_selection_reference",
        ),
    )

    reference_row_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        default=lambda: generate_prefixed_id("tref"),
    )
    selection_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("scene_target_selections.selection_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    reference_id: Mapped[str] = mapped_column(String(255), nullable=False)
    source_candidate_id: Mapped[str] = mapped_column(
        String(255), nullable=False
    )
    frame_index: Mapped[int] = mapped_column(Integer, nullable=False)
    bbox_xyxy: Mapped[list[float]] = mapped_column(JSON, nullable=False)
    crop_artifact: Mapped[str] = mapped_column(String(2048), nullable=False)
    quality: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    metadata_: Mapped[dict] = mapped_column(
        "metadata", JSON, default=dict, nullable=False
    )

    selection = relationship("SceneTargetSelection")


class EarlierAnchorProposal(Base, TimestampMixin):
    """Blind appearance proposal; never an automatic identity decision."""

    __tablename__ = "earlier_anchor_proposals"
    __table_args__ = (
        UniqueConstraint(
            "selection_id",
            "candidate_id",
            name="uq_earlier_anchor_proposal_candidate",
        ),
    )

    proposal_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        default=lambda: generate_prefixed_id("eap"),
    )
    selection_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("scene_target_selections.selection_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    candidate_id: Mapped[str] = mapped_column(String(255), nullable=False)
    retrieval_rank: Mapped[int] = mapped_column(Integer, nullable=False)
    retrieval_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    prototype_similarity: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    decision_state: Mapped[str] = mapped_column(
        String(64), default="PENDING_USER_CONFIRMATION", nullable=False
    )
    artifacts: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    metadata_: Mapped[dict] = mapped_column(
        "metadata", JSON, default=dict, nullable=False
    )

    selection = relationship("SceneTargetSelection")
