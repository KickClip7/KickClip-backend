from pydantic import BaseModel, Field

from app.domains.timeline.schema import TimelineEventRead


class SessionStartRequest(BaseModel):
    match_id: str


class SessionStartResponse(BaseModel):
    session_id: str
    total_events: int
    video_url: str | None = None


class SessionChatRequest(BaseModel):
    session_id: str
    message: str = Field(min_length=1, max_length=2000)


class SessionChatResponse(BaseModel):
    needs_clarification: bool
    current_clips: list[TimelineEventRead] = Field(default_factory=list)
    reply_text: str | None = None
    clarification_question: str | None = None
    clarification_options: list[str] | None = None


class SessionStateResponse(BaseModel):
    current_clips: list[TimelineEventRead] = Field(default_factory=list)
    all_events: list[TimelineEventRead] = Field(default_factory=list)
