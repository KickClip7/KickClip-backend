from typing import List, Literal, Optional

from pydantic import BaseModel


class EditIntent(BaseModel):
    intent_type: Literal["filter", "build", "remove", "adjust", "confirm"]
    label: Optional[str] = None
    half: Optional[int] = None
    target_duration: Optional[int] = None
    target_clip_count: Optional[int] = None
    target_index: Optional[int] = None
    adjust_delta_sec: Optional[int] = None
    needs_clarification: bool = False
    clarification_question: Optional[str] = None
    clarification_options: Optional[List[str]] = None
