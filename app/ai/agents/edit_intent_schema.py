from typing import List, Literal, Optional

from pydantic import BaseModel


class LabelCount(BaseModel):
    """라벨 하나에 대해 요청받은 개수.

    '슈팅 2개와 코너킥 2개'처럼 라벨마다 개수가 다른 요청은 target_clip_count(전체 개수
    하나짜리 정수)로는 표현할 수 없어서 이 항목으로 따로 담는다.
    """

    label: str
    count: int


class EditIntent(BaseModel):
    intent_type: Literal[
        "filter",
        "build",
        "add",
        "remove",
        "adjust",
        "confirm",
        "chitchat",
        "unsupported",
    ]
    labels: Optional[List[str]] = None
    # 요청받았지만 현재 모델이 만들어내지 않는 장면 표현(예: 프리킥).
    # 비슷한 라벨로 대체하지 않고 그대로 사용자에게 알리기 위해 남긴다.
    unsupported_terms: Optional[List[str]] = None
    half: Optional[int] = None
    target_duration: Optional[int] = None
    target_clip_count: Optional[int] = None
    # 라벨별 개수를 따로 지정한 경우에만 채운다. 채워지면 labels보다 우선한다.
    label_counts: Optional[List[LabelCount]] = None
    target_index: Optional[int] = None
    adjust_delta_sec: Optional[int] = None
    needs_clarification: bool = False
    clarification_question: Optional[str] = None
    clarification_options: Optional[List[str]] = None
    response_text: Optional[str] = None
