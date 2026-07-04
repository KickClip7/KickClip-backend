from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ResolvedPlayerProfile:
    display_name: str
    real_team: str | None
    position: str | None
    nationality: str | None
    traits: list[str]
    summary: str
    identity_status: str
    profile_source: str
    aliases: list[str]


KNOWN_PLAYER_PROFILES: dict[str, ResolvedPlayerProfile] = {
    "손흥민": ResolvedPlayerProfile(
        display_name="손흥민",
        real_team="Tottenham Hotspur",
        position="Forward / Left wing",
        nationality="Korea Republic",
        traits=["양발 슈팅", "스프린트 침투", "박스 안 결정력"],
        summary="빠른 전환, 양발 마무리, 뒷공간 침투 장면을 중심으로 숏폼 하이라이트를 구성하기 좋습니다.",
        identity_status="RAG_RESOLVED",
        profile_source="rule_based_dictionary",
        aliases=["son", "sonny", "heung-min son", "son heung-min"],
    ),
    "이강인": ResolvedPlayerProfile(
        display_name="이강인",
        real_team="Paris Saint-Germain",
        position="Attacking midfielder / Winger",
        nationality="Korea Republic",
        traits=["탈압박", "왼발 킥", "전진 패스", "세트피스"],
        summary="볼 터치, 전진 패스, 세트피스 장면을 중심으로 하이라이트를 구성하기 좋습니다.",
        identity_status="RAG_RESOLVED",
        profile_source="rule_based_dictionary",
        aliases=["kangin", "kang-in lee", "lee kang-in"],
    ),
    "김민재": ResolvedPlayerProfile(
        display_name="김민재",
        real_team="Bayern Munich",
        position="Centre-back",
        nationality="Korea Republic",
        traits=["대인 수비", "인터셉트", "전진 수비", "공중볼"],
        summary="수비 개입, 인터셉트, 압박 차단 장면을 중심으로 하이라이트를 구성하기 좋습니다.",
        identity_status="RAG_RESOLVED",
        profile_source="rule_based_dictionary",
        aliases=["kim min-jae", "minjae", "min-jae kim"],
    ),
    "황희찬": ResolvedPlayerProfile(
        display_name="황희찬",
        real_team="Wolverhampton Wanderers",
        position="Forward / Winger",
        nationality="Korea Republic",
        traits=["저돌적 돌파", "압박", "박스 침투", "마무리"],
        summary="돌파, 압박, 박스 안 침투 장면을 중심으로 숏폼 하이라이트를 구성하기 좋습니다.",
        identity_status="RAG_RESOLVED",
        profile_source="rule_based_dictionary",
        aliases=["hwang hee-chan", "heechan", "hee-chan hwang"],
    ),
}


class PlayerProfileResolver:
    """초기 선수 프로필 resolver.

    실제 RAG/외부 검색을 붙이기 전까지는 작은 dictionary와 사용자 문장 파싱으로
    Player identity를 연결한다.
    """

    def resolve(self, user_message: str) -> ResolvedPlayerProfile:
        normalized = (user_message or "").strip().lower()

        for key, profile in KNOWN_PLAYER_PROFILES.items():
            if key.lower() in normalized:
                return profile
            if any(alias.lower() in normalized for alias in profile.aliases):
                return profile

        display_name = self._extract_name_candidate(user_message)
        return ResolvedPlayerProfile(
            display_name=display_name or "이름 미확인 선수",
            real_team=None,
            position=None,
            nationality=None,
            traits=[],
            summary="사용자 입력을 바탕으로 선수 이름을 임시 연결했습니다. 추후 프로필 검색/RAG 검증이 필요합니다.",
            identity_status="USER_LABELED",
            profile_source="user_message",
            aliases=[display_name] if display_name else [],
        )

    @staticmethod
    def _extract_name_candidate(user_message: str) -> str | None:
        text = (user_message or "").strip()
        if not text:
            return None

        patterns = [
            r"(?:이 선수는|이 선수|선수는|선수)\s*([가-힣A-Za-z\-\s]{2,30}?)(?:이야|야|입니다|이에요|예요|임|이다|$)",
            r"([가-힣]{2,5}?)(?:이야|야|입니다|이에요|예요|임|이다)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()

        if len(text) <= 30:
            return text

        return None