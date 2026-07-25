from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.domains.agent.player_profile_resolver import PlayerProfileResolver
from app.domains.agent.rule_based_planner import RuleBasedClipPlanner
from app.domains.agent.schema import (
    AgentClipPlanItemResponse,
    AgentClipPlanRequest,
    AgentClipPlanResponse,
    PlayerProfileResolveRequest,
    PlayerProfileResolveResponse,
)
from app.domains.clip_plan.repository import ClipPlanRepository
from app.domains.match.repository import MatchRepository
from app.domains.player.repository import PlayerRepository
from app.domains.project.repository import ProjectRepository
from app.domains.timeline.dev_context import (
    resolve_agent_match_id,
    select_dev_timeline_events,
)
from app.domains.timeline.repository import TimelineEventRepository


class AgentService:
    def __init__(self, db: Session):
        self.db = db
        self.match_repository = MatchRepository(db)
        self.timeline_event_repository = TimelineEventRepository(db)
        self.player_repository = PlayerRepository(db)
        self.clip_plan_repository = ClipPlanRepository(db)
        self.project_repository = ProjectRepository(db)
        self.planner = RuleBasedClipPlanner()
        self.profile_resolver = PlayerProfileResolver()

    def create_clip_plan(self, data: AgentClipPlanRequest) -> AgentClipPlanResponse:
        settings = get_settings()
        project = self.project_repository.get_by_id(data.project_id)
        if project is None:
            raise ValueError("Project not found")
        match_id = project.match_id
        if data.match_id is not None:
            requested_match_id = resolve_agent_match_id(data.match_id, settings)
            if requested_match_id != match_id:
                raise ValueError("Project does not belong to requested match")
        match = self.match_repository.get_by_id(match_id)
        if match is None:
            raise ValueError("Match not found")

        events = select_dev_timeline_events(
            self.timeline_event_repository.list_by_match(match_id),
            match_id=match_id,
            settings=settings,
        )
        players = self.player_repository.list_by_match(match_id)

        planned = self.planner.build_plan(
            prompt=data.prompt,
            events=events,
            players=players,
            target_duration_sec=data.target_duration_sec,
            selected_player_id=data.selected_player_id,
            options=data.options,
        )

        try:
            clip_plan = self.clip_plan_repository.create_plan(
                project_id=project.project_id,
                mode=data.mode or "AGENT_GENERATED",
                summary=planned.summary,
                target_duration_sec=planned.target_duration_sec,
                actual_duration_sec=planned.total_duration_sec,
                created_by="agent",
                options={
                    **data.options,
                    "prompt": data.prompt,
                    "requested_match_id": data.match_id,
                    "project_id": project.project_id,
                    "parsed_labels": planned.parsed_labels,
                    "parsed_half": planned.parsed_half,
                    "selected_player_id": planned.selected_player_id,
                    "planner": "rule_based_v1",
                },
            )

            response_items: list[AgentClipPlanItemResponse] = []
            for order_index, item in enumerate(planned.items):
                self.clip_plan_repository.create_item(
                    clip_plan_id=clip_plan.clip_plan_id,
                    timeline_event_id=item.timeline_event_id,
                    start_sec=item.start_sec,
                    end_sec=item.end_sec,
                    duration_sec=item.duration_sec,
                    order_index=order_index,
                    reason=item.reason,
                    metadata_={"source": "agent_rule_based_v1"},
                )
                response_items.append(
                    AgentClipPlanItemResponse(
                        timeline_event_id=item.timeline_event_id,
                        start_sec=item.start_sec,
                        end_sec=item.end_sec,
                        duration_sec=item.duration_sec,
                        reason=item.reason,
                    )
                )

            self.db.commit()

            return AgentClipPlanResponse(
                clip_plan_id=clip_plan.clip_plan_id,
                summary=planned.summary,
                total_duration_sec=planned.total_duration_sec,
                items=response_items,
            )

        except Exception:
            self.db.rollback()
            raise

    def resolve_player_profile(
        self,
        data: PlayerProfileResolveRequest,
    ) -> PlayerProfileResolveResponse:
        match = self.match_repository.get_by_id(data.match_id)
        if match is None:
            raise ValueError("Match not found")

        player = self.player_repository.get_by_id(data.player_id)
        if player is None or player.match_id != data.match_id:
            raise ValueError("Player not found")

        profile = self.profile_resolver.resolve(data.user_message)

        metadata = dict(player.metadata_ or {})
        metadata["resolved_profile"] = {
            "display_name": profile.display_name,
            "real_team": profile.real_team,
            "position": profile.position,
            "nationality": profile.nationality,
            "traits": profile.traits,
            "summary": profile.summary,
            "profile_source": profile.profile_source,
        }
        metadata["aliases"] = sorted(
            set([*(metadata.get("aliases") or []), *profile.aliases, profile.display_name])
        )

        try:
            self.player_repository.update_profile(
                player=player,
                display_name=profile.display_name,
                identity_status=profile.identity_status,
                profile_source=profile.profile_source,
                metadata_=metadata,
            )
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise

        return PlayerProfileResolveResponse(
            player_id=player.player_id,
            display_name=profile.display_name,
            real_team=profile.real_team,
            position=profile.position,
            nationality=profile.nationality,
            traits=profile.traits,
            summary=profile.summary,
            identity_status=profile.identity_status,
        )
