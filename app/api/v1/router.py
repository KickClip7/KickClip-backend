from fastapi import APIRouter, Depends

from app.api.v1 import (
    action_spotting,
    agent,
    analysis_jobs,
    artifacts,
    auth,
    clip_plans,
    event_candidate_ranking_v1_1_2a,
    health,
    highlights,
    matches,
    media,
    players,
    projects,
    renders,
    scene_target_reviewability,
    session,
    studio,
    timelines,
    tracking,
)
from app.domains.auth.dependencies import get_current_user
from app.domains.highlight.event_candidate_ranking_v1_1_2a_integration import (
    install_v112a_integration,
)


api_router = APIRouter()
authenticated = [Depends(get_current_user)]
install_v112a_integration(highlights)

api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(
    artifacts.router,
    prefix="/artifacts",
    tags=["artifacts"],
)
api_router.include_router(
    highlights.router,
    tags=["highlight"],
    dependencies=authenticated,
)
api_router.include_router(
    event_candidate_ranking_v1_1_2a.router,
    tags=["highlight"],
    dependencies=authenticated,
)
api_router.include_router(
    scene_target_reviewability.router,
    tags=["scene-target-reviewability"],
    dependencies=authenticated,
)
api_router.include_router(
    action_spotting.router,
    prefix="/action-spotting",
    tags=["action-spotting"],
    dependencies=authenticated,
)
api_router.include_router(projects.router, prefix="/projects", tags=["projects"], dependencies=authenticated)
api_router.include_router(matches.router, prefix="/matches", tags=["matches"], dependencies=authenticated)
api_router.include_router(studio.router, prefix="/studio", tags=["studio"], dependencies=authenticated)
api_router.include_router(media.router, prefix="/media", tags=["media"])
api_router.include_router(timelines.router, tags=["timelines"], dependencies=authenticated)
api_router.include_router(players.router, tags=["players"], dependencies=authenticated)
api_router.include_router(agent.router, prefix="/agent", tags=["agent"], dependencies=authenticated)
api_router.include_router(clip_plans.router, prefix="/clip-plans", tags=["clip-plans"], dependencies=authenticated)
api_router.include_router(session.router, prefix="/session", tags=["session"], dependencies=authenticated)
api_router.include_router(renders.router, tags=["renders"], dependencies=authenticated)
api_router.include_router(
    tracking.router,
    prefix="/tracking",
    tags=["tracking"],
    dependencies=authenticated,
)

# analysis_jobs router는 다음 두 경로를 동시에 제공해야 하므로 prefix 없이 연결한다.
# POST /api/v1/matches/{match_id}/analysis-jobs
# GET  /api/v1/analysis-jobs/{job_id}
api_router.include_router(analysis_jobs.router, dependencies=authenticated)
