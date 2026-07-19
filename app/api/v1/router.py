from fastapi import APIRouter, Depends

from app.api.v1 import (
    action_spotting,
    agent,
    analysis_jobs,
    auth,
    clip_plans,
    health,
    matches,
    media,
    players,
    projects,
    renders,
    studio,
    timelines,
)
from app.domains.auth.dependencies import get_current_user


api_router = APIRouter()
authenticated = [Depends(get_current_user)]

api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
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
api_router.include_router(renders.router, tags=["renders"], dependencies=authenticated)

# analysis_jobs router는 다음 두 경로를 동시에 제공해야 하므로 prefix 없이 연결한다.
# POST /api/v1/matches/{match_id}/analysis-jobs
# GET  /api/v1/analysis-jobs/{job_id}
api_router.include_router(analysis_jobs.router, dependencies=authenticated)
