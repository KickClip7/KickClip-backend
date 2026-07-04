from fastapi import APIRouter

from app.api.v1 import (
    agent,
    analysis_jobs,
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


api_router = APIRouter()

api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(projects.router, prefix="/projects", tags=["projects"])
api_router.include_router(matches.router, prefix="/matches", tags=["matches"])
api_router.include_router(studio.router, prefix="/studio", tags=["studio"])
api_router.include_router(media.router, prefix="/media", tags=["media"])
api_router.include_router(timelines.router, tags=["timelines"])
api_router.include_router(players.router, tags=["players"])
api_router.include_router(agent.router, prefix="/agent", tags=["agent"])
api_router.include_router(clip_plans.router, prefix="/clip-plans", tags=["clip-plans"])
api_router.include_router(renders.router, tags=["renders"])

# analysis_jobs router는 다음 두 경로를 동시에 제공해야 하므로 prefix 없이 연결한다.
# POST /api/v1/matches/{match_id}/analysis-jobs
# GET  /api/v1/analysis-jobs/{job_id}
api_router.include_router(analysis_jobs.router)
