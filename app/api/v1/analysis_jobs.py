from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.ai.runtime.job_runner import run_analysis_job_background
from app.db.session import get_db
from app.domains.analysis.schema import (
    AnalysisJobCreateRequest,
    AnalysisJobCreateResponse,
    AnalysisJobStatusResponse,
)
from app.domains.analysis.service import AnalysisJobService


router = APIRouter()


@router.post(
    "/matches/{match_id}/analysis-jobs",
    response_model=AnalysisJobCreateResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["analysis-jobs"],
    summary="분석 작업 생성",
)
def create_analysis_job(
    match_id: str,
    payload: AnalysisJobCreateRequest,
    background_tasks: BackgroundTasks,
    auto_start: bool = Query(
        default=True,
        description="true면 job 생성 후 dummy JobRunner를 background에서 실행한다.",
    ),
    db: Session = Depends(get_db),
) -> AnalysisJobCreateResponse:
    service = AnalysisJobService(db)

    try:
        response = service.create_analysis_job_for_match(
            match_id=match_id,
            data=payload,
        )
    except ValueError as exc:
        detail = str(exc)

        if detail == "Match not found":
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=detail,
            ) from exc

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail,
        ) from exc

    if auto_start:
        background_tasks.add_task(
            run_analysis_job_background,
            response.analysis_job_id,
        )

    return response


@router.get(
    "/analysis-jobs/{job_id}",
    response_model=AnalysisJobStatusResponse,
    tags=["analysis-jobs"],
    summary="분석 작업 상태 조회",
)
def get_analysis_job(
    job_id: str,
    db: Session = Depends(get_db),
) -> AnalysisJobStatusResponse:
    service = AnalysisJobService(db)
    response = service.get_analysis_job_status(job_id)

    if response is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis job not found",
        )

    return response