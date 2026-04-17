"""
Jobs Router - Async training job management.

Endpoints:
  GET /api/job/{job_id}  - Poll job status and get results when done
"""

from fastapi import APIRouter, HTTPException
from app.services.job_store import job_store

router = APIRouter(prefix="/api")


@router.get("/job/{job_id}", summary="Poll training job status")
async def get_job(job_id: str):
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return job
