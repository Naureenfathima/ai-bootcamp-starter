"""
health.py — Health check endpoints.

Every production service needs a /health endpoint so load balancers,
monitoring tools, and deployment platforms can verify the service is alive.
"""

from fastapi import APIRouter
from pydantic import BaseModel
from app.config import settings

router = APIRouter(tags=["Health"])


class HealthResponse(BaseModel):
    status: str
    version: str
    app_name: str


@router.get("/health", response_model=HealthResponse, summary="Health check")
async def health_check():
    """
    Returns service status. Called by load balancers and monitoring tools.
    Should always return 200 if the service is running.
    """
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        app_name=settings.app_name,
    )


@router.get("/", include_in_schema=False)
async def root():
    return {"message": f"Welcome to {settings.app_name}. Visit /docs for the API."}
