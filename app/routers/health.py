"""
Health check endpoints.

Two-tier health checks following cloud-native best practices:
  - /health/live  → Is the process alive? (lightweight)
  - /health/ready → Is the service ready to accept traffic? (deep check)
"""

from fastapi import APIRouter, Depends

from app.models.schemas import HealthResponse
from app.services.llm_service import LLMService, get_llm_service

router = APIRouter(prefix="/health", tags=["Health"])


@router.get(
    "/live",
    response_model=HealthResponse,
    summary="Liveness check",
    description="Returns 200 if the FastAPI process is running. No dependency checks.",
)
async def liveness() -> HealthResponse:
    """Simple liveness probe — confirms the process is alive."""
    return HealthResponse(status="alive")


@router.get(
    "/ready",
    response_model=HealthResponse,
    summary="Readiness check",
    description="Deep health check: verifies Ollama connection, lists available models, and detects GPU.",
)
async def readiness(
    service: LLMService = Depends(get_llm_service),
) -> HealthResponse:
    """
    Readiness probe — checks all critical dependencies.

    This endpoint is what other services should poll to know
    whether the LLM microservice is ready to handle requests.
    """
    ollama_ok, model_names = await service.check_ollama_health()
    gpu_ok = service.check_gpu()

    if ollama_ok:
        status = "ready"
    else:
        status = "degraded"

    return HealthResponse(
        status=status,
        ollama_connected=ollama_ok,
        models_available=model_names,
        gpu_detected=gpu_ok,
        uptime_seconds=round(service.uptime_seconds, 2),
    )
