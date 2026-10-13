"""
Models router — list and inspect available LLM models.

Queries Ollama for installed models and their metadata,
including size, quantization level, and parameter count.
"""

from fastapi import APIRouter, Depends

from app.config import get_settings, Settings
from app.models.schemas import ModelInfo, ModelsListResponse
from app.services.llm_service import LLMService, get_llm_service

router = APIRouter(prefix="/api/models", tags=["Models"])


@router.get(
    "",
    response_model=ModelsListResponse,
    summary="List all available models",
    description=(
        "Returns all models installed in Ollama along with their "
        "size, quantization level, and parameter count. Also shows "
        "which models are configured as defaults for daily/coding/power use."
    ),
)
async def list_models(
    service: LLMService = Depends(get_llm_service),
    settings: Settings = Depends(get_settings),
) -> ModelsListResponse:
    """List all models available in Ollama with their metadata."""
    models = await service.list_models()
    return ModelsListResponse(
        models=models,
        default_daily=settings.model_daily,
        default_coding=settings.model_coding,
        default_power_coding=settings.model_power_coding,
    )


@router.get(
    "/active",
    response_model=dict,
    summary="Show active model configuration",
    description="Shows which models are configured as defaults for each use case.",
)
async def active_models(
    settings: Settings = Depends(get_settings),
) -> dict:
    """Return the currently configured default models for each purpose."""
    return {
        "daily": {
            "model": settings.model_daily,
            "description": "General conversation, summarization, Q&A",
            "temperature": settings.default_temperature_chat,
        },
        "coding": {
            "model": settings.model_coding,
            "description": "Fast code generation, debugging, boilerplate",
            "temperature": settings.default_temperature_code,
        },
        "power_coding": {
            "model": settings.model_power_coding,
            "description": "Complex coding tasks, hybrid GPU/CPU mode",
            "temperature": settings.default_temperature_code,
        },
        "context_window": settings.default_num_ctx,
        "max_tokens": settings.max_tokens,
    }
