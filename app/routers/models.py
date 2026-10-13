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
        "max_tokens_chat": settings.max_tokens_chat,
        "max_tokens_code": settings.max_tokens_code,
        "optimizations": {
            "smart_routing_enabled": settings.smart_routing_enabled,
            "response_cache_enabled": settings.response_cache_enabled,
            "keep_alive": settings.ollama_keep_alive,
            "max_loaded_models": settings.ollama_max_loaded_models,
        },
    }


@router.post(
    "/preload",
    summary="Preload a model into GPU memory",
    description="Preheats model weights in Ollama ahead of time for low first-token latency.",
)
async def preload_model(
    model: str | None = None,
    service: LLMService = Depends(get_llm_service),
    settings: Settings = Depends(get_settings),
) -> dict:
    """Preload model weights into VRAM."""
    target_model = model or settings.model_coding
    success = await service.preload_model(target_model)
    return {
        "status": "success" if success else "failed",
        "model": target_model,
        "preloaded": success,
    }


@router.post(
    "/unload",
    summary="Unload models from GPU VRAM",
    description="Evicts models from GPU memory immediately, dropping VRAM to 0 MB.",
)
async def unload_models() -> dict:
    """Force eviction of models from VRAM."""
    from app.services.engine_manager import get_engine_manager
    mgr = get_engine_manager()
    unloaded = await mgr.unload_models()
    return {
        "status": "success",
        "unloaded_models": unloaded,
        "count": len(unloaded),
    }
