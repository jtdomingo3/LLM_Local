"""
FastAPI application factory and main server configuration.

Wires together routers, middleware, lifespan lifecycle events,
CORS settings, and custom error handlers.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.middleware.logging_middleware import RequestLoggingMiddleware
from app.routers import chat, code, health, models, openai_compat
from app.services.engine_manager import get_engine_manager
from app.services.llm_service import get_llm_service

# Configure standard logging
settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("app.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifecycle manager.
    Auto-starts headless engine if needed, probes GPU, and manages cleanup.
    """
    logger.info("=" * 60)
    logger.info("Starting LLM Local Microservice (Standalone Mode)")
    logger.info("Ollama Backend URL: %s", settings.ollama_base_url)
    logger.info("Default Daily Model: %s", settings.model_daily)
    logger.info("Default Coding Model: %s", settings.model_coding)
    logger.info("Context Window (default): %d tokens", settings.default_num_ctx)
    logger.info("=" * 60)

    # 1. Automatically ensure inference backend is running
    engine_mgr = get_engine_manager()
    await engine_mgr.ensure_running()

    # 2. Probe Backend & GPU
    service = get_llm_service()
    ollama_ok, models_found = await service.check_ollama_health()
    gpu_ok = service.check_gpu()

    if ollama_ok:
        logger.info("[OK] Connected to local inference engine.")
        logger.info("Available models (%d): %s", len(models_found), ", ".join(models_found) or "None")
    else:
        logger.warning(
            "[WARN] Inference engine backend is not responsive at %s.",
            settings.ollama_base_url,
        )

    if gpu_ok:
        logger.info("[OK] NVIDIA GPU detected and ready for acceleration.")
    else:
        logger.warning("[WARN] No NVIDIA GPU detected with nvidia-smi. Running on CPU fallback.")

    yield

    logger.info("Shutting down LLM Local Microservice...")
    engine_mgr.shutdown()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    app = FastAPI(
        title="LLM Local Microservice",
        description=(
            "High-performance local LLM microservice wrapping Ollama with "
            "hardware-optimized configurations for RTX 3050 (4GB VRAM) and 24GB DDR5 RAM."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # CORS configuration - permissive for local multi-service interop
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Telemetry and logging middleware
    app.add_middleware(RequestLoggingMiddleware)

    # Register API routers
    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(code.router)
    app.include_router(models.router)
    app.include_router(openai_compat.router)

    # Root summary route
    @app.get("/", summary="Root index", tags=["General"])
    async def root():
        return {
            "name": "LLM Local Microservice",
            "version": "1.0.0",
            "status": "online",
            "documentation": "/docs",
            "health": "/health/ready",
            "endpoints": {
                "chat": "/api/chat",
                "chat_stream": "/api/chat/stream",
                "code": "/api/code",
                "code_stream": "/api/code/stream",
                "code_review": "/api/code/review",
                "code_explain": "/api/code/explain",
                "models": "/api/models",
            },
        }

    # Global unhandled error handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled error processing request: %s", exc)
        return JSONResponse(
            status_code=500,
            content={
                "detail": f"Internal server error: {exc}",
                "path": request.url.path,
            },
        )

    return app


app = create_app()
