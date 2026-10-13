"""
OpenAI-compatible endpoints (/v1/chat/completions, /v1/completions, /v1/models).

Allows standard tools like Continue.dev (VS Code), Cursor, Aider, Chatbox,
and other IDE extensions to connect directly to OUR microservice instead of
calling Ollama raw.

By routing through our server, every request is constrained by our memory-reduction
pipeline (context window clamped to 4096 tokens, deterministic temperature for code,
and global Q8_0 KV cache + Flash Attention).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from openai import OpenAIError
from pydantic import BaseModel, ConfigDict, Field

from app.config import Settings, get_settings
from app.services.llm_service import LLMService, get_llm_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["OpenAI Compatible"])


class OpenAIChatRequest(BaseModel):
    """OpenAI-format chat completion request."""
    model: str | None = Field(default=None, description="Model ID")
    messages: list[dict[str, Any]] = Field(..., description="Message history")
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1)
    stream: bool | None = Field(default=False)

    model_config = ConfigDict(extra="allow")


class OpenAICompletionRequest(BaseModel):
    """OpenAI-format legacy/autocomplete completion request."""
    model: str | None = Field(default=None, description="Model ID")
    prompt: str | list[str] = Field(..., description="Prompt string or tokens")
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1)
    stream: bool | None = Field(default=False)

    model_config = ConfigDict(extra="allow")


@router.get("/models", summary="OpenAI-compatible models list")
async def list_openai_models(
    service: LLMService = Depends(get_llm_service),
    settings: Settings = Depends(get_settings),
):
    """
    Returns installed models in OpenAI's standard /v1/models format
    so Continue.dev, Cursor, and IDE extensions can discover them.
    """
    await service._ensure_backend()
    models = await service.list_models()
    model_list = [
        {
            "id": m.name,
            "object": "model",
            "created": 1700000000,
            "owned_by": "llm-local",
            "permission": [],
            "root": m.name,
            "parent": None,
        }
        for m in models
    ]

    # Ensure defaults are included even if Ollama list is empty
    known_ids = {m["id"] for m in model_list}
    for default_model in [settings.model_coding, settings.model_daily, settings.model_power_coding]:
        if default_model not in known_ids:
            model_list.append({
                "id": default_model,
                "object": "model",
                "created": 1700000000,
                "owned_by": "llm-local",
            })
            known_ids.add(default_model)

    return {"object": "list", "data": model_list}


@router.post("/chat/completions", summary="OpenAI-compatible chat completion")
async def chat_completions(
    request: OpenAIChatRequest,
    http_request: Request,
    service: LLMService = Depends(get_llm_service),
    settings: Settings = Depends(get_settings),
):
    """
    OpenAI-compatible /v1/chat/completions endpoint.
    Used by Continue.dev, Cursor, Aider, and ChatGPT-style apps.
    Enforces num_ctx=4096 memory limit to protect 4GB VRAM.
    """
    await service._ensure_backend()
    target_model = request.model or settings.model_coding
    temp = request.temperature if request.temperature is not None else settings.default_temperature_code
    max_tok = request.max_tokens or settings.max_tokens

    # Dynamically scale num_ctx based on prompt length (1024, 2048, or 4096) to save VRAM
    computed_ctx = service._compute_num_ctx(request.messages, max_tok)
    extra_body = {
        "options": {
            "num_ctx": computed_ctx,
        }
    }

    try:
        if request.stream:
            stream = await service.client.chat.completions.create(
                model=target_model,
                messages=request.messages,
                temperature=temp,
                max_tokens=max_tok,
                stream=True,
                extra_body=extra_body,
            )

            async def event_generator():
                try:
                    async for chunk in stream:
                        if await http_request.is_disconnected():
                            break
                        # Dump as standard OpenAI chunk JSON
                        chunk_json = chunk.model_dump_json()
                        yield f"data: {chunk_json}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    logger.error("Error streaming to OpenAI-compatible client: %s", e)
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"

            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        # Synchronous generation
        completion = await service.client.chat.completions.create(
            model=target_model,
            messages=request.messages,
            temperature=temp,
            max_tokens=max_tok,
            stream=False,
            extra_body=extra_body,
        )
        return JSONResponse(content=completion.model_dump())

    except OpenAIError as e:
        logger.error("Ollama backend error in /v1/chat/completions: %s", e)
        raise HTTPException(status_code=502, detail=f"Ollama backend error: {e}")
    except Exception as e:
        logger.exception("Unexpected error in /v1/chat/completions: %s", e)
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


@router.post("/completions", summary="OpenAI-compatible legacy/tab-autocomplete completion")
async def completions(
    request: OpenAICompletionRequest,
    http_request: Request,
    service: LLMService = Depends(get_llm_service),
    settings: Settings = Depends(get_settings),
):
    """
    OpenAI-compatible /v1/completions endpoint.
    Used by Continue.dev for inline Tab Autocomplete.
    """
    await service._ensure_backend()
    target_model = request.model or settings.model_coding
    temp = request.temperature if request.temperature is not None else settings.default_temperature_code
    max_tok = request.max_tokens or 256  # Autocomplete typically needs smaller token lengths
    prompt_text = request.prompt if isinstance(request.prompt, str) else "\n".join(request.prompt)

    # Dynamic num_ctx calculation for autocomplete
    computed_ctx = service._compute_num_ctx(prompt_text, max_tok)
    extra_body = {
        "options": {
            "num_ctx": computed_ctx,
        }
    }

    try:
        prompt_text = request.prompt if isinstance(request.prompt, str) else "\n".join(request.prompt)

        if request.stream:
            stream = await service.client.completions.create(
                model=target_model,
                prompt=prompt_text,
                temperature=temp,
                max_tokens=max_tok,
                stream=True,
                extra_body=extra_body,
            )

            async def event_generator():
                try:
                    async for chunk in stream:
                        if await http_request.is_disconnected():
                            break
                        yield f"data: {chunk.model_dump_json()}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    logger.error("Error streaming autocomplete: %s", e)
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"

            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )

        completion = await service.client.completions.create(
            model=target_model,
            prompt=prompt_text,
            temperature=temp,
            max_tokens=max_tok,
            stream=False,
            extra_body=extra_body,
        )
        return JSONResponse(content=completion.model_dump())

    except OpenAIError as e:
        logger.error("Ollama backend error in /v1/completions: %s", e)
        raise HTTPException(status_code=502, detail=f"Ollama backend error: {e}")
    except Exception as e:
        logger.exception("Unexpected error in /v1/completions: %s", e)
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")
