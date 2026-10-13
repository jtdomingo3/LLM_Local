"""
Chat router — general conversation endpoints.

Provides both synchronous (JSON) and streaming (SSE) chat generation.
Default model: phi4-mini (3.8B, full GPU, ~25 tok/s on RTX 3050).
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from openai import OpenAIError

from app.models.schemas import ChatRequest, ChatResponse
from app.services.llm_service import LLMService, get_llm_service

router = APIRouter(prefix="/api/chat", tags=["Chat"])


@router.post(
    "",
    response_model=ChatResponse,
    summary="Generate a chat response",
    description=(
        "Send a prompt and receive a complete JSON response. "
        "Uses phi4-mini by default for fast, general-purpose conversation."
    ),
)
async def chat(
    request: ChatRequest,
    service: LLMService = Depends(get_llm_service),
) -> ChatResponse:
    """
    Generate a complete chat response.

    The response includes the full generated text, token usage,
    and latency metrics. For real-time token streaming, use /api/chat/stream.
    """
    try:
        return await service.generate_chat(request)
    except OpenAIError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama backend error: {e}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal error: {e}",
        )


@router.post(
    "/stream",
    summary="Stream a chat response (SSE)",
    description=(
        "Send a prompt and receive tokens as they are generated via "
        "Server-Sent Events (SSE). Each event contains a JSON object "
        "with a 'token' field. The stream ends with a {'done': true} event."
    ),
)
async def chat_stream(
    request: ChatRequest,
    http_request: Request,
    service: LLMService = Depends(get_llm_service),
) -> StreamingResponse:
    """
    Stream chat tokens in real-time via SSE.

    This is ideal for frontends that want to display text as it's
    being generated, providing a much faster perceived response time.

    SSE format per chunk:
        data: {"token": "Hello", "model": "phi4-mini"}

    Final chunk:
        data: {"done": true, "model": "phi4-mini"}
    """

    async def event_stream():
        async for event in service.generate_chat_stream(request):
            # Check if client disconnected to save resources
            if await http_request.is_disconnected():
                break
            yield event

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
