"""
Code router — coding assistant endpoints.

Provides code generation, streaming, review, and explanation.
Default model: qwen2.5-coder:3b (3B, full GPU, ~28 tok/s on RTX 3050).
Temperature defaults to 0.2 for deterministic, reliable code output.
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from openai import OpenAIError

from app.models.schemas import CodeRequest, CodeResponse
from app.services.llm_service import LLMService, get_llm_service

router = APIRouter(prefix="/api/code", tags=["Code"])


@router.post(
    "",
    response_model=CodeResponse,
    summary="Generate code",
    description=(
        "Send a coding task and receive generated code. "
        "Uses qwen2.5-coder:3b by default. Specify 'language' for "
        "targeted output (e.g., 'python', 'javascript', 'rust')."
    ),
)
async def generate_code(
    request: CodeRequest,
    service: LLMService = Depends(get_llm_service),
) -> CodeResponse:
    """
    Generate code based on a natural language description.

    Optimized with low temperature (0.2) for deterministic output
    and a coding-specific system prompt for best results.
    """
    try:
        return await service.generate_code(request)
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
    summary="Stream code generation (SSE)",
    description=(
        "Stream generated code tokens in real-time via Server-Sent Events. "
        "Ideal for IDE integrations that display code as it's written."
    ),
)
async def code_stream(
    request: CodeRequest,
    http_request: Request,
    service: LLMService = Depends(get_llm_service),
) -> StreamingResponse:
    """
    Stream code tokens in real-time via SSE.

    SSE format per chunk:
        data: {"token": "def ", "model": "qwen2.5-coder:3b"}

    Final chunk:
        data: {"done": true, "model": "qwen2.5-coder:3b"}
    """

    async def event_stream():
        async for event in service.generate_code_stream(request):
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


@router.post(
    "/review",
    response_model=CodeResponse,
    summary="Review code",
    description=(
        "Submit code for review. The model will analyze it for bugs, "
        "security issues, and suggest improvements with refactored code."
    ),
)
async def review_code(
    request: CodeRequest,
    service: LLMService = Depends(get_llm_service),
) -> CodeResponse:
    """
    Review code and provide structured feedback.

    Returns:
    - Issues found (bugs, security, logic errors)
    - Suggested improvements
    - Refactored version of the code
    """
    try:
        return await service.review_code(request)
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
    "/explain",
    response_model=CodeResponse,
    summary="Explain code",
    description=(
        "Submit a code snippet and receive a detailed explanation "
        "of what it does, step-by-step logic, and key concepts."
    ),
)
async def explain_code(
    request: CodeRequest,
    service: LLMService = Depends(get_llm_service),
) -> CodeResponse:
    """
    Explain a code snippet in detail.

    Returns:
    - High-level purpose
    - Step-by-step walkthrough
    - Key patterns, algorithms, or techniques used
    """
    try:
        return await service.explain_code(request)
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
