"""
Core LLM Service — the brain of the microservice.

Handles all interaction with Ollama via the OpenAI-compatible API.
Supports both synchronous (JSON) and streaming (SSE) generation.

Uses AsyncOpenAI client for non-blocking I/O so the FastAPI server
can handle multiple concurrent requests efficiently.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from collections.abc import AsyncGenerator

import httpx
from openai import AsyncOpenAI, OpenAIError

from app.config import Settings, get_settings
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    CodeRequest,
    CodeResponse,
    ModelInfo,
    TokenUsage,
)

logger = logging.getLogger(__name__)


class LLMService:
    """
    Service layer for LLM inference via Ollama.

    This uses the OpenAI-compatible endpoint that Ollama exposes at /v1,
    meaning the same code works if you swap to OpenAI, Azure, or any
    other OpenAI-compatible backend — just change the base_url.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.client = AsyncOpenAI(
            base_url=self.settings.ollama_base_url,
            api_key=self.settings.ollama_api_key,
        )
        self._start_time = time.time()

    async def _ensure_backend(self) -> None:
        """Ensure local inference backend is active before processing request."""
        from app.services.engine_manager import get_engine_manager
        mgr = get_engine_manager()
        if not await mgr.is_healthy():
            logger.info("Local engine not running. Launching headless sub-engine on-demand...")
            await mgr.ensure_running()

    # ---------------------------------------------------------------
    # Chat generation
    # ---------------------------------------------------------------

    async def generate_chat(self, request: ChatRequest) -> ChatResponse:
        """Generate a chat response (non-streaming, full JSON response)."""
        await self._ensure_backend()
        model = request.model or self.settings.model_daily
        messages = self._build_messages(
            prompt=request.prompt,
            system_prompt=request.system_prompt or "You are a helpful, concise assistant.",
            history=request.messages,
        )

        start = time.perf_counter()

        try:
            completion = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )

            elapsed_ms = (time.perf_counter() - start) * 1000
            content = completion.choices[0].message.content or ""

            usage = TokenUsage()
            if completion.usage:
                usage = TokenUsage(
                    prompt_tokens=completion.usage.prompt_tokens or 0,
                    completion_tokens=completion.usage.completion_tokens or 0,
                    total_tokens=completion.usage.total_tokens or 0,
                )

            logger.info(
                "Chat generated | model=%s | tokens=%d | latency=%.0fms",
                model, usage.total_tokens, elapsed_ms,
            )

            return ChatResponse(
                model=model,
                response=content,
                usage=usage,
                latency_ms=round(elapsed_ms, 2),
            )

        except OpenAIError as e:
            logger.error("Ollama backend error: %s", e)
            raise
        except Exception as e:
            logger.error("Unexpected error during chat generation: %s", e)
            raise

    async def generate_chat_stream(self, request: ChatRequest) -> AsyncGenerator[str, None]:
        """Generate a chat response as an SSE stream, yielding tokens as they arrive."""
        await self._ensure_backend()
        model = request.model or self.settings.model_daily
        messages = self._build_messages(
            prompt=request.prompt,
            system_prompt=request.system_prompt or "You are a helpful, concise assistant.",
            history=request.messages,
        )

        try:
            stream = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=True,
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    # SSE format: data: <json>\n\n
                    yield f"data: {json.dumps({'token': token, 'model': model})}\n\n"

            # Signal end of stream
            yield f"data: {json.dumps({'done': True, 'model': model})}\n\n"

        except OpenAIError as e:
            logger.error("Streaming error: %s", e)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        except Exception as e:
            logger.error("Unexpected streaming error: %s", e)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    # ---------------------------------------------------------------
    # Code generation
    # ---------------------------------------------------------------

    async def generate_code(self, request: CodeRequest) -> CodeResponse:
        """Generate code (non-streaming, full JSON response)."""
        await self._ensure_backend()
        model = request.model or self.settings.model_coding

        system_prompt = request.system_prompt or self._build_code_system_prompt(
            request.language
        )
        messages = self._build_messages(
            prompt=request.prompt,
            system_prompt=system_prompt,
        )

        start = time.perf_counter()

        try:
            completion = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )

            elapsed_ms = (time.perf_counter() - start) * 1000
            content = completion.choices[0].message.content or ""

            usage = TokenUsage()
            if completion.usage:
                usage = TokenUsage(
                    prompt_tokens=completion.usage.prompt_tokens or 0,
                    completion_tokens=completion.usage.completion_tokens or 0,
                    total_tokens=completion.usage.total_tokens or 0,
                )

            logger.info(
                "Code generated | model=%s | lang=%s | tokens=%d | latency=%.0fms",
                model, request.language, usage.total_tokens, elapsed_ms,
            )

            return CodeResponse(
                model=model,
                language=request.language,
                response=content,
                usage=usage,
                latency_ms=round(elapsed_ms, 2),
            )

        except OpenAIError as e:
            logger.error("Ollama backend error during code gen: %s", e)
            raise
        except Exception as e:
            logger.error("Unexpected error during code generation: %s", e)
            raise

    async def generate_code_stream(self, request: CodeRequest) -> AsyncGenerator[str, None]:
        """Generate code as an SSE stream."""
        await self._ensure_backend()
        model = request.model or self.settings.model_coding

        system_prompt = request.system_prompt or self._build_code_system_prompt(
            request.language
        )
        messages = self._build_messages(
            prompt=request.prompt,
            system_prompt=system_prompt,
        )

        try:
            stream = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=True,
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    yield f"data: {json.dumps({'token': token, 'model': model})}\n\n"

            yield f"data: {json.dumps({'done': True, 'model': model})}\n\n"

        except OpenAIError as e:
            logger.error("Code streaming error: %s", e)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        except Exception as e:
            logger.error("Unexpected code streaming error: %s", e)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    # ---------------------------------------------------------------
    # Code review & explain (specialized code endpoints)
    # ---------------------------------------------------------------

    async def review_code(self, request: CodeRequest) -> CodeResponse:
        """Review code and provide feedback."""
        lang_hint = f" ({request.language})" if request.language else ""
        review_request = CodeRequest(
            prompt=request.prompt,
            language=request.language,
            model=request.model,
            system_prompt=(
                f"You are an expert code reviewer. Analyze the following{lang_hint} code and provide:\n"
                "1. **Issues**: bugs, security vulnerabilities, or logical errors\n"
                "2. **Improvements**: performance, readability, and best practices\n"
                "3. **Refactored Code**: an improved version if applicable\n"
                "Be concise and specific."
            ),
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=False,
        )
        return await self.generate_code(review_request)

    async def explain_code(self, request: CodeRequest) -> CodeResponse:
        """Explain what a code snippet does."""
        lang_hint = f" ({request.language})" if request.language else ""
        explain_request = CodeRequest(
            prompt=request.prompt,
            language=request.language,
            model=request.model,
            system_prompt=(
                f"You are an expert programmer and teacher. Explain the following{lang_hint} code:\n"
                "1. **Purpose**: what the code does at a high level\n"
                "2. **Step-by-step**: walk through the logic line by line\n"
                "3. **Key concepts**: highlight any patterns, algorithms, or techniques used\n"
                "Be clear and concise. Use examples if helpful."
            ),
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=False,
        )
        return await self.generate_code(explain_request)

    # ---------------------------------------------------------------
    # Model management & health
    # ---------------------------------------------------------------

    async def list_models(self) -> list[ModelInfo]:
        """List all models available in Ollama."""
        try:
            # Query Ollama's native API (not OpenAI-compatible) for model details
            async with httpx.AsyncClient(timeout=10.0) as http_client:
                response = await http_client.get("http://localhost:11434/api/tags")
                response.raise_for_status()
                data = response.json()

            models = []
            for m in data.get("models", []):
                size_bytes = m.get("size", 0)
                size_gb = size_bytes / (1024**3)
                details = m.get("details", {})

                models.append(ModelInfo(
                    name=m.get("name", "unknown"),
                    size_bytes=size_bytes,
                    size_display=f"{size_gb:.2f} GB",
                    quantization=details.get("quantization_level", "unknown"),
                    family=details.get("family", "unknown"),
                    parameter_size=details.get("parameter_size", "unknown"),
                ))

            return models

        except Exception as e:
            logger.error("Failed to list models: %s", e)
            return []

    async def check_ollama_health(self) -> tuple[bool, list[str]]:
        """
        Check if Ollama is reachable and return available model names.

        Returns:
            Tuple of (is_connected, list_of_model_names)
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as http_client:
                response = await http_client.get("http://localhost:11434/api/tags")
                response.raise_for_status()
                data = response.json()

            model_names = [m["name"] for m in data.get("models", [])]
            return True, model_names

        except Exception as e:
            logger.warning("Ollama health check failed: %s", e)
            return False, []

    def check_gpu(self) -> bool:
        """Check if an NVIDIA GPU is detected via nvidia-smi."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0 and bool(result.stdout.strip())
        except Exception:
            return False

    @property
    def uptime_seconds(self) -> float:
        """Get the service uptime in seconds."""
        return time.time() - self._start_time

    # ---------------------------------------------------------------
    # Private helpers
    # ---------------------------------------------------------------

    def _build_messages(
        self,
        prompt: str,
        system_prompt: str,
        history: list | None = None,
    ) -> list[dict[str, str]]:
        """Build the messages array for the OpenAI-compatible API."""
        messages = [{"role": "system", "content": system_prompt}]

        # Append conversation history if provided
        if history:
            for msg in history:
                messages.append({"role": msg.role, "content": msg.content})

        # Append the current user prompt
        messages.append({"role": "user", "content": prompt})
        return messages

    def _build_code_system_prompt(self, language: str | None = None) -> str:
        """Build a system prompt optimized for code generation."""
        lang_instruction = ""
        if language:
            lang_instruction = f" Write code in {language}."

        return (
            "You are an expert programmer. Write clean, efficient, well-documented code."
            f"{lang_instruction}"
            " Include brief explanations of your approach."
            " Follow best practices and handle edge cases."
        )


# ---------------------------------------------------------------
# Singleton instance for dependency injection
# ---------------------------------------------------------------

_llm_service: LLMService | None = None


def get_llm_service() -> LLMService:
    """Get or create the singleton LLM service instance."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
