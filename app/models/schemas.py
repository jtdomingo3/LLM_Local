"""
Pydantic schemas for request/response validation.

All models use strict validation to protect the microservice
from malformed payloads before hitting the LLM backend.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------
# Shared / Common
# ---------------------------------------------------------------

class Message(BaseModel):
    """A single message in a conversation."""
    role: str = Field(
        default="user",
        description="Message role: 'system', 'user', or 'assistant'",
    )
    content: str = Field(
        ...,
        min_length=1,
        description="The message content",
    )


class TokenUsage(BaseModel):
    """Token usage statistics for a generation request."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


# ---------------------------------------------------------------
# Chat Schemas
# ---------------------------------------------------------------

class ChatRequest(BaseModel):
    """Request body for /api/chat endpoints."""
    prompt: str = Field(
        ...,
        min_length=1,
        description="The input prompt or question for the LLM",
        examples=["Explain Python decorators in 3 sentences"],
    )
    model: str | None = Field(
        default=None,
        description="Model name. Defaults to the daily driver (phi4-mini)",
    )
    system_prompt: str | None = Field(
        default=None,
        description="Optional system instruction to guide the model's behavior",
        examples=["You are a helpful, concise assistant."],
    )
    messages: list[Message] | None = Field(
        default=None,
        description="Optional conversation history. If provided, 'prompt' is appended as the latest user message",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature. Higher = more creative, lower = more deterministic",
    )
    max_tokens: int = Field(
        default=2048,
        ge=1,
        le=32768,
        description="Maximum number of tokens to generate",
    )
    stream: bool = Field(
        default=False,
        description="If true, response is streamed via SSE",
    )


class ChatResponse(BaseModel):
    """Response body for /api/chat endpoints."""
    model: str = Field(description="The model that generated this response")
    response: str = Field(description="The generated text")
    usage: TokenUsage = Field(default_factory=TokenUsage)
    latency_ms: float = Field(
        default=0.0,
        description="Time taken to generate the response in milliseconds",
    )


# ---------------------------------------------------------------
# Code Schemas
# ---------------------------------------------------------------

class CodeRequest(BaseModel):
    """Request body for /api/code endpoints."""
    prompt: str = Field(
        ...,
        min_length=1,
        description="The coding task or code to review/explain",
        examples=["Write a Python function to merge two sorted lists"],
    )
    language: str | None = Field(
        default=None,
        description="Target programming language (e.g., 'python', 'javascript', 'rust')",
        examples=["python"],
    )
    model: str | None = Field(
        default=None,
        description="Model name. Defaults to the coding model (qwen2.5-coder:3b)",
    )
    system_prompt: str | None = Field(
        default=None,
        description="Optional system instruction. Overrides the default coding prompt",
    )
    temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        description="Sampling temperature. Low values (0.1-0.3) recommended for code",
    )
    max_tokens: int = Field(
        default=2048,
        ge=1,
        le=32768,
        description="Maximum number of tokens to generate",
    )
    stream: bool = Field(
        default=False,
        description="If true, response is streamed via SSE",
    )


class CodeResponse(BaseModel):
    """Response body for /api/code endpoints."""
    model: str = Field(description="The model that generated this response")
    language: str | None = Field(
        default=None,
        description="The target programming language",
    )
    response: str = Field(description="The generated code or review")
    usage: TokenUsage = Field(default_factory=TokenUsage)
    latency_ms: float = Field(
        default=0.0,
        description="Time taken to generate the response in milliseconds",
    )


# ---------------------------------------------------------------
# Health Schemas
# ---------------------------------------------------------------

class HealthResponse(BaseModel):
    """Response body for /health endpoints."""
    status: str = Field(description="Service status: 'alive', 'ready', or 'degraded'")
    ollama_connected: bool = Field(
        default=False,
        description="Whether the Ollama backend is reachable",
    )
    models_available: list[str] = Field(
        default_factory=list,
        description="List of models currently available in Ollama",
    )
    gpu_detected: bool = Field(
        default=False,
        description="Whether an NVIDIA GPU was detected",
    )
    uptime_seconds: float = Field(
        default=0.0,
        description="How long the service has been running",
    )
    cache_stats: dict[str, Any] | None = Field(
        default=None,
        description="Response cache performance and size metrics",
    )
    loaded_models: list[str] = Field(
        default_factory=list,
        description="Models currently resident in GPU/host memory",
    )


# ---------------------------------------------------------------
# Model Info Schemas
# ---------------------------------------------------------------

class ModelInfo(BaseModel):
    """Information about an available model."""
    name: str = Field(description="Model name as known to Ollama")
    size_bytes: int = Field(default=0, description="Model size in bytes")
    size_display: str = Field(default="", description="Human-readable model size")
    quantization: str = Field(default="unknown", description="Quantization level")
    family: str = Field(default="unknown", description="Model family")
    parameter_size: str = Field(default="unknown", description="Parameter count")


class ModelsListResponse(BaseModel):
    """Response body for /api/models endpoints."""
    models: list[ModelInfo] = Field(default_factory=list)
    default_daily: str = Field(description="Default model for daily use")
    default_coding: str = Field(description="Default model for coding")
    default_power_coding: str = Field(description="Default model for power coding")
