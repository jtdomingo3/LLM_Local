"""
Unit tests for chat endpoints.
"""

from unittest.mock import AsyncMock

import pytest
from openai import APIConnectionError
from starlette.testclient import TestClient

from app.main import app
from app.models.schemas import ChatResponse, TokenUsage
from app.services.llm_service import get_llm_service


@pytest.fixture
def client():
    return TestClient(app)


def test_chat_empty_prompt_validation_failure(client):
    """Test validation fails when prompt is empty."""
    response = client.post("/api/chat", json={"prompt": ""})
    assert response.status_code == 422


def test_chat_successful_response(client):
    """Test successful JSON chat generation."""
    mock_service = AsyncMock()
    mock_service.generate_chat.return_value = ChatResponse(
        model="phi4-mini",
        response="This is a test response.",
        usage=TokenUsage(prompt_tokens=10, completion_tokens=15, total_tokens=25),
        latency_ms=120.5,
    )

    app.dependency_overrides[get_llm_service] = lambda: mock_service
    try:
        response = client.post(
            "/api/chat",
            json={
                "prompt": "Hello world",
                "system_prompt": "Be concise",
                "temperature": 0.5,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "phi4-mini"
        assert data["response"] == "This is a test response."
        assert data["usage"]["total_tokens"] == 25
        assert data["latency_ms"] == 120.5
        mock_service.generate_chat.assert_called_once()
    finally:
        app.dependency_overrides.clear()


def test_chat_backend_error_handling(client):
    """Test backend failure returns HTTP 502."""
    mock_service = AsyncMock()
    mock_service.generate_chat.side_effect = APIConnectionError(request=None)

    app.dependency_overrides[get_llm_service] = lambda: mock_service
    try:
        response = client.post("/api/chat", json={"prompt": "Test error"})
        assert response.status_code == 502
        assert "Ollama backend error" in response.json()["detail"]
    finally:
        app.dependency_overrides.clear()


def test_chat_streaming(client):
    """Test SSE streaming chat output."""
    mock_service = AsyncMock()

    async def mock_generator(req):
        yield 'data: {"token": "Hello", "model": "phi4-mini"}\n\n'
        yield 'data: {"token": " world", "model": "phi4-mini"}\n\n'
        yield 'data: {"done": true, "model": "phi4-mini"}\n\n'

    mock_service.generate_chat_stream = mock_generator

    app.dependency_overrides[get_llm_service] = lambda: mock_service
    try:
        response = client.post("/api/chat/stream", json={"prompt": "Say hello world"})
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        text = response.text
        assert "Hello" in text
        assert "world" in text
        assert '"done": true' in text
    finally:
        app.dependency_overrides.clear()
