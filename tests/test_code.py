"""
Unit tests for code generation, review, and explanation endpoints.
"""

from unittest.mock import AsyncMock

import pytest
from starlette.testclient import TestClient

from app.main import app
from app.models.schemas import CodeResponse, TokenUsage
from app.services.llm_service import get_llm_service


@pytest.fixture
def client():
    return TestClient(app)


def test_code_generate_success(client):
    """Test generating code via /api/code."""
    mock_service = AsyncMock()
    mock_service.generate_code.return_value = CodeResponse(
        model="qwen2.5-coder:3b",
        language="python",
        response="def add(a: int, b: int) -> int:\n    return a + b",
        usage=TokenUsage(prompt_tokens=15, completion_tokens=20, total_tokens=35),
        latency_ms=95.0,
    )

    app.dependency_overrides[get_llm_service] = lambda: mock_service
    try:
        response = client.post(
            "/api/code",
            json={
                "prompt": "Write a python function to add two numbers",
                "language": "python",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "qwen2.5-coder:3b"
        assert data["language"] == "python"
        assert "def add" in data["response"]
        assert data["usage"]["total_tokens"] == 35
    finally:
        app.dependency_overrides.clear()


def test_code_review_success(client):
    """Test reviewing code via /api/code/review."""
    mock_service = AsyncMock()
    mock_service.review_code.return_value = CodeResponse(
        model="qwen2.5-coder:3b",
        language="python",
        response="Code Review:\n- Use type hints\n- Handle None inputs",
        usage=TokenUsage(prompt_tokens=25, completion_tokens=30, total_tokens=55),
        latency_ms=110.0,
    )

    app.dependency_overrides[get_llm_service] = lambda: mock_service
    try:
        response = client.post(
            "/api/code/review",
            json={
                "prompt": "def divide(a, b): return a / b",
                "language": "python",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "Code Review:" in data["response"]
    finally:
        app.dependency_overrides.clear()


def test_code_explain_success(client):
    """Test explaining code via /api/code/explain."""
    mock_service = AsyncMock()
    mock_service.explain_code.return_value = CodeResponse(
        model="qwen2.5-coder:3b",
        language="javascript",
        response="This function performs asynchronous debouncing.",
        usage=TokenUsage(prompt_tokens=20, completion_tokens=25, total_tokens=45),
        latency_ms=85.0,
    )

    app.dependency_overrides[get_llm_service] = lambda: mock_service
    try:
        response = client.post(
            "/api/code/explain",
            json={
                "prompt": "const debounce = (fn, delay) => { ... }",
                "language": "javascript",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "debouncing" in data["response"]
    finally:
        app.dependency_overrides.clear()


def test_code_stream_success(client):
    """Test code SSE streaming output."""
    mock_service = AsyncMock()

    async def mock_code_generator(req):
        yield 'data: {"token": "def ", "model": "qwen2.5-coder:3b"}\n\n'
        yield 'data: {"token": "main():", "model": "qwen2.5-coder:3b"}\n\n'
        yield 'data: {"done": true, "model": "qwen2.5-coder:3b"}\n\n'

    mock_service.generate_code_stream = mock_code_generator

    app.dependency_overrides[get_llm_service] = lambda: mock_service
    try:
        response = client.post(
            "/api/code/stream",
            json={"prompt": "Write main function in python", "language": "python"},
        )
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        assert "def " in response.text
        assert "main():" in response.text
    finally:
        app.dependency_overrides.clear()
