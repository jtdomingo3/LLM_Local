"""
Unit tests for the OpenAI compatibility router (/v1/chat/completions, /v1/models).
Used by Continue.dev, Cursor, Aider, etc.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from starlette.testclient import TestClient

from app.main import app
from app.models.schemas import ModelInfo
from app.services.llm_service import get_llm_service


@pytest.fixture
def client():
    return TestClient(app)


def test_v1_models(client):
    """Test /v1/models returns OpenAI format model list."""
    mock_service = AsyncMock()
    mock_service.list_models.return_value = [
        ModelInfo(
            name="qwen2.5-coder:3b",
            size_bytes=2100000000,
            size_display="2.1 GB",
            quantization="Q4_K_M",
            family="qwen2",
            parameter_size="3B",
        )
    ]

    app.dependency_overrides[get_llm_service] = lambda: mock_service
    try:
        response = client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        ids = [m["id"] for m in data["data"]]
        assert "qwen2.5-coder:3b" in ids
    finally:
        app.dependency_overrides.clear()


def test_v1_chat_completions_sync(client):
    """Test /v1/chat/completions non-streaming returns standard OpenAI format."""
    mock_service = AsyncMock()
    mock_completion = MagicMock()
    mock_completion.model_dump.return_value = {
        "id": "chatcmpl-test-123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "qwen2.5-coder:3b",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "print('hello world')"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
    }

    mock_service.client.chat.completions.create = AsyncMock(return_value=mock_completion)
    mock_service._compute_num_ctx = MagicMock(return_value=2048)

    app.dependency_overrides[get_llm_service] = lambda: mock_service
    try:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen2.5-coder:3b",
                "messages": [{"role": "user", "content": "write hello world"}],
                "stream": False,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["choices"][0]["message"]["content"] == "print('hello world')"
    finally:
        app.dependency_overrides.clear()


def test_v1_completions_sync(client):
    """Test /v1/completions tab-autocomplete non-streaming."""
    mock_service = AsyncMock()
    mock_completion = MagicMock()
    mock_completion.model_dump.return_value = {
        "id": "cmpl-test-123",
        "object": "text_completion",
        "created": 1700000000,
        "model": "qwen2.5-coder:3b",
        "choices": [
            {
                "index": 0,
                "text": "def calculate_sum():",
                "finish_reason": "stop",
            }
        ],
    }

    mock_service.client.completions.create = AsyncMock(return_value=mock_completion)
    mock_service._compute_num_ctx = MagicMock(return_value=1024)

    app.dependency_overrides[get_llm_service] = lambda: mock_service
    try:
        response = client.post(
            "/v1/completions",
            json={
                "model": "qwen2.5-coder:3b",
                "prompt": "def calculate",
                "stream": False,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["choices"][0]["text"] == "def calculate_sum():"
    finally:
        app.dependency_overrides.clear()
