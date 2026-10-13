"""
Unit tests for models endpoints.
"""

from unittest.mock import AsyncMock

import pytest
from starlette.testclient import TestClient

from app.main import app
from app.models.schemas import ModelInfo
from app.services.llm_service import get_llm_service


@pytest.fixture
def client():
    return TestClient(app)


def test_list_models_success(client):
    """Test listing models from /api/models."""
    mock_service = AsyncMock()
    mock_service.list_models.return_value = [
        ModelInfo(
            name="phi4-mini",
            size_bytes=2400000000,
            size_display="2.4 GB",
            quantization="Q4_K_M",
            family="phi",
            parameter_size="3.8B",
        ),
        ModelInfo(
            name="qwen2.5-coder:3b",
            size_bytes=2100000000,
            size_display="2.1 GB",
            quantization="Q4_K_M",
            family="qwen2",
            parameter_size="3B",
        ),
    ]

    app.dependency_overrides[get_llm_service] = lambda: mock_service
    try:
        response = client.get("/api/models")
        assert response.status_code == 200
        data = response.json()
        assert len(data["models"]) == 2
        assert data["models"][0]["name"] == "phi4-mini"
        assert data["default_daily"] == "phi4-mini"
        assert data["default_coding"] == "qwen2.5-coder:3b"
    finally:
        app.dependency_overrides.clear()


def test_active_models_configuration(client):
    """Test /api/models/active returns active model presets."""
    response = client.get("/api/models/active")
    assert response.status_code == 200
    data = response.json()
    assert "daily" in data
    assert "coding" in data
    assert "power_coding" in data
    assert data["daily"]["model"] == "phi4-mini"
    assert data["coding"]["model"] == "qwen2.5-coder:3b"
    assert data["context_window"] == 4096
