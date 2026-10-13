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
    assert "optimizations" in data
    assert data["optimizations"]["smart_routing_enabled"] is True


def test_preload_model_endpoint(client):
    """Test /api/models/preload warms up the model."""
    mock_service = AsyncMock()
    mock_service.preload_model.return_value = True

    app.dependency_overrides[get_llm_service] = lambda: mock_service
    try:
        response = client.post("/api/models/preload", params={"model": "qwen2.5-coder:3b"})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["model"] == "qwen2.5-coder:3b"
        assert data["preloaded"] is True
    finally:
        app.dependency_overrides.clear()


def test_unload_models_endpoint(client):
    """Test /api/models/unload evicts models from memory."""
    from unittest.mock import patch

    with patch("app.services.engine_manager.EngineManager.unload_models", new_callable=AsyncMock) as mock_unload:
        mock_unload.return_value = ["phi4-mini", "qwen2.5-coder:3b"]
        response = client.post("/api/models/unload")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["count"] == 2
        assert "phi4-mini" in data["unloaded_models"]
