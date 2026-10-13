"""
Unit tests for health check endpoints and root summary.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from starlette.testclient import TestClient

from app.main import app
from app.services.llm_service import get_llm_service


@pytest.fixture
def client():
    return TestClient(app)


def test_root_endpoint(client):
    """Test the root summary endpoint returns service info."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "LLM Local Microservice"
    assert data["status"] == "online"
    assert "endpoints" in data
    assert "/api/chat" in data["endpoints"].values()


def test_liveness_probe(client):
    """Test /health/live returns alive without checking dependencies."""
    response = client.get("/health/live")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "alive"


def test_readiness_probe_success(client):
    """Test /health/ready returns ready when Ollama is available."""
    mock_service = AsyncMock()
    mock_service.check_ollama_health.return_value = (True, ["phi4-mini", "qwen2.5-coder:3b"])
    mock_service.check_gpu = Mock(return_value=True)
    mock_service.uptime_seconds = 12.34

    app.dependency_overrides[get_llm_service] = lambda: mock_service
    try:
        response = client.get("/health/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert data["ollama_connected"] is True
        assert "phi4-mini" in data["models_available"]
        assert data["gpu_detected"] is True
    finally:
        app.dependency_overrides.clear()


def test_readiness_probe_degraded(client):
    """Test /health/ready returns degraded when Ollama is unreachable."""
    mock_service = AsyncMock()
    mock_service.check_ollama_health.return_value = (False, [])
    mock_service.check_gpu = Mock(return_value=False)
    mock_service.uptime_seconds = 5.0

    app.dependency_overrides[get_llm_service] = lambda: mock_service
    try:
        response = client.get("/health/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["ollama_connected"] is False
    finally:
        app.dependency_overrides.clear()
