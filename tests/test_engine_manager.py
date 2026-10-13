"""
Unit tests for EngineManager (headless inference engine lifecycle).
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.engine_manager import EngineManager


@pytest.mark.asyncio
async def test_engine_manager_is_healthy_false_on_connection_error():
    """Verify is_healthy returns False when port is closed."""
    mgr = EngineManager(base_url="http://127.0.0.1:59999")
    healthy = await mgr.is_healthy()
    assert healthy is False


@pytest.mark.asyncio
async def test_engine_manager_already_running():
    """Verify ensure_running does not start a new process if already healthy."""
    mgr = EngineManager(base_url="http://localhost:11434")
    with patch.object(mgr, "is_healthy", AsyncMock(return_value=True)):
        with patch.object(mgr, "_start_engine") as mock_start:
            started = await mgr.ensure_running()
            assert started is True
            mock_start.assert_not_called()


@pytest.mark.asyncio
async def test_engine_manager_shutdown():
    """Verify shutdown terminates the managed process."""
    mgr = EngineManager()
    mock_proc = MagicMock()
    mgr.process = mock_proc
    mgr._we_started_it = True

    mgr.shutdown()

    mock_proc.terminate.assert_called_once()
    assert mgr.process is None
    assert mgr._we_started_it is False
