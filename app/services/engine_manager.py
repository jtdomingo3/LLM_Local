"""
Engine Manager — Automates headless lifecycle of the local LLM inference engine.

Ensures the microservice is self-contained:
- Probes if the engine backend is active
- If not running, automatically launches the headless engine with
  CUDA Flash Attention (OLLAMA_FLASH_ATTENTION=1) and KV cache quantization (q8_0)
- Cleans up and terminates the engine process when the microservice shuts down
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time

import httpx

from app.config import Settings, get_settings

logger = logging.getLogger("app.engine")


class EngineManager:
    """
    Manages the lifecycle of the local LLM backend.
    Enables python run.py to stand alone as a single command.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        settings: Settings | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.settings = settings or get_settings()
        self.process: subprocess.Popen | None = None
        self._we_started_it = False

    async def is_healthy(self) -> bool:
        """Quickly check if the backend is responsive."""
        try:
            async with httpx.AsyncClient(timeout=1.5) as client:
                res = await client.get(f"{self.base_url}/api/tags")
                return res.status_code == 200
        except Exception:
            return False

    def _find_binary(self) -> str | None:
        """Locate the ollama binary on the system."""
        # Check PATH first
        bin_path = shutil.which("ollama")
        if bin_path:
            return bin_path

        # Check default Windows user directory
        user_profile = os.environ.get("USERPROFILE", "")
        default_path = os.path.join(
            user_profile, "AppData", "Local", "Programs", "Ollama", "ollama.exe"
        )
        if os.path.exists(default_path):
            return default_path

        # Check Program Files
        prog_path = r"C:\Program Files\Ollama\ollama.exe"
        if os.path.exists(prog_path):
            return prog_path

        return None

    async def ensure_running(self) -> bool:
        """
        Check if the engine is running. If not, automatically launch
        it headless with hardware optimizations.
        """
        if await self.is_healthy():
            logger.info("[OK] Local inference engine is already active.")
            return True

        logger.info("Local inference engine is not running. Launching headless sub-engine...")
        return self._start_engine()

    def _start_engine(self) -> bool:
        """Start the headless engine process."""
        binary = self._find_binary()
        if not binary:
            logger.warning(
                "Could not find ollama binary to auto-start. Backend may be unavailable."
            )
            return False

        # Apply our research hardware optimizations in the subprocess environment
        env = os.environ.copy()
        env["OLLAMA_FLASH_ATTENTION"] = "1"
        env["OLLAMA_KV_CACHE_TYPE"] = "q8_0"
        env["OLLAMA_KEEP_ALIVE"] = self.settings.ollama_keep_alive
        env["OLLAMA_MAX_LOADED_MODELS"] = str(self.settings.ollama_max_loaded_models)
        env["OLLAMA_NUM_PARALLEL"] = str(self.settings.ollama_num_parallel)

        # On Windows, suppress console popup window for clean background execution
        creationflags = 0
        if os.name == "nt":
            creationflags = 0x08000000  # CREATE_NO_WINDOW

        try:
            self.process = subprocess.Popen(
                [binary, "serve"],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creationflags,
            )
            self._we_started_it = True
            logger.info("Started headless inference engine (PID: %d)", self.process.pid)

            # Wait up to 10 seconds for the backend to initialize
            start_time = time.time()
            while time.time() - start_time < 10.0:
                time.sleep(0.5)
                try:
                    with httpx.Client(timeout=1.0) as client:
                        res = client.get(f"{self.base_url}/api/tags")
                        if res.status_code == 200:
                            logger.info("[OK] Headless engine is ready and accepting requests.")
                            return True
                except Exception:
                    pass

            logger.warning("Engine process launched, but health probe timed out after 10s.")
            return False

        except Exception as e:
            logger.error("Failed to start headless inference engine: %s", e)
            return False

    async def get_loaded_models(self) -> list[str]:
        """Fetch list of currently loaded models in memory via /api/ps."""
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                res = await client.get(f"{self.base_url}/api/ps")
                if res.status_code == 200:
                    data = res.json()
                    return [m.get("name") or m.get("model", "") for m in data.get("models", [])]
        except Exception as e:
            logger.debug("Failed to query loaded models from engine: %s", e)
        return []

    async def unload_models(self) -> list[str]:
        """
        Forcefully unloads all currently loaded models from VRAM/RAM
        by issuing keep_alive=0 requests.
        """
        loaded = await self.get_loaded_models()
        unloaded = []
        if not loaded:
            return unloaded

        async with httpx.AsyncClient(timeout=5.0) as client:
            for model_name in loaded:
                if not model_name:
                    continue
                try:
                    res = await client.post(
                        f"{self.base_url}/api/generate",
                        json={"model": model_name, "keep_alive": 0},
                    )
                    if res.status_code == 200:
                        unloaded.append(model_name)
                        logger.info("Unloaded model from VRAM: %s", model_name)
                except Exception as e:
                    logger.warning("Failed to unload model %s: %s", model_name, e)
        return unloaded

    def _unload_models_sync(self) -> list[str]:
        """Synchronously evict all active models from VRAM (used during shutdown)."""
        unloaded = []
        try:
            with httpx.Client(timeout=2.0) as client:
                res = client.get(f"{self.base_url}/api/ps")
                if res.status_code == 200:
                    models = [
                        m.get("name") or m.get("model", "")
                        for m in res.json().get("models", [])
                    ]
                    for model_name in models:
                        if model_name:
                            client.post(
                                f"{self.base_url}/api/generate",
                                json={"model": model_name, "keep_alive": 0},
                                timeout=3.0,
                            )
                            unloaded.append(model_name)
        except Exception as e:
            logger.debug("Error during sync model unload: %s", e)
        return unloaded

    def shutdown(self) -> None:
        """Clean up the headless engine process and VRAM on server shutdown."""
        # First attempt to free GPU memory gracefully
        try:
            self._unload_models_sync()
        except Exception:
            pass

        if self._we_started_it and self.process:
            logger.info("Shutting down managed inference engine (PID: %d)...", self.process.pid)
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
            self.process = None
            self._we_started_it = False
            logger.info("[OK] Managed inference engine stopped.")


# Singleton instance
_engine_manager: EngineManager | None = None


def get_engine_manager() -> EngineManager:
    """Get the singleton EngineManager instance."""
    global _engine_manager
    if _engine_manager is None:
        _engine_manager = EngineManager()
    return _engine_manager
