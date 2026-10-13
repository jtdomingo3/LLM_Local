"""
Configuration module for the LLM Local Microservice.

Loads settings from .env file using pydantic-settings.
All optimization defaults from our research are baked in here.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ---------------------------------------------------------------
    # Ollama connection
    # ---------------------------------------------------------------
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_api_key: str = "ollama"

    # ---------------------------------------------------------------
    # Server config
    # ---------------------------------------------------------------
    service_host: str = "0.0.0.0"
    service_port: int = 8100

    # ---------------------------------------------------------------
    # Model configuration
    # Optimized for RTX 3050 4GB VRAM + 24GB DDR5 RAM:
    #   - phi4-mini: 3.8B, ~2.5 GB VRAM, full GPU, ~25 tok/s
    #   - qwen2.5-coder:3b: 3B, ~2.2 GB VRAM, full GPU, ~28 tok/s
    #   - qwen2.5-coder:7b: 7B, ~4.5 GB hybrid GPU/CPU, ~8 tok/s
    # ---------------------------------------------------------------
    model_daily: str = "phi4-mini"
    model_coding: str = "qwen2.5-coder:3b"
    model_power_coding: str = "qwen2.5-coder:7b"

    # ---------------------------------------------------------------
    # Engine optimization
    # Keep-alive: 90s releases VRAM quickly during idle periods
    # Max loaded models: 1 strictly prevents VRAM spillover on 4GB GPUs
    # Num parallel: 1 prevents multiple concurrent context allocations
    # ---------------------------------------------------------------
    ollama_keep_alive: str = "90s"
    ollama_max_loaded_models: int = 1
    ollama_num_parallel: int = 1

    # ---------------------------------------------------------------
    # Generation defaults
    # Context window: 4096 is safe for 4GB VRAM with q8_0 KV cache
    # Temperature: 0.7 for chat (creative), 0.2 for code (deterministic)
    # Default token generation caps: reduced to 2048 to prevent runaway memory
    # ---------------------------------------------------------------
    default_num_ctx: int = 4096
    default_temperature_chat: float = 0.7
    default_temperature_code: float = 0.2
    max_tokens: int = 4096
    max_tokens_chat: int = 2048
    max_tokens_code: int = 2048

    # ---------------------------------------------------------------
    # Response Caching (RAM-saving & zero-GPU token reuse)
    # ---------------------------------------------------------------
    response_cache_enabled: bool = True
    response_cache_max_size: int = 256
    response_cache_ttl_seconds: int = 3600

    # ---------------------------------------------------------------
    # Smart Request Routing (routes tasks to right-sized model)
    # ---------------------------------------------------------------
    smart_routing_enabled: bool = True

    # ---------------------------------------------------------------
    # Logging
    # ---------------------------------------------------------------
    log_level: str = "INFO"

    @property
    def model_map(self) -> dict[str, str]:
        """Map of purpose to model name for easy lookup."""
        return {
            "daily": self.model_daily,
            "coding": self.model_coding,
            "power_coding": self.model_power_coding,
        }


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance. Reads .env only once."""
    return Settings()
