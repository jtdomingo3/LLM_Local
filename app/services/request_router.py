"""
Smart Request Router — Heuristic complexity classifier for optimal model routing.

Directs incoming tasks to the lightest capable model:
- Simple/everyday code -> qwen2.5-coder:3b (runs 100% on GPU, ~28 tok/s, saves VRAM)
- Heavy/architectural code -> qwen2.5-coder:7b (hybrid GPU/CPU, maximum power)
- Chat/Q&A -> phi4-mini (100% GPU, ~25 tok/s, low memory footprint)
"""

from __future__ import annotations

import logging
import re
from typing import Literal

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

ComplexityLevel = Literal["simple", "complex"]

# Keywords indicating complex programming tasks requiring deeper reasoning
COMPLEX_PATTERNS = [
    r"\barchitecture\b",
    r"\bconcurrency\b",
    r"\bmultithreading\b",
    r"\brace condition\b",
    r"\bdeadlock\b",
    r"\bmemory leak\b",
    r"\bprofiling\b",
    r"\boptimization\b",
    r"\bdistributed\b",
    r"\bmicroservice architecture\b",
    r"\bcompiler\b",
    r"\babstract syntax tree\b",
    r"\bast\b",
    r"\bmeta-programming\b",
    r"\bdesign pattern\b",
    r"\bcomplex algorithm\b",
    r"\bdynamic programming\b",
    r"\bgraph traversal\b",
    r"\brefactor entire\b",
]

# Keywords indicating straightforward coding tasks
SIMPLE_PATTERNS = [
    r"\bwrite a function to\b",
    r"\bregex\b",
    r"\bformat\b",
    r"\bfix typo\b",
    r"\bboilerplate\b",
    r"\bdocstring\b",
    r"\bunit test for\b",
    r"\bjson parse\b",
    r"\bcsv\b",
    r"\bhello world\b",
    r"\bsimple\b",
    r"\blist comprehension\b",
]


class RequestRouter:
    """Classifies prompt complexity and selects the best model for resource efficiency."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._compiled_complex = [re.compile(p, re.IGNORECASE) for p in COMPLEX_PATTERNS]
        self._compiled_simple = [re.compile(p, re.IGNORECASE) for p in SIMPLE_PATTERNS]

    def classify_complexity(self, prompt: str) -> ComplexityLevel:
        """
        Classify prompt complexity based on prompt length and keyword heuristics.

        Returns:
            'simple' or 'complex'
        """
        prompt_len = len(prompt.strip())

        # Long prompts (> 1000 characters / ~250 words) often involve complex contexts
        if prompt_len > 1000:
            return "complex"

        # Explicit simple pattern check
        for pattern in self._compiled_simple:
            if pattern.search(prompt):
                return "simple"

        # Explicit complex pattern check
        complex_hits = sum(1 for pattern in self._compiled_complex if pattern.search(prompt))
        if complex_hits >= 1:
            return "complex"

        return "simple"

    def route_code_model(self, prompt: str, requested_model: str | None = None) -> str:
        """
        Select the optimal code model for the prompt.
        If a specific model is explicitly requested (not None and not 'auto'), respect it.
        """
        if requested_model and requested_model.lower() not in ("auto", "default", ""):
            return requested_model

        if not self.settings.smart_routing_enabled:
            return self.settings.model_coding

        complexity = self.classify_complexity(prompt)
        if complexity == "complex":
            logger.info("Smart Router: Classified task as COMPLEX -> routing to %s", self.settings.model_power_coding)
            return self.settings.model_power_coding

        logger.info("Smart Router: Classified task as SIMPLE -> routing to %s", self.settings.model_coding)
        return self.settings.model_coding

    def route_chat_model(self, prompt: str, requested_model: str | None = None) -> str:
        """Select the optimal model for chat."""
        if requested_model and requested_model.lower() not in ("auto", "default", ""):
            return requested_model
        return self.settings.model_daily


# Singleton instance
_request_router: RequestRouter | None = None


def get_request_router(settings: Settings | None = None) -> RequestRouter:
    """Get the singleton RequestRouter instance."""
    global _request_router
    if _request_router is None:
        _request_router = RequestRouter(settings or get_settings())
    return _request_router
