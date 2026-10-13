"""
Unit tests for RequestRouter and dynamic context calculation.
"""

from app.config import Settings
from app.services.request_router import RequestRouter
from app.services.llm_service import LLMService


def test_classify_complexity_simple():
    """Verify simple prompts classify as 'simple'."""
    router = RequestRouter()
    assert router.classify_complexity("Write a function to add two numbers") == "simple"
    assert router.classify_complexity("Fix typo in docstring") == "simple"
    assert router.classify_complexity("def hello_world(): pass") == "simple"


def test_classify_complexity_complex():
    """Verify architectural or difficult prompts classify as 'complex'."""
    router = RequestRouter()
    assert router.classify_complexity("Design a microservice architecture with async queue") == "complex"
    assert router.classify_complexity("Fix concurrency deadlock and race condition in multithreading code") == "complex"
    # Long prompts > 1000 chars
    long_prompt = "Refactor this code: " + ("x = 1\n" * 300)
    assert router.classify_complexity(long_prompt) == "complex"


def test_route_code_model():
    """Verify routing selects 3b for simple and 7b for complex tasks."""
    settings = Settings(
        model_coding="qwen2.5-coder:3b",
        model_power_coding="qwen2.5-coder:7b",
        smart_routing_enabled=True,
    )
    router = RequestRouter(settings=settings)

    # Simple prompt routes to 3B model
    m_simple = router.route_code_model("Write a python regex to parse emails")
    assert m_simple == "qwen2.5-coder:3b"

    # Complex prompt routes to 7B power model
    m_complex = router.route_code_model("Resolve concurrency race condition in our distributed cache architecture")
    assert m_complex == "qwen2.5-coder:7b"

    # Explicit override takes precedence
    m_explicit = router.route_code_model("Write hello world", requested_model="custom-coder:1.5b")
    assert m_explicit == "custom-coder:1.5b"


def test_compute_num_ctx():
    """Verify dynamic context allocation brackets."""
    service = LLMService()

    # Short query + small max_tokens -> 1024 bracket
    ctx_small = service._compute_num_ctx(
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=256,
    )
    assert ctx_small == 1024

    # Medium query + 1024 max_tokens -> 2048 bracket
    ctx_med = service._compute_num_ctx(
        messages=[{"role": "user", "content": "Write a python script " * 50}],
        max_tokens=1024,
    )
    assert ctx_med == 2048

    # Long query -> clamped at 4096 (default_num_ctx)
    ctx_large = service._compute_num_ctx(
        messages=[{"role": "user", "content": "Long context " * 500}],
        max_tokens=2048,
    )
    assert ctx_large == 4096
