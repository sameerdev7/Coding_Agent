import pytest

from agent.config import Settings


def test_defaults_without_any_env_file():
    settings = Settings(_env_file=None)
    assert settings.llm_provider == "groq"
    assert settings.max_iterations == 15
    assert settings.groq_api_key is None


def test_require_key_for_raises_when_missing():
    settings = Settings(_env_file=None)
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        settings.require_key_for("groq")


def test_require_key_for_returns_configured_key():
    settings = Settings(_env_file=None, groq_api_key="test-key-123")
    assert settings.require_key_for("groq") == "test-key-123"


def test_max_iterations_must_be_positive():
    with pytest.raises(ValueError):
        Settings(_env_file=None, max_iterations=0)


def test_provider_and_sandbox_are_validated():
    with pytest.raises(ValueError):
        Settings(_env_file=None, llm_provider="openai")
    with pytest.raises(ValueError):
        Settings(_env_file=None, sandbox="chroot")


def test_safe_defaults():
    settings = Settings(_env_file=None)
    assert settings.sandbox == "auto"
    assert settings.protect_tests is True
