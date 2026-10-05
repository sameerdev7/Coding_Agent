from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed, validated config — replaces scattered os.environ.get() calls.

    Reads from a .env file in the current directory (if present) and from real
    environment variables, with real env vars taking precedence.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_provider: Literal["groq", "gemini"] = "groq"
    groq_api_key: str | None = None
    gemini_api_key: str | None = None
    max_iterations: int = Field(default=15, gt=0)
    sandbox: Literal["auto", "docker", "local"] = "auto"
    protect_tests: bool = True

    def require_key_for(self, provider: str) -> str:
        key = {"groq": self.groq_api_key, "gemini": self.gemini_api_key}.get(provider)
        if not key:
            raise ValueError(
                f"no API key configured for provider '{provider}' — set "
                f"{provider.upper()}_API_KEY in .env or the environment"
            )
        return key
