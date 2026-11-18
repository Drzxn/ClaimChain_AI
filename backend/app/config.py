"""Application configuration.

All settings come from environment variables (or a local ``.env`` file).
Nothing here reads a secret from source code, and no default value is a real
credential — every key defaults to an empty string so a missing key fails
loudly at the point of use rather than silently authenticating as nobody.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed view over the process environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Application -------------------------------------------------------
    app_name: str = "ClaimChain AI"
    app_env: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"
    api_prefix: str = "/api"

    # CORS origin of the Vite dev server.
    frontend_origin: str = "http://localhost:5173"

    # --- LLM (Google Gemini) ----------------------------------------------
    # Empty by default: the foundation makes no LLM calls. Stage 1 onward
    # validates presence at the call site.
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash-lite"

    # --- Search (Exa) ------------------------------------------------------
    exa_api_key: str = ""

    # --- Database ----------------------------------------------------------
    # Matches the docker-compose service, published on host port 5433 so it
    # cannot collide with a host PostgreSQL on 5432.
    database_url: str = (
        "postgresql+psycopg://claimchain:claimchain@localhost:5433/claimchain"
    )

    # --- Pipeline controls -------------------------------------------------
    # Inert in the foundation. Declared here so the values have one home and
    # are not invented ad hoc when the stages land.
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_max_repair_retries: int = Field(default=1, ge=0, le=3)
    max_subclaims: int = Field(default=8, ge=1, le=20)
    max_candidates_per_subclaim: int = Field(default=8, ge=1, le=20)
    per_claim_token_budget: int = Field(default=60_000, ge=1_000)

    # Fixture mode: replay recorded responses instead of live API calls.
    # Deferred to a later phase; the flag exists so the contract is stable.
    use_fixtures: bool = False

    # --- Derived -----------------------------------------------------------
    @property
    def is_development(self) -> bool:
        return self.app_env == "development"

    @property
    def llm_configured(self) -> bool:
        """Whether an LLM key is present. Never logs or returns the key."""
        return bool(self.gemini_api_key.strip())

    @property
    def search_configured(self) -> bool:
        """Whether a search key is present. Never logs or returns the key."""
        return bool(self.exa_api_key.strip())


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor.

    Cached so the environment is parsed once per process. Tests clear the
    cache via ``get_settings.cache_clear()`` after monkeypatching the env.
    """
    return Settings()
