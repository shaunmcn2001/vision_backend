"""Application configuration driven by environment variables."""

from functools import lru_cache
from typing import Iterable

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration sourced from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        enable_decoding=False,
        extra="ignore",
    )

    gcp_project: str = Field(default="baradine-farm", env="GCP_PROJECT")
    gcp_region: str = Field(default="australia-southeast1", env="GCP_REGION")
    google_credentials_path: str | None = Field(
        default=None, env="GOOGLE_APPLICATION_CREDENTIALS"
    )
    cors_origins: list[str] = Field(default_factory=list, env="CORS_ORIGINS")
    max_pixels_for_direct: float = Field(default=3e7, env="MAX_PIXELS_FOR_DIRECT")
    tile_session_ttl_minutes: int = Field(default=60, env="TILE_SESSION_TTL_MINUTES")

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: str | Iterable[str] | None) -> list[str]:
        if value is None or value == "":
            return []
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return [origin.strip() for origin in value if origin.strip()]


@lru_cache()
def get_settings() -> Settings:
    """Cached settings instance."""
    return Settings()
