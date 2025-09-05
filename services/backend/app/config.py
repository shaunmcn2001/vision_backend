# services/backend/app/config.py
import os
from pydantic_settings import BaseSettings  # <-- changed

class Settings(BaseSettings):
    GCP_PROJECT: str = os.getenv("GCP_PROJECT", "baradine-farm")
    GCP_REGION: str = os.getenv("GCP_REGION", "australia-southeast1")

    GCS_BUCKET: str | None = os.getenv("GCS_BUCKET")
    CLOUD_SQL_INSTANCE: str | None = os.getenv("CLOUD_SQL_INSTANCE")
    DB_USER: str | None = os.getenv("DB_USER")
    DB_PASS: str | None = os.getenv("DB_PASS")
    DB_NAME: str | None = os.getenv("DB_NAME")
    CLOUD_TASKS_QUEUE: str | None = os.getenv("CLOUD_TASKS_QUEUE")
    BACKFILL_START: str | None = os.getenv("BACKFILL_START")
    EE_CREDENTIALS_SECRET: str | None = os.getenv("EE_CREDENTIALS_SECRET")
    SENTRY_DSN: str | None = os.getenv("SENTRY_DSN")
