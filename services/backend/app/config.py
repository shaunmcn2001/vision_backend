import os
from pydantic import BaseSettings

class Settings(BaseSettings):
    GCP_PROJECT: str = os.getenv("GCP_PROJECT", "baradine-farm")
    GCP_REGION: str = os.getenv("GCP_REGION", "australia-southeast1")
