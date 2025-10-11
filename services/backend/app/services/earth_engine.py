# app/services/earth_engine.py
import os

import ee
from functools import lru_cache
from app.services.ee_patches import apply_ee_runtime_patches
from app.services.ee_debug import debug_trace, debug_wrap  # noqa: F401

apply_ee_runtime_patches()


@lru_cache(maxsize=1)
def ensure_ee():
    # Uses the GOOGLE_APPLICATION_CREDENTIALS file from env
    ee.Initialize(opt_url="https://earthengine.googleapis.com")
    return ee


def ee_health():
    ensure_ee()
    return {"ee_initialized": True, "project": os.getenv("GCP_PROJECT")}
