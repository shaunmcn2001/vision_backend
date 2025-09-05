# app/services/earth_engine.py
import ee, os
from functools import lru_cache

@lru_cache(maxsize=1)
def ensure_ee():
    # Uses the GOOGLE_APPLICATION_CREDENTIALS file from env
    ee.Initialize(opt_url='https://earthengine.googleapis.com')
    return ee

def ee_health():
    ensure_ee()
    return {"ee_initialized": True, "project": os.getenv("GCP_PROJECT")}
