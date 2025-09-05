# services/backend/app/services/earth_engine.py
import ee, threading

_init_lock = threading.Lock()
_initialized = False

def ensure_ee():
    """Initialize Earth Engine once per process using ADC from GOOGLE_APPLICATION_CREDENTIALS."""
    global _initialized
    if _initialized:
        return
    with _init_lock:
        if _initialized:
            return
        ee.Initialize()   # uses /secrets/ee-key.json you mounted
        _initialized = True

def ee_ping():
    """Lightweight request to confirm EE is usable."""
    ensure_ee()
    # Return project quota info or simple collection size as a smoke test
    col = ee.ImageCollection('COPERNICUS/S2_SR').filterDate('2020-01-01', '2020-01-02')
    return {'count': col.size().getInfo()}
