# --- Safety constants for API back-compat (s2_indices.py, etc.) ---
import os

if not hasattr(globals(), "DEFAULT_CV_THRESHOLD"):
    DEFAULT_CV_THRESHOLD = float(os.getenv("ZONES_CV_THRESHOLD", "0"))

if not hasattr(globals(), "DEFAULT_SMOOTH_RADIUS_M"):
    DEFAULT_SMOOTH_RADIUS_M = 0
if not hasattr(globals(), "DEFAULT_OPEN_RADIUS_M"):
    DEFAULT_OPEN_RADIUS_M = 0
if not hasattr(globals(), "DEFAULT_CLOSE_RADIUS_M"):
    DEFAULT_CLOSE_RADIUS_M = 0
if not hasattr(globals(), "DEFAULT_SIMPLIFY_TOL_M"):
    DEFAULT_SIMPLIFY_TOL_M = 5
if not hasattr(globals(), "DEFAULT_SIMPLIFY_BUFFER_M"):
    DEFAULT_SIMPLIFY_BUFFER_M = 3
if not hasattr(globals(), "DEFAULT_N_CLASSES"):
    DEFAULT_N_CLASSES = 5
if not hasattr(globals(), "DEFAULT_CLOUD_PROB_MAX"):
    DEFAULT_CLOUD_PROB_MAX = 100
