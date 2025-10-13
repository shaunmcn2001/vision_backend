from app.services import zones as zone_service

# Safe defaults if the service module doesn't export legacy constants
CV_DEFAULT = getattr(zone_service, "DEFAULT_CV_THRESHOLD", 0.0)
SIMPLIFY_TOL_DEFAULT = getattr(zone_service, "DEFAULT_SIMPLIFY_TOL_M", 5)
SIMPLIFY_BUFFER_DEFAULT = getattr(zone_service, "DEFAULT_SIMPLIFY_BUFFER_M", 3)
SMOOTH_RADIUS_DEFAULT = getattr(zone_service, "DEFAULT_SMOOTH_RADIUS_M", 0)
OPEN_RADIUS_DEFAULT = getattr(zone_service, "DEFAULT_OPEN_RADIUS_M", 0)
CLOSE_RADIUS_DEFAULT = getattr(zone_service, "DEFAULT_CLOSE_RADIUS_M", 0)
N_CLASSES_DEFAULT = getattr(zone_service, "DEFAULT_N_CLASSES", 5)
CLOUD_PROB_DEFAULT = getattr(zone_service, "DEFAULT_CLOUD_PROB_MAX", 100)
