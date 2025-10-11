from __future__ import annotations
import traceback
import inspect
import logging

logger = logging.getLogger(__name__)


def debug_trace(error: Exception, prefix: str = "EE TRACE") -> None:
    """
    Pretty-print the call stack and offending line when an Earth Engine error occurs.
    Example output:
      [EE TRACE] Invalid argument specified for ee.List(): 1
      at zones_core.py:224 (build_zone_thresholds)
      line:   l2 = ee.List(l).sort()

    Logs to console + FastAPI logs.
    """
    tb = traceback.extract_tb(error.__traceback__)
    for frame in reversed(tb):
        # find the first frame inside our backend code
        if "services/backend" in frame.filename:
            fname = frame.filename.split("services/backend")[-1].lstrip("/")
            module_name = inspect.getmodulename(frame.filename) or "unknown"
            print(f"[{prefix}] {error}")
            print(f"  at {fname}:{frame.lineno} ({frame.name})")
            try:
                with open(frame.filename, "r", encoding="utf-8") as f:
                    src = f.readlines()[frame.lineno - 1].strip()
                    print(f"  line:   {src}")
            except Exception:
                pass
            logger.error(
                f"[{prefix}] {error} at {fname}:{frame.lineno} ({frame.name}) [module={module_name}]"
            )
            break
    else:
        # fallback if no internal frame matched
        print(f"[{prefix}] {error}")
        traceback.print_exc()
        logger.exception("[%s] %s", prefix, error)


def debug_wrap(func):
    """
    Decorator: wrap any function that calls Earth Engine.
    On failure, prints EE traceback info.

    Usage:
        @debug_wrap
        def my_zone_fn(...):
            ...
    """

    def _inner(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            debug_trace(exc)
            raise

    return _inner
