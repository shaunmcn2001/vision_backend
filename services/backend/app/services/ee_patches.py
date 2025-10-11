# services/backend/app/services/ee_patches.py
from __future__ import annotations
import logging
import ee

logger = logging.getLogger(__name__)

# keep originals so we don't recurse
_ORIG_ee_List = ee.List
_ORIG_fromImages = getattr(ee.ImageCollection, "fromImages", None)

def _safe_ee_list(x):
    """
    GLOBAL PATCH for ee.List:
    - Avoids 'Invalid argument specified for ee.List(): 1' by wrap-then-flatten.
      * If x is ee.List  -> [x].flatten() == x
      * If x is scalar   -> [x]
    """
    try:
        return _ORIG_ee_List([x]).flatten()
    except Exception as exc:  # ultra-defensive
        logger.warning("ee.List patch failed for %r (%s); returning []", x, exc)
        return _ORIG_ee_List([])

def _safe_from_images(images):
    """
    Safe wrapper for ee.ImageCollection.fromImages:
    - Normalizes 'images' to a list (using patched ee.List)
    - Forces each element to ee.Image; invalids become a 0-mask image
    - Always returns a valid ImageCollection
    """
    lst = _safe_ee_list(images)
    def _to_image(im):
        try:
            return ee.Image(im)
        except Exception:
            return ee.Image(0).updateMask(ee.Image(0))
    coerced = lst.map(lambda i: _to_image(i))
    return ee.ImageCollection(coerced)

_PATCHED = False

def apply_ee_runtime_patches():
    """Idempotent: call once at process start (or import for side-effect)."""
    global _PATCHED
    if _PATCHED:
        return
    ee.List = _safe_ee_list  # type: ignore[attr-defined]
    if _ORIG_fromImages is not None:
        ee.ImageCollection.fromImages = _safe_from_images  # type: ignore[attr-defined]
    _PATCHED = True
    logger.info("Applied EE patches: safe ee.List + safe ImageCollection.fromImages")
