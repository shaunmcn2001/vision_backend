# ---------- test_gee_connection.py ----------
import ee
from app import gee

# initialise your service account credentials (same as your app)
gee.initialize()

# Small square test area (change coords if you want)
geom = ee.Geometry.Polygon([
    [[153.0, -27.5],
     [153.01, -27.5],
     [153.01, -27.51],
     [153.0, -27.51],
     [153.0, -27.5]]
])

print("Testing Sentinel-2 access...")
col = gee.monthly_sentinel2_collection(
    aoi=geom,
    start="2024-03-01",
    end="2024-06-30",
    cloud_prob_max=80
)

# 1. How many monthly images?
count = col.size().getInfo()
print(f"Monthly images found: {count}")

# 2. Check NDVI stats on first image
if count > 0:
    img = col.first().normalizedDifference(["B8", "B4"]).rename("NDVI")
    stats = img.reduceRegion(
        ee.Reducer.minMax(),
        geom,
        scale=20,
        bestEffort=True,
        maxPixels=1e9
    ).getInfo()
    print("NDVI min/max:", stats)
else:
    print("No images found – geometry or dates might be outside Sentinel-2 coverage.")
