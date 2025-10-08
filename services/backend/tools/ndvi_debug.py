from __future__ import annotations

import argparse
import json

import ee


ee.Initialize()


def main() -> None:
    parser = argparse.ArgumentParser("NDVI debug")
    parser.add_argument("--aoi-geojson", required=True, help="Path to AOI GeoJSON")
    parser.add_argument("--months", required=True, nargs="+", help="YYYY-MM ...")
    parser.add_argument("--scale", type=int, default=10)
    args = parser.parse_args()

    with open(args.aoi_geojson, "r", encoding="utf-8") as handle:
        gj = json.load(handle)

    aoi = ee.FeatureCollection([ee.Feature(None, {"geom": gj["geometry"]})]).geometry()

    from app.services.zones import build_monthly_ndvi_collection

    ic = build_monthly_ndvi_collection(aoi, args.months)
    region = ee.FeatureCollection([ee.Feature(aoi)]).geometry().buffer(5).bounds(1)

    def _diag(img):
        ym = img.get("ym")
        bands = img.bandNames()
        mask_n = img.mask().bandNames().size()
        rng = img.reduceRegion(
            ee.Reducer.minMax(),
            region,
            args.scale,
            maxPixels=1e9,
            bestEffort=True,
            tileScale=4,
        )
        vv = img.mask().reduceRegion(
            ee.Reducer.sum(),
            region,
            args.scale,
            maxPixels=1e9,
            bestEffort=True,
            tileScale=4,
        )
        hist = img.reduceRegion(
            ee.Reducer.fixedHistogram(0.0, 1.0, 64),
            region,
            args.scale,
            maxPixels=1e9,
            bestEffort=True,
            tileScale=4,
        )
        return ee.Feature(
            None,
            {
                "ym": ym,
                "bands": bands,
                "mask_bands": mask_n,
                "NDVI_min": rng.get("NDVI_min"),
                "NDVI_max": rng.get("NDVI_max"),
                "valid_sum": ee.Number(vv.values().reduce(ee.Reducer.sum())),
                "hist": hist.get("NDVI"),
            },
        )

    fc = ic.map(_diag)
    out = ee.FeatureCollection(fc).aggregate_array("properties").getInfo()
    print(json.dumps({"per_month": out}, indent=2))


if __name__ == "__main__":
    main()
