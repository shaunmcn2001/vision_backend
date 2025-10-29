import { useEffect, useRef } from "react";
import maplibregl, { Map as MapLibreMap } from "maplibre-gl";

import type { GeometryInput, TileResponse } from "../lib/api";
import "maplibre-gl/dist/maplibre-gl.css";

export type MapProps = {
  aoi: GeometryInput | null;
  tile: TileResponse | null;
};

function extractGeometry(geometry: GeometryInput | null) {
  if (!geometry) return null;
  if (geometry.type === "FeatureCollection") {
    return geometry.features[0]?.geometry ?? null;
  }
  return geometry as any;
}

function computeBounds(geometry: GeometryInput | null) {
  const geom = extractGeometry(geometry);
  if (!geom || !("coordinates" in geom)) return null;
  const coords = (geom.coordinates as number[][][] | number[][][][]);
  const flat: number[][] = [];
  function flatten(arr: any) {
    if (typeof arr[0] === "number") {
      flat.push(arr as number[]);
    } else {
      arr.forEach(flatten);
    }
  }
  flatten(coords);
  if (!flat.length) return null;
  let minLng = flat[0][0];
  let minLat = flat[0][1];
  let maxLng = flat[0][0];
  let maxLat = flat[0][1];
  flat.forEach(([lng, lat]) => {
    minLng = Math.min(minLng, lng);
    minLat = Math.min(minLat, lat);
    maxLng = Math.max(maxLng, lng);
    maxLat = Math.max(maxLat, lat);
  });
  return [minLng, minLat, maxLng, maxLat] as const;
}

export function Map({ aoi, tile }: MapProps) {
  const mapRef = useRef<HTMLDivElement | null>(null);
  const mapInstance = useRef<MapLibreMap | null>(null);

  useEffect(() => {
    if (!mapRef.current || mapInstance.current) return;
    const map = new maplibregl.Map({
      container: mapRef.current,
      style: "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
      center: [146.9, -32.1],
      zoom: 5,
      attributionControl: true
    });
    map.addControl(new maplibregl.NavigationControl(), "top-right");
    mapInstance.current = map;
  }, []);

  useEffect(() => {
    const map = mapInstance.current;
    if (!map || !aoi) return;
    const sourceId = "aoi";
    const geometry = extractGeometry(aoi);
    if (!geometry) return;
    const featureCollection = {
      type: "FeatureCollection" as const,
      features: [{ type: "Feature" as const, geometry, properties: {} }]
    };
    if (!map.getSource(sourceId)) {
      map.addSource(sourceId, {
        type: "geojson",
        data: featureCollection
      });
      map.addLayer({
        id: "aoi-fill",
        type: "fill",
        source: sourceId,
        paint: {
          "fill-color": "#2563eb",
          "fill-opacity": 0.2
        }
      });
      map.addLayer({
        id: "aoi-outline",
        type: "line",
        source: sourceId,
        paint: {
          "line-color": "#1e3a8a",
          "line-width": 2
        }
      });
    } else {
      const source = map.getSource(sourceId) as maplibregl.GeoJSONSource;
      source.setData(featureCollection);
    }
    const bounds = computeBounds(aoi);
    if (bounds) {
      map.fitBounds([
        [bounds[0], bounds[1]],
        [bounds[2], bounds[3]]
      ], { padding: 40, duration: 500 });
    }
  }, [aoi]);

  useEffect(() => {
    const map = mapInstance.current;
    if (!map) return;
    const sourceId = "product-tile";
    const layerId = "product-tile-layer";
    if (map.getLayer(layerId)) {
      map.removeLayer(layerId);
    }
    if (map.getSource(sourceId)) {
      map.removeSource(sourceId);
    }
    if (!tile) return;
    map.addSource(sourceId, {
      type: "raster",
      tiles: [window.location.origin + tile.urlTemplate],
      tileSize: 256
    });
    map.addLayer({
      id: layerId,
      type: "raster",
      source: sourceId,
      paint: {
        "raster-opacity": 0.85
      }
    });
  }, [tile]);

  return <div ref={mapRef} className="h-full w-full rounded-lg border border-slate-200" />;
}
