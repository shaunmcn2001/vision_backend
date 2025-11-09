import type { Feature, FeatureCollection, Geometry, MultiPolygon, Polygon as GeoPolygon } from "geojson";

import type { GeometryInput } from "./api";

export type LngLat = { lon: number; lat: number };

type Position = [number, number];
type Ring = Position[];
type Polygon = Ring[];

function pointInRing(point: LngLat, ring: Ring): boolean {
  if (ring.length < 3) return false;
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const xi = ring[i][0];
    const yi = ring[i][1];
    const xj = ring[j][0];
    const yj = ring[j][1];
    const intersect = yi > point.lat !== yj > point.lat && point.lon < ((xj - xi) * (point.lat - yi)) / (yj - yi + 1e-12) + xi;
    if (intersect) inside = !inside;
  }
  return inside;
}

function pointInPolygon(point: LngLat, polygon: Polygon): boolean {
  if (!polygon.length) return false;
  const [outer, ...holes] = polygon;
  if (!pointInRing(point, outer)) return false;
  return holes.every((hole) => !pointInRing(point, hole));
}

function extractGeometries(aoi: GeometryInput | null): Polygon[] {
  if (!aoi) return [];
  if ((aoi as any).type === "FeatureCollection") {
    const collection = aoi as FeatureCollection;
    return collection.features.flatMap((feature) => extractGeometries((feature as Feature).geometry as Geometry));
  }
  if ((aoi as any).type === "Feature") {
    return extractGeometries((aoi as Feature).geometry as Geometry);
  }
  const geom = aoi as Geometry;
  if (geom.type === "Polygon") {
    return (geom as GeoPolygon).coordinates as Polygon[];
  }
  if (geom.type === "MultiPolygon") {
    return (geom as MultiPolygon).coordinates.flat() as Polygon[];
  }
  return [];
}

export function pointInsideGeometry(point: LngLat, geometry: GeometryInput | null): boolean {
  const polygons = extractGeometries(geometry);
  if (!polygons.length) return false;
  return polygons.some((polygon) => pointInPolygon(point, polygon));
}
