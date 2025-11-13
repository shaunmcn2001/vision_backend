import type { Feature, FeatureCollection, Geometry, MultiPolygon, Polygon } from "geojson";
import shp from "shpjs";

function geometryToPolygonList(geometry: Geometry | null | undefined): Polygon["coordinates"][] {
  if (!geometry) {
    return [];
  }
  if (geometry.type === "Polygon") {
    return [geometry.coordinates];
  }
  if (geometry.type === "MultiPolygon") {
    return geometry.coordinates;
  }
  return [];
}

export function collectionToGeometry(collection: FeatureCollection): Geometry {
  const polygonCoordinates: Polygon["coordinates"][] = [];
  for (const feature of collection.features) {
    polygonCoordinates.push(...geometryToPolygonList(feature.geometry));
  }
  if (polygonCoordinates.length === 0) {
    throw new Error("AOI geometry must contain polygons");
  }
  if (polygonCoordinates.length === 1) {
    return { type: "Polygon", coordinates: polygonCoordinates[0] } satisfies Polygon;
  }
  return { type: "MultiPolygon", coordinates: polygonCoordinates } satisfies MultiPolygon;
}

function featureToGeometry(feature: Feature): Geometry | null {
  if (!feature.geometry) {
    return null;
  }
  if (feature.geometry.type === "Polygon" || feature.geometry.type === "MultiPolygon") {
    return feature.geometry;
  }
  throw new Error("AOI geometry must contain polygons");
}

export async function parseShapefileZip(file: File): Promise<Geometry | null> {
  const buffer = await file.arrayBuffer();
  const geojson = await shp(buffer);
  if (!geojson) {
    return null;
  }
  if ((geojson as FeatureCollection).type === "FeatureCollection") {
    return collectionToGeometry(geojson as FeatureCollection);
  }
  if ((geojson as Feature).type === "Feature") {
    return featureToGeometry(geojson as Feature);
  }
  const geometry = geojson as Geometry;
  if (geometry.type === "Polygon" || geometry.type === "MultiPolygon") {
    return geometry;
  }
  throw new Error("AOI geometry must contain polygons");
}
