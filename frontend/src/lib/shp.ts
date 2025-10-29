import type { FeatureCollection, Geometry } from "geojson";
import shp from "shpjs";

export async function parseShapefileZip(file: File): Promise<Geometry | null> {
  const buffer = await file.arrayBuffer();
  const geojson = await shp(buffer);
  if (!geojson) {
    return null;
  }
  if ((geojson as FeatureCollection).type === "FeatureCollection") {
    const collection = geojson as FeatureCollection;
    return collection.features[0]?.geometry ?? null;
  }
  return (geojson as { geometry?: Geometry }).geometry ?? (geojson as Geometry);
}
