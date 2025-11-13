import { iter } from "but-unzip";
import type { Geometry, MultiPolygon, Polygon, Position } from "geojson";

function parseCoordinateString(value: string | null | undefined): Position[] {
  if (!value) {
    return [];
  }
  return value
    .trim()
    .split(/\s+/)
    .map((tuple) => tuple.split(",", 3).slice(0, 2).map(Number) as Position)
    .filter((position) => position.length === 2 && !Number.isNaN(position[0]) && !Number.isNaN(position[1]));
}

function ensureClosedRing(ring: Position[]): Position[] {
  if (ring.length === 0) {
    return ring;
  }
  const [firstLon, firstLat] = ring[0];
  const [lastLon, lastLat] = ring[ring.length - 1];
  if (firstLon !== lastLon || firstLat !== lastLat) {
    return [...ring, [firstLon, firstLat]];
  }
  return ring;
}

function polygonElementToCoordinates(polygon: Element): Polygon["coordinates"] | null {
  const rings: Position[][] = [];

  const outer = polygon.querySelector("outerBoundaryIs > LinearRing > coordinates, LinearRing > coordinates");
  const outerRing = ensureClosedRing(parseCoordinateString(outer?.textContent));
  if (outerRing.length === 0) {
    return null;
  }
  rings.push(outerRing);

  const innerBoundaries = polygon.querySelectorAll("innerBoundaryIs > LinearRing > coordinates");
  for (const boundary of innerBoundaries) {
    const innerRing = ensureClosedRing(parseCoordinateString(boundary.textContent));
    if (innerRing.length) {
      rings.push(innerRing);
    }
  }

  return rings;
}

function documentPolygons(doc: Document): Polygon["coordinates"][] {
  const polygons: Polygon["coordinates"][] = [];
  const polygonElements = doc.querySelectorAll("Polygon");
  for (const element of polygonElements) {
    const coords = polygonElementToCoordinates(element);
    if (coords) {
      polygons.push(coords);
    }
  }
  return polygons;
}

export function kmlDocumentToGeometry(doc: Document): Geometry | null {
  if (doc.querySelector("parsererror")) {
    throw new Error("Unable to parse KML document");
  }
  const polygons = documentPolygons(doc);
  if (polygons.length === 0) {
    return null;
  }
  if (polygons.length === 1) {
    return { type: "Polygon", coordinates: polygons[0] } satisfies Polygon;
  }
  return { type: "MultiPolygon", coordinates: polygons } satisfies MultiPolygon;
}

async function extractKmzKml(buffer: ArrayBuffer): Promise<string | null> {
  const decoder = new TextDecoder();
  for (const entry of iter(new Uint8Array(buffer))) {
    if (entry.filename.toLowerCase().endsWith(".kml")) {
      const bytes = await entry.read();
      return decoder.decode(bytes);
    }
  }
  return null;
}

export async function parseKmlOrKmz(file: File): Promise<Geometry | null> {
  const extension = file.name.split(".").pop()?.toLowerCase();
  let kmlText: string | null = null;
  if (extension === "kmz") {
    const buffer = await file.arrayBuffer();
    kmlText = await extractKmzKml(buffer);
  } else {
    kmlText = await file.text();
  }
  if (!kmlText) {
    return null;
  }
  const document = new DOMParser().parseFromString(kmlText, "application/vnd.google-earth.kml+xml");
  return kmlDocumentToGeometry(document);
}
