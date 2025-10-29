import { z } from "zod";

import type { GeometryInput, SeasonInput } from "./api";

export const GeoJsonSchema = z.object({
  type: z.enum(["Polygon", "MultiPolygon"]),
  coordinates: z.any()
});

export function parseGeoJsonText(text: string): GeometryInput {
  const value = JSON.parse(text);
  return GeoJsonSchema.parse(value);
}

export const SeasonCsvSchema = z.object({
  field_name: z.string().optional(),
  field_id: z.string().optional(),
  crop: z.string().optional(),
  sowing_date: z.string().min(1),
  harvest_date: z.string().min(1),
  emergence_date: z.string().optional(),
  flowering_date: z.string().optional(),
  yield_asset: z.string().optional(),
  soil_asset: z.string().optional()
});

function normaliseHeader(header: string): string {
  return header.trim().toLowerCase().replace(/\s+/g, "_");
}

export async function parseSeasonCsv(file: File): Promise<SeasonInput[]> {
  const text = await file.text();
  const [headerLine, ...rows] = text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  if (!headerLine) {
    return [];
  }
  const headers = headerLine.split(",").map(normaliseHeader);
  const seasons: SeasonInput[] = [];
  for (const row of rows) {
    const cells = row.split(",");
    const record: Record<string, string> = {};
    headers.forEach((header, index) => {
      record[header] = cells[index]?.trim() ?? "";
    });
    const parsed = SeasonCsvSchema.safeParse(record);
    if (parsed.success) {
      const { data } = parsed;
      seasons.push({
        fieldName: data.field_name || null,
        fieldId: data.field_id || null,
        crop: data.crop || null,
        sowingDate: data.sowing_date,
        harvestDate: data.harvest_date,
        emergenceDate: data.emergence_date || null,
        floweringDate: data.flowering_date || null,
        yieldAsset: data.yield_asset || null,
        soilAsset: data.soil_asset || null
      });
    }
  }
  return seasons;
}

export function validateDateRange(start: string, end: string): boolean {
  if (!start || !end) return false;
  return new Date(start) <= new Date(end);
}
