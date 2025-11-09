export type ProductType = "ndvi-month" | "imagery-daily" | "zones-basic" | "zones-advanced";

export interface Geometry {
  type: string;
  coordinates: unknown;
}

export interface TileResponse {
  token: string;
  urlTemplate: string;
  minZoom: number;
  maxZoom: number;
}

export interface TilePreview {
  tile: TileResponse;
}

export interface NdviMonthItem {
  name: string;
  tile: TileResponse;
  meanNdvi?: number | null;
}

export interface NdviMonthResult {
  items: NdviMonthItem[];
  mean: TilePreview;
}

export interface ImageryDay {
  date: string;
  tile: TileResponse;
  cloudPct: number;
}

export interface ImagerySummary {
  count: number;
  avgCloudPct: number;
}

export interface ImageryResult {
  days: ImageryDay[];
  summary: ImagerySummary;
}

export interface BasicZoneDownloads {
  rasterGeotiff: string;
  vectorShp: string;
  statsCsv: string;
}

export interface BasicZonesResult {
  preview: TilePreview;
  downloads: BasicZoneDownloads;
}

export interface AdvancedZoneDownloads {
  zonesGeotiff: string;
  vectorsShp: string;
  vectorsDissolvedShp: string;
  statsCsv: string;
  statsDissolvedCsv: string;
}

export interface AdvancedPreview {
  composite: TilePreview;
  zones: TilePreview;
}

export interface AdvancedZonesResult {
  preview: AdvancedPreview;
  downloads: AdvancedZoneDownloads;
}

export interface SeasonRow {
  fieldName?: string | null;
  fieldId?: string | null;
  crop?: string | null;
  sowingDate: string;
  harvestDate: string;
  emergenceDate?: string | null;
  floweringDate?: string | null;
  yieldAsset?: string | null;
  soilAsset?: string | null;
}

export type ProductResult =
  | { type: "ndvi-month"; data: NdviMonthResult }
  | { type: "imagery-daily"; data: ImageryResult }
  | { type: "zones-basic"; data: BasicZonesResult }
  | { type: "zones-advanced"; data: AdvancedZonesResult };
