import type { FeatureCollection, Geometry, Polygon, MultiPolygon } from "geojson";

export type TileResponse = {
  token: string;
  urlTemplate: string;
  minZoom: number;
  maxZoom: number;
  expiresAt: string;
};

export type NdviMonthItem = {
  name: string;
  tile: TileResponse;
  meanNdvi?: number | null;
};

export type NdviYearlyAverage = {
  year: number;
  meanNdvi?: number | null;
};

export type NdviMonthlyAverage = {
  year: number;
  month: number;
  label: string;
  meanNdvi?: number | null;
};

export type NdviCsvDownloads = Record<string, string>;

export type NdviMonthResult = {
  items: NdviMonthItem[];
  mean: TileResponse;
  downloads: DownloadsMap;
  overallMeanNdvi?: number | null;
  yearlyAverages: NdviYearlyAverage[];
  lastYearMonthlyAverages: NdviMonthlyAverage[];
  csvDownloads: NdviCsvDownloads;
};

export type WeatherForecastDay = {
  date: string;
  tempMinC: number;
  tempMaxC: number;
  precipitationMm: number;
  windAvgKmh: number;
  windMaxKmh: number;
  description: string;
  icon?: string | null;
  gdd: number;
  sprayOk: boolean;
};

export type WeatherSeriesPoint = {
  date: string;
  value: number;
  cumulative: number;
};

export type SprayRecommendation = {
  bestDays: string[];
  hasWindow: boolean;
};

export type WeatherForecastResponse = {
  location: { lat: number; lon: number };
  forecast: WeatherForecastDay[];
  gddBaseC: number;
  chart: {
    gdd: WeatherSeriesPoint[];
    precipitation: WeatherSeriesPoint[];
  };
  sprayRecommendation: SprayRecommendation;
  precipitationTile?: TileResponse | null;
};

export type ImageryDay = {
  date: string;
  tile?: TileResponse;
  cloudPct: number;
};

export type ImageryResult = {
  days: ImageryDay[];
  summary: {
    count: number;
    avgCloudPct: number;
  };
};

export type DownloadsMap = Record<string, string>;

export type BasicZonesResult = {
  preview: {
    tile: TileResponse;
  };
  downloads: {
    rasterGeotiff: string;
    vectorShp: string;
    statsCsv: string;
  };
  vectorsGeojson: FeatureCollection;
  classCount: number;
};

export type AdvancedZonesResult = {
  preview: {
    composite: TileResponse;
    zones: TileResponse;
  };
  downloads: {
    zonesGeotiff: string;
    vectorsShp: string;
    vectorsDissolvedShp: string;
    statsCsv: string;
    statsDissolvedCsv: string;
  };
  vectorsGeojson: FeatureCollection;
  classCount: number;
};

export type SeasonInput = {
  fieldName?: string | null;
  fieldId?: string | null;
  crop?: string | null;
  sowingDate: string;
  harvestDate: string;
  emergenceDate?: string | null;
  floweringDate?: string | null;
  yieldAsset?: string | null;
  soilAsset?: string | null;
};

async function fetchJson<T>(url: string, body: unknown): Promise<T> {
  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(body)
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    const message = typeof detail.detail === "string" ? detail.detail : "Request failed";
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

export type GeometryInput = Polygon | MultiPolygon | FeatureCollection;

export function requestNdviMonth(payload: {
  aoi: GeometryInput;
  start: string;
  end: string;
  clamp?: [number, number];
}): Promise<NdviMonthResult> {
  return fetchJson<NdviMonthResult>("/api/products/ndvi-month", payload);
}

export function requestImageryDaily(payload: {
  aoi: GeometryInput;
  start: string;
  end: string;
  bands?: string[];
}): Promise<ImageryResult> {
  return fetchJson<ImageryResult>("/api/products/imagery/daily", payload);
}

export function requestBasicZones(payload: {
  aoi: GeometryInput;
  start: string;
  end: string;
  nClasses: number;
}): Promise<BasicZonesResult> {
  return fetchJson<BasicZonesResult>("/api/products/zones/basic", payload);
}

export function requestAdvancedZones(payload: {
  aoi: GeometryInput;
  breaks: number[];
  seasons: SeasonInput[];
}): Promise<AdvancedZonesResult> {
  return fetchJson<AdvancedZonesResult>("/api/products/zones/advanced", payload);
}

export function requestWeatherForecast(payload: {
  aoi: GeometryInput;
  baseTempC?: number;
}): Promise<WeatherForecastResponse> {
  return fetchJson<WeatherForecastResponse>("/api/weather/forecast", payload);
}
