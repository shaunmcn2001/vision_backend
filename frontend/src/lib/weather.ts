import type { MapLayerConfig } from "../components/Map";
import type { WeatherForecastDay, WeatherForecastResponse } from "./api";

const ICON_LOOKUP: Record<string, string> = {
  "01": "☀️",
  "02": "🌤️",
  "03": "⛅️",
  "04": "☁️",
  "09": "🌦️",
  "10": "🌧️",
  "11": "⛈️",
  "13": "❄️",
  "50": "🌫️"
};

function iconFromCode(code?: string | null, description?: string): string {
  if (!code) {
    return description?.includes("rain") ? "🌧️" : "☁️";
  }
  const key = code.slice(0, 2);
  return ICON_LOOKUP[key] ?? "☁️";
}

function formatDateLabel(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleDateString(undefined, {
    weekday: "short",
    month: "short",
    day: "numeric"
  });
}

export const WeatherService = {
  iconFor(day: WeatherForecastDay) {
    return iconFromCode(day.icon ?? undefined, day.description);
  },
  dateLabel(value: string) {
    return formatDateLabel(value);
  },
  buildPrecipitationLayer(response: WeatherForecastResponse | null): MapLayerConfig | null {
    if (!response?.precipitationTile) {
      return null;
    }
    return {
      id: "precipitation-overlay",
      type: "raster",
      visible: false,
      tile: response.precipitationTile,
      opacity: 0.6,
    };
  }
};
