import { useState } from "react";
import { Label } from "@radix-ui/react-label";

import type { GeometryInput } from "../lib/api";
import { parseShapefileZip } from "../lib/shp";
import { parseGeoJsonText } from "../lib/validators";

export type AOIInputProps = {
  value: GeometryInput | null;
  onChange: (geometry: GeometryInput | null) => void;
};

export function AOIInput({ value, onChange }: AOIInputProps) {
  const [rawGeoJson, setRawGeoJson] = useState<string>("");
  const [error, setError] = useState<string | null>(null);

  async function handleFileInput(file: File | undefined) {
    if (!file) return;
    try {
      const geometry = await parseShapefileZip(file);
      if (!geometry) {
        setError("Unable to read shapefile geometry.");
        return;
      }
      onChange({
        type: "FeatureCollection",
        features: [
          {
            type: "Feature",
            geometry,
            properties: {}
          }
        ]
      });
      setError(null);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to parse shapefile";
      setError(message);
    }
  }

  function handleGeoJsonSubmit() {
    try {
      const geometry = parseGeoJsonText(rawGeoJson);
      onChange(geometry);
      setError(null);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Invalid GeoJSON";
      setError(message);
    }
  }

  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <Label className="text-sm font-medium text-slate-700">Upload AOI (Shapefile ZIP)</Label>
        <input
          type="file"
          accept=".zip"
          className="block w-full text-sm text-slate-700"
          onChange={(event) => handleFileInput(event.target.files?.[0])}
        />
      </div>
      <div className="space-y-2">
        <Label className="text-sm font-medium text-slate-700">Paste AOI GeoJSON</Label>
        <textarea
          value={rawGeoJson}
          onChange={(event) => setRawGeoJson(event.target.value)}
          rows={6}
          className="w-full rounded-md border border-slate-200 bg-white p-3 text-sm text-slate-800 shadow-sm focus:border-slate-400 focus:outline-none"
          placeholder="Paste Polygon or MultiPolygon GeoJSON"
        />
        <button
          type="button"
          onClick={handleGeoJsonSubmit}
          className="rounded-md bg-slate-900 px-3 py-2 text-sm font-semibold text-white hover:bg-slate-800"
        >
          Use GeoJSON
        </button>
      </div>
      {value ? (
        <p className="text-xs text-emerald-600">AOI ready.</p>
      ) : (
        <p className="text-xs text-slate-500">Upload a shapefile or paste GeoJSON to continue.</p>
      )}
      {error ? <p className="text-xs text-rose-600">{error}</p> : null}
    </div>
  );
}
