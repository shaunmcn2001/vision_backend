import { useState } from "react";

import type {
  AdvancedZonesResult,
  BasicZonesResult,
  GeometryInput,
  ImageryResult,
  NdviMonthResult,
  SeasonInput
} from "./lib/api";
import {
  requestAdvancedZones,
  requestBasicZones,
  requestImageryDaily,
  requestNdviMonth
} from "./lib/api";
import { validateDateRange } from "./lib/validators";
import { AOIInput } from "./components/AOIInput";
import { Map, type MapLayerConfig } from "./components/Map";
import { ProductForm, type ProductKind } from "./components/ProductForm";

import "./index.css";

type ProductResult =
  | { type: "ndvi-month"; data: NdviMonthResult }
  | { type: "imagery"; data: ImageryResult }
  | { type: "zones-basic"; data: BasicZonesResult }
  | { type: "zones-advanced"; data: AdvancedZonesResult };

type LayerEntry = MapLayerConfig & {
  label: string;
  description?: string;
};

const NAV_ITEMS = [
  { label: "Cadastre" },
  { label: "Property Reports" },
  { label: "Grazing Maps", active: true },
  { label: "SmartMaps" }
];

const PRODUCT_OPTIONS: { id: ProductKind; name: string; description: string }[] = [
  {
    id: "ndvi-month",
    name: "NDVI Month",
    description: "Monthly NDVI composites for the selected boundary."
  },
  {
    id: "imagery",
    name: "Daily Imagery",
    description: "Sentinel-2 true colour mosaics with cloud statistics."
  },
  {
    id: "zones-basic",
    name: "Basic Zones",
    description: "Quantile NDVI zones with GeoTIFF, SHP, and CSV downloads."
  },
  {
    id: "zones-advanced",
    name: "Advanced Zones",
    description: "Season-aware NDVI analytics for agronomy teams."
  }
];

function formatMonthLabel(name: string): string {
  const stripped = name.replace("ndvi_", "");
  const [year, month] = stripped.split("-");
  if (!year || !month) return name;
  const date = new Date(Number(year), Number(month) - 1, 1);
  if (Number.isNaN(date.getTime())) return name;
  return date.toLocaleDateString(undefined, { month: "short", year: "numeric" });
}

function formatDateLabel(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleDateString(undefined, {
    day: "numeric",
    month: "short",
    year: "numeric"
  });
}

function formatDownloadLabel(key: string): string {
  return key
    .replace(/([A-Z])/g, " $1")
    .replace(/_/g, " ")
    .replace(/^\w/, (char) => char.toUpperCase());
}

export default function App() {
  const [product, setProduct] = useState<ProductKind>("ndvi-month");
  const [aoi, setAoi] = useState<GeometryInput | null>(null);
  const [startDate, setStartDate] = useState<string>("");
  const [endDate, setEndDate] = useState<string>("");
  const [clamp, setClamp] = useState<[number, number] | undefined>();
  const [imageryBands, setImageryBands] = useState<string[]>(["B4", "B3", "B2"]);
  const [breaksText, setBreaksText] = useState<string>("-1.0,-0.3,0.3,1.0");
  const [seasons, setSeasons] = useState<SeasonInput[]>([]);
  const [layers, setLayers] = useState<LayerEntry[]>([]);
  const [downloads, setDownloads] = useState<Record<string, string>>({});
  const [result, setResult] = useState<ProductResult | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const downloadEntries = Object.entries(downloads);

  function clearOutputs() {
    setLayers([]);
    setDownloads({});
    setResult(null);
    setError(null);
  }

  function handleSelectProduct(next: ProductKind) {
    setProduct(next);
    clearOutputs();
  }

  function handleAoiChange(geometry: GeometryInput | null) {
    setAoi(geometry);
    clearOutputs();
  }

  function toggleLayer(layerId: string) {
    setLayers((current) =>
      current.map((layer) =>
        layer.id === layerId ? { ...layer, visible: !layer.visible } : layer
      )
    );
  }

  async function runNdviMonth() {
    if (!aoi) throw new Error("Provide an AOI");
    const response = await requestNdviMonth({
      aoi,
      start: startDate,
      end: endDate,
      clamp
    });
    const layerEntries: LayerEntry[] = [
      {
        id: "ndvi-mean",
        label: "NDVI Mean",
        description: "Average NDVI for the selected period.",
        tile: response.mean,
        visible: true
      },
      ...response.items.map((item) => ({
        id: `ndvi-${item.name}`,
        label: formatMonthLabel(item.name),
        description: "Monthly NDVI composite.",
        tile: item.tile,
        visible: false
      }))
    ];
    setLayers(layerEntries);
    setDownloads({});
    setResult({ type: "ndvi-month", data: response });
  }

  async function runImagery() {
    if (!aoi) throw new Error("Provide an AOI");
    const response = await requestImageryDaily({
      aoi,
      start: startDate,
      end: endDate,
      bands: imageryBands
    });
    const imageryLayers: LayerEntry[] = response.days
      .filter((day) => Boolean(day.tile))
      .map((day, index) => ({
        id: `imagery-${day.date}`,
        label: formatDateLabel(day.date),
        description: `Cloud cover ${Math.round(day.cloudPct)}%`,
        tile: day.tile!,
        visible: index === 0
      }));
    if (!imageryLayers.length) {
      throw new Error("No imagery tiles returned for the selected range.");
    }
    setLayers(imageryLayers);
    setDownloads({});
    setResult({ type: "imagery", data: response });
  }

  async function runBasicZones() {
    if (!aoi) throw new Error("Provide an AOI");
    const response = await requestBasicZones({
      aoi,
      start: startDate,
      end: endDate
    });
    const layerEntries: LayerEntry[] = [
      {
        id: "zones-basic-preview",
        label: "Zones Preview",
        description: "Quantile NDVI zones before download.",
        tile: response.preview.tile,
        visible: true
      }
    ];
    setLayers(layerEntries);
    setDownloads({ ...response.downloads });
    setResult({ type: "zones-basic", data: response });
  }

  async function runAdvancedZones() {
    if (!aoi) throw new Error("Provide an AOI");
    const breaks = breaksText
      .split(",")
      .map((value) => parseFloat(value.trim()))
      .filter((value) => !Number.isNaN(value));
    if (breaks.length !== 4) {
      throw new Error("Advanced zones require four break values.");
    }
    if (!seasons.length) {
      throw new Error("Upload at least one season row.");
    }
    const response = await requestAdvancedZones({
      aoi,
      breaks,
      seasons
    });
    const layerEntries: LayerEntry[] = [
      {
        id: "zones-advanced",
        label: "Zones Preview",
        description: "Season-aware NDVI classes.",
        tile: response.preview.zones,
        visible: true
      },
      {
        id: "zones-advanced-composite",
        label: "Composite Preview",
        description: "Median NDVI composite for the configured seasons.",
        tile: response.preview.composite,
        visible: false
      }
    ];
    setLayers(layerEntries);
    setDownloads({ ...response.downloads });
    setResult({ type: "zones-advanced", data: response });
  }

  async function handleSubmit() {
    if (!aoi) {
      setError("AOI is required");
      return;
    }
    if (!validateDateRange(startDate, endDate) && product !== "zones-advanced") {
      setError("Choose a valid date range");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      if (product === "ndvi-month") await runNdviMonth();
      if (product === "imagery") await runImagery();
      if (product === "zones-basic") await runBasicZones();
      if (product === "zones-advanced") await runAdvancedZones();
    } catch (err) {
      const message = err instanceof Error ? err.message : "Request failed";
      setError(message);
    } finally {
      setLoading(false);
    }
  }

  function renderResultSummary() {
    if (!result) return null;
    if (result.type === "imagery") {
      return (
        <p className="text-xs text-slate-500">
          {result.data.summary.count} scenes • Avg cloud{" "}
          {result.data.summary.avgCloudPct.toFixed(1)}%
        </p>
      );
    }
    if (result.type === "ndvi-month") {
      return (
        <p className="text-xs text-slate-500">
          {result.data.items.length} monthly layers plus a mean composite ready to
          view.
        </p>
      );
    }
    if (result.type === "zones-basic") {
      return (
        <p className="text-xs text-slate-500">
          Previewing quantile zones; download detailed outputs below.
        </p>
      );
    }
    if (result.type === "zones-advanced") {
      return (
        <p className="text-xs text-slate-500">
          Composite and zone previews are available together for inspection.
        </p>
      );
    }
    return null;
  }

  return (
    <div className="flex min-h-screen flex-col bg-slate-100">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex w-full max-w-6xl items-center justify-between px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-emerald-100 text-lg font-semibold text-emerald-600">
              V
            </div>
            <div>
              <p className="text-lg font-semibold text-slate-900">Praedia</p>
              <p className="text-xs text-slate-500">Vision NDVI Builder</p>
            </div>
          </div>
          <nav className="hidden rounded-full border border-slate-200 bg-slate-50 p-1 text-sm font-semibold text-slate-500 shadow-sm md:flex">
            {NAV_ITEMS.map((item) => {
              const isActive = Boolean(item.active);
              return (
                <button
                  key={item.label}
                  type="button"
                  className={`rounded-full px-4 py-2 ${
                    isActive
                      ? "bg-white text-slate-900 shadow"
                      : "text-slate-500 hover:text-slate-700"
                  }`}
                >
                  {item.label}
                </button>
              );
            })}
          </nav>
          <div className="hidden text-xs font-semibold uppercase tracking-[0.2em] text-slate-400 md:block">
            NSW • QLD • SA • VIC
          </div>
        </div>
      </header>
      <main className="flex flex-1 flex-col lg:flex-row">
        <section className="flex-1 bg-slate-200/40">
          <div className="h-[420px] w-full border-b border-slate-200 lg:h-full lg:border-none">
            <Map aoi={aoi} layers={layers} />
          </div>
        </section>
        <aside className="w-full border-t border-slate-200 bg-white lg:max-w-md lg:border-l lg:border-t-0">
          <div className="h-full overflow-y-auto p-6">
            <div className="space-y-8">
              <section className="space-y-4">
                <div>
                  <h2 className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                    Input
                  </h2>
                  <p className="text-sm text-slate-600">
                    Upload paddock boundaries, choose a workflow, and generate NDVI
                    products.
                  </p>
                </div>
                <div className="space-y-4">
                  <div className="flex flex-wrap gap-2">
                    {PRODUCT_OPTIONS.map((option) => {
                      const isActive = option.id === product;
                      return (
                        <button
                          key={option.id}
                          type="button"
                          onClick={() => handleSelectProduct(option.id)}
                          className={`rounded-full border px-4 py-2 text-sm font-semibold transition ${
                            isActive
                              ? "border-slate-900 bg-slate-900 text-white shadow-sm"
                              : "border-slate-200 bg-white text-slate-600 hover:border-slate-300 hover:text-slate-800"
                          }`}
                        >
                          {option.name}
                        </button>
                      );
                    })}
                  </div>
                  <div className="space-y-6 rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
                    <AOIInput value={aoi} onChange={handleAoiChange} />
                    <ProductForm
                      product={product}
                      startDate={startDate}
                      endDate={endDate}
                      clamp={clamp}
                      onClampChange={setClamp}
                      onStartDateChange={setStartDate}
                      onEndDateChange={setEndDate}
                      imageryBands={imageryBands}
                      onImageryBandsChange={setImageryBands}
                      breaksText={breaksText}
                      onBreaksChange={setBreaksText}
                      seasons={seasons}
                      onSeasonsChange={setSeasons}
                      onSubmit={handleSubmit}
                      disabled={loading}
                      hasAoi={Boolean(aoi)}
                    />
                    {error ? <p className="text-xs text-rose-600">{error}</p> : null}
                  </div>
                </div>
              </section>
              <section className="space-y-4">
                <div>
                  <h2 className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                    Layers
                  </h2>
                  <p className="text-sm text-slate-600">
                    Toggle map overlays returned by the selected workflow.
                  </p>
                </div>
                <div className="space-y-3 rounded-2xl border border-slate-200 bg-slate-50 p-4">
                  {layers.length ? (
                    <div className="space-y-2">
                      {layers.map((layer) => (
                        <label
                          key={layer.id}
                          className="flex items-start justify-between gap-4 rounded-lg border border-transparent bg-white px-3 py-2 shadow-sm"
                        >
                          <span>
                            <span className="block text-sm font-semibold text-slate-800">
                              {layer.label}
                            </span>
                            {layer.description ? (
                              <span className="text-xs text-slate-500">{layer.description}</span>
                            ) : null}
                          </span>
                          <input
                            type="checkbox"
                            checked={layer.visible}
                            onChange={() => toggleLayer(layer.id)}
                            className="mt-1 h-4 w-4 accent-slate-900"
                          />
                        </label>
                      ))}
                      {renderResultSummary()}
                    </div>
                  ) : (
                    <p className="text-xs text-slate-500">
                      Run any workflow to reveal toggleable layers on the map.
                    </p>
                  )}
                </div>
              </section>
              <section className="space-y-4">
                <div>
                  <h2 className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                    Export Features
                  </h2>
                  <p className="text-sm text-slate-600">
                    Download artefacts like GeoTIFFs, shapefiles, and statistics CSVs.
                  </p>
                </div>
                <div className="space-y-3 rounded-2xl border border-dashed border-slate-300 bg-white p-4">
                  {downloadEntries.length ? (
                    <ul className="space-y-2">
                      {downloadEntries.map(([key, url]) => (
                        <li
                          key={key}
                          className="flex items-center justify-between rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-700"
                        >
                          <span>{formatDownloadLabel(key)}</span>
                          <a
                            href={url}
                            target="_blank"
                            rel="noreferrer"
                            className="rounded-md bg-slate-900 px-3 py-1 text-xs font-semibold text-white hover:bg-slate-800"
                          >
                            Download
                          </a>
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="text-xs text-slate-500">
                      Generate Basic or Advanced zones to surface download links for GeoTIFF,
                      shapefile, and CSV outputs.
                    </p>
                  )}
                </div>
              </section>
            </div>
          </div>
        </aside>
      </main>
    </div>
  );
}
