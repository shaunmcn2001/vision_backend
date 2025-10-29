import { useState } from "react";

import type {
  AdvancedZonesResult,
  BasicZonesResult,
  GeometryInput,
  ImageryDay,
  ImageryResult,
  NdviMonthResult,
  SeasonInput,
  TileResponse
} from "./lib/api";
import {
  requestAdvancedZones,
  requestBasicZones,
  requestImageryDaily,
  requestNdviMonth
} from "./lib/api";
import { validateDateRange } from "./lib/validators";
import { AOIInput } from "./components/AOIInput";
import { DownloadsList } from "./components/DownloadsList";
import { Map } from "./components/Map";
import { ProductForm, type ProductKind } from "./components/ProductForm";
import { DaysTable } from "./components/DaysTable";

import "./index.css";

const PRODUCTS: { id: ProductKind; name: string; description: string }[] = [
  {
    id: "ndvi-month",
    name: "NDVI Month",
    description: "Monthly NDVI rasters with period mean preview."
  },
  {
    id: "imagery",
    name: "Daily Imagery",
    description: "Daily Sentinel-2 RGB mosaics with cloud percentages."
  },
  {
    id: "zones-basic",
    name: "Basic NDVI Zones",
    description: "Five quantile zones with raster/vector downloads."
  },
  {
    id: "zones-advanced",
    name: "Advanced Zones",
    description: "Stage-aware long-term zones with downloads."
  }
];

 type ProductResult =
  | { type: "ndvi-month"; data: NdviMonthResult }
  | { type: "imagery"; data: ImageryResult }
  | { type: "zones-basic"; data: BasicZonesResult }
  | { type: "zones-advanced"; data: AdvancedZonesResult };

export default function App() {
  const [product, setProduct] = useState<ProductKind>("ndvi-month");
  const [aoi, setAoi] = useState<GeometryInput | null>(null);
  const [startDate, setStartDate] = useState<string>("");
  const [endDate, setEndDate] = useState<string>("");
  const [clamp, setClamp] = useState<[number, number] | undefined>();
  const [imageryBands, setImageryBands] = useState<string[]>(["B4", "B3", "B2"]);
  const [breaksText, setBreaksText] = useState<string>("-1.0,-0.3,0.3,1.0");
  const [seasons, setSeasons] = useState<SeasonInput[]>([]);
  const [result, setResult] = useState<ProductResult | null>(null);
  const [activeTile, setActiveTile] = useState<TileResponse | null>(null);
  const [selectedDay, setSelectedDay] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  function resetResults() {
    setResult(null);
    setActiveTile(null);
    setSelectedDay(null);
    setError(null);
  }

  function handleSelectProduct(next: ProductKind) {
    setProduct(next);
    resetResults();
  }

  async function runNdviMonth() {
    if (!aoi) throw new Error("Provide an AOI");
    const response = await requestNdviMonth({
      aoi,
      start: startDate,
      end: endDate,
      clamp
    });
    setResult({ type: "ndvi-month", data: response });
    setActiveTile(response.mean);
  }

  async function runImagery() {
    if (!aoi) throw new Error("Provide an AOI");
    const response = await requestImageryDaily({
      aoi,
      start: startDate,
      end: endDate,
      bands: imageryBands
    });
    setResult({ type: "imagery", data: response });
    const firstTile = response.days.find((day) => day.tile)?.tile ?? null;
    setActiveTile(firstTile ?? null);
    setSelectedDay(firstTile ? response.days[0]?.date ?? null : null);
  }

  async function runBasicZones() {
    if (!aoi) throw new Error("Provide an AOI");
    const response = await requestBasicZones({
      aoi,
      start: startDate,
      end: endDate
    });
    setResult({ type: "zones-basic", data: response });
    setActiveTile(response.preview.tile);
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
    setResult({ type: "zones-advanced", data: response });
    setActiveTile(response.preview.zones);
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

  function renderResult() {
    if (!result) return null;
    if (result.type === "ndvi-month") {
      return (
        <div className="space-y-4">
          <h3 className="text-sm font-semibold text-slate-700">Monthly Scenes</h3>
          <div className="grid grid-cols-2 gap-2">
            {result.data.items.map((item) => (
              <button
                key={item.name}
                type="button"
                onClick={() => setActiveTile(item.tile)}
                className="rounded-md border border-slate-200 px-3 py-2 text-left text-sm text-slate-700 hover:border-slate-400"
              >
                {item.name}
              </button>
            ))}
          </div>
          <p className="text-xs text-slate-500">Mean layer is rendered on the map by default.</p>
        </div>
      );
    }
    if (result.type === "imagery") {
      return (
        <DaysTable
          days={result.data.days}
          selectedDate={selectedDay}
          onSelect={(day: ImageryDay) => {
            if (day.tile) {
              setActiveTile(day.tile);
              setSelectedDay(day.date);
            }
          }}
        />
      );
    }
    if (result.type === "zones-basic") {
      return <DownloadsList downloads={result.data.downloads} />;
    }
    if (result.type === "zones-advanced") {
      return (
        <div className="space-y-4">
          <div className="flex gap-2">
            <button
              type="button"
              onClick={() => setActiveTile(result.data.preview.zones)}
              className="rounded-md border border-slate-200 px-3 py-1 text-sm text-slate-700 hover:border-slate-400"
            >
              Zones Preview
            </button>
            <button
              type="button"
              onClick={() => setActiveTile(result.data.preview.composite)}
              className="rounded-md border border-slate-200 px-3 py-1 text-sm text-slate-700 hover:border-slate-400"
            >
              Composite Preview
            </button>
          </div>
          <DownloadsList downloads={result.data.downloads} />
        </div>
      );
    }
    return null;
  }

  return (
    <div className="flex min-h-screen flex-col bg-slate-50">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
          <div>
            <h1 className="text-xl font-bold text-slate-900">Vision</h1>
            <p className="text-sm text-slate-600">Earth Engine products for agricultural valuation.</p>
          </div>
        </div>
      </header>
      <main className="mx-auto grid w-full max-w-6xl flex-1 grid-cols-1 gap-6 px-6 py-6 md:grid-cols-[380px_1fr]">
        <aside className="space-y-6">
          <section className="space-y-3">
            <h2 className="text-sm font-semibold text-slate-700">Products</h2>
            <div className="grid grid-cols-1 gap-3">
              {PRODUCTS.map((option) => {
                const isActive = option.id === product;
                return (
                  <button
                    key={option.id}
                    type="button"
                    onClick={() => handleSelectProduct(option.id)}
                    className={`rounded-lg border px-3 py-3 text-left transition ${
                      isActive
                        ? "border-slate-900 bg-slate-900 text-white"
                        : "border-slate-200 bg-white text-slate-800 hover:border-slate-400"
                    }`}
                  >
                    <div className="text-sm font-semibold">{option.name}</div>
                    <div className={`mt-1 text-xs ${isActive ? "text-slate-200" : "text-slate-500"}`}>
                      {option.description}
                    </div>
                  </button>
                );
              })}
            </div>
          </section>
          <section className="space-y-4 rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
            <AOIInput value={aoi} onChange={(geometry) => {
              setAoi(geometry);
              resetResults();
            }} />
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
          </section>
        </aside>
        <section className="flex flex-col gap-4">
          <div className="h-[420px] rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
            <Map aoi={aoi} tile={activeTile} />
          </div>
          <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
            {renderResult()}
          </div>
        </section>
      </main>
    </div>
  );
}
