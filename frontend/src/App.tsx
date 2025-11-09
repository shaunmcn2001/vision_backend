import { ChangeEvent, FormEvent, useEffect, useMemo, useState } from "react";

import type {
  AdvancedZonesResult,
  BasicZonesResult,
  GeometryInput,
  NdviMonthResult,
  SeasonInput,
  WeatherForecastResponse
} from "./lib/api";
import {
  requestAdvancedZones,
  requestBasicZones,
  requestNdviMonth,
  requestWeatherForecast
} from "./lib/api";
import { validateDateRange } from "./lib/validators";
import { AOIInput } from "./components/AOIInput";
import { Map, type MapLayerConfig, type LegendConfig } from "./components/Map";
import { NdviTrendChart } from "./components/NdviTrendChart";
import { WeatherForecast } from "./components/WeatherForecast";
import { WeatherCharts } from "./components/WeatherCharts";
import { WeatherService } from "./lib/weather";
import { NoteModal } from "./components/NoteModal";
import { NotesPanel } from "./components/NotesPanel";
import { loadNotes, notesStorageKey, saveNotes, type FieldNote, type NoteCategoryId } from "./lib/notes";
import { pointInsideGeometry } from "./lib/geometry";

import "./index.css";

const NDVI_PALETTE = ["#f9f5d7", "#f6cf75", "#ee964b", "#4f9d69", "#226f54", "#193b48"];
const ZONE_PALETTE_BASE = [
  "#f5e6b3",
  "#d1ce7a",
  "#7fb285",
  "#4f8f8c",
  "#3a6b82",
  "#325272",
  "#273b5b",
  "#1e2a44",
  "#151c31"
];
const DEFAULT_CLAMP: [number, number] = [-0.2, 0.8];

function toDateInput(value: Date): string {
  const iso = value.toISOString();
  return iso.slice(0, 10);
}

function currentSeasonRange() {
  const today = new Date();
  const end = toDateInput(today);
  const startAnchor = new Date(today.getFullYear(), today.getMonth() - 5, 1);
  return { start: toDateInput(startAnchor), end };
}

function formatMonthLabel(name: string): string {
  const stripped = name.replace("ndvi_", "");
  const [year, month] = stripped.split("-");
  if (!year || !month) return name;
  const date = new Date(Number(year), Number(month) - 1, 1);
  if (Number.isNaN(date.getTime())) return name;
  return date.toLocaleDateString(undefined, { month: "short", year: "numeric" });
}

function formatDownloadLabel(key: string): string {
  return key
    .replace(/_/g, " ")
    .replace(/\bndvi\b/i, "NDVI")
    .replace(/\bgeotiff\b/i, "GeoTIFF")
    .replace(/\braw\b/i, "Raw")
    .replace(/\bcolour\b/i, "Colour")
    .replace(/\b\w/g, (match) => match.toUpperCase());
}

function zonePalette(count: number): string[] {
  if (count <= ZONE_PALETTE_BASE.length) {
    return ZONE_PALETTE_BASE.slice(0, count);
  }
  return Array.from({ length: count }, (_, index) => {
    const position = (index / Math.max(count - 1, 1)) * (ZONE_PALETTE_BASE.length - 1);
    const lower = Math.floor(position);
    const upper = Math.min(ZONE_PALETTE_BASE.length - 1, Math.ceil(position));
    if (lower === upper) return ZONE_PALETTE_BASE[lower];
    return ZONE_PALETTE_BASE[upper];
  });
}

interface LayerEntry extends MapLayerConfig {
  label: string;
  subtitle?: string;
  group?: "ndvi" | "vra" | "weather";
}

type SeasonDraft = {
  crop: string;
  sowingDate: string;
  harvestDate: string;
  yieldAsset?: string;
  soilAsset?: string;
};

type VraPrescription = {
  id: string;
  mode: "basic" | "advanced";
  title: string;
  createdAt: string;
  classCount: number;
  downloads: Record<string, string>;
  layerIds: string[];
};

const EMPTY_SEASON: SeasonDraft = {
  crop: "",
  sowingDate: "",
  harvestDate: "",
  yieldAsset: "",
  soilAsset: ""
};

export default function App() {
  const defaults = useMemo(currentSeasonRange, []);
  const [aoi, setAoi] = useState<GeometryInput | null>(null);
  const [startDate, setStartDate] = useState<string>(defaults.start);
  const [endDate, setEndDate] = useState<string>(defaults.end);
  const [clamp, setClamp] = useState<[number, number]>(DEFAULT_CLAMP);

  const [layers, setLayers] = useState<LayerEntry[]>([]);
  const [vraLayers, setVraLayers] = useState<LayerEntry[]>([]);
  const [weatherLayers, setWeatherLayers] = useState<LayerEntry[]>([]);

  const [legendConfig, setLegendConfig] = useState<LegendConfig | null>(null);

  const [ndviResult, setNdviResult] = useState<NdviMonthResult | null>(null);
  const [downloads, setDownloads] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [weather, setWeather] = useState<WeatherForecastResponse | null>(null);
  const [weatherLoading, setWeatherLoading] = useState(false);
  const [weatherError, setWeatherError] = useState<string | null>(null);

  const [notes, setNotes] = useState<FieldNote[]>([]);
  const [notesVisible, setNotesVisible] = useState(true);
  const [noteMode, setNoteMode] = useState(false);
  const [noteModalOpen, setNoteModalOpen] = useState(false);
  const [pendingLocation, setPendingLocation] = useState<{ lat: number; lon: number } | null>(null);
  const [noteError, setNoteError] = useState<string | null>(null);
  const [selectedNoteId, setSelectedNoteId] = useState<string | null>(null);

  const [vraMode, setVraMode] = useState<"basic" | "advanced">("basic");
  const [vraZoneCount, setVraZoneCount] = useState(5);
  const [vraStartDate, setVraStartDate] = useState(defaults.start);
  const [vraEndDate, setVraEndDate] = useState(defaults.end);
  const [vraBreaksText, setVraBreaksText] = useState("-0.3,0.0,0.3,0.6");
  const [vraSeasons, setVraSeasons] = useState<SeasonDraft[]>([]);
  const [seasonDraft, setSeasonDraft] = useState<SeasonDraft>(EMPTY_SEASON);
  const [vraLoading, setVraLoading] = useState(false);
  const [vraError, setVraError] = useState<string | null>(null);
  const [prescriptions, setPrescriptions] = useState<VraPrescription[]>([]);

  const noteKey = useMemo(() => notesStorageKey(aoi), [aoi]);

  useEffect(() => {
    const stored = loadNotes(noteKey);
    setNotes(stored);
  }, [noteKey]);

  useEffect(() => {
    saveNotes(noteKey, notes);
  }, [noteKey, notes]);

  const mapLayers = useMemo(() => [...layers, ...weatherLayers, ...vraLayers], [layers, weatherLayers, vraLayers]);
  const precipitationLayer = weatherLayers.find((layer) => layer.id === "precipitation-overlay");
  const downloadEntries = Object.entries(downloads);

  const trendPoints = useMemo(
    () =>
      ndviResult?.items.map((item) => ({
        label: formatMonthLabel(item.name),
        value: typeof item.meanNdvi === "number" ? item.meanNdvi : null
      })) ?? [],
    [ndviResult]
  );

  function handleAoiChange(geometry: GeometryInput | null) {
    setAoi(geometry);
    setLayers([]);
    setVraLayers([]);
    setWeatherLayers([]);
    setLegendConfig(null);
    setDownloads({});
    setNdviResult(null);
    setWeather(null);
    setNotes([]);
    setPrescriptions([]);
    setSelectedNoteId(null);
    setNoteMode(false);
    setNoteError(null);
  }

  function handleClampChange(index: 0 | 1, value: string) {
    const parsed = Number(value);
    if (Number.isNaN(parsed)) return;
    setClamp((current) => {
      const next = [...current] as [number, number];
      next[index] = parsed;
      return next;
    });
  }

  function toggleLayer(id: string) {
    setLayers((current) => current.map((layer) => (layer.id === id ? { ...layer, visible: !layer.visible } : layer)));
  }

  function toggleVraLayers(layerIds: string[]) {
    setVraLayers((current) => {
      const shouldShow = !layerIds.some((layerId) => current.find((layer) => layer.id === layerId)?.visible);
      return current.map((layer) =>
        layerIds.includes(layer.id) ? { ...layer, visible: shouldShow } : layer
      );
    });
    if (layerIds.length) {
      setLegendConfig((prev) => prev);
    }
  }

  function toggleWeatherOverlay() {
    setWeatherLayers((current) =>
      current.map((layer) =>
        layer.id === "precipitation-overlay" ? { ...layer, visible: !layer.visible } : layer
      )
    );
  }

  function handleToggleNotesVisibility() {
    setNotesVisible((value) => !value);
  }

  function handleStartNoteMode() {
    if (!aoi) {
      setNoteError("Upload a field before adding notes.");
      return;
    }
    setNoteMode((value) => !value);
    setNoteError(null);
  }

  function handleSelectNote(noteId: string) {
    setNotesVisible(true);
    setSelectedNoteId(noteId);
    setNoteMode(false);
  }

  function downloadBlob(filename: string, content: string, type: string) {
    const blob = new Blob([content], { type });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    link.click();
    URL.revokeObjectURL(url);
  }

  function handleExportJson() {
    downloadBlob("field-notes.json", JSON.stringify(notes, null, 2), "application/json");
  }

  function handleExportCsv() {
    const header = ["id", "lat", "lon", "category", "text", "createdAt"];
    const rows = notes.map((note) => [
      note.id,
      note.lat.toFixed(6),
      note.lon.toFixed(6),
      note.category,
      note.text.replace(/"/g, '""'),
      note.createdAt
    ]);
    const csv = [header.join(","), ...rows.map((row) => row.map((cell, index) => (index === 4 ? `"${cell}"` : cell)).join(","))].join("\n");
    downloadBlob("field-notes.csv", csv, "text/csv");
  }

  function handleSeasonDraftChange(event: ChangeEvent<HTMLInputElement | HTMLSelectElement>) {
    const { name, value } = event.target;
    setSeasonDraft((current) => ({ ...current, [name]: value }));
  }

  function addSeason() {
    if (!seasonDraft.sowingDate || !seasonDraft.harvestDate) {
      setVraError("Provide sowing and harvest dates for the season.");
      return;
    }
    setVraSeasons((current) => [...current, seasonDraft]);
    setSeasonDraft(EMPTY_SEASON);
    setVraError(null);
  }

  function removeSeason(index: number) {
    setVraSeasons((current) => current.filter((_, idx) => idx !== index));
  }

  async function handleMapClickForNote(position: { lat: number; lon: number }) {
    if (!noteMode) return;
    if (!aoi || !pointInsideGeometry({ lat: position.lat, lon: position.lon }, aoi)) {
      setNoteError("Notes must be dropped inside the active field boundary.");
      return;
    }
    setNoteError(null);
    setPendingLocation(position);
    setNoteModalOpen(true);
  }

  function handleSaveNote(payload: { text: string; category: NoteCategoryId; photo?: string | null }) {
    if (!pendingLocation) return;
    const newNote: FieldNote = {
      id: typeof crypto !== "undefined" && "randomUUID" in crypto ? crypto.randomUUID() : `note-${Date.now()}`,
      lat: pendingLocation.lat,
      lon: pendingLocation.lon,
      text: payload.text,
      category: payload.category,
      photo: payload.photo ?? null,
      createdAt: new Date().toISOString()
    };
    setNotes((current) => [newNote, ...current]);
    setNoteMode(false);
    setPendingLocation(null);
    setNoteModalOpen(false);
    setSelectedNoteId(newNote.id);
  }

  function handleCancelNoteModal() {
    setPendingLocation(null);
    setNoteModalOpen(false);
  }

  async function handleWeatherFetch(event: FormEvent) {
    event.preventDefault();
    if (!aoi) {
      setWeatherError("Upload a field boundary to fetch weather.");
      return;
    }
    setWeatherLoading(true);
    setWeatherError(null);
    try {
      const response = await requestWeatherForecast({ aoi });
      setWeather(response);
      const precipitationLayerConfig = WeatherService.buildPrecipitationLayer(response);
      if (precipitationLayerConfig) {
        setWeatherLayers([
          {
            ...precipitationLayerConfig,
            label: "Precipitation radar",
            subtitle: "RainViewer overlay",
            visible: false,
            group: "weather"
          }
        ]);
      } else {
        setWeatherLayers([]);
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unable to load weather.";
      setWeatherError(message);
    } finally {
      setWeatherLoading(false);
    }
  }

  async function handleGenerateNdvi(event: FormEvent) {
    event.preventDefault();
    if (!aoi) {
      setError("Upload or draw a field boundary to continue.");
      return;
    }
    if (!validateDateRange(startDate, endDate)) {
      setError("Choose a valid date range.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const response = await requestNdviMonth({
        aoi,
        start: startDate,
        end: endDate,
        clamp
      });
      const monthLayers: LayerEntry[] = response.items.map((item, index) => ({
        id: item.name,
        label: formatMonthLabel(item.name),
        subtitle: "Monthly composite",
        type: "raster",
        tile: item.tile,
        visible: index === response.items.length - 1,
        group: "ndvi"
      }));
      const meanLayer: LayerEntry = {
        id: "ndvi-mean",
        label: "Season mean",
        subtitle: `${response.items.length || 1} months combined`,
        type: "raster",
        tile: response.mean,
        visible: true,
        group: "ndvi"
      };
      setLayers([meanLayer, ...monthLayers]);
      setDownloads({ ...(response.downloads ?? {}) });
      setNdviResult(response);
      setLegendConfig({
        type: "gradient",
        title: "NDVI",
        min: clamp[0],
        max: clamp[1],
        colors: NDVI_PALETTE
      });
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to fetch NDVI.";
      setError(message);
    } finally {
      setLoading(false);
    }
  }

  function attachPrescriptionLayers(
    title: string,
    downloadsMap: Record<string, string>,
    classCount: number,
    layersToAdd: LayerEntry[]
  ) {
    const id = `vra-${Date.now()}`;
    const paletteColors = layersToAdd.find((layer) => layer.palette)?.palette ?? zonePalette(classCount);
    const legendEntries = paletteColors.map((color, index) => ({
      label: `Zone ${index + 1}${index === 0 ? " (Lowest)" : index === paletteColors.length - 1 ? " (Highest)" : ""}`,
      color
    }));
    const nextLayers = layersToAdd.map((layer, index) => ({ ...layer, id: `${id}-${layer.id}-${index}` }));
    setVraLayers((current) => [...current, ...nextLayers]);
    setLegendConfig({
      type: "discrete",
      title: "VRA Zones",
      entries: legendEntries
    });
    setPrescriptions((current) => [
      {
        id,
        mode: vraMode,
        title,
        createdAt: new Date().toISOString(),
        classCount,
        downloads: downloadsMap,
        layerIds: nextLayers.map((layer) => layer.id)
      },
      ...current
    ]);
  }

  async function handleGenerateBasicZones(event: FormEvent) {
    event.preventDefault();
    if (!aoi) {
      setVraError("Upload a field before generating zones.");
      return;
    }
    if (!validateDateRange(vraStartDate, vraEndDate)) {
      setVraError("Choose a valid date range for zones.");
      return;
    }
    setVraLoading(true);
    setVraError(null);
    try {
      const response: BasicZonesResult = await requestBasicZones({
        aoi,
        start: vraStartDate,
        end: vraEndDate,
        nClasses: vraZoneCount
      });
      const downloadsMap: Record<string, string> = { ...response.downloads };
      const palette = zonePalette(response.classCount ?? vraZoneCount);
      const layersToAdd: LayerEntry[] = [
        {
          id: "preview",
          label: `${response.classCount}-class zones preview`,
          subtitle: "Raster preview",
          type: "raster",
          tile: response.preview.tile,
          visible: false,
          group: "vra"
        },
        {
          id: "vectors",
          label: "Zone polygons",
          subtitle: "Vector prescription",
          type: "vector",
          geoJson: response.vectorsGeojson,
          palette,
          visible: true,
          group: "vra"
        }
      ];
      attachPrescriptionLayers(
        `${response.classCount} NDVI classes`,
        downloadsMap,
        response.classCount ?? vraZoneCount,
        layersToAdd
      );
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to generate basic zones.";
      setVraError(message);
    } finally {
      setVraLoading(false);
    }
  }

  function buildSeasonPayloads(): SeasonInput[] {
    return vraSeasons.map((season, index) => ({
      fieldName: null,
      fieldId: null,
      crop: season.crop || null,
      sowingDate: season.sowingDate,
      harvestDate: season.harvestDate,
      emergenceDate: null,
      floweringDate: null,
      yieldAsset: season.yieldAsset || null,
      soilAsset: season.soilAsset || null
      // A future enhancement can upload yield/soil rasters and assign their GCS paths to yieldAsset/soilAsset.
    }));
  }

  async function handleGenerateAdvancedZones(event: FormEvent) {
    event.preventDefault();
    if (!aoi) {
      setVraError("Upload a field before generating zones.");
      return;
    }
    const breaks = vraBreaksText
      .split(",")
      .map((value) => parseFloat(value.trim()))
      .filter((value) => !Number.isNaN(value));
    if (breaks.length !== 4) {
      setVraError("Advanced zones require four break values.");
      return;
    }
    if (!vraSeasons.length) {
      setVraError("Add at least one season for advanced zones.");
      return;
    }
    setVraLoading(true);
    setVraError(null);
      try {
        const seasonsPayload = buildSeasonPayloads();
        const response: AdvancedZonesResult = await requestAdvancedZones({
          aoi,
          breaks,
          seasons: seasonsPayload
        });
        const downloadsMap: Record<string, string> = { ...response.downloads };
        const palette = zonePalette(response.classCount ?? breaks.length + 1);
        const layersToAdd: LayerEntry[] = [
          {
            id: "zones-preview",
            label: "Advanced zones",
          subtitle: "Preview",
          type: "raster",
          tile: response.preview.zones,
          visible: true,
          group: "vra"
        },
        {
          id: "composite-preview",
          label: "Composite",
          subtitle: "Season composite",
          type: "raster",
          tile: response.preview.composite,
          visible: false,
          group: "vra"
        },
        {
          id: "vectors",
          label: "Zone polygons",
          subtitle: "Vector prescription",
          type: "vector",
          geoJson: response.vectorsGeojson,
          palette,
          visible: false,
          group: "vra"
        }
      ];
      attachPrescriptionLayers(
        `${response.classCount} advanced classes`,
        downloadsMap,
        response.classCount ?? palette.length,
        layersToAdd
      );
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to generate advanced zones.";
      setVraError(message);
    } finally {
      setVraLoading(false);
    }
  }

  return (
    <div className="flex min-h-screen bg-slate-100 text-slate-900">
      <aside className="flex w-full max-w-md flex-col gap-6 border-r border-slate-200 bg-white/90 px-6 py-8 backdrop-blur">
        <div className="space-y-1">
          <p className="text-xs font-semibold uppercase tracking-[0.3em] text-slate-400">Field monitoring</p>
          <h1 className="text-2xl font-semibold text-slate-900">NDVI crop insight</h1>
          <p className="text-sm text-slate-500">
            Upload a paddock boundary, generate NDVI composites, scout the field, and build VRA prescriptions like OneSoil.
          </p>
        </div>

        <section className="rounded-2xl border border-slate-100 bg-white/95 p-5 shadow-sm">
          <div className="mb-4 flex items-center justify-between">
            <div>
              <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Field boundary</p>
              <h2 className="text-lg font-semibold text-slate-900">Upload or draw</h2>
            </div>
            {aoi ? <span className="text-xs font-semibold text-emerald-600">Ready</span> : null}
          </div>
          <AOIInput value={aoi} onChange={handleAoiChange} />
        </section>

        <section className="rounded-2xl border border-slate-100 bg-white/95 p-5 shadow-sm">
          <div className="mb-4 flex items-center justify-between">
            <div>
              <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">NDVI Month</p>
              <h2 className="text-lg font-semibold text-slate-900">Growing season</h2>
            </div>
            {ndviResult ? (
              <span className="text-xs text-slate-500">{ndviResult.items.length} months</span>
            ) : null}
          </div>
          <form className="space-y-4" onSubmit={handleGenerateNdvi}>
            <div className="grid grid-cols-2 gap-3">
              <label className="space-y-1 text-sm text-slate-600">
                <span className="text-xs font-semibold uppercase tracking-wide text-slate-500">Start date</span>
                <input
                  type="date"
                  value={startDate}
                  onChange={(event) => setStartDate(event.target.value)}
                  className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm text-slate-900"
                />
              </label>
              <label className="space-y-1 text-sm text-slate-600">
                <span className="text-xs font-semibold uppercase tracking-wide text-slate-500">End date</span>
                <input
                  type="date"
                  value={endDate}
                  onChange={(event) => setEndDate(event.target.value)}
                  className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm text-slate-900"
                />
              </label>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <label className="space-y-1 text-sm text-slate-600">
                <span className="text-xs font-semibold uppercase tracking-wide text-slate-500">Clamp min</span>
                <input
                  type="number"
                  step="0.05"
                  value={clamp[0]}
                  onChange={(event) => handleClampChange(0, event.target.value)}
                  className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm text-slate-900"
                />
              </label>
              <label className="space-y-1 text-sm text-slate-600">
                <span className="text-xs font-semibold uppercase tracking-wide text-slate-500">Clamp max</span>
                <input
                  type="number"
                  step="0.05"
                  value={clamp[1]}
                  onChange={(event) => handleClampChange(1, event.target.value)}
                  className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm text-slate-900"
                />
              </label>
            </div>
            <button
              type="submit"
              disabled={loading || !aoi}
              className="w-full rounded-xl bg-emerald-600 px-4 py-3 text-sm font-semibold uppercase tracking-wide text-white shadow-lg shadow-emerald-600/30 transition hover:bg-emerald-500 disabled:cursor-not-allowed disabled:bg-slate-300"
            >
              {loading ? "Generating..." : "Generate NDVI"}
            </button>
            {error ? <p className="text-sm text-rose-600">{error}</p> : null}
          </form>
        </section>

        <section className="rounded-2xl border border-slate-100 bg-white/95 p-5 shadow-sm">
          <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
            <div>
              <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">VRA Maps</p>
              <h2 className="text-lg font-semibold text-slate-900">Productivity zones</h2>
            </div>
            <div className="flex gap-2">
              <button
                type="button"
                onClick={() => setVraMode("basic")}
                className={`rounded-full px-4 py-1 text-xs font-semibold uppercase tracking-wide ${
                  vraMode === "basic" ? "bg-emerald-600 text-white" : "bg-slate-200 text-slate-600"
                }`}
              >
                Basic
              </button>
              <button
                type="button"
                onClick={() => setVraMode("advanced")}
                className={`rounded-full px-4 py-1 text-xs font-semibold uppercase tracking-wide ${
                  vraMode === "advanced" ? "bg-emerald-600 text-white" : "bg-slate-200 text-slate-600"
                }`}
              >
                Advanced
              </button>
            </div>
          </div>
          {/* Future enhancement: expose an export CRS selector once the backend accepts an export_crs parameter so shapefiles can match machinery coordinate systems out of the box. */}
          {vraMode === "basic" ? (
            <form className="space-y-4" onSubmit={handleGenerateBasicZones}>
              <div className="grid grid-cols-2 gap-3">
                <label className="space-y-1 text-sm text-slate-600">
                  <span className="text-xs font-semibold uppercase tracking-wide text-slate-500">Zones</span>
                  <input
                    type="number"
                    min={3}
                    max={9}
                    value={vraZoneCount}
                    onChange={(event) => setVraZoneCount(Number(event.target.value))}
                    className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm text-slate-900"
                  />
                </label>
                <label className="space-y-1 text-sm text-slate-600">
                  <span className="text-xs font-semibold uppercase tracking-wide text-slate-500">Date range</span>
                  <input
                    type="text"
                    readOnly
                    value={`${vraStartDate} → ${vraEndDate}`}
                    className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm text-slate-900"
                  />
                </label>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <input
                  type="date"
                  value={vraStartDate}
                  onChange={(event) => setVraStartDate(event.target.value)}
                  className="rounded-lg border border-slate-200 px-3 py-2 text-sm text-slate-900"
                />
                <input
                  type="date"
                  value={vraEndDate}
                  onChange={(event) => setVraEndDate(event.target.value)}
                  className="rounded-lg border border-slate-200 px-3 py-2 text-sm text-slate-900"
                />
              </div>
              <button
                type="submit"
                disabled={vraLoading || !aoi}
                className="w-full rounded-xl bg-slate-900 px-4 py-2 text-sm font-semibold uppercase tracking-wide text-white shadow-lg shadow-slate-900/20 disabled:cursor-not-allowed disabled:bg-slate-400"
              >
                {vraLoading ? "Building zones..." : "Generate basic zones"}
              </button>
            </form>
          ) : (
            <div className="space-y-4">
              <form className="space-y-3" onSubmit={handleGenerateAdvancedZones}>
                <label className="space-y-1 text-sm text-slate-600">
                  <span className="text-xs font-semibold uppercase tracking-wide text-slate-500">Breaks (4 values)</span>
                  <input
                    type="text"
                    value={vraBreaksText}
                    onChange={(event) => setVraBreaksText(event.target.value)}
                    className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm text-slate-900"
                  />
                </label>
                <div className="rounded-xl border border-slate-200 p-3">
                  <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Season builder</p>
                  <p className="text-xs text-slate-500">
                    Add seasons with crop + dates. Future yield/soil uploads can populate the asset fields (stored via SeasonInput.yieldAsset/soilAsset).
                  </p>
                  <div className="mt-3 grid grid-cols-2 gap-3">
                    <input
                      name="crop"
                      value={seasonDraft.crop}
                      onChange={handleSeasonDraftChange}
                      placeholder="Crop"
                      className="rounded-lg border border-slate-200 px-3 py-2 text-sm"
                    />
                    <input
                      type="date"
                      name="sowingDate"
                      value={seasonDraft.sowingDate}
                      onChange={handleSeasonDraftChange}
                      className="rounded-lg border border-slate-200 px-3 py-2 text-sm"
                    />
                    <input
                      type="date"
                      name="harvestDate"
                      value={seasonDraft.harvestDate}
                      onChange={handleSeasonDraftChange}
                      className="rounded-lg border border-slate-200 px-3 py-2 text-sm"
                    />
                    <input
                      name="yieldAsset"
                      value={seasonDraft.yieldAsset}
                      onChange={handleSeasonDraftChange}
                      placeholder="Yield layer ref"
                      className="rounded-lg border border-slate-200 px-3 py-2 text-sm"
                    />
                    <input
                      name="soilAsset"
                      value={seasonDraft.soilAsset}
                      onChange={handleSeasonDraftChange}
                      placeholder="Soil layer ref"
                      className="rounded-lg border border-slate-200 px-3 py-2 text-sm"
                    />
                  </div>
                  <button
                    type="button"
                    className="mt-3 rounded-xl border border-dashed border-emerald-300 px-3 py-1 text-xs font-semibold uppercase tracking-wide text-emerald-700"
                    onClick={addSeason}
                  >
                    Add season
                  </button>
                  <ul className="mt-3 space-y-2 text-sm">
                    {vraSeasons.map((season, index) => (
                      <li key={`${season.crop}-${index}`} className="flex items-center justify-between rounded-lg bg-slate-50 px-3 py-2">
                        <div>
                          <p className="font-semibold text-slate-800">{season.crop || `Season ${index + 1}`}</p>
                          <p className="text-xs text-slate-500">{season.sowingDate} → {season.harvestDate}</p>
                        </div>
                        <button type="button" className="text-xs text-rose-600" onClick={() => removeSeason(index)}>
                          Remove
                        </button>
                      </li>
                    ))}
                  </ul>
                </div>
                <button
                  type="submit"
                  disabled={vraLoading || !aoi}
                  className="w-full rounded-xl bg-slate-900 px-4 py-2 text-sm font-semibold uppercase tracking-wide text-white shadow-lg shadow-slate-900/20 disabled:cursor-not-allowed disabled:bg-slate-400"
                >
                  {vraLoading ? "Building advanced zones..." : "Generate advanced zones"}
                </button>
              </form>
            </div>
          )}
          {vraError ? <p className="mt-3 text-sm text-rose-600">{vraError}</p> : null}
          {prescriptions.length ? (
            <div className="mt-4 space-y-3">
              {prescriptions.map((entry) => {
                const visible = entry.layerIds.some((layerId) => vraLayers.find((layer) => layer.id === layerId)?.visible);
                return (
                  <div key={entry.id} className="rounded-2xl border border-slate-100 bg-white px-4 py-3 shadow-sm">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm font-semibold text-slate-900">{entry.title}</p>
                        <p className="text-xs text-slate-500">{entry.classCount} classes • {new Date(entry.createdAt).toLocaleString()}</p>
                      </div>
                      <button
                        type="button"
                        onClick={() => toggleVraLayers(entry.layerIds)}
                        className={`rounded-full px-3 py-1 text-xs font-semibold uppercase tracking-wide ${
                          visible ? "bg-emerald-600 text-white" : "bg-slate-200 text-slate-600"
                        }`}
                      >
                        {visible ? "Hide" : "Show"}
                      </button>
                    </div>
                    <div className="mt-2 space-x-2 text-xs">
                      {Object.entries(entry.downloads).map(([label, url]) => (
                        <a key={label} href={url} target="_blank" rel="noreferrer" className="text-emerald-600 hover:underline">
                          {formatDownloadLabel(label)}
                        </a>
                      ))}
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <p className="mt-4 text-sm text-slate-500">Run a VRA workflow to store prescriptions. Each entry can be exported for machinery consoles.</p>
          )}
        </section>

        <section className="rounded-2xl border border-slate-100 bg-white/95 p-5 shadow-sm">
          <div className="mb-4 flex items-center justify-between">
            <div>
              <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Downloads</p>
              <h2 className="text-lg font-semibold text-slate-900">NDVI outputs</h2>
            </div>
            <span className="text-xs text-slate-500">Raw + coloured</span>
          </div>
          {downloadEntries.length ? (
            <div className="space-y-2">
              {downloadEntries.map(([label, url]) => (
                <a
                  key={label}
                  href={url}
                  target="_blank"
                  rel="noreferrer"
                  className="flex items-center justify-between rounded-xl border border-slate-200 px-4 py-2 text-sm font-medium text-slate-700 hover:border-emerald-200 hover:bg-emerald-50"
                >
                  <span>{formatDownloadLabel(label)}</span>
                  <span className="text-xs uppercase tracking-wide text-emerald-600">Download</span>
                </a>
              ))}
            </div>
          ) : (
            <p className="text-sm text-slate-500">Downloads appear once NDVI tiles are generated.</p>
          )}
        </section>

        <section className="rounded-2xl border border-slate-100 bg-white/95 p-5 shadow-sm">
          <div className="mb-4 flex items-center justify-between">
            <div>
              <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Trend</p>
              <h2 className="text-lg font-semibold text-slate-900">NDVI over time</h2>
            </div>
            <span className="text-xs text-slate-500">Avg per month</span>
          </div>
          <NdviTrendChart points={trendPoints} clamp={clamp} />
        </section>

        <section className="rounded-2xl border border-slate-100 bg-white/95 p-5 shadow-sm">
          <div className="mb-4 flex items-center justify-between">
            <div>
              <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Weather</p>
              <h2 className="text-lg font-semibold text-slate-900">5-day field outlook</h2>
            </div>
            {weather ? <span className="text-xs text-slate-500">{weather.forecast.length} days</span> : null}
          </div>
          <form className="space-y-3" onSubmit={handleWeatherFetch}>
            <button
              type="submit"
              disabled={weatherLoading || !aoi}
              className="w-full rounded-xl bg-slate-900 px-4 py-3 text-sm font-semibold uppercase tracking-wide text-white shadow-lg shadow-slate-900/20 transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:bg-slate-300"
            >
              {weatherLoading ? "Loading forecast..." : weather ? "Refresh weather" : "Load weather"}
            </button>
            {weatherError ? <p className="text-sm text-rose-600">{weatherError}</p> : null}
          </form>
          {weather ? (
            <div className="mt-4 space-y-6">
              <WeatherForecast days={weather.forecast} recommendation={weather.sprayRecommendation} />
              {precipitationLayer ? (
                <div className="flex items-center justify-between rounded-xl border border-slate-100 bg-slate-50 px-4 py-3">
                  <div>
                    <p className="text-sm font-semibold text-slate-700">Precipitation overlay</p>
                    <p className="text-xs text-slate-500">RainViewer radar tiles</p>
                  </div>
                  <button
                    type="button"
                    onClick={toggleWeatherOverlay}
                    className={`rounded-full px-4 py-2 text-xs font-semibold uppercase tracking-wide transition ${
                      precipitationLayer.visible
                        ? "bg-emerald-600 text-white"
                        : "bg-white text-slate-600 shadow"
                    }`}
                  >
                    {precipitationLayer.visible ? "Hide overlay" : "Show overlay"}
                  </button>
                </div>
              ) : null}
              <WeatherCharts chart={weather.chart} base={weather.gddBaseC} />
            </div>
          ) : (
            <p className="mt-4 text-sm text-slate-500">
              Fetch the forecast to view weather, precipitation, and spray recommendations for this field.
            </p>
          )}
        </section>

        <NotesPanel
          notes={notes}
          notesVisible={notesVisible}
          noteMode={noteMode}
          onToggleVisibility={handleToggleNotesVisibility}
          onStartNote={handleStartNoteMode}
          onSelectNote={handleSelectNote}
          onExportCsv={handleExportCsv}
          onExportJson={handleExportJson}
        />
        {noteError ? <p className="text-sm text-rose-600">{noteError}</p> : null}
      </aside>

      <section className="relative flex flex-1 flex-col">
        <div className="flex-1">
          <Map
            aoi={aoi}
            layers={mapLayers}
            legend={legendConfig}
            noteMode={noteMode}
            notes={notes}
            showNotes={notesVisible}
            selectedNoteId={selectedNoteId}
            onMapClickForNote={handleMapClickForNote}
            onSelectNote={setSelectedNoteId}
          />
        </div>
      </section>

      <NoteModal
        open={noteModalOpen}
        location={pendingLocation}
        onClose={handleCancelNoteModal}
        onSave={handleSaveNote}
      />
    </div>
  );
}
