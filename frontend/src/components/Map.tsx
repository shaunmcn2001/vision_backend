import { useEffect, useRef } from "react";
import maplibregl, { Map as MapLibreMap } from "maplibre-gl";
import type { FeatureCollection } from "geojson";

import type { GeometryInput, TileResponse } from "../lib/api";
import { categoryMeta, type FieldNote } from "../lib/notes";
import "maplibre-gl/dist/maplibre-gl.css";

export type LegendConfig =
  | {
      type: "gradient";
      title: string;
      min: number;
      max: number;
      colors: string[];
    }
  | {
      type: "discrete";
      title: string;
      entries: { label: string; color: string }[];
    };

export type MapProps = {
  aoi: GeometryInput | null;
  layers: MapLayerConfig[];
  legend?: LegendConfig | null;
  noteMode?: boolean;
  notes?: FieldNote[];
  showNotes?: boolean;
  selectedNoteId?: string | null;
  onMapClickForNote?: (lngLat: { lat: number; lon: number }) => void;
  onSelectNote?: (noteId: string) => void;
};

export type MapLayerConfig = {
  id: string;
  type: "raster" | "vector";
  visible: boolean;
  tile?: TileResponse;
  geoJson?: FeatureCollection;
  palette?: string[];
  name?: string;
  opacity?: number;
};

function extractGeometry(geometry: GeometryInput | null) {
  if (!geometry) return null;
  if (geometry.type === "FeatureCollection") {
    return geometry.features[0]?.geometry ?? null;
  }
  return geometry as any;
}

function computeBounds(geometry: GeometryInput | null) {
  const geom = extractGeometry(geometry);
  if (!geom || !("coordinates" in geom)) return null;
  const coords = (geom.coordinates as number[][][] | number[][][][]);
  const flat: number[][] = [];
  function flatten(arr: any) {
    if (typeof arr[0] === "number") {
      flat.push(arr as number[]);
    } else {
      arr.forEach(flatten);
    }
  }
  flatten(coords);
  if (!flat.length) return null;
  let minLng = flat[0][0];
  let minLat = flat[0][1];
  let maxLng = flat[0][0];
  let maxLat = flat[0][1];
  flat.forEach(([lng, lat]) => {
    minLng = Math.min(minLng, lng);
    minLat = Math.min(minLat, lat);
    maxLng = Math.max(maxLng, lng);
    maxLat = Math.max(maxLat, lat);
  });
  return [minLng, minLat, maxLng, maxLat] as const;
}

export function Map({
  aoi,
  layers,
  legend,
  noteMode = false,
  notes = [],
  showNotes = true,
  selectedNoteId,
  onMapClickForNote,
  onSelectNote
}: MapProps) {
  const mapRef = useRef<HTMLDivElement | null>(null);
  const mapInstance = useRef<MapLibreMap | null>(null);
  const noteMarkers = useRef<Map<string, { marker: maplibregl.Marker; popup: maplibregl.Popup }>>(new Map());

  useEffect(() => {
    if (!mapRef.current || mapInstance.current) return;
    const map = new maplibregl.Map({
      container: mapRef.current,
      style: "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
      center: [146.9, -32.1],
      zoom: 5,
      attributionControl: true
    });
    map.addControl(new maplibregl.NavigationControl(), "top-right");
    mapInstance.current = map;
  }, []);

  useEffect(() => {
    const map = mapInstance.current;
    if (!map || !aoi) return;
    const sourceId = "aoi";
    const geometry = extractGeometry(aoi);
    if (!geometry) return;
    const featureCollection = {
      type: "FeatureCollection" as const,
      features: [{ type: "Feature" as const, geometry, properties: {} }]
    };
    if (!map.getSource(sourceId)) {
      map.addSource(sourceId, {
        type: "geojson",
        data: featureCollection
      });
      map.addLayer({
        id: "aoi-fill",
        type: "fill",
        source: sourceId,
        paint: {
          "fill-color": "#2563eb",
          "fill-opacity": 0.2
        }
      });
      map.addLayer({
        id: "aoi-outline",
        type: "line",
        source: sourceId,
        paint: {
          "line-color": "#1e3a8a",
          "line-width": 2
        }
      });
    } else {
      const source = map.getSource(sourceId) as maplibregl.GeoJSONSource;
      source.setData(featureCollection);
    }
    const bounds = computeBounds(aoi);
    if (bounds) {
      map.fitBounds([
        [bounds[0], bounds[1]],
        [bounds[2], bounds[3]]
      ], { padding: 40, duration: 500 });
    }
  }, [aoi]);

  useEffect(() => {
    const map = mapInstance.current;
    if (!map) return;
    const layerPrefix = "product-layer-";
    const sourcePrefix = "product-source-";

    const style = map.getStyle();
    if (style?.layers) {
      style.layers
        .filter((layer) => layer.id.startsWith(layerPrefix))
        .forEach((layer) => map.removeLayer(layer.id));
    }
    if (style?.sources) {
      Object.keys(style.sources)
        .filter((sourceId) => sourceId.startsWith(sourcePrefix))
        .forEach((sourceId) => map.removeSource(sourceId));
    }

    layers.forEach((layer) => {
      if (!layer.visible) return;
      const sourceId = `${sourcePrefix}${layer.id}`;
      if (layer.type === "raster" && layer.tile) {
        const tileUrl = layer.tile.urlTemplate.startsWith("http")
          ? layer.tile.urlTemplate
          : `${window.location.origin}${layer.tile.urlTemplate}`;
        map.addSource(sourceId, {
          type: "raster",
          tiles: [tileUrl],
          tileSize: 256
        });
        map.addLayer({
          id: `${layerPrefix}${layer.id}`,
          type: "raster",
          source: sourceId,
          paint: {
            "raster-opacity": layer.opacity ?? 0.85
          }
        });
      } else if (layer.type === "vector" && layer.geoJson) {
        map.addSource(sourceId, {
          type: "geojson",
          data: layer.geoJson
        });
        const palette = (layer.palette ?? ["#440154", "#30678d", "#35b779", "#fde725", "#f4f18f"]).map((color) =>
          color.startsWith("#") ? color : `#${color}`
        );
        const colorExpression: maplibregl.ExpressionSpecification = ["match", ["get", "zone"]];
        palette.forEach((color, index) => {
          colorExpression.push(index + 1, color);
        });
        colorExpression.push("#6b7280");
        map.addLayer({
          id: `${layerPrefix}${layer.id}-fill`,
          type: "fill",
          source: sourceId,
          paint: {
            "fill-color": colorExpression,
            "fill-opacity": 0.4
          }
        });
        map.addLayer({
          id: `${layerPrefix}${layer.id}-outline`,
          type: "line",
          source: sourceId,
          paint: {
            "line-color": "#1f2937",
            "line-width": 1
          }
        });
      }
    });
  }, [layers]);

  useEffect(() => {
    const map = mapInstance.current;
    if (!map) return;
    const handler = (event: maplibregl.MapMouseEvent & maplibregl.EventData) => {
      if (!noteMode || !onMapClickForNote) return;
      onMapClickForNote({ lat: event.lngLat.lat, lon: event.lngLat.lng });
    };
    if (noteMode && onMapClickForNote) {
      map.getCanvas().style.cursor = "crosshair";
      map.on("click", handler);
    } else {
      map.getCanvas().style.cursor = "";
    }
    return () => {
      map.off("click", handler);
      if (!noteMode) {
        map.getCanvas().style.cursor = "";
      }
    };
  }, [noteMode, onMapClickForNote]);

  useEffect(() => {
    const map = mapInstance.current;
    if (!map) return;
    noteMarkers.current.forEach(({ marker, popup }) => {
      popup.remove();
      marker.remove();
    });
    noteMarkers.current.clear();
    if (!showNotes || !notes.length) return;
    notes.forEach((note) => {
      const iconMeta = categoryMeta(note.category);
      const el = document.createElement("div");
      el.style.width = "30px";
      el.style.height = "30px";
      el.style.borderRadius = "50%";
      el.style.display = "flex";
      el.style.alignItems = "center";
      el.style.justifyContent = "center";
      el.style.fontSize = "18px";
      el.style.color = "#fff";
      el.style.border = "2px solid #fff";
      el.style.boxShadow = "0 4px 12px rgba(15,23,42,0.3)";
      el.style.backgroundColor = iconMeta.color;
      el.textContent = iconMeta.icon;

      const popupContent = document.createElement("div");
      popupContent.style.maxWidth = "240px";
      popupContent.style.fontFamily = "Inter, system-ui, sans-serif";

      const title = document.createElement("div");
      title.style.fontWeight = "600";
      title.style.marginBottom = "4px";
      title.textContent = `${iconMeta.icon} ${iconMeta.label}`;
      popupContent.appendChild(title);

      const body = document.createElement("div");
      body.style.fontSize = "13px";
      body.style.color = "#1f2937";
      body.style.marginBottom = "6px";
      body.style.whiteSpace = "pre-line";
      body.textContent = note.text;
      popupContent.appendChild(body);

      const timestamp = document.createElement("div");
      timestamp.style.fontSize = "11px";
      timestamp.style.color = "#64748b";
      timestamp.textContent = new Date(note.createdAt).toLocaleString();
      popupContent.appendChild(timestamp);
      if (note.photo) {
        const img = document.createElement("img");
        img.src = note.photo;
        img.alt = "Note";
        img.style.width = "100%";
        img.style.borderRadius = "8px";
        img.style.marginTop = "6px";
        popupContent.appendChild(img);
      }
      const popup = new maplibregl.Popup({ closeButton: true, offset: 16 }).setDOMContent(popupContent);
      const marker = new maplibregl.Marker({ element: el })
        .setLngLat([note.lon, note.lat])
        .setPopup(popup)
        .addTo(map);
      el.addEventListener("click", () => {
        popup.addTo(map);
        onSelectNote?.(note.id);
      });
      noteMarkers.current.set(note.id, { marker, popup });
    });
  }, [notes, showNotes, onSelectNote]);

  useEffect(() => {
    const map = mapInstance.current;
    if (!map || !selectedNoteId || !showNotes) return;
    const entry = noteMarkers.current.get(selectedNoteId);
    if (!entry) return;
    const lngLat = entry.marker.getLngLat();
    entry.popup.setLngLat(lngLat).addTo(map);
    map.flyTo({ center: lngLat, zoom: Math.max(map.getZoom(), 15), essential: true });
  }, [selectedNoteId, showNotes]);

  return (
    <div className="relative h-full w-full">
      <div ref={mapRef} className="h-full w-full" />
      {legend ? <LegendOverlay legend={legend} /> : null}
    </div>
  );
}

function LegendOverlay({ legend }: { legend: LegendConfig }) {
  if (!legend) return null;
  if (legend.type === "gradient") {
    const gradient = `linear-gradient(to right, ${legend.colors
      .map((color) => (color.startsWith("#") ? color : `#${color}`))
      .join(", ")})`;
    return (
      <div className="pointer-events-none absolute left-4 top-4 rounded-md bg-white/90 p-3 text-xs text-slate-700 shadow">
        <p className="mb-2 font-semibold uppercase tracking-wide text-slate-500">{legend.title}</p>
        <div className="flex items-center gap-2">
          <span>{legend.min}</span>
          <div className="h-2 w-32 rounded-sm" style={{ background: gradient }} />
          <span>{legend.max}</span>
        </div>
      </div>
    );
  }

  return (
    <div className="pointer-events-none absolute left-4 top-4 rounded-md bg-white/90 p-3 text-xs text-slate-700 shadow">
      <p className="mb-2 font-semibold uppercase tracking-wide text-slate-500">{legend.title}</p>
      <ul className="space-y-1">
        {legend.entries.map((entry) => (
          <li key={entry.label} className="flex items-center gap-2">
            <span
              className="inline-block h-3 w-3 rounded-sm"
              style={{ background: entry.color.startsWith("#") ? entry.color : `#${entry.color}` }}
            />
            <span>{entry.label}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
