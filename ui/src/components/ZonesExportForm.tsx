import React, { useMemo, useState } from "react";

function parseThresholds(input) {
  if (!input.trim()) return null;
  const parts = input.split(/[\s,]+/).map(s => s.trim()).filter(Boolean);
  if (!parts.length) return null;
  const vals = parts.map(Number);
  if (vals.some(v => Number.isNaN(v))) return null;
  return vals;
}

export default function ZonesExportForm() {
  const [aoiName, setAoiName] = useState("MyField");
  const [aoiGeojson, setAoiGeojson] = useState("");
  const [monthsStart, setMonthsStart] = useState("2024-05");
  const [monthsEnd, setMonthsEnd] = useState("2024-09");
  const [nClasses, setNClasses] = useState(5);
  const [ndviMin, setNdviMin] = useState("0.35");
  const [ndviMax, setNdviMax] = useState("0.73");
  const [customText, setCustomText] = useState("");
  const [cloudProb, setCloudProb] = useState("80");
  const [mmuHa, setMmuHa] = useState("1");
  const [smoothM, setSmoothM] = useState("0");
  const [includeStats, setIncludeStats] = useState(true);
  const [mode, setMode] = useState("linear");

  const customParsed = useMemo(() => parseThresholds(customText), [customText]);

  const previewEdges = useMemo(() => {
    if (customParsed && customParsed.length === nClasses + 1) {
      return [...customParsed].sort((a, b) => a - b);
    }
    const lo = parseFloat(ndviMin);
    const hi = parseFloat(ndviMax);
    if (!Number.isFinite(lo) || !Number.isFinite(hi) || hi <= lo) return [];
    const step = (hi - lo) / nClasses;
    return Array.from({ length: nClasses + 1 }, (_, i) => +(lo + i * step).toFixed(4));
  }, [customParsed, nClasses, ndviMin, ndviMax]);

  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  async function submit() {
    setBusy(true);
    setError(null);
    setResult(null);

    let geo;
    try {
      geo = JSON.parse(aoiGeojson);
    } catch {
      setBusy(false);
      setError("AOI GeoJSON invalid");
      return;
    }

    const monthsBetween = (a, b) => {
      const out = [];
      const [ay, am] = a.split("-").map(Number);
      const [by, bm] = b.split("-").map(Number);
      let y = ay, m = am;
      while (y < by || (y === by && m <= bm)) {
        out.push(`${y}-${String(m).padStart(2, "0")}`);
        m++; if (m > 12) { m = 1; y++; }
      }
      return out;
    };

    const payload = {
      aoi_geojson: geo,
      aoi_name: aoiName,
      months: monthsBetween(monthsStart, monthsEnd),
      export_target: "zip",
      method: "ndvi_linear",
      mode,
      n_classes: nClasses,
      cloud_prob_max: Number(cloudProb),
      mmu_ha: Number(mmuHa),
      smooth_radius_m: Number(smoothM),
      include_zonal_stats: includeStats,
    };

    if (customParsed && customParsed.length === nClasses + 1) {
      payload.custom_thresholds = customParsed.map(Number);
    } else {
      payload.ndvi_min = Number(ndviMin);
      payload.ndvi_max = Number(ndviMax);
    }

    try {
      const res = await fetch("/api/zones/production", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!res.ok) {
        setError(data?.detail || "Export failed");
      } else {
        setResult(data);
      }
    } catch (e) {
      setError(e.message || "Network error");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="max-w-3xl mx-auto p-4 space-y-6">
      <h1 className="text-2xl font-semibold">Production Zones Export</h1>
      <div className="grid gap-4">
        <label className="block">
          <span>AOI Name</span>
          <input className="w-full border p-2" value={aoiName} onChange={e => setAoiName(e.target.value)} />
        </label>
        <label className="block">
          <span>AOI GeoJSON</span>
          <textarea className="w-full border p-2 font-mono text-sm" rows={8}
            value={aoiGeojson} onChange={e => setAoiGeojson(e.target.value)} />
        </label>
        <div className="grid grid-cols-2 gap-4">
          <input value={monthsStart} onChange={e => setMonthsStart(e.target.value)} className="border p-2" />
          <input value={monthsEnd} onChange={e => setMonthsEnd(e.target.value)} className="border p-2" />
        </div>
        <div className="flex gap-4">
          <label><input type="radio" checked={nClasses===3} onChange={()=>setNClasses(3)} />3 zones</label>
          <label><input type="radio" checked={nClasses===5} onChange={()=>setNClasses(5)} />5 zones</label>
        </div>
        <div className="grid grid-cols-3 gap-4">
          <input placeholder="NDVI min" value={ndviMin} onChange={e=>setNdviMin(e.target.value)} className="border p-2" />
          <input placeholder="NDVI max" value={ndviMax} onChange={e=>setNdviMax(e.target.value)} className="border p-2" />
          <input placeholder="Custom thresholds" value={customText} onChange={e=>setCustomText(e.target.value)} className="border p-2" />
        </div>
        <div>Preview: {previewEdges.join(", ")}</div>
        <button onClick={submit} disabled={busy} className="bg-black text-white px-4 py-2 rounded">{busy?"Exporting…":"Export Zones"}</button>
        {error && <div className="text-red-600">{error}</div>}
        {result && <pre className="text-xs whitespace-pre-wrap border p-2">{JSON.stringify(result,null,2)}</pre>}
      </div>
    </div>
  );
}
