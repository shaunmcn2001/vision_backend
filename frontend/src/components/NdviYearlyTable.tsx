import type { NdviYearlyAverage } from "../lib/api";

export type NdviYearlyTableProps = {
  averages: NdviYearlyAverage[];
};

function downloadBlob(filename: string, content: string, type: string) {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

function formatValue(value: number | null | undefined) {
  if (typeof value !== "number") return "—";
  return value.toFixed(3);
}

export function NdviYearlyTable({ averages }: NdviYearlyTableProps) {
  if (!averages.length) {
    return (
      <div className="rounded-lg border border-slate-100 bg-slate-50 px-4 py-3 text-sm text-slate-500">
        Yearly NDVI history is unavailable for this field.
      </div>
    );
  }

  const sorted = [...averages].sort((a, b) => a.year - b.year);

  function handleDownloadCsv() {
    const header = ["year", "mean_ndvi"];
    const rows = sorted.map((entry) => [
      entry.year.toString(),
      typeof entry.meanNdvi === "number" ? entry.meanNdvi.toFixed(4) : ""
    ]);
    const csv = [header.join(","), ...rows.map((row) => row.join(","))].join("\n");
    downloadBlob("ndvi_yearly_history.csv", csv, "text/csv");
  }

  function handleDownloadJson() {
    const payload = sorted.map((entry) => ({ year: entry.year, meanNdvi: entry.meanNdvi ?? null }));
    downloadBlob("ndvi_yearly_history.json", JSON.stringify(payload, null, 2), "application/json");
  }

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="text-sm font-semibold text-slate-700">10-year yearly averages</p>
        <div className="flex gap-2">
          <button
            type="button"
            onClick={handleDownloadCsv}
            className="rounded-md border border-slate-200 px-3 py-1 text-xs font-semibold uppercase tracking-wide text-slate-600 hover:border-emerald-200 hover:text-emerald-700"
          >
            Download CSV
          </button>
          <button
            type="button"
            onClick={handleDownloadJson}
            className="rounded-md border border-slate-200 px-3 py-1 text-xs font-semibold uppercase tracking-wide text-slate-600 hover:border-emerald-200 hover:text-emerald-700"
          >
            Download JSON
          </button>
        </div>
      </div>
      <div className="overflow-hidden rounded-lg border border-slate-200">
        <table className="min-w-full divide-y divide-slate-200 text-sm">
          <thead className="bg-slate-50 text-left">
            <tr>
              <th className="px-3 py-2 font-semibold text-slate-600">Year</th>
              <th className="px-3 py-2 font-semibold text-slate-600">Mean NDVI</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100 bg-white">
            {sorted.map((entry) => (
              <tr key={entry.year}>
                <td className="px-3 py-2 text-slate-700">{entry.year}</td>
                <td className="px-3 py-2 text-slate-700">{formatValue(entry.meanNdvi)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
