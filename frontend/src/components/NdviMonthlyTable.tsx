export type NdviMonthlyTableRow = {
  label: string;
  formattedLabel: string;
  year: number;
  month: number;
  meanNdvi?: number | null;
};

export type NdviMonthlyTableProps = {
  rows: NdviMonthlyTableRow[];
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

export function NdviMonthlyTable({ rows }: NdviMonthlyTableProps) {
  if (!rows.length) {
    return (
      <div className="rounded-lg border border-slate-100 bg-slate-50 px-4 py-3 text-sm text-slate-500">
        Monthly NDVI stats for the last year will appear after running NDVI.
      </div>
    );
  }

  const ordered = [...rows].sort((a, b) => a.year - b.year || a.month - b.month);

  function handleDownloadCsv() {
    const header = ["label", "year", "month", "mean_ndvi"];
    const dataRows = ordered.map((entry) => [
      entry.label,
      entry.year.toString(),
      entry.month.toString().padStart(2, "0"),
      typeof entry.meanNdvi === "number" ? entry.meanNdvi.toFixed(4) : ""
    ]);
    const csv = [header.join(","), ...dataRows.map((row) => row.join(","))].join("\n");
    downloadBlob("ndvi_last_year_monthly.csv", csv, "text/csv");
  }

  function handleDownloadJson() {
    const payload = ordered.map((entry) => ({
      label: entry.label,
      year: entry.year,
      month: entry.month,
      meanNdvi: entry.meanNdvi ?? null
    }));
    downloadBlob("ndvi_last_year_monthly.json", JSON.stringify(payload, null, 2), "application/json");
  }

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="text-sm font-semibold text-slate-700">Last-year monthly averages</p>
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
              <th className="px-3 py-2 font-semibold text-slate-600">Month</th>
              <th className="px-3 py-2 font-semibold text-slate-600">Mean NDVI</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100 bg-white">
            {ordered.map((entry) => (
              <tr key={entry.label}>
                <td className="px-3 py-2 text-slate-700">{entry.formattedLabel}</td>
                <td className="px-3 py-2 text-slate-700">{formatValue(entry.meanNdvi)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
