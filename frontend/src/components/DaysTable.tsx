import type { ImageryDay } from "../lib/api";

export type DaysTableProps = {
  days: ImageryDay[];
  selectedDate: string | null;
  onSelect: (day: ImageryDay) => void;
};

export function DaysTable({ days, selectedDate, onSelect }: DaysTableProps) {
  if (!days.length) return null;
  return (
    <div className="overflow-hidden rounded-lg border border-slate-200">
      <table className="min-w-full divide-y divide-slate-200 text-sm">
        <thead className="bg-slate-50 text-left">
          <tr>
            <th className="px-3 py-2 font-semibold text-slate-600">Date</th>
            <th className="px-3 py-2 font-semibold text-slate-600">Cloud %</th>
            <th className="px-3 py-2" />
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-100 bg-white">
          {days.map((day) => {
            const isActive = selectedDate === day.date;
            return (
              <tr key={day.date} className={isActive ? "bg-slate-100" : undefined}>
                <td className="px-3 py-2 text-slate-700">{day.date}</td>
                <td className="px-3 py-2 text-slate-600">{day.cloudPct.toFixed(1)}</td>
                <td className="px-3 py-2 text-right">
                  <button
                    type="button"
                    onClick={() => onSelect(day)}
                    className="rounded-md bg-slate-900 px-3 py-1 text-xs font-semibold text-white hover:bg-slate-800"
                  >
                    View
                  </button>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
