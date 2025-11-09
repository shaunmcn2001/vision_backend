import type { WeatherForecastResponse } from "../lib/api";

interface WeatherChartsProps {
  chart: WeatherForecastResponse["chart"];
  base: number;
}

const WIDTH_STEP = 68;
const PADDING_X = 36;
const HEIGHT = 160;

function buildLinePath(points: { x: number; y: number }[]): string {
  if (!points.length) return "";
  return points.map((point, index) => `${index === 0 ? "M" : "L"} ${point.x} ${point.y}`).join(" ");
}

export function WeatherCharts({ chart, base }: WeatherChartsProps) {
  const gddSeries = chart.gdd;
  const precipSeries = chart.precipitation;
  if (!gddSeries.length || !precipSeries.length) {
    return <p className="text-sm text-slate-500">Weather chart will appear after loading the forecast.</p>;
  }
  const segments = Math.max(gddSeries.length - 1, 1);
  const innerWidth = segments * WIDTH_STEP;
  const width = innerWidth + PADDING_X * 2;
  const maxGdd = Math.max(...gddSeries.map((point) => point.cumulative), 1);
  const maxPrecip = Math.max(...precipSeries.map((point) => point.value), 1);
  const step = gddSeries.length > 1 ? innerWidth / (gddSeries.length - 1) : 0;

  const points = gddSeries.map((point, index) => {
    const ratio = point.cumulative / maxGdd;
    const x = PADDING_X + index * step;
    const y = HEIGHT - 40 - ratio * 80;
    return { x, y };
  });

  const gddPath = buildLinePath(points);

  return (
    <div className="space-y-6">
      <div>
        <div className="mb-3 flex items-center justify-between">
          <p className="text-sm font-semibold text-slate-700">GDD accumulation</p>
          <p className="text-xs text-slate-500">Base {base.toFixed(0)}°C</p>
        </div>
        <svg viewBox={`0 0 ${width} ${HEIGHT}`} className="h-40 w-full" role="img" aria-label="GDD chart">
          <defs>
            <linearGradient id="gddGradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#a5b4fc" />
              <stop offset="100%" stopColor="#4c1d95" />
            </linearGradient>
          </defs>
          <rect x="0" y="0" width={width} height={HEIGHT} fill="#f8fafc" rx="16" />
          <path d={gddPath} fill="none" stroke="url(#gddGradient)" strokeWidth={4} strokeLinecap="round" />
          {points.map((point, index) => (
            <g key={`${point.x}-${index}`}>
              <circle cx={point.x} cy={point.y} r={5} fill="#fff" stroke="#312e81" strokeWidth={1.5} />
              <text x={point.x} y={HEIGHT - 12} textAnchor="middle" className="text-[10px] fill-slate-500">
                {new Date(gddSeries[index].date).toLocaleDateString(undefined, { weekday: "short" })}
              </text>
            </g>
          ))}
        </svg>
      </div>
      <div>
        <div className="mb-3 flex items-center justify-between">
          <p className="text-sm font-semibold text-slate-700">Daily precipitation</p>
          <p className="text-xs text-slate-500">Next five days</p>
        </div>
        <div className="flex items-end gap-3 rounded-2xl border border-slate-100 bg-slate-50 p-4">
          {precipSeries.map((point) => {
            const heightPct = Math.max(point.value / maxPrecip, 0.05);
            return (
              <div key={point.date} className="flex flex-1 flex-col items-center gap-2">
                <div className="w-full rounded-full bg-gradient-to-t from-sky-200 to-sky-500" style={{ height: `${heightPct * 120}px` }} />
                <p className="text-xs font-semibold text-slate-600">{point.value.toFixed(1)} mm</p>
                <p className="text-[11px] text-slate-500">
                  {new Date(point.date).toLocaleDateString(undefined, { weekday: "short" })}
                </p>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
