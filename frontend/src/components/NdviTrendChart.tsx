import { useId, useMemo } from "react";

export type TrendPoint = {
  label: string;
  value: number | null;
};

type NdviTrendChartProps = {
  points: TrendPoint[];
  clamp: [number, number];
};

const CHART_HEIGHT = 160;
const LEFT_PADDING = 28;
const RIGHT_PADDING = 28;
const TOP_PADDING = 16;
const BOTTOM_PADDING = 34;

export function NdviTrendChart({ points, clamp }: NdviTrendChartProps) {
  const gradientId = useId();
  const areaGradientId = `${gradientId}-area`;

  const chart = useMemo(() => {
    if (!points.length) {
      return {
        coords: [] as Array<{ label: string; value: number | null; x: number; y: number | null }>,
        width: LEFT_PADDING + RIGHT_PADDING + 200,
        linePath: "",
        areaPath: "",
        hasValues: false
      };
    }
    const [minClamp, maxClamp] = clamp;
    const range = Math.max(maxClamp - minClamp, 0.0001);
    const innerWidth = Math.max(points.length - 1, 1) * 74;
    const width = LEFT_PADDING + RIGHT_PADDING + innerWidth;
    const step = innerWidth / Math.max(points.length - 1, 1);

    const coords = points.map((point, index) => {
      const raw = typeof point.value === "number" ? point.value : null;
      const clampedValue = raw == null ? null : Math.min(Math.max(raw, minClamp), maxClamp);
      const normalised = clampedValue == null ? null : (clampedValue - minClamp) / range;
      const availableHeight = CHART_HEIGHT - TOP_PADDING - BOTTOM_PADDING;
      const y = normalised == null ? null : TOP_PADDING + (1 - normalised) * availableHeight;
      const x = LEFT_PADDING + index * step;
      return { label: point.label, value: raw, x, y };
    });

    const series = coords.filter((coord) => coord.y != null) as Array<{
      label: string;
      value: number;
      x: number;
      y: number;
    }>;

    const linePath = series
      .map((coord, idx) => `${idx === 0 ? "M" : "L"} ${coord.x} ${coord.y}`)
      .join(" ");

    const areaPath = series.length
      ? `${linePath} L ${series[series.length - 1].x} ${CHART_HEIGHT - BOTTOM_PADDING} L ${series[0].x} ${CHART_HEIGHT - BOTTOM_PADDING} Z`
      : "";

    return {
      coords,
      width,
      linePath,
      areaPath,
      hasValues: series.length > 0
    };
  }, [points, clamp]);

  if (!points.length) {
    return (
      <div className="rounded-xl border border-slate-100 bg-slate-50 p-4 text-sm text-slate-500">
        Upload a field and run NDVI to unlock the season trend.
      </div>
    );
  }

  return (
    <div>
      <svg
        viewBox={`0 0 ${chart.width} ${CHART_HEIGHT}`}
        className="h-40 w-full"
        role="img"
        aria-label="NDVI trend chart"
      >
        <defs>
          <linearGradient id={gradientId} x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#f6cf75" />
            <stop offset="50%" stopColor="#4f9d69" />
            <stop offset="100%" stopColor="#226f54" />
          </linearGradient>
          <linearGradient id={areaGradientId} x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="rgba(79, 157, 105, 0.25)" />
            <stop offset="100%" stopColor="rgba(34, 111, 84, 0.05)" />
          </linearGradient>
        </defs>
        <rect
          x="0"
          y="0"
          width={chart.width}
          height={CHART_HEIGHT}
          fill="#f8fafc"
          rx="12"
        />
        <line
          x1={LEFT_PADDING}
          x2={chart.width - RIGHT_PADDING}
          y1={TOP_PADDING}
          y2={TOP_PADDING}
          stroke="#e2e8f0"
          strokeDasharray="4 4"
        />
        <line
          x1={LEFT_PADDING}
          x2={chart.width - RIGHT_PADDING}
          y1={CHART_HEIGHT - BOTTOM_PADDING}
          y2={CHART_HEIGHT - BOTTOM_PADDING}
          stroke="#e2e8f0"
        />
        {chart.areaPath && chart.hasValues ? (
          <path d={chart.areaPath} fill={`url(#${areaGradientId})`} />
        ) : null}
        {chart.linePath && chart.hasValues ? (
          <path
            d={chart.linePath}
            fill="none"
            stroke={`url(#${gradientId})`}
            strokeWidth={3}
            strokeLinecap="round"
          />
        ) : null}
        {chart.coords.map((coord, index) =>
          coord.y != null ? (
            <g key={`${coord.label}-${index}`}>
              <circle cx={coord.x} cy={coord.y} r={5} fill="#fff" stroke="#0f172a" strokeWidth={1.5} />
              <circle cx={coord.x} cy={coord.y} r={3} fill="#0f172a" />
            </g>
          ) : null
        )}
        <text
          x={LEFT_PADDING}
          y={TOP_PADDING - 4}
          className="text-[10px] fill-slate-500"
        >
          {clamp[1].toFixed(2)}
        </text>
        <text
          x={LEFT_PADDING}
          y={CHART_HEIGHT - BOTTOM_PADDING + 14}
          className="text-[10px] fill-slate-500"
        >
          {clamp[0].toFixed(2)}
        </text>
      </svg>
      <div className="mt-4 grid grid-cols-2 gap-2 text-sm text-slate-600">
        {points.map((point, index) => (
          <div
            key={`${point.label}-${index}`}
            className="rounded-lg border border-slate-100 bg-slate-50 px-3 py-2"
          >
            <p className="text-xs uppercase tracking-wide text-slate-500">{point.label}</p>
            <p className="text-base font-semibold text-slate-900">
              {typeof point.value === "number" ? point.value.toFixed(2) : "—"}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}
