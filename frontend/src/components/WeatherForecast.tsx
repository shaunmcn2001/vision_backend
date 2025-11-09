import type { SprayRecommendation, WeatherForecastDay } from "../lib/api";
import { WeatherService } from "../lib/weather";

interface WeatherForecastProps {
  days: WeatherForecastDay[];
  recommendation: SprayRecommendation;
}

function formatTemp(min: number, max: number): string {
  return `${Math.round(min)}° / ${Math.round(max)}°C`;
}

function formatRain(value: number): string {
  return `${value.toFixed(1)} mm`;
}

function formatWind(value: number): string {
  return `${value.toFixed(1)} km/h`;
}

export function WeatherForecast({ days, recommendation }: WeatherForecastProps) {
  if (!days.length) {
    return <p className="text-sm text-slate-500">Load a field to view the five-day outlook.</p>;
  }
  return (
    <div className="space-y-3">
      {days.map((day) => {
        const sprayReady = day.sprayOk;
        return (
          <div
            key={day.date}
            className={`flex items-center justify-between rounded-2xl border px-4 py-3 shadow-sm transition ${
              sprayReady
                ? "border-emerald-200 bg-emerald-50/70 text-emerald-900"
                : "border-slate-100 bg-white"
            }`}
          >
            <div className="flex items-center gap-4">
              <div className="text-3xl" aria-hidden>
                {WeatherService.iconFor(day)}
              </div>
              <div>
                <p className="text-sm font-semibold uppercase tracking-wide text-slate-500">
                  {WeatherService.dateLabel(day.date)}
                </p>
                <p className="text-lg font-semibold text-slate-900">{day.description}</p>
                <p className="text-xs text-slate-500">{formatTemp(day.tempMinC, day.tempMaxC)}</p>
              </div>
            </div>
            <div className="text-right text-sm text-slate-600">
              <p>Rain • {formatRain(day.precipitationMm)}</p>
              <p>Wind • {formatWind(day.windAvgKmh)}</p>
              <p>GDD • {day.gdd.toFixed(1)}</p>
              {sprayReady ? (
                <span className="mt-1 inline-flex rounded-full bg-white px-2 py-0.5 text-xs font-semibold text-emerald-600">
                  Spray window
                </span>
              ) : null}
            </div>
          </div>
        );
      })}
      {!recommendation.hasWindow ? (
        <div className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-700">
          No ideal spray window in the next five days — monitor wind and rainfall before scheduling.
        </div>
      ) : (
        <div className="rounded-xl border border-slate-100 bg-slate-50 px-4 py-3 text-sm text-slate-600">
          Best days: {recommendation.bestDays.map((day) => WeatherService.dateLabel(day)).join(", ") || "TBD"}
        </div>
      )}
    </div>
  );
}
