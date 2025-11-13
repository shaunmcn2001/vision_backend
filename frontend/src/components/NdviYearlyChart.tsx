import { useMemo } from "react";

import type { NdviYearlyAverage } from "../lib/api";
import { NdviTrendChart } from "./NdviTrendChart";
import type { TrendPoint } from "./NdviTrendChart";

export type NdviYearlyChartProps = {
  averages: NdviYearlyAverage[];
  clamp: [number, number];
};

export function NdviYearlyChart({ averages, clamp }: NdviYearlyChartProps) {
  const points = useMemo<TrendPoint[]>(
    () =>
      [...averages]
        .sort((a, b) => a.year - b.year)
        .map((average) => ({
          label: average.year.toString(),
          value: typeof average.meanNdvi === "number" ? average.meanNdvi : null
        })),
    [averages]
  );

  return <NdviTrendChart points={points} clamp={clamp} />;
}
