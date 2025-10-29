import { ChangeEvent } from "react";
import { Label } from "@radix-ui/react-label";

import type { SeasonInput } from "../lib/api";
import { parseSeasonCsv } from "../lib/validators";

export type ProductKind = "ndvi-month" | "imagery" | "zones-basic" | "zones-advanced";

export type ProductFormProps = {
  product: ProductKind;
  startDate: string;
  endDate: string;
  clamp: [number, number] | undefined;
  onClampChange: (clamp: [number, number] | undefined) => void;
  onStartDateChange: (value: string) => void;
  onEndDateChange: (value: string) => void;
  imageryBands: string[];
  onImageryBandsChange: (bands: string[]) => void;
  breaksText: string;
  onBreaksChange: (value: string) => void;
  seasons: SeasonInput[];
  onSeasonsChange: (seasons: SeasonInput[]) => void;
  onSubmit: () => void;
  disabled: boolean;
  hasAoi: boolean;
};

export function ProductForm(props: ProductFormProps) {
  const {
    product,
    startDate,
    endDate,
    clamp,
    onClampChange,
    onStartDateChange,
    onEndDateChange,
    imageryBands,
    onImageryBandsChange,
    breaksText,
    onBreaksChange,
    seasons,
    onSeasonsChange,
    onSubmit,
    disabled,
    hasAoi
  } = props;

  function handleClampChange(index: number, value: string) {
    if (!value) {
      onClampChange(undefined);
      return;
    }
    const parsed = parseFloat(value);
    if (Number.isNaN(parsed)) return;
    const next: [number, number] = clamp ? [...clamp] as [number, number] : [0, 1];
    next[index] = parsed;
    onClampChange(next);
  }

  async function handleSeasonCsv(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    const parsed = await parseSeasonCsv(file);
    onSeasonsChange(parsed);
  }

  function renderClampFields() {
    if (product !== "ndvi-month") return null;
    return (
      <div className="grid grid-cols-2 gap-3">
        <div className="space-y-1">
          <Label className="text-xs text-slate-600">Clamp Min</Label>
          <input
            type="number"
            step="0.01"
            value={clamp?.[0] ?? ""}
            onChange={(event) => handleClampChange(0, event.target.value)}
            className="w-full rounded-md border border-slate-200 px-2 py-1 text-sm"
            placeholder="0"
          />
        </div>
        <div className="space-y-1">
          <Label className="text-xs text-slate-600">Clamp Max</Label>
          <input
            type="number"
            step="0.01"
            value={clamp?.[1] ?? ""}
            onChange={(event) => handleClampChange(1, event.target.value)}
            className="w-full rounded-md border border-slate-200 px-2 py-1 text-sm"
            placeholder="1"
          />
        </div>
      </div>
    );
  }

  function renderImageryFields() {
    if (product !== "imagery") return null;
    return (
      <div className="space-y-1">
        <Label className="text-xs text-slate-600">Bands (comma separated)</Label>
        <input
          type="text"
          value={imageryBands.join(",")}
          onChange={(event) => onImageryBandsChange(
            event.target.value
              .split(",")
              .map((band) => band.trim())
              .filter(Boolean)
          )}
          className="w-full rounded-md border border-slate-200 px-2 py-1 text-sm"
          placeholder="B4,B3,B2"
        />
      </div>
    );
  }

  function renderBreaksFields() {
    if (product !== "zones-advanced") return null;
    return (
      <div className="space-y-1">
        <Label className="text-xs text-slate-600">Fixed Breaks</Label>
        <input
          type="text"
          value={breaksText}
          onChange={(event) => onBreaksChange(event.target.value)}
          className="w-full rounded-md border border-slate-200 px-2 py-1 text-sm"
          placeholder="-1.0,-0.3,0.3,1.0"
        />
      </div>
    );
  }

  function renderSeasonUpload() {
    if (product !== "zones-advanced") return null;
    return (
      <div className="space-y-1">
        <Label className="text-xs text-slate-600">Seasons CSV</Label>
        <input
          type="file"
          accept=".csv"
          onChange={handleSeasonCsv}
          className="w-full text-sm"
        />
        <p className="text-xs text-slate-500">Loaded seasons: {seasons.length}</p>
      </div>
    );
  }

  return (
    <form
      className="space-y-4"
      onSubmit={(event) => {
        event.preventDefault();
        onSubmit();
      }}
    >
      <div className="grid grid-cols-2 gap-3">
        <div className="space-y-1">
          <Label className="text-xs text-slate-600">Start Date</Label>
          <input
            type="date"
            value={startDate}
            onChange={(event) => onStartDateChange(event.target.value)}
            className="w-full rounded-md border border-slate-200 px-2 py-1 text-sm"
            required
          />
        </div>
        <div className="space-y-1">
          <Label className="text-xs text-slate-600">End Date</Label>
          <input
            type="date"
            value={endDate}
            onChange={(event) => onEndDateChange(event.target.value)}
            className="w-full rounded-md border border-slate-200 px-2 py-1 text-sm"
            required
          />
        </div>
      </div>
      {renderClampFields()}
      {renderImageryFields()}
      {renderBreaksFields()}
      {renderSeasonUpload()}
      <button
        type="submit"
        disabled={disabled || !hasAoi}
        className="w-full rounded-md bg-slate-900 px-3 py-2 text-sm font-semibold text-white transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:bg-slate-400"
      >
        {disabled ? "Working…" : "Run Product"}
      </button>
    </form>
  );
}
