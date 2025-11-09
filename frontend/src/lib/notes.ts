import type { GeometryInput } from "./api";

export type NoteCategoryId = "pest" | "disease" | "weed" | "water" | "nutrient" | "observation";

export interface FieldNote {
  id: string;
  lat: number;
  lon: number;
  text: string;
  category: NoteCategoryId;
  photo?: string | null;
  createdAt: string;
}

export const NOTE_CATEGORIES: { id: NoteCategoryId; label: string; icon: string; color: string }[] = [
  { id: "pest", label: "Pest", icon: "🪲", color: "#ef4444" },
  { id: "disease", label: "Disease", icon: "🧫", color: "#f97316" },
  { id: "weed", label: "Weed", icon: "🌿", color: "#22c55e" },
  { id: "water", label: "Water Stress", icon: "💧", color: "#0ea5e9" },
  { id: "nutrient", label: "Nutrient", icon: "🧪", color: "#a855f7" },
  { id: "observation", label: "Observation", icon: "📍", color: "#475569" }
];

export function categoryMeta(id: NoteCategoryId) {
  return NOTE_CATEGORIES.find((category) => category.id === id) ?? NOTE_CATEGORIES[NOTE_CATEGORIES.length - 1];
}

function geometryHash(geometry: GeometryInput | null): string {
  const raw = JSON.stringify(geometry ?? {});
  let hash = 0;
  for (let i = 0; i < raw.length; i += 1) {
    hash = (hash * 31 + raw.charCodeAt(i)) >>> 0; // eslint-disable-line no-bitwise
  }
  return hash.toString(16);
}

export function notesStorageKey(geometry: GeometryInput | null): string {
  return `field-notes:${geometryHash(geometry)}`;
}

export function loadNotes(key: string): FieldNote[] {
  if (typeof window === "undefined") return [];
  try {
    const stored = window.localStorage.getItem(key);
    if (!stored) return [];
    const parsed = JSON.parse(stored);
    if (!Array.isArray(parsed)) return [];
    return parsed as FieldNote[];
  } catch (error) {
    return [];
  }
}

export function saveNotes(key: string, notes: FieldNote[]): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(key, JSON.stringify(notes));
  } catch (error) {
    // ignore storage errors
  }
}
