import { FormEvent, useEffect, useState } from "react";

import { NOTE_CATEGORIES, type NoteCategoryId } from "../lib/notes";

type NoteModalProps = {
  open: boolean;
  location: { lat: number; lon: number } | null;
  onClose: () => void;
  onSave: (payload: { text: string; category: NoteCategoryId; photo?: string | null }) => void;
};

export function NoteModal({ open, location, onClose, onSave }: NoteModalProps) {
  const [text, setText] = useState("");
  const [category, setCategory] = useState<NoteCategoryId>("observation");
  const [photo, setPhoto] = useState<string | null>(null);
  const [loadingPhoto, setLoadingPhoto] = useState(false);

  useEffect(() => {
    if (!open) {
      setText("");
      setCategory("observation");
      setPhoto(null);
      setLoadingPhoto(false);
    }
  }, [open]);

  function handleFileChange(file: File | undefined) {
    if (!file) {
      setPhoto(null);
      return;
    }
    setLoadingPhoto(true);
    const reader = new FileReader();
    reader.onload = () => {
      setPhoto(typeof reader.result === "string" ? reader.result : null);
      setLoadingPhoto(false);
    };
    reader.onerror = () => setLoadingPhoto(false);
    reader.readAsDataURL(file);
  }

  function handleSubmit(event: FormEvent) {
    event.preventDefault();
    if (!text.trim()) return;
    onSave({ text: text.trim(), category, photo });
  }

  if (!open || !location) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/60 px-4 py-8">
      <div className="w-full max-w-lg rounded-2xl bg-white p-6 shadow-2xl">
        <div className="mb-4 flex items-start justify-between">
          <div>
            <p className="text-xs uppercase tracking-wide text-slate-400">New field note</p>
            <h3 className="text-xl font-semibold text-slate-900">Capture observation</h3>
            <p className="text-xs text-slate-500">{location.lat.toFixed(5)}, {location.lon.toFixed(5)}</p>
          </div>
          <button type="button" onClick={onClose} className="text-slate-500 transition hover:text-slate-900">
            ✕
          </button>
        </div>
        <form className="space-y-4" onSubmit={handleSubmit}>
          <div className="space-y-1">
            <label className="text-xs font-semibold uppercase tracking-wide text-slate-500">Category</label>
            <select
              className="w-full rounded-xl border border-slate-200 px-3 py-2 text-sm text-slate-900"
              value={category}
              onChange={(event) => setCategory(event.target.value as NoteCategoryId)}
            >
              {NOTE_CATEGORIES.map((item) => (
                <option key={item.id} value={item.id}>
                  {item.label}
                </option>
              ))}
            </select>
          </div>
          <div className="space-y-1">
            <label className="text-xs font-semibold uppercase tracking-wide text-slate-500">Notes</label>
            <textarea
              className="w-full rounded-xl border border-slate-200 px-3 py-2 text-sm text-slate-900"
              rows={4}
              required
              value={text}
              onChange={(event) => setText(event.target.value)}
              placeholder="Describe what you observed..."
            />
          </div>
          <div className="space-y-1">
            <label className="text-xs font-semibold uppercase tracking-wide text-slate-500">Photo</label>
            <input
              type="file"
              accept="image/*"
              onChange={(event) => handleFileChange(event.target.files?.[0])}
              className="text-sm"
            />
            {loadingPhoto ? <p className="text-xs text-slate-500">Loading preview…</p> : null}
            {photo ? <img src={photo} alt="Note" className="mt-2 h-32 w-full rounded-lg object-cover" /> : null}
          </div>
          <div className="flex justify-end gap-3">
            <button
              type="button"
              onClick={onClose}
              className="rounded-xl border border-slate-200 px-4 py-2 text-sm font-semibold text-slate-600"
            >
              Cancel
            </button>
            <button
              type="submit"
              className="rounded-xl bg-emerald-600 px-4 py-2 text-sm font-semibold text-white shadow-lg shadow-emerald-600/30"
            >
              Save note
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
