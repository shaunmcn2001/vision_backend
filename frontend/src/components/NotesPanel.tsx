import { NOTE_CATEGORIES, categoryMeta, type FieldNote } from "../lib/notes";

interface NotesPanelProps {
  notes: FieldNote[];
  notesVisible: boolean;
  noteMode: boolean;
  onToggleVisibility: () => void;
  onStartNote: () => void;
  onSelectNote: (id: string) => void;
  onExportJson: () => void;
  onExportCsv: () => void;
}

function formatTimestamp(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString(undefined, {
    weekday: "short",
    hour: "2-digit",
    minute: "2-digit",
    month: "short",
    day: "numeric"
  });
}

export function NotesPanel({
  notes,
  notesVisible,
  noteMode,
  onToggleVisibility,
  onStartNote,
  onSelectNote,
  onExportCsv,
  onExportJson
}: NotesPanelProps) {
  return (
    <section className="rounded-2xl border border-slate-100 bg-white/95 p-5 shadow-sm">
      <div className="mb-4 flex items-center justify-between">
        <div>
          <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Notes</p>
          <h2 className="text-lg font-semibold text-slate-900">Field scouting</h2>
          <p className="text-xs text-slate-500">{notes.length} {notes.length === 1 ? "note" : "notes"} in this field</p>
        </div>
        <div className="flex flex-col items-end gap-2">
          <button
            type="button"
            onClick={onStartNote}
            className={`rounded-xl px-4 py-2 text-xs font-semibold uppercase tracking-wide transition ${
              noteMode ? "bg-emerald-600 text-white" : "bg-slate-900 text-white hover:bg-slate-700"
            }`}
          >
            {noteMode ? "Click map to drop" : "Add note"}
          </button>
          <button
            type="button"
            onClick={onToggleVisibility}
            className={`text-[11px] font-semibold uppercase tracking-wide ${notesVisible ? "text-emerald-600" : "text-slate-500"}`}
          >
            {notesVisible ? "Hide markers" : "Show markers"}
          </button>
        </div>
      </div>
      {notes.length ? (
        <div className="space-y-3">
          {notes.map((note) => {
            const meta = categoryMeta(note.category);
            return (
              <button
                key={note.id}
                type="button"
                onClick={() => onSelectNote(note.id)}
                className="flex w-full gap-3 rounded-2xl border border-slate-100 bg-white px-4 py-3 text-left shadow-sm transition hover:border-emerald-200"
              >
                <div
                  className="flex h-12 w-12 items-center justify-center rounded-xl text-2xl"
                  style={{ backgroundColor: `${meta.color}22` }}
                >
                  {meta.icon}
                </div>
                <div className="flex-1">
                  <p className="text-sm font-semibold text-slate-900">{meta.label}</p>
                  <p className="text-xs text-slate-500">{note.text.slice(0, 80)}{note.text.length > 80 ? "…" : ""}</p>
                  <p className="text-[11px] text-slate-400">{formatTimestamp(note.createdAt)}</p>
                </div>
                {note.photo ? (
                  <img src={note.photo} alt="Note" className="h-12 w-16 rounded-lg object-cover" />
                ) : null}
              </button>
            );
          })}
        </div>
      ) : (
        <p className="text-sm text-slate-500">No notes yet. Enable Add Note mode to place your first marker.</p>
      )}
      <div className="mt-4 flex flex-wrap gap-2">
        <button
          type="button"
          onClick={onExportJson}
          className="rounded-xl border border-slate-200 px-4 py-2 text-xs font-semibold uppercase tracking-wide text-slate-600"
        >
          Export JSON
        </button>
        <button
          type="button"
          onClick={onExportCsv}
          className="rounded-xl border border-slate-200 px-4 py-2 text-xs font-semibold uppercase tracking-wide text-slate-600"
        >
          Export CSV
        </button>
      </div>
      <div className="mt-4 text-xs text-slate-400">
        Categories: {NOTE_CATEGORIES.map((cat) => `${cat.icon} ${cat.label}`).join(" · ")}
      </div>
    </section>
  );
}
