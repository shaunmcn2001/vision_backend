export type DownloadsListProps = {
  downloads: Record<string, string>;
};

function formatLabel(key: string): string {
  return key
    .replace(/([A-Z])/g, " $1")
    .replace(/_/g, " ")
    .replace(/^\w/, (c) => c.toUpperCase());
}

export function DownloadsList({ downloads }: DownloadsListProps) {
  const entries = Object.entries(downloads);
  if (!entries.length) return null;
  return (
    <div className="space-y-3">
      <h3 className="text-sm font-semibold text-slate-700">Downloads</h3>
      <ul className="space-y-2">
        {entries.map(([key, url]) => (
          <li key={key} className="flex items-center justify-between rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-700">
            <span>{formatLabel(key)}</span>
            <a
              href={url}
              target="_blank"
              rel="noreferrer"
              className="rounded-md bg-slate-900 px-3 py-1 text-xs font-semibold text-white hover:bg-slate-800"
            >
              Download
            </a>
          </li>
        ))}
      </ul>
    </div>
  );
}
