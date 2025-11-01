# Vision Frontend

React + Vite dashboard that mirrors the Praedia design to explore Vision backend products. The UI combines a full-screen map with a workflow card for uploading shapefiles, selecting NDVI workflows, toggling returned layers, and downloading artefacts.

## Features
- AOI upload via zipped shapefile or direct GeoJSON paste.
- Product selector for NDVI composites, daily imagery, and zone generation.
- MapLibre map rendering with toggleable raster layers per result.
- Download panel for GeoTIFF, shapefile, and CSV exports.

## Getting Started

### Prerequisites
- Node.js 18+
- Backend API running locally or remotely (defaults to same origin).

### Installation
```bash
cd frontend
npm install
```

### Development
```bash
npm run dev
```
The dev server runs on `http://localhost:5173`. Use `--host` flags in `.env` if you need LAN access.

### Build
```bash
npm run build
npm run preview   # optional production preview
```

## Project Structure
- `src/App.tsx` – Root layout combining map, controls, and workflow logic.
- `src/components/` – Input card, MapLibre wrapper, download list, and shared UI pieces.
- `src/lib/` – API clients, shapefile parser, and validators.
- `index.css` / `tailwind.config.ts` – Styling that matches the Praedia-inspired design.

## Environment Configuration
Create an `.env` file when pointing to a remote API:
```
VITE_API_BASE_URL=https://my-backend.example.com
```
Update `request*` helpers in `src/lib/api.ts` if additional headers (such as API keys) are required.

## Linting & Testing
The project ships with Vitest configuration. Add tests under `src/` and run:
```bash
npm test
```

## Tips
- Layer toggles drive map updates without reloading tiles; keep IDs stable when extending products.
- To add new workflows, extend the product union in `src/App.tsx` and centralise API calls inside `src/lib/api.ts`.
