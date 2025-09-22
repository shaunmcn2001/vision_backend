import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from app.api.export import sentinel2_router as export_shapefile_router
from app.api.routes import router as api_router
from app.api.fields import router as fields_router
from app.api.fields_upload import router as fields_upload_router
from app.api.tiles import router as tiles_router
from app.api.s2_indices import router as s2_indices_router
from app import gee
import os

app = FastAPI(
    title="Agri NDVI Backend",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Allow all origins (adjust later if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger(__name__)


@app.on_event("startup")
def _startup_init_gee() -> None:
    """Initialise Earth Engine once the application starts."""
    try:
        gee.initialize()
        logger.info("Earth Engine initialised for Sentinel-2 exports")
    except Exception as exc:  # pragma: no cover - best effort
        logger.warning("Earth Engine initialisation skipped: %s", exc)

@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "service": "vision-backend",
        "project": os.getenv("GCP_PROJECT"),
        "region": os.getenv("GCP_REGION"),
    }

@app.get("/")
def root():
    return {"service": "vision-backend", "status": "running"}


@app.get("/ui", response_class=HTMLResponse)
def export_ui():
    return HTMLResponse(
        content=r"""
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Sentinel-2 Index Export</title>
    <style>
      :root {
        color-scheme: light dark;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      }
      body {
        margin: 0;
        background: #f7f7f7;
      }
      .container {
        max-width: 720px;
        margin: 4rem auto;
        padding: 2.75rem;
        background: #ffffff;
        border-radius: 12px;
        box-shadow: 0 12px 40px rgba(15, 23, 42, 0.08);
      }
      h1 {
        margin-top: 0;
        margin-bottom: 1rem;
        font-size: 1.875rem;
        line-height: 1.2;
        color: #1f2933;
        text-align: center;
      }
      p.instructions {
        color: #475569;
        font-size: 0.95rem;
        margin-bottom: 2rem;
        text-align: center;
      }
      form {
        display: grid;
        gap: 1.35rem;
      }
      label {
        display: grid;
        gap: 0.5rem;
        font-weight: 600;
        color: #1f2933;
      }
      input[type="date"],
      input[type="text"],
      input[type="file"],
      select {
        padding: 0.65rem 0.75rem;
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        font: inherit;
        background: #ffffff;
      }
      input[type="file"] {
        padding: 0.5rem;
      }
      select[multiple] {
        min-height: 10rem;
      }
      .hint {
        font-weight: 400;
        font-size: 0.85rem;
        color: #64748b;
      }
      button {
        padding: 0.85rem 1.6rem;
        border: none;
        border-radius: 999px;
        background: #2563eb;
        color: #ffffff;
        font-weight: 600;
        cursor: pointer;
        transition: background 0.2s ease;
        font: inherit;
      }
      button:hover {
        background: #1d4ed8;
      }
      button:disabled {
        background: #94a3b8;
        cursor: not-allowed;
      }
      #status {
        min-height: 1.75rem;
        font-size: 0.95rem;
        text-align: center;
      }
      #status.success {
        color: #047857;
      }
      #status.error {
        color: #b91c1c;
      }
      #status.pending {
        color: #334155;
      }
    </style>
  </head>
  <body>
    <main class="container">
      <h1>Sentinel-2 Index Export</h1>
      <p class="instructions">
        Upload a zipped shapefile bundle containing .shp, .dbf, and .shx files, choose your
        date range, pick the vegetation indices to include, and select the export destination.
        The workflow will queue one export per month for every index, so a broader date range
        or more indices means more GeoTIFFs will be built for you.
      </p>
      <form id="export-form" novalidate>
        <label for="file">
          Shapefile archive (.zip)
          <input id="file" name="file" type="file" accept=".zip" required />
          <span class="hint">Include the .shp, .dbf, and .shx members in a single ZIP file.</span>
        </label>
        <label for="start_date">
          Start date
          <input id="start_date" name="start_date" type="date" required />
        </label>
        <label for="end_date">
          End date
          <input id="end_date" name="end_date" type="date" required />
          <span class="hint">Every month that overlaps the range will be exported.</span>
        </label>
        <label for="aoi_name">
          AOI name
          <input id="aoi_name" name="aoi_name" type="text" placeholder="Used in file names" required />
        </label>
        <label for="indices">
          Vegetation indices to export
          <select id="indices" name="indices" multiple size="10" required>
            <option value="NDVI" selected>NDVI</option>
            <option value="EVI">EVI</option>
            <option value="GNDVI">GNDVI</option>
            <option value="NDRE">NDRE</option>
            <option value="SAVI">SAVI</option>
            <option value="MSAVI">MSAVI</option>
            <option value="VARI">VARI</option>
            <option value="MCARI">MCARI</option>
            <option value="NDWI_McFeeters">NDWI (McFeeters)</option>
            <option value="NDWI_Gao">NDWI (Gao)</option>
            <option value="NDMI">NDMI</option>
            <option value="MSI">MSI</option>
            <option value="GVMI">GVMI</option>
            <option value="NBR">NBR</option>
            <option value="PSRI">PSRI</option>
            <option value="ARI">ARI</option>
            <option value="CRI">CRI</option>
            <option value="BSI">BSI</option>
            <option value="SBI">SBI</option>
            <option value="NDSI_Soil">NDSI (Soil)</option>
            <option value="NDTI">NDTI</option>
            <option value="PRI">PRI</option>
          </select>
          <span class="hint">Hold Ctrl (Windows/Linux) or ⌘ (macOS) to choose multiple indices.</span>
        </label>
        <label for="export_target">
          Export destination
          <select id="export_target" name="export_target">
            <option value="zip" selected>Download ZIP</option>
            <option value="gcs">Google Cloud Storage</option>
            <option value="drive">Google Drive</option>
          </select>
        </label>
        <label for="api_key">
          API key (optional)
          <input id="api_key" name="api_key" type="text" placeholder="x-api-key value" />
        </label>
        <button type="submit">Start export</button>
      </form>
      <div id="status" role="status" aria-live="polite"></div>
    </main>
    <script>
      const form = document.getElementById('export-form');
      const statusEl = document.getElementById('status');
      const submitButton = form.querySelector('button[type="submit"]');

      const { origin, pathname } = window.location;
      const basePath = pathname.replace(/\/[^/]*$/, '/') || '/';
      const buildUrl = (path) => {
        const cleaned = path.replace(/^\//, '');
        return new URL(`${basePath}${cleaned}`, origin);
      };

      const POLL_INTERVAL_MS = 5000;

      const setStatus = (message, type = 'pending') => {
        statusEl.textContent = message;
        statusEl.className = type;
      };

      const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

      const readError = async (response) => {
        try {
          const data = await response.json();
          if (data) {
            if (Array.isArray(data.detail)) {
              return data.detail.join(', ');
            }
            if (data.detail) {
              return data.detail;
            }
            if (typeof data.message === 'string') {
              return data.message;
            }
            if (typeof data.error === 'string') {
              return data.error;
            }
          }
        } catch (error) {
          // ignore JSON parsing failures and fall back to status text
        }
        return response.statusText || `HTTP ${response.status}`;
      };

      const deriveMonths = (startValue, endValue) => {
        if (!startValue || !endValue) {
          throw new Error('Please choose both start and end dates.');
        }
        const parseIsoDate = (value) => {
          const parts = value.split('-').map((part) => Number.parseInt(part, 10));
          if (parts.length < 3 || parts.some((part) => Number.isNaN(part))) {
            throw new Error('Dates must be provided in YYYY-MM-DD format.');
          }
          const [year, month, day] = parts;
          return new Date(Date.UTC(year, month - 1, day));
        };
        const startDate = parseIsoDate(startValue);
        const endDate = parseIsoDate(endValue);
        if (startDate > endDate) {
          throw new Error('The start date must be on or before the end date.');
        }
        const months = [];
        const current = new Date(Date.UTC(startDate.getUTCFullYear(), startDate.getUTCMonth(), 1));
        const last = new Date(Date.UTC(endDate.getUTCFullYear(), endDate.getUTCMonth(), 1));
        while (current <= last) {
          const year = current.getUTCFullYear();
          const month = String(current.getUTCMonth() + 1).padStart(2, '0');
          months.push(`${year}-${month}`);
          current.setUTCMonth(current.getUTCMonth() + 1);
        }
        return months;
      };

      const uploadAndFetchGeometry = async (file, aoiName, headers) => {
        const uploadUrl = buildUrl('api/fields/upload');
        const formData = new FormData();
        formData.append('file', file);
        if (aoiName) {
          formData.append('name', aoiName);
        }
        const uploadResponse = await fetch(uploadUrl, {
          method: 'POST',
          headers,
          body: formData,
        });
        if (!uploadResponse.ok) {
          throw new Error(await readError(uploadResponse));
        }
        const uploadData = await uploadResponse.json();
        const fieldId = uploadData.id || uploadData.field_id;
        if (!fieldId) {
          throw new Error('Upload succeeded but no field identifier was returned.');
        }

        const detailResponse = await fetch(buildUrl(`api/fields/${fieldId}`), {
          method: 'GET',
          headers,
        });
        if (!detailResponse.ok) {
          throw new Error(await readError(detailResponse));
        }
        const detail = await detailResponse.json();
        if (!detail.geometry) {
          throw new Error('Field geometry was not returned by the server.');
        }
        return detail.geometry;
      };

      const updateProgress = (job, exportTarget) => {
        const items = Array.isArray(job.items) ? job.items : [];
        const total = items.length;
        const completed = items.filter((item) => item.status === 'completed').length;
        const failed = items.filter((item) => item.status === 'failed').length;
        const active = items.filter((item) => !['completed', 'failed'].includes(item.status)).length;

        let message = 'Checking export status…';
        let type = 'pending';

        switch (job.state) {
          case 'pending':
            message = `Export job queued… ${completed} of ${total} ready${failed ? ` (${failed} failed)` : ''}.`;
            break;
          case 'running':
            message = `Building exports: ${completed} of ${total} ready${failed ? ` (${failed} failed)` : ''}${active ? `, ${active} in progress` : ''}.`;
            break;
          case 'partial':
            if (exportTarget === 'zip') {
              message = `Exports ready with warnings: ${completed} of ${total} available (${failed} failed). Successful files will download as a ZIP.`;
            } else {
              message = `Exports ready with warnings: ${completed} of ${total} available (${failed} failed). Destination links will follow.`;
            }
            break;
          case 'completed':
            message = `Exports ready: ${completed} of ${total} available.`;
            break;
          case 'failed':
            message = job.error ? `Export failed: ${job.error}` : 'Export failed.';
            type = 'error';
            break;
          default:
            message = `Job status: ${job.state || 'unknown'}.`;
        }

        setStatus(message, type);
      };

      const pollJob = async (jobId, exportTarget, headers) => {
        const statusUrl = buildUrl(`export/s2/${jobId}/status`);
        while (true) {
          const response = await fetch(statusUrl, { headers });
          if (!response.ok) {
            throw new Error(await readError(response));
          }
          const job = await response.json();
          updateProgress(job, exportTarget);

          if (job.state === 'failed') {
            throw new Error(job.error || 'Export job failed.');
          }
          if (job.state === 'completed' || job.state === 'partial') {
            return job;
          }

          await sleep(POLL_INTERVAL_MS);
        }
      };

      const downloadZip = async (jobId, headers) => {
        const response = await fetch(buildUrl(`export/s2/${jobId}/download`), {
          method: 'GET',
          headers,
        });
        if (!response.ok) {
          throw new Error(await readError(response));
        }
        const blob = await response.blob();
        const disposition = response.headers.get('content-disposition') || '';
        const match = disposition.match(/filename\*?=(?:UTF-8''|"?)([^";]+)/i);
        const filename = match ? decodeURIComponent(match[1]) : 'sentinel2_indices.zip';

        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        link.remove();
        window.URL.revokeObjectURL(url);
      };

      const fetchExportSummary = async (jobId, headers) => {
        const response = await fetch(buildUrl(`export/s2/${jobId}/download`), {
          method: 'GET',
          headers,
        });
        if (!response.ok) {
          throw new Error(await readError(response));
        }
        return response.json();
      };

      let isProcessing = false;

      form.addEventListener('submit', async (event) => {
        event.preventDefault();
        if (isProcessing) {
          return;
        }

        const fileInput = document.getElementById('file');
        const startInput = document.getElementById('start_date');
        const endInput = document.getElementById('end_date');
        const aoiNameInput = document.getElementById('aoi_name');
        const indicesSelect = document.getElementById('indices');
        const exportTargetSelect = document.getElementById('export_target');
        const apiKeyInput = document.getElementById('api_key');

        if (!fileInput.files.length) {
          setStatus('Please select a zipped shapefile bundle before submitting.', 'error');
          fileInput.focus();
          return;
        }

        if (!form.reportValidity()) {
          setStatus('Please complete the highlighted fields before submitting.', 'error');
          return;
        }

        const selectedIndices = Array.from(indicesSelect.selectedOptions).map((option) => option.value);
        if (!selectedIndices.length) {
          setStatus('Select at least one vegetation index to export.', 'error');
          indicesSelect.focus();
          return;
        }

        let months;
        try {
          months = deriveMonths(startInput.value, endInput.value);
        } catch (error) {
          setStatus(error instanceof Error ? error.message : String(error), 'error');
          return;
        }

        const apiKey = apiKeyInput.value.trim();
        const headers = apiKey ? { 'x-api-key': apiKey } : {};
        const exportTarget = exportTargetSelect.value;
        const aoiName = aoiNameInput.value.trim();
        const monthCount = months.length;
        const indexCount = selectedIndices.length;

        setStatus('Uploading shapefile to prepare AOI geometry…', 'pending');
        isProcessing = true;
        submitButton.disabled = true;

        try {
          const geometry = await uploadAndFetchGeometry(fileInput.files[0], aoiName, headers);

          setStatus('Queueing Sentinel-2 export job…', 'pending');

          const queueResponse = await fetch(buildUrl('export/s2/indices'), {
            method: 'POST',
            headers: { ...headers, 'Content-Type': 'application/json' },
            body: JSON.stringify({
              aoi_geojson: geometry,
              months,
              indices: selectedIndices,
              export_target: exportTarget,
              aoi_name: aoiName,
            }),
          });

          if (!queueResponse.ok) {
            throw new Error(await readError(queueResponse));
          }

          const data = await queueResponse.json();
          const jobId = data.job_id;
          if (!jobId) {
            throw new Error('Export job did not return an identifier.');
          }

          setStatus('Export job queued. Building composites…', 'pending');

          const finalStatus = await pollJob(jobId, exportTarget, headers);

          if (exportTarget === 'zip') {
            setStatus('Exports ready. Preparing ZIP download…', 'pending');
            await downloadZip(jobId, headers);
          } else {
            const summary = await fetchExportSummary(jobId, headers);
            if (summary && Array.isArray(summary.items)) {
              console.groupCollapsed('Sentinel-2 export destinations');
              summary.items.forEach((item) => {
                const destination = item.signed_url || item.destination_uri || 'pending';
                console.log(`${item.month} ${item.index}: ${destination}`);
              });
              console.groupEnd();
            } else {
              console.info('Export summary', summary);
            }
          }

          const items = Array.isArray(finalStatus.items) ? finalStatus.items : [];
          const successful = items.filter((item) => item.status === 'completed').length;
          const failed = items.filter((item) => item.status === 'failed').length;
          const total = items.length;
          const monthLabel = monthCount === 1 ? 'month' : 'months';
          const indexLabel = indexCount === 1 ? 'index' : 'indices';
          const destinationNote = exportTarget === 'zip' ? '' : ' Destinations have been logged to the browser console.';

          if (failed > 0) {
            const warningPrefix = exportTarget === 'zip' ? 'Download finished with warnings' : 'Exports finished with warnings';
            setStatus(
              `${warningPrefix}: ${successful} of ${total} exports succeeded (${failed} failed) across ${monthCount} ${monthLabel} × ${indexCount} ${indexLabel}.${destinationNote}`,
              'success'
            );
          } else {
            const completionPrefix = exportTarget === 'zip' ? 'Download complete' : 'Exports complete';
            const summary = exportTarget === 'zip'
              ? `${successful} Sentinel-2 exports built for ${monthCount} ${monthLabel} × ${indexCount} ${indexLabel}`
              : `${successful} Sentinel-2 exports available for ${monthCount} ${monthLabel} × ${indexCount} ${indexLabel}`;
            setStatus(
              `${completionPrefix}! ${summary}.${destinationNote}`,
              'success'
            );
          }

          form.reset();
        } catch (error) {
          console.error('Export workflow failed', error);
          const message = error instanceof Error ? error.message : String(error);
          setStatus(`Export failed: ${message}`, 'error');
        } finally {
          submitButton.disabled = false;
          isProcessing = false;
        }
      });
    </script>
  </body>
</html>
        """
    )


# Simple API key middleware (protects everything except health/docs)
@app.middleware("http")
async def require_api_key(req: Request, call_next):
    public_paths = {
        "/healthz",
        "/docs",
        "/redoc",
        "/openapi.json",           # needed for Swagger UI to load
        "/docs/oauth2-redirect",   # docs assets
        "/docs/swagger-ui",
        "/docs/swagger-ui-init.js",
        "/docs/swagger-ui-bundle.js",
        "/docs/swagger-ui.css",
    }
    if req.url.path in public_paths or req.url.path.startswith("/docs/swagger-ui"):
        return await call_next(req)
    if req.url.path in {"/", "/ui"}:
        return await call_next(req)
    if req.headers.get("x-api-key") != os.getenv("API_KEY"):
        return JSONResponse(status_code=401, content={"detail": "Invalid API key"})
    return await call_next(req)

# Mount API routes
app.include_router(api_router, prefix="/api")
app.include_router(fields_router, prefix="/api/fields")
app.include_router(fields_upload_router, prefix="/api/fields")
app.include_router(tiles_router, prefix="/api")
app.include_router(s2_indices_router)
app.include_router(export_shapefile_router)

