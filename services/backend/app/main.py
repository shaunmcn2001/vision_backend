from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from app.api.routes import router as api_router
from app.api.fields import router as fields_router
from app.api.fields_upload import router as fields_upload_router
from app.api.tiles import router as tiles_router
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
    <title>NDVI Export</title>
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
        max-width: 640px;
        margin: 4rem auto;
        padding: 2.5rem;
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
        gap: 1.25rem;
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
      button {
        font: inherit;
      }
      input[type="date"],
      input[type="text"],
      input[type="file"] {
        padding: 0.65rem 0.75rem;
        border: 1px solid #cbd5e1;
        border-radius: 8px;
      }
      input[type="file"] {
        padding: 0.5rem;
      }
      button {
        padding: 0.85rem 1.5rem;
        border: none;
        border-radius: 999px;
        background: #2563eb;
        color: #ffffff;
        font-weight: 600;
        cursor: pointer;
        transition: background 0.2s ease;
      }
      button:hover {
        background: #1d4ed8;
      }
      #status {
        min-height: 1.5rem;
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
      <h1>Download Monthly NDVI</h1>
      <p class="instructions">
        Upload a zipped shapefile bundle containing .shp, .dbf, and .shx files, set your
        date range, and submit to receive a ZIP file of monthly NDVI GeoTIFFs.
      </p>
      <form id="export-form" novalidate>
        <label for="file">
          Shapefile archive (.zip)
          <input id="file" name="file" type="file" accept=".zip" required />
        </label>
        <label for="start_date">
          Start date
          <input id="start_date" name="start_date" type="date" required />
        </label>
        <label for="end_date">
          End date
          <input id="end_date" name="end_date" type="date" required />
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
      const status = document.getElementById('status');

      form.addEventListener('submit', async (event) => {
        event.preventDefault();
        status.textContent = 'Uploading shapefile and starting export…';
        status.className = 'pending';

        const fileInput = document.getElementById('file');
        if (!fileInput.files.length) {
          status.textContent = 'Please select a zipped shapefile bundle before submitting.';
          status.className = 'error';
          fileInput.focus();
          return;
        }

        if (!form.reportValidity()) {
          status.textContent = 'Please complete the highlighted fields before submitting.';
          status.className = 'error';
          return;
        }

        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        formData.append('start_date', document.getElementById('start_date').value);
        formData.append('end_date', document.getElementById('end_date').value);

        const apiKey = document.getElementById('api_key').value.trim();
        const headers = apiKey ? { 'x-api-key': apiKey } : {};

        try {
          const response = await fetch('/api/export/export', {
            method: 'POST',
            headers,
            body: formData,
          });

          if (!response.ok) {
            let message = `Export failed with status ${response.status}`;
            try {
              const data = await response.json();
              if (data && data.detail) {
                message = Array.isArray(data.detail) ? data.detail.join(', ') : data.detail;
              }
            } catch (parseError) {
              // ignore JSON parsing issues and fall back to default message
            }
            status.textContent = message;
            status.className = 'error';
            return;
          }

          const blob = await response.blob();
          const disposition = response.headers.get('content-disposition') || '';
          const match = disposition.match(/filename\*?=(?:UTF-8''|"?)([^";]+)/i);
          const filename = match ? decodeURIComponent(match[1]) : 'ndvi_export.zip';

          const url = window.URL.createObjectURL(blob);
          const link = document.createElement('a');
          link.href = url;
          link.download = filename;
          document.body.appendChild(link);
          link.click();
          link.remove();
          window.URL.revokeObjectURL(url);

          form.reset();
          status.textContent = 'Export successful! Your download should begin shortly.';
          status.className = 'success';
        } catch (error) {
          status.textContent = `Unexpected error: ${error}`;
          status.className = 'error';
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
    public_paths = (
        "/healthz",
        "/docs",
        "/redoc",
        "/openapi.json",           # needed for Swagger UI to load
        "/docs/oauth2-redirect",   # docs assets
        "/docs/swagger-ui",
        "/docs/swagger-ui-init.js",
        "/docs/swagger-ui-bundle.js",
        "/docs/swagger-ui.css",
        "/",
        "/ui",
    )
    if any(req.url.path.startswith(p) for p in public_paths):
        return await call_next(req)
    if req.headers.get("x-api-key") != os.getenv("API_KEY"):
        raise HTTPException(status_code=401, detail="Invalid API key")
    return await call_next(req)

# Mount API routes
app.include_router(api_router, prefix="/api")
app.include_router(fields_router, prefix="/api/fields")
app.include_router(fields_upload_router, prefix="/api/fields")
app.include_router(tiles_router, prefix="/api")

