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
from app.api.zones import router as zones_router
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
      * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
      }

      body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        background-color: #f5f5f5;
        padding: 20px;
      }

      .container {
        max-width: 840px;
        margin: 0 auto;
        background: #ffffff;
        border-radius: 16px;
        box-shadow: 0 20px 50px rgba(15, 23, 42, 0.12);
        padding: 40px;
      }

      .header {
        text-align: center;
        margin-bottom: 32px;
      }

      .title {
        font-size: 32px;
        font-weight: 600;
        color: #1f2933;
        margin-bottom: 16px;
      }

      .description {
        color: #475569;
        line-height: 1.6;
        font-size: 15px;
        max-width: 580px;
        margin: 0 auto;
      }

      form {
        display: grid;
        gap: 24px;
      }

      .form-group {
        display: flex;
        flex-direction: column;
        gap: 12px;
      }

      .label {
        font-weight: 600;
        color: #1f2933;
        font-size: 14px;
      }

      .hidden {
        display: none !important;
      }

      .toggle-row {
        display: flex;
        align-items: center;
        gap: 12px;
      }

      .nested-panel {
        margin-top: 12px;
        padding: 16px;
        border-radius: 12px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
      }

      .nested-panel .grid {
        display: grid;
        gap: 16px;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      }

      .nested-panel .subfield {
        display: flex;
        flex-direction: column;
        gap: 8px;
      }

      .results-panel {
        margin-top: 24px;
        padding: 16px;
        border-radius: 12px;
        border: 1px solid #d8dee9;
        background: #f1f5f9;
      }

      .results-panel.hidden {
        display: none;
      }

      .results-panel h3 {
        font-size: 16px;
        font-weight: 600;
        color: #1f2933;
        margin-bottom: 12px;
      }

      .zone-artifacts {
        list-style: none;
        display: grid;
        gap: 10px;
        margin: 0;
        padding: 0;
      }

      .zone-artifacts li {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 14px;
        border-radius: 10px;
        background: #ffffff;
        border: 1px solid #e2e8f0;
      }

      .artifact-label {
        font-weight: 600;
        color: #1f2933;
        font-size: 14px;
      }

      .artifact-output {
        display: flex;
        gap: 12px;
        align-items: center;
        font-size: 13px;
        color: #475569;
      }

      .artifact-output a {
        color: #2563eb;
        font-weight: 600;
        text-decoration: none;
      }

      .artifact-output a:hover {
        text-decoration: underline;
      }

      .status-text {
        color: #475569;
      }

      .zones-info {
        margin-top: 14px;
        font-size: 13px;
        color: #475569;
      }

      .file-upload {
        border: 2px dashed #d0d7e3;
        border-radius: 12px;
        padding: 24px;
        display: flex;
        align-items: center;
        gap: 12px;
        background: #f8fafc;
        cursor: pointer;
        transition: all 0.2s ease;
      }

      .file-upload:focus,
      .file-upload:hover {
        border-color: #4285f4;
        background: #f4f8ff;
      }

      .file-upload-button {
        background: #4a5568;
        color: #ffffff;
        border: none;
        padding: 10px 18px;
        border-radius: 8px;
        font-size: 13px;
        cursor: pointer;
        transition: background 0.2s ease;
      }

      .file-upload-button:hover {
        background: #2d3748;
      }

      .file-status {
        color: #475569;
        font-size: 14px;
        flex: 1;
      }

      .file-help {
        color: #6b7280;
        font-size: 13px;
      }

      .input-field,
      .select-field {
        width: 100%;
        padding: 12px;
        border: 1px solid #d0d7e3;
        border-radius: 10px;
        font-size: 14px;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
      }

      .input-field:focus,
      .select-field:focus {
        outline: none;
        border-color: #4285f4;
        box-shadow: 0 0 0 3px rgba(66, 133, 244, 0.15);
      }

      .help-text {
        color: #64748b;
        font-size: 13px;
        line-height: 1.5;
      }

      .checkbox-controls {
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
      }

      .control-button {
        background: #ffffff;
        border: 1px solid #4285f4;
        color: #4285f4;
        padding: 8px 18px;
        border-radius: 999px;
        font-size: 12px;
        cursor: pointer;
        transition: all 0.2s ease;
      }

      .control-button:hover {
        background: #4285f4;
        color: #ffffff;
      }

      .checkbox-group {
        border: 1px solid #d0d7e3;
        border-radius: 10px;
        max-height: 220px;
        overflow-y: auto;
        background: #ffffff;
      }

      .checkbox-item {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 10px 14px;
        border-bottom: 1px solid #f1f5f9;
      }

      .checkbox-item:last-child {
        border-bottom: none;
      }

      .checkbox-item input {
        width: 18px;
        height: 18px;
      }

      .checkbox-item label {
        color: #1f2933;
        font-size: 14px;
        cursor: pointer;
      }

      .checkbox-help {
        color: #6b7280;
        font-size: 12px;
      }

      .export-button {
        width: 100%;
        background: #4285f4;
        color: #ffffff;
        border: none;
        padding: 16px;
        border-radius: 12px;
        font-size: 16px;
        font-weight: 600;
        cursor: pointer;
        transition: background 0.2s ease, transform 0.2s ease;
      }

      .export-button:hover {
        background: #3367d6;
        transform: translateY(-1px);
      }

      .export-button:disabled {
        background: #94a3b8;
        cursor: not-allowed;
        transform: none;
      }

      .status-message {
        text-align: center;
        font-size: 14px;
        margin-top: 12px;
        font-weight: 500;
        display: none;
      }

      .status-message.pending {
        color: #4285f4;
      }

      .status-message.success {
        color: #0f9d58;
      }

      .status-message.error {
        color: #d93025;
      }

      @media (max-width: 640px) {
        body {
          padding: 16px;
        }

        .container {
          padding: 28px 20px;
        }

        .title {
          font-size: 26px;
        }
        .checkbox-controls {
          gap: 8px;
        }
      }
    </style>
   </head>
   <body>
     <div class="container">
       <div class="header">
         <h1 class="title">Sentinel-2 Index Export</h1>
         <p class="description">
           Upload a zipped shapefile bundle (containing .shp, .dbf, and .shx files), choose the
           date range to cover, pick the vegetation indices you need, and decide where the exports
           should be delivered. One export is created per month for every index you select.
         </p>
       </div>

       <form id="exportForm" novalidate>
         <div class="form-group">
           <label class="label" for="fileInput">Shapefile archive (.zip)</label>
           <div class="file-upload" id="fileUpload" role="button" tabindex="0">
             <input type="file" id="fileInput" name="file" accept=".zip" style="display: none" required />
             <button type="button" class="file-upload-button">Choose file</button>
             <span class="file-status">No file chosen</span>
           </div>
           <div class="file-help">Include the .shp, .dbf, and .shx members in a single ZIP file.</div>
         </div>

         <div class="form-group">
           <label class="label" for="startDate">Start date</label>
           <input type="date" class="input-field" id="startDate" name="start_date" required />
         </div>

         <div class="form-group">
           <label class="label" for="endDate">End date</label>
           <input type="date" class="input-field" id="endDate" name="end_date" required />
           <div class="help-text">Every month that overlaps the range will be exported.</div>
         </div>

         <div class="form-group">
           <label class="label" for="aoiName">Paddock name</label>
           <input type="text" class="input-field" id="aoiName" name="aoi_name" placeholder="Used in file names" required />
         </div>

         <div class="form-group">
           <label class="label" for="indicesGroup">Vegetation indices to export</label>
           <div class="checkbox-controls">
             <button type="button" class="control-button" id="selectAllIndices">Select all</button>
             <button type="button" class="control-button" id="clearAllIndices">Clear all</button>
           </div>
           <div class="checkbox-group" id="indicesGroup">
             <div class="checkbox-item">
               <input type="checkbox" id="index-ndvi" name="indices" value="NDVI" checked />
               <label for="index-ndvi">NDVI</label>
             </div>
             <div class="checkbox-item">
               <input type="checkbox" id="index-evi" name="indices" value="EVI" />
               <label for="index-evi">EVI</label>
             </div>
             <div class="checkbox-item">
               <input type="checkbox" id="index-gndvi" name="indices" value="GNDVI" />
               <label for="index-gndvi">GNDVI</label>
             </div>
             <div class="checkbox-item">
               <input type="checkbox" id="index-ndre" name="indices" value="NDRE" />
               <label for="index-ndre">NDRE</label>
             </div>
             <div class="checkbox-item">
               <input type="checkbox" id="index-savi" name="indices" value="SAVI" />
               <label for="index-savi">SAVI</label>
             </div>
             <div class="checkbox-item">
               <input type="checkbox" id="index-msavi" name="indices" value="MSAVI" />
               <label for="index-msavi">MSAVI</label>
             </div>
             <div class="checkbox-item">
               <input type="checkbox" id="index-vari" name="indices" value="VARI" />
               <label for="index-vari">VARI</label>
             </div>
             <div class="checkbox-item">
               <input type="checkbox" id="index-mcari" name="indices" value="MCARI" />
               <label for="index-mcari">MCARI</label>
             </div>
             <div class="checkbox-item">
               <input type="checkbox" id="index-ndwi-mcfeeters" name="indices" value="NDWI_McFeeters" />
               <label for="index-ndwi-mcfeeters">NDWI (McFeeters)</label>
             </div>
             <div class="checkbox-item">
               <input type="checkbox" id="index-ndwi-gao" name="indices" value="NDWI_Gao" />
               <label for="index-ndwi-gao">NDWI (Gao)</label>
             </div>
             <div class="checkbox-item">
               <input type="checkbox" id="index-ndmi" name="indices" value="NDMI" />
               <label for="index-ndmi">NDMI</label>
             </div>
             <div class="checkbox-item">
               <input type="checkbox" id="index-msi" name="indices" value="MSI" />
               <label for="index-msi">MSI</label>
             </div>
             <div class="checkbox-item">
               <input type="checkbox" id="index-gvmi" name="indices" value="GVMI" />
               <label for="index-gvmi">GVMI</label>
             </div>
             <div class="checkbox-item">
               <input type="checkbox" id="index-nbr" name="indices" value="NBR" />
               <label for="index-nbr">NBR</label>
             </div>
             <div class="checkbox-item">
               <input type="checkbox" id="index-psri" name="indices" value="PSRI" />
               <label for="index-psri">PSRI</label>
             </div>
             <div class="checkbox-item">
               <input type="checkbox" id="index-ari" name="indices" value="ARI" />
               <label for="index-ari">ARI</label>
             </div>
             <div class="checkbox-item">
               <input type="checkbox" id="index-cri" name="indices" value="CRI" />
               <label for="index-cri">CRI</label>
             </div>
             <div class="checkbox-item">
               <input type="checkbox" id="index-bsi" name="indices" value="BSI" />
               <label for="index-bsi">BSI</label>
             </div>
             <div class="checkbox-item">
               <input type="checkbox" id="index-sbi" name="indices" value="SBI" />
               <label for="index-sbi">SBI</label>
             </div>
             <div class="checkbox-item">
               <input type="checkbox" id="index-ndsi-soil" name="indices" value="NDSI_Soil" />
               <label for="index-ndsi-soil">NDSI (Soil)</label>
             </div>
             <div class="checkbox-item">
               <input type="checkbox" id="index-ndti" name="indices" value="NDTI" />
               <label for="index-ndti">NDTI</label>
             </div>
             <div class="checkbox-item">
               <input type="checkbox" id="index-pri" name="indices" value="PRI" />
               <label for="index-pri">PRI</label>
             </div>
           </div>
           <div class="checkbox-help">
             Use Select all / Clear all or hold Ctrl (Windows/Linux) or ⌘ (macOS) to toggle multiple entries quickly.
           </div>
         </div>

         <div class="form-group">
           <label class="label" for="exportDestination">Export destination</label>
           <select class="select-field" id="exportDestination" name="export_target">
             <option value="zip" selected>Download ZIP</option>
             <option value="gcs">Google Cloud Storage</option>
             <option value="drive">Google Drive</option>
           </select>
         </div>

         <div class="form-group">
           <label class="label" for="zonesToggle">Production zones</label>
           <div class="toggle-row">
             <input type="checkbox" id="zonesToggle" />
             <label for="zonesToggle">Enable productivity zones for the selected period</label>
           </div>
           <div class="help-text">Compute stable NDVI-based production zones and export a raster, shapefile, and optional zonal statistics table.</div>
           <div id="zonesOptions" class="nested-panel hidden" aria-hidden="true">
             <div class="grid">
               <div class="subfield">
                 <label class="label" for="zonesClasses">Number of classes</label>
                 <select class="select-field" id="zonesClasses">
                   <option value="3" selected>3 classes</option>
                   <option value="5">5 classes</option>
                 </select>
               </div>
               <div class="subfield">
                 <label class="label" for="zonesCv">Stability (CV) threshold</label>
                 <input type="number" class="input-field" id="zonesCv" step="0.01" min="0" value="0.25" />
               </div>
               <div class="subfield">
                 <label class="label" for="zonesMmu">Minimum mapping unit (ha)</label>
                 <input type="number" class="input-field" id="zonesMmu" step="0.1" min="0" value="0.5" />
               </div>
               <div class="subfield">
                 <label class="label" for="zonesSmooth">Smooth kernel (px)</label>
                 <input type="number" class="input-field" id="zonesSmooth" step="1" min="0" value="1" />
               </div>
               <div class="subfield">
                 <label class="label" for="zonesSimplify">Simplify tolerance (m)</label>
                 <input type="number" class="input-field" id="zonesSimplify" step="1" min="0" value="5" />
               </div>
             </div>
           </div>
         </div>

         <div class="form-group">
           <label class="label" for="apiKey">API key (optional)</label>
           <input type="text" class="input-field" id="apiKey" name="api_key" placeholder="x-api-key value" />
           <div class="help-text">Only required when API key enforcement is enabled for the backend.</div>
         </div>

         <button type="submit" class="export-button">Start export</button>
         <div class="status-message" id="statusMessage" role="status" aria-live="polite"></div>
         <div id="zonesPanel" class="results-panel hidden" aria-live="polite">
           <h3>Production zones</h3>
           <ul class="zone-artifacts">
             <li>
               <span class="artifact-label">Raster</span>
               <span class="artifact-output" data-zone-artifact="raster">
                 <span class="status-text">Not requested</span>
               </span>
             </li>
             <li>
               <span class="artifact-label">Shapefile</span>
               <span class="artifact-output" data-zone-artifact="vectors">
                 <span class="status-text">Not requested</span>
               </span>
             </li>
             <li>
               <span class="artifact-label">Zonal stats CSV</span>
               <span class="artifact-output" data-zone-artifact="zonal_stats">
                 <span class="status-text">Not requested</span>
               </span>
             </li>
           </ul>
           <div class="zones-info" id="zonesInfo"></div>
         </div>
       </form>
     </div>

     <script>
       const form = document.getElementById('exportForm');
       const fileUpload = document.getElementById('fileUpload');
       const fileInput = document.getElementById('fileInput');
       const fileStatus = document.querySelector('.file-status');
       const exportButton = document.querySelector('.export-button');
       const statusMessage = document.getElementById('statusMessage');
       const selectAllButton = document.getElementById('selectAllIndices');
       const clearAllButton = document.getElementById('clearAllIndices');
       const startDateInput = document.getElementById('startDate');
       const endDateInput = document.getElementById('endDate');
       const aoiNameInput = document.getElementById('aoiName');
       const exportDestinationSelect = document.getElementById('exportDestination');
       const apiKeyInput = document.getElementById('apiKey');
       const zonesToggle = document.getElementById('zonesToggle');
       const zonesOptions = document.getElementById('zonesOptions');
       const zonesClasses = document.getElementById('zonesClasses');
       const zonesCvInput = document.getElementById('zonesCv');
       const zonesMmuInput = document.getElementById('zonesMmu');
       const zonesSmoothInput = document.getElementById('zonesSmooth');
       const zonesSimplifyInput = document.getElementById('zonesSimplify');
       const zonesPanel = document.getElementById('zonesPanel');
       const zonesInfo = document.getElementById('zonesInfo');
       const zoneArtifactElements = {
         raster: document.querySelector('[data-zone-artifact="raster"]'),
         vectors: document.querySelector('[data-zone-artifact="vectors"]'),
         zonal_stats: document.querySelector('[data-zone-artifact="zonal_stats"]'),
       };

       const { origin, pathname } = window.location;
       const lastSlashIndex = pathname.lastIndexOf('/');
       const basePath = lastSlashIndex >= 0 ? pathname.slice(0, lastSlashIndex + 1) || '/' : '/';

       const buildUrl = (path) => {
         const cleaned = path.replace(/^\/+/, '');
         return new URL(`${basePath}${cleaned}`, origin);
       };

       const POLL_INTERVAL_MS = 5000;
       let zonesEnabledGlobal = false;
       let zoneEndpointData = null;
       let zoneJobData = null;
       let currentExportTarget = 'zip';

       const getIndicesCheckboxes = () => Array.from(document.querySelectorAll('input[name="indices"]'));

       const toggleZoneOptions = () => {
         const enabled = zonesToggle.checked;
         if (enabled) {
           zonesOptions.classList.remove('hidden');
           zonesOptions.setAttribute('aria-hidden', 'false');
         } else {
           zonesOptions.classList.add('hidden');
           zonesOptions.setAttribute('aria-hidden', 'true');
         }
       };

       const resetZonePanel = () => {
         zonesPanel.classList.add('hidden');
         zonesInfo.textContent = '';
         Object.values(zoneArtifactElements).forEach((el) => {
           if (!el) {
             return;
           }
           el.innerHTML = '<span class="status-text">Not requested</span>';
         });
       };

       const setZoneArtifactStatus = (kind, info) => {
         const container = zoneArtifactElements[kind];
         if (!container) {
           return;
         }
         container.innerHTML = '';
         const textSpan = document.createElement('span');
         textSpan.className = 'status-text';
         textSpan.textContent = info.text;
         container.appendChild(textSpan);
         if (info.url) {
           const link = document.createElement('a');
           link.href = info.url;
           link.target = '_blank';
           link.rel = 'noopener';
           link.textContent = info.filename || 'Download';
           container.appendChild(link);
         }
       };

       const renderZonePanel = () => {
         if (!zonesEnabledGlobal) {
           resetZonePanel();
           return;
         }

         zonesPanel.classList.remove('hidden');

         const zoneData = zoneJobData || {};
         const endpointData = zoneEndpointData || {};
         const zonePaths = (zoneData.paths && zoneData.paths) || {};
         const endpointPaths = (endpointData.paths && endpointData.paths) || {};
         const metadata = zoneData.metadata || endpointData.metadata || {};

         const usedMonths = Array.isArray(metadata.used_months) ? metadata.used_months : [];
         const skippedMonths = Array.isArray(metadata.skipped_months) ? metadata.skipped_months : [];
         const mmuApplied = metadata.mmu_applied;

         const infoParts = [];
         if (usedMonths.length) {
           infoParts.push(`Period: ${usedMonths[0]} – ${usedMonths[usedMonths.length - 1]}`);
         }
         if (skippedMonths.length) {
           infoParts.push(`Skipped: ${skippedMonths.join(', ')}`);
         }
         if (typeof mmuApplied === 'boolean') {
           infoParts.push(mmuApplied ? 'MMU applied' : 'MMU skipped (AOI smaller than threshold)');
         }
         const overallStatus = zoneData.status || endpointData.status;
         const overallError = zoneData.error || endpointData.error;
         if (overallStatus === 'failed' || overallError) {
           const errorText = overallError ? String(overallError) : 'Unknown error';
           infoParts.push(`Error: ${errorText}`);
         }
         zonesInfo.textContent = infoParts.join(' · ');

         const includeStats = !(
           (zonePaths.zonal_stats === null || typeof zonePaths.zonal_stats === 'undefined') &&
           (endpointPaths.zonal_stats === null || typeof endpointPaths.zonal_stats === 'undefined')
         );

         const computeStatus = (kind) => {
           if (kind === 'zonal_stats' && !includeStats) {
             return { text: 'Not requested' };
           }

           const jobTask = (zoneData.tasks && zoneData.tasks[kind]) || {};
           const endpointTask = (endpointData.tasks && endpointData.tasks[kind]) || {};
           const overallStatus = zoneData.status || endpointData.status;
           const overallError = zoneData.error || endpointData.error;
           if (overallStatus === 'failed' || overallError) {
             return { text: overallError ? `Failed: ${overallError}` : 'Failed' };
           }
           const error = jobTask.error || endpointTask.error;
           if (error) {
             return { text: `Failed: ${error}` };
           }

           const downloadUrl =
             jobTask.signed_url ||
             jobTask.destination_uri ||
             endpointTask.signed_url ||
             endpointTask.destination_uri ||
             null;
           const filenameSource =
             (zonePaths && zonePaths[kind]) ||
             (endpointPaths && endpointPaths[kind]) ||
             null;
           const filename = filenameSource ? String(filenameSource).split('/').pop() : undefined;

           if (downloadUrl) {
             return { text: 'Ready', url: downloadUrl, filename };
           }

           if (currentExportTarget === 'zip') {
             if (zoneData.status === 'completed') {
               return {
                 text: filename ? `Included (${filename})` : 'Included in ZIP',
               };
             }
             const zoneStatus = zoneData.status || endpointTask.state || 'Preparing…';
             if (zoneStatus === 'failed' && overallError) {
               return { text: `Failed: ${overallError}` };
             }
             return { text: `Status: ${zoneStatus}` };
           }

           const taskState = jobTask.state || endpointTask.state;
           return { text: taskState ? `Status: ${taskState}` : 'Pending…' };
         };

         ['raster', 'vectors', 'zonal_stats'].forEach((kind) => {
           setZoneArtifactStatus(kind, computeStatus(kind));
         });
       };

       const primeZonePanel = () => {
         zonesEnabledGlobal = true;
         zoneEndpointData = null;
         zoneJobData = null;
         zonesPanel.classList.remove('hidden');
         zonesInfo.textContent = 'Preparing production zones…';
         ['raster', 'vectors', 'zonal_stats'].forEach((kind) => {
           setZoneArtifactStatus(kind, { text: 'Queued…' });
         });
       };

       const updateZonePanelFromEndpoint = (data) => {
         zoneEndpointData = data || null;
         renderZonePanel();
       };

       const updateZonePanelFromJob = (zone) => {
         zoneJobData = zone || null;
         renderZonePanel();
       };

       const updateFileStatus = (text) => {
         fileStatus.textContent = text || 'No file chosen';
       };

       const setStatus = (message, type = 'pending') => {
         if (!message) {
           statusMessage.textContent = '';
           statusMessage.className = 'status-message';
           statusMessage.style.display = 'none';
           return;
         }
         statusMessage.textContent = message;
         statusMessage.className = `status-message ${type}`;
         statusMessage.style.display = 'block';
       };

       const setProcessingState = (processing) => {
         exportButton.disabled = processing;
         exportButton.textContent = processing ? 'Processing…' : 'Start export';
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
           // Ignore JSON parsing failures and fall back to status text.
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
         const uploadUrl = buildUrl('export/s2/indices/aoi');
         const formData = new FormData();
         formData.append('file', file);
         if (aoiName) {
           formData.append('aoi_name', aoiName);
         }
         const uploadResponse = await fetch(uploadUrl, {
           method: 'POST',
           headers,
           body: formData,
         });
         if (!uploadResponse.ok) {
           throw new Error(await readError(uploadResponse));
         }
         const payload = await uploadResponse.json();
         if (!payload || !payload.geometry) {
           throw new Error('AOI geometry was not returned by the server.');
         }
         return payload;
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
        if (job.zone_exports) {
          updateZonePanelFromJob(job.zone_exports);
        }
       };

       const pollJob = async (jobId, exportTarget, headers) => {
         const statusUrl = buildUrl(`export/s2/indices/${jobId}/status`);
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
         const response = await fetch(buildUrl(`export/s2/indices/${jobId}/download`), {
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
         const response = await fetch(buildUrl(`export/s2/indices/${jobId}/download`), {
           method: 'GET',
           headers,
         });
         if (!response.ok) {
           throw new Error(await readError(response));
         }
         return response.json();
       };

       const queueZonesProduction = async (
         geometry,
         aoiName,
         months,
         zoneParams,
         exportTarget,
         headers
       ) => {
         const response = await fetch(buildUrl('api/zones/production'), {
           method: 'POST',
           headers: { ...headers, 'Content-Type': 'application/json' },
           body: JSON.stringify({
             aoi_geojson: geometry,
             aoi_name: aoiName,
             months,
             cloud_prob_max: 40,
             n_classes: zoneParams.n_classes,
             cv_mask_threshold: zoneParams.cv_mask_threshold,
             mmu_ha: zoneParams.mmu_ha,
             smooth_kernel_px: zoneParams.smooth_kernel_px,
             simplify_tol_m: zoneParams.simplify_tol_m,
             export_target: exportTarget,
             include_zonal_stats: zoneParams.include_zonal_stats,
           }),
         });
         if (!response.ok) {
           throw new Error(await readError(response));
         }
         const data = await response.json();
         updateZonePanelFromEndpoint(data);
         return data;
       };

       let isProcessing = false;

       const setIndicesSelection = (checked) => {
         getIndicesCheckboxes().forEach((checkbox) => {
           checkbox.checked = checked;
         });
         const firstCheckbox = getIndicesCheckboxes()[0];
         if (firstCheckbox) {
           firstCheckbox.focus();
         }
       };

       if (selectAllButton) {
         selectAllButton.addEventListener('click', () => setIndicesSelection(true));
       }

       if (clearAllButton) {
         clearAllButton.addEventListener('click', () => setIndicesSelection(false));
       }

       const setDefaultDates = () => {
         const today = new Date();
         const threeMonthsAgo = new Date(today.getFullYear(), today.getMonth() - 3, today.getDate());
         endDateInput.value = today.toISOString().split('T')[0];
         startDateInput.value = threeMonthsAgo.toISOString().split('T')[0];
       };

       setDefaultDates();
       toggleZoneOptions();

       if (zonesToggle) {
         zonesToggle.addEventListener('change', () => {
           toggleZoneOptions();
           if (!zonesToggle.checked) {
             zonesEnabledGlobal = false;
             resetZonePanel();
           }
         });
       }

       fileUpload.addEventListener('click', () => {
         fileInput.click();
       });

       fileUpload.addEventListener('keydown', (event) => {
         if (event.key === 'Enter' || event.key === ' ') {
           event.preventDefault();
           fileInput.click();
         }
       });

       fileInput.addEventListener('change', (event) => {
         const file = event.target.files[0];
         updateFileStatus(file ? file.name : 'No file chosen');
       });

       form.addEventListener('submit', async (event) => {
         event.preventDefault();
         if (isProcessing) {
           return;
         }

         if (!fileInput.files.length) {
           setStatus('Please select a zipped shapefile bundle before submitting.', 'error');
           fileUpload.focus();
           return;
         }

         if (!form.reportValidity()) {
           setStatus('Please complete the highlighted fields before submitting.', 'error');
           return;
         }

         const selectedIndices = getIndicesCheckboxes()
           .filter((checkbox) => checkbox.checked)
           .map((checkbox) => checkbox.value);

         if (!selectedIndices.length) {
           setStatus('Select at least one vegetation index to export.', 'error');
           const firstCheckbox = getIndicesCheckboxes()[0];
           if (firstCheckbox) {
             firstCheckbox.focus();
           }
           return;
         }

         let months;
         try {
           months = deriveMonths(startDateInput.value, endDateInput.value);
         } catch (error) {
           setStatus(error instanceof Error ? error.message : String(error), 'error');
           return;
         }

        const apiKey = apiKeyInput.value.trim();
        const headers = apiKey ? { 'x-api-key': apiKey } : {};
        const exportTarget = exportDestinationSelect.value;
        const zonesEnabled = zonesToggle.checked;
        let zoneParams = null;
        if (zonesEnabled) {
          const nClasses = parseInt(zonesClasses.value, 10);
          const cvThreshold = parseFloat(zonesCvInput.value);
          const mmu = parseFloat(zonesMmuInput.value);
          const smooth = parseInt(zonesSmoothInput.value, 10);
          const simplify = parseFloat(zonesSimplifyInput.value);
          if (!Number.isInteger(nClasses) || ![3, 5].includes(nClasses)) {
            setStatus('Choose either 3 or 5 production classes.', 'error');
            return;
          }
          if ([cvThreshold, mmu, smooth, simplify].some((value) => Number.isNaN(value) || value < 0)) {
            setStatus('Production zone parameters must be non-negative numbers.', 'error');
            return;
          }
          zoneParams = {
            n_classes: nClasses,
            cv_mask_threshold: cvThreshold,
            mmu_ha: mmu,
            smooth_kernel_px: smooth,
            simplify_tol_m: simplify,
            include_zonal_stats: true,
          };
        }
        const aoiName = aoiNameInput.value.trim();
        const monthCount = months.length;
        const indexCount = selectedIndices.length;

        currentExportTarget = exportTarget;
        if (zonesEnabled) {
          primeZonePanel();
        } else {
          zonesEnabledGlobal = false;
          resetZonePanel();
        }

        setStatus('Uploading shapefile to prepare AOI geometry…', 'pending');
        isProcessing = true;
        setProcessingState(true);

         try {
           const aoiPayload = await uploadAndFetchGeometry(fileInput.files[0], aoiName, headers);
          const { geometry, aoi_name: sanitizedAoiName } = aoiPayload;
          const jobAoiName = sanitizedAoiName || aoiName;
          if (sanitizedAoiName && sanitizedAoiName !== aoiName) {
            console.info(`AOI name sanitised to "${sanitizedAoiName}" for export filenames.`);
          }

          setStatus('Queueing Sentinel-2 export job…', 'pending');

          let zonePromise = null;
          if (zonesEnabled && zoneParams) {
            zonePromise = queueZonesProduction(
              geometry,
              jobAoiName,
              months,
              zoneParams,
              exportTarget,
              headers
            ).catch((error) => {
              const message = error instanceof Error ? error.message : String(error);
              zonesPanel.classList.remove('hidden');
              ['raster', 'vectors', 'zonal_stats'].forEach((kind) => {
                setZoneArtifactStatus(kind, { text: `Failed: ${message}` });
              });
              zonesInfo.textContent = 'Production zones request failed.';
              return null;
            });
          }

          const queueResponse = await fetch(buildUrl('export/s2/indices'), {
            method: 'POST',
            headers: { ...headers, 'Content-Type': 'application/json' },
            body: JSON.stringify({
              aoi_geojson: geometry,
              months,
              indices: selectedIndices,
              export_target: exportTarget,
              aoi_name: jobAoiName,
              production_zones: zonesEnabled
                ? {
                    enabled: true,
                    n_classes: zoneParams.n_classes,
                    cv_mask_threshold: zoneParams.cv_mask_threshold,
                    mmu_ha: zoneParams.mmu_ha,
                    smooth_kernel_px: zoneParams.smooth_kernel_px,
                    simplify_tol_m: zoneParams.simplify_tol_m,
                  }
                : { enabled: false },
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

          if (zonePromise) {
            await zonePromise;
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
          setDefaultDates();
          updateFileStatus('No file chosen');
          getIndicesCheckboxes().forEach((checkbox) => {
            checkbox.checked = false;
          });
          const defaultCheckbox = document.getElementById('index-ndvi');
          if (defaultCheckbox) {
            defaultCheckbox.checked = true;
          }
          zonesToggle.checked = false;
          toggleZoneOptions();
          zonesEnabledGlobal = false;
          resetZonePanel();
        } catch (error) {
           console.error('Export workflow failed', error);
           const message = error instanceof Error ? error.message : String(error);
           setStatus(`Export failed: ${message}`, 'error');
         } finally {
           isProcessing = false;
           setProcessingState(false);
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
app.include_router(zones_router, prefix="/api")
app.include_router(export_shapefile_router)

