from __future__ import annotations
import os
import ee

def initialize_ee():
    """
    Initialize Google Earth Engine using a Service Account.
    Prefers GOOGLE_APPLICATION_CREDENTIALS (path to ee-key.json) and GCP_PROJECT.
    """
    key_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "/etc/secrets/ee-key.json")
    project = os.getenv("GCP_PROJECT", None)

    if not key_path or not os.path.exists(key_path):
        raise FileNotFoundError(f"Missing service-account key file at {key_path}. "
                                "Mount ee-key.json and set GOOGLE_APPLICATION_CREDENTIALS.")

    # When the key path is provided, the underlying email/account is read from the JSON.
    credentials = ee.ServiceAccountCredentials(None, key_path)
    ee.Initialize(credentials=credentials, project=project)
    return {"mode": "service_account", "project": project, "key_path": key_path}
