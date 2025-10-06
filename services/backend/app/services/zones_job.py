from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, Optional
from uuid import uuid4

_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = threading.Lock()


def start_job(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> str:
    job_id = uuid4().hex
    with _jobs_lock:
        _jobs[job_id] = {
            "status": "queued",
            "detail": None,
            "started": time.time(),
        }

    def run() -> None:
        try:
            _update(job_id, status="running")
            detail = fn(*args, **kwargs)
            _update(job_id, status="done", detail=detail)
        except Exception as exc:  # pragma: no cover - defensive
            _update(job_id, status="error", detail=str(exc))

    threading.Thread(target=run, daemon=True).start()
    return job_id


def job_status(job_id: str) -> Optional[Dict[str, Any]]:
    with _jobs_lock:
        return _jobs.get(job_id)


def _update(job_id: str, **updates: Any) -> None:
    with _jobs_lock:
        if job_id not in _jobs:
            return
        record = dict(_jobs[job_id])
        record.update(updates)
        _jobs[job_id] = record


__all__ = ["job_status", "start_job"]
