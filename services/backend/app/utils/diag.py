from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, List


class PipelineError(RuntimeError):
    def __init__(self, code: str, msg: str, hints: Optional[str] = None, ctx: Optional[Dict[str, Any]] = None):
        super().__init__(msg)
        self.code = code
        self.hints = hints
        self.ctx = ctx or {}


@dataclass
class StageMetric:
    name: str
    details: Dict[str, Any]


class Guard:
    def __init__(self):
        self._stages: Dict[str, StageMetric] = {}
        self._warnings: List[str] = []

    def record(self, stage: str, **kv):
        self._stages[stage] = StageMetric(stage, kv)

    def warn(self, msg: str):
        self._warnings.append(msg)

    def require(self, cond: bool, code: str, msg: str, hints: Optional[str] = None, **ctx):
        if not cond:
            raise PipelineError(code=code, msg=f"{code}: {msg}", hints=hints, ctx=ctx)

    def diagnostics_payload(self) -> Dict[str, Any]:
        return {
            "stages": {k: v.details for k, v in self._stages.items()},
            "warnings": list(self._warnings),
        }
