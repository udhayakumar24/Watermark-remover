"""
jobs.py — tiny in-process job manager.

Long operations (tracking, rendering) run on worker threads and report progress into
a shared dict that the browser polls. One job at a time per stage keeps memory use
predictable on small machines.
"""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class Job:
    id: str
    kind: str
    state: str = "queued"          # queued | running | done | error | cancelled
    progress: float = 0.0
    current: int = 0
    total: int = 0
    stage: str = ""
    result: Any = None
    error: Optional[str] = None
    started: float = field(default_factory=time.time)
    finished: Optional[float] = None
    _cancel: bool = field(default=False, repr=False)

    def cancel(self) -> None:
        self._cancel = True

    @property
    def cancelled(self) -> bool:
        return self._cancel

    def public(self) -> dict:
        elapsed = (self.finished or time.time()) - self.started
        rate = self.current / elapsed if elapsed > 0.5 and self.current else 0.0
        eta = None
        if rate > 0 and self.total > self.current:
            eta = (self.total - self.current) / rate
        return {
            "id": self.id,
            "kind": self.kind,
            "state": "cancelled" if self._cancel and self.state != "done" else self.state,
            "progress": round(self.progress, 4),
            "current": self.current,
            "total": self.total,
            "stage": self.stage,
            "result": self.result,
            "error": self.error,
            "elapsed": round(elapsed, 1),
            "fps": round(rate, 2),
            "eta": round(eta, 1) if eta is not None else None,
        }


class JobManager:
    def __init__(self, max_workers: int = 1):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._sem = threading.Semaphore(max_workers)

    def submit(self, kind: str, fn: Callable[[Job], Any]) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], kind=kind)
        with self._lock:
            self._jobs[job.id] = job

        def runner() -> None:
            with self._sem:
                job.state = "running"
                try:
                    job.result = fn(job)
                    if job._cancel:
                        job.state = "cancelled"
                    else:
                        job.state = "done"
                        job.progress = 1.0
                except Exception as exc:
                    job.state = "error"
                    job.error = f"{type(exc).__name__}: {exc}"
                    traceback.print_exc()
                finally:
                    job.finished = time.time()

        threading.Thread(target=runner, daemon=True, name=f"job-{job.id}").start()
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def latest(self, kind: str) -> Optional[Job]:
        with self._lock:
            matches = [j for j in self._jobs.values() if j.kind == kind]
        return max(matches, key=lambda j: j.started) if matches else None
