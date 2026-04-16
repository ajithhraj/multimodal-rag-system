from __future__ import annotations

from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
import threading
from typing import TYPE_CHECKING
from uuid import uuid4

from multimodal_rag.api.schemas import IngestJobState

if TYPE_CHECKING:
    from multimodal_rag.engine import MultimodalRAG


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class IngestJobRecord:
    job_id: str
    status: IngestJobState
    tenant_id: str
    collection: str | None
    paths: list[str]
    submitted_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    result: dict[str, int] | None = None
    error: str | None = None

    def copy(self) -> "IngestJobRecord":
        return replace(self, paths=list(self.paths), result=dict(self.result) if self.result else None)


class IngestJobManager:
    def __init__(
        self,
        max_workers: int = 1,
        ttl_seconds: int = 86400,
        max_retained: int = 200,
    ):
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="mmrag-ingest")
        self._ttl = timedelta(seconds=ttl_seconds)
        self._max_retained = max_retained
        self._jobs: dict[str, IngestJobRecord] = {}
        self._futures: dict[str, Future[None]] = {}
        self._lock = threading.Lock()

    def _cleanup_locked(self, now: datetime) -> None:
        expired_ids: list[str] = []
        for job_id, job in self._jobs.items():
            if job.finished_at is None:
                continue
            if now - job.finished_at > self._ttl:
                expired_ids.append(job_id)
        for job_id in expired_ids:
            self._jobs.pop(job_id, None)

        if len(self._jobs) <= self._max_retained:
            return

        removable = sorted(
            (
                (job.finished_at, job.submitted_at, job_id)
                for job_id, job in self._jobs.items()
                if job.finished_at is not None
            ),
            key=lambda item: (item[0], item[1]),
        )
        while len(self._jobs) > self._max_retained and removable:
            _, _, job_id = removable.pop(0)
            self._jobs.pop(job_id, None)

    def submit(
        self,
        engine: "MultimodalRAG",
        paths: list[Path],
        collection: str | None,
        tenant_id: str,
    ) -> IngestJobRecord:
        submitted_at = _utc_now()
        record = IngestJobRecord(
            job_id=uuid4().hex,
            status="queued",
            tenant_id=tenant_id,
            collection=collection,
            paths=[str(path) for path in paths],
            submitted_at=submitted_at,
        )
        with self._lock:
            self._cleanup_locked(submitted_at)
            self._jobs[record.job_id] = record

        future = self._executor.submit(self._run_job, record.job_id, engine, paths, collection, tenant_id)
        with self._lock:
            self._futures[record.job_id] = future
        return record.copy()

    def _run_job(
        self,
        job_id: str,
        engine: "MultimodalRAG",
        paths: list[Path],
        collection: str | None,
        tenant_id: str,
    ) -> None:
        started_at = _utc_now()
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return
            record.status = "running"
            record.started_at = started_at

        try:
            result = engine.ingest_paths(paths, collection=collection, tenant_id=tenant_id)
            finished_at = _utc_now()
            with self._lock:
                record = self._jobs.get(job_id)
                if record is None:
                    return
                record.status = "completed"
                record.finished_at = finished_at
                record.result = dict(result)
                record.error = None
                self._cleanup_locked(finished_at)
        except Exception as exc:
            finished_at = _utc_now()
            with self._lock:
                record = self._jobs.get(job_id)
                if record is None:
                    return
                record.status = "failed"
                record.finished_at = finished_at
                record.result = None
                record.error = str(exc)
                self._cleanup_locked(finished_at)
        finally:
            with self._lock:
                self._futures.pop(job_id, None)

    def get(self, job_id: str) -> IngestJobRecord | None:
        with self._lock:
            self._cleanup_locked(_utc_now())
            record = self._jobs.get(job_id)
            if record is None:
                return None
            return record.copy()

    def list(self, limit: int = 50) -> list[IngestJobRecord]:
        with self._lock:
            self._cleanup_locked(_utc_now())
            records = sorted(
                self._jobs.values(),
                key=lambda job: job.submitted_at,
                reverse=True,
            )
            return [job.copy() for job in records[:limit]]

    def cancel(self, job_id: str) -> IngestJobRecord | None:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return None
            if record.status in {"completed", "failed", "cancelled"}:
                return record.copy()

            future = self._futures.get(job_id)
            if future is None:
                return record.copy()

            if future.cancel():
                finished_at = _utc_now()
                record.status = "cancelled"
                record.finished_at = finished_at
                record.result = None
                record.error = "Cancelled by request."
                self._futures.pop(job_id, None)
                self._cleanup_locked(finished_at)
            return record.copy()

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
