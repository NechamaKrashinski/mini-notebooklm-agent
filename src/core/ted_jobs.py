"""Thin CRUD over the ted_jobs table.

Deliberately thin: only existence, display status and timestamps. The talk's
content (brief, script, feedback) lives in the graph checkpoint, the single
source of truth; read it with agents.ted.get_values(job_id).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from core.db import get_conn, lock

Status = Literal["pending_approval", "audio_ready", "failed"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create(job_id: str) -> None:
    now = _now()
    with lock:
        conn = get_conn()
        conn.execute(
            "INSERT INTO ted_jobs (job_id, status, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (job_id, "pending_approval", now, now),
        )
        conn.commit()


def set_status(job_id: str, status: Status) -> None:
    with lock:
        conn = get_conn()
        conn.execute(
            "UPDATE ted_jobs SET status = ?, updated_at = ? WHERE job_id = ?",
            (status, _now(), job_id),
        )
        conn.commit()


def get(job_id: str) -> dict | None:
    with lock:
        row = get_conn().execute("SELECT * FROM ted_jobs WHERE job_id = ?", (job_id,)).fetchone()
    return dict(row) if row else None


def list_jobs(status: Status | None = None) -> list[dict]:
    query = "SELECT * FROM ted_jobs"
    params: tuple = ()
    if status:
        query += " WHERE status = ?"
        params = (status,)
    with lock:
        rows = get_conn().execute(query + " ORDER BY created_at DESC", params).fetchall()
    return [dict(r) for r in rows]


def delete(job_id: str) -> bool:
    """Delete the job row and its checkpoint thread: deleting a talk deletes all of it."""
    from agents.ted.checkpoint import get_saver

    get_saver().delete_thread(job_id)
    with lock:
        conn = get_conn()
        cur = conn.execute("DELETE FROM ted_jobs WHERE job_id = ?", (job_id,))
        conn.commit()
    return cur.rowcount > 0
