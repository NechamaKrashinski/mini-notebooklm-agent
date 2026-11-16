"""Application database (SQLite): product tables like ted_jobs.

Kept in its own file, separate from the LangGraph checkpoints: the checkpoint
file is owned by the graph runtime (its schema and contents are LangGraph's),
while this file holds the app's own bookkeeping. Mixing them would couple our
schema to a library's internals.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

DATA_DIR = Path("data")
APP_DB_PATH = DATA_DIR / "app.sqlite"

SCHEMA = """
CREATE TABLE IF NOT EXISTS ted_jobs (
    job_id     TEXT PRIMARY KEY,
    status     TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""

# One shared connection; FastAPI runs sync endpoints in a threadpool, so every
# access goes through this lock.
lock = threading.RLock()
_conn: sqlite3.Connection | None = None


def get_conn() -> sqlite3.Connection:
    global _conn
    with lock:
        if _conn is None:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(APP_DB_PATH, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(SCHEMA)
            _conn = conn
        return _conn
