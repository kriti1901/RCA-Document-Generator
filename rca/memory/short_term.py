"""Short-term memory: the working state of one RCA session.

LangGraph saves the whole session state to SQLite after every step, keyed by
the session id. That is what lets a session survive a browser refresh or a
restart, and be picked up again days later. For production, swap SqliteSaver
for the Postgres checkpointer; nothing else changes.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver


def make_checkpointer(db_path: Path | str) -> SqliteSaver:
    if str(db_path) != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False, timeout=30)
    if str(db_path) != ":memory:":
        conn.execute("PRAGMA journal_mode=WAL")
    saver = SqliteSaver(conn)
    saver.setup()
    return saver
