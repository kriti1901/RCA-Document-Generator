"""Long-term memory: what the app remembers across sessions.

- incidents: every finished RCA (facts, text, version, Confluence link)
- information_gaps: what a user declined to provide, and who, so the next
  session on that incident asks only for what is still missing
- org_knowledge: service names and people seen before
- audit_log: who answered, declined, approved and published what
- a full-text index over past incidents, for "similar incidents"

It is a plain SQLite file. For production the same tables move to Postgres,
with pgvector replacing the full-text index for semantic search.
"""

from __future__ import annotations

import json
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

_SCHEMA = """
CREATE TABLE IF NOT EXISTS incidents (
    incident_key TEXT PRIMARY KEY,
    incident_id TEXT, title TEXT, severity TEXT, services TEXT, started_at TEXT,
    summary TEXT, root_cause TEXT, status TEXT, version INTEGER,
    facts_json TEXT, drafts_json TEXT, notes_json TEXT, extra_json TEXT,
    pdf_path TEXT, confluence_page_id TEXT, confluence_url TEXT,
    created_by TEXT, created_at TEXT, updated_at TEXT
);
CREATE VIRTUAL TABLE IF NOT EXISTS incidents_fts USING fts5(
    incident_key UNINDEXED, title, summary, root_cause, services
);
CREATE TABLE IF NOT EXISTS information_gaps (
    incident_key TEXT, gap_id TEXT, label TEXT, section TEXT,
    status TEXT, declined_by TEXT, declined_at TEXT, resolved_at TEXT, version INTEGER,
    PRIMARY KEY (incident_key, gap_id)
);
CREATE TABLE IF NOT EXISTS org_knowledge (
    kind TEXT, key TEXT, value TEXT, source_incident TEXT, updated_at TEXT,
    PRIMARY KEY (kind, key)
);
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    incident_key TEXT, session_id TEXT, at TEXT, actor TEXT, event TEXT, detail TEXT
);
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY, incident_key TEXT, title TEXT, status TEXT,
    user_name TEXT, created_at TEXT, updated_at TEXT
);
"""

_STOPWORDS = frozenset(
    "the and for with that this from was were are has have had not but into over under after before "
    "when what which who how why all any our its their about above below then than them they there "
    "incident issue problem error errors failed failure outage service services system".split()
)


def _keywords(text: str, limit: int = 12) -> list[str]:
    words = []
    for word in re.findall(r"[A-Za-z0-9][A-Za-z0-9_]{2,}", text.lower()):
        if word not in _STOPWORDS and word not in words:
            words.append(word)
    return words[:limit]


class LongTermMemory:
    def __init__(self, db_path: Path | str) -> None:
        self.path = str(db_path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        # An in-memory database lives only as long as its connection, so keep one open.
        self._shared = sqlite3.connect(self.path, check_same_thread=False) if self.path == ":memory:" else None
        with self._db() as db:
            db.executescript(_SCHEMA)

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        conn = self._shared or sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            if self._shared is None:
                conn.close()

    # ---- incidents -------------------------------------------------------

    def save_incident(self, record: dict, gaps: list[dict], now: str) -> None:
        """Insert or update a finished RCA, refresh its search entry and sync its open gaps."""
        key = record["incident_key"]
        with self._db() as db:
            existing = db.execute("SELECT created_at, created_by FROM incidents WHERE incident_key = ?",
                                  (key,)).fetchone()
            created_at = existing["created_at"] if existing else now
            created_by = existing["created_by"] if existing else record.get("created_by", "")
            db.execute(
                """INSERT OR REPLACE INTO incidents VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (key, record.get("incident_id", ""), record.get("title", ""), record.get("severity", ""),
                 record.get("services", ""), record.get("started_at", ""), record.get("summary", ""),
                 record.get("root_cause", ""), record.get("status", ""), int(record.get("version", 1)),
                 json.dumps(record.get("facts", {})), json.dumps(record.get("drafts", {})),
                 json.dumps(record.get("notes", [])), json.dumps(record.get("extra", {})),
                 record.get("pdf_path", ""), record.get("confluence_page_id", ""),
                 record.get("confluence_url", ""), created_by, created_at, now),
            )
            db.execute("DELETE FROM incidents_fts WHERE incident_key = ?", (key,))
            db.execute("INSERT INTO incidents_fts VALUES (?,?,?,?,?)",
                       (key, record.get("title", ""), record.get("summary", ""),
                        record.get("root_cause", ""), record.get("services", "")))
            current = {gap["id"] for gap in gaps}
            db.execute(
                "UPDATE information_gaps SET status = 'resolved', resolved_at = ? "
                "WHERE incident_key = ? AND status = 'declined' AND gap_id NOT IN (%s)"
                % ",".join("?" * len(current)),
                (now, key, *current),
            )
            for gap in gaps:
                db.execute(
                    "INSERT OR REPLACE INTO information_gaps VALUES (?,?,?,?,?,?,?,?,?)",
                    (key, gap["id"], gap.get("label", ""), gap.get("section", ""), "declined",
                     gap.get("by", ""), gap.get("at", ""), "", int(record.get("version", 1))),
                )

    def get_incident(self, incident_key: str) -> dict | None:
        with self._db() as db:
            row = db.execute("SELECT * FROM incidents WHERE incident_key = ?", (incident_key,)).fetchone()
        if row is None:
            return None
        record = dict(row)
        for column, name in (("facts_json", "facts"), ("drafts_json", "drafts"),
                             ("notes_json", "notes"), ("extra_json", "extra")):
            record[name] = json.loads(record.pop(column) or "null")
        return record

    def list_incidents(self, limit: int = 50) -> list[dict]:
        with self._db() as db:
            rows = db.execute(
                """SELECT i.incident_key, i.incident_id, i.title, i.severity, i.status, i.version,
                          i.confluence_url, i.pdf_path, i.updated_at,
                          (SELECT COUNT(*) FROM information_gaps g
                            WHERE g.incident_key = i.incident_key AND g.status = 'declined') AS open_gaps
                     FROM incidents i ORDER BY i.updated_at DESC LIMIT ?""", (limit,)).fetchall()
        return [dict(row) for row in rows]

    def find_similar(self, text: str, exclude_key: str = "", limit: int = 3) -> list[dict]:
        """Past incidents that share keywords with `text`, best match first."""
        words = _keywords(text)
        if not words:
            return []
        query = " OR ".join(f'"{word}"' for word in words)
        with self._db() as db:
            rows = db.execute(
                """SELECT i.incident_key, i.incident_id, i.title, i.severity, i.started_at,
                          i.root_cause, i.confluence_url
                     FROM incidents_fts f JOIN incidents i ON i.incident_key = f.incident_key
                    WHERE incidents_fts MATCH ? AND i.incident_key != ?
                    ORDER BY bm25(incidents_fts) LIMIT ?""", (query, exclude_key, limit)).fetchall()
        return [dict(row) for row in rows]

    def open_gaps(self, incident_key: str) -> list[dict]:
        with self._db() as db:
            rows = db.execute(
                "SELECT gap_id, label, section, declined_by, declined_at, version FROM information_gaps "
                "WHERE incident_key = ? AND status = 'declined' ORDER BY rowid", (incident_key,)).fetchall()
        return [dict(row) for row in rows]

    # ---- organisation knowledge -----------------------------------------

    def remember(self, kind: str, key: str, now: str, value: str = "", source: str = "") -> None:
        key = key.strip()
        if not key:
            return
        with self._db() as db:
            db.execute("INSERT OR REPLACE INTO org_knowledge VALUES (?,?,?,?,?)",
                       (kind, key, value, source, now))

    def recall(self, kind: str, limit: int = 50) -> list[str]:
        with self._db() as db:
            rows = db.execute("SELECT key FROM org_knowledge WHERE kind = ? ORDER BY updated_at DESC LIMIT ?",
                              (kind, limit)).fetchall()
        return [row["key"] for row in rows]

    # ---- audit trail and sessions ---------------------------------------

    def log(self, incident_key: str, session_id: str, at: str, actor: str, event: str, detail: str = "") -> None:
        with self._db() as db:
            db.execute("INSERT INTO audit_log (incident_key, session_id, at, actor, event, detail) "
                       "VALUES (?,?,?,?,?,?)", (incident_key, session_id, at, actor, event, detail))

    def audit(self, incident_key: str) -> list[dict]:
        with self._db() as db:
            rows = db.execute("SELECT at, actor, event, detail FROM audit_log WHERE incident_key = ? ORDER BY id",
                              (incident_key,)).fetchall()
        return [dict(row) for row in rows]

    def upsert_session(self, session_id: str, incident_key: str, title: str, status: str,
                       user_name: str, now: str) -> None:
        with self._db() as db:
            existing = db.execute("SELECT created_at FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
            db.execute("INSERT OR REPLACE INTO sessions VALUES (?,?,?,?,?,?,?)",
                       (session_id, incident_key, title, status, user_name,
                        existing["created_at"] if existing else now, now))

    def list_sessions(self, limit: int = 20) -> list[dict]:
        with self._db() as db:
            rows = db.execute("SELECT * FROM sessions ORDER BY updated_at DESC LIMIT ?", (limit,)).fetchall()
        return [dict(row) for row in rows]
