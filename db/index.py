from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

from core.models import AthleteProfile, Artifact, Clip
from core.paths import get_db_path

SCHEMA_VERSION = 1


def _utc_now() -> datetime:
    return datetime.utcnow()


def _format_timestamp(value: datetime | None = None) -> str:
    return (value or _utc_now()).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_timestamp(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")


def _connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = db_path or get_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path | None = None) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_versions (version INTEGER PRIMARY KEY, applied_at TEXT)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS athletes (id TEXT PRIMARY KEY, name TEXT, created_at TEXT)"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS clips (
                id TEXT PRIMARY KEY,
                athlete_id TEXT,
                source_path TEXT,
                created_at TEXT,
                session_id TEXT,
                FOREIGN KEY (athlete_id) REFERENCES athletes(id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS artifacts (
                id TEXT PRIMARY KEY,
                clip_id TEXT,
                kind TEXT,
                path TEXT,
                created_at TEXT,
                meta_json TEXT,
                FOREIGN KEY (clip_id) REFERENCES clips(id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS corpus_exports (
                id TEXT PRIMARY KEY,
                athlete_id TEXT,
                clip_id TEXT,
                feature_set TEXT,
                path TEXT,
                created_at TEXT
            )
            """
        )
        conn.execute(
            "INSERT OR IGNORE INTO schema_versions (version, applied_at) VALUES (?, ?)",
            (SCHEMA_VERSION, _format_timestamp()),
        )


def list_athletes(db_path: Path | None = None) -> list[AthleteProfile]:
    with _connect(db_path) as conn:
        rows = conn.execute("SELECT id, name, created_at FROM athletes ORDER BY created_at ASC").fetchall()
    athletes: list[AthleteProfile] = []
    for row in rows:
        athletes.append(
            AthleteProfile(
                id=row["id"],
                name=row["name"],
                created_at=_parse_timestamp(row["created_at"]),
            )
        )
    return athletes


def create_athlete(name: str, db_path: Path | None = None) -> AthleteProfile:
    athlete_id = f"ath_{uuid.uuid4().hex[:12]}"
    created_at = _format_timestamp()
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT INTO athletes (id, name, created_at) VALUES (?, ?, ?)",
            (athlete_id, name, created_at),
        )
    return AthleteProfile(id=athlete_id, name=name, created_at=_parse_timestamp(created_at))


def get_or_create_athlete(name: str, db_path: Path | None = None) -> AthleteProfile:
    with _connect(db_path) as conn:
        row = conn.execute("SELECT id, name, created_at FROM athletes WHERE name = ?", (name,)).fetchone()
        if row:
            return AthleteProfile(
                id=row["id"],
                name=row["name"],
                created_at=_parse_timestamp(row["created_at"]),
            )
    return create_athlete(name, db_path)


def create_clip(
    athlete_id: str,
    source_path: str,
    session_id: str | None = None,
    db_path: Path | None = None,
) -> Clip:
    clip_id = f"clip_{uuid.uuid4().hex[:12]}"
    created_at = _format_timestamp()
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO clips (id, athlete_id, source_path, created_at, session_id)
            VALUES (?, ?, ?, ?, ?)
            """,
            (clip_id, athlete_id, source_path, created_at, session_id),
        )
    return Clip(
        id=clip_id,
        athlete_id=athlete_id,
        source_path=source_path,
        created_at=_parse_timestamp(created_at),
        session_id=session_id,
    )


def add_artifact(
    clip_id: str,
    kind: str,
    path: str,
    meta: dict[str, object] | None = None,
    db_path: Path | None = None,
) -> Artifact:
    artifact_id = f"art_{uuid.uuid4().hex[:12]}"
    created_at = _format_timestamp()
    meta_json = json.dumps(meta, indent=2) if meta else None
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO artifacts (id, clip_id, kind, path, created_at, meta_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (artifact_id, clip_id, kind, path, created_at, meta_json),
        )
    return Artifact(
        id=artifact_id,
        clip_id=clip_id,
        kind=kind,
        path=path,
        created_at=_parse_timestamp(created_at),
        meta=meta,
    )


def record_corpus_export(
    athlete_id: str,
    clip_id: str,
    feature_set: str,
    path: str,
    db_path: Path | None = None,
) -> None:
    export_id = f"corp_{uuid.uuid4().hex[:12]}"
    created_at = _format_timestamp()
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO corpus_exports (id, athlete_id, clip_id, feature_set, path, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (export_id, athlete_id, clip_id, feature_set, path, created_at),
        )
