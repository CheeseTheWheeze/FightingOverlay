from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class AthleteProfile:
    id: str
    name: str
    created_at: datetime


@dataclass(frozen=True)
class Session:
    id: str
    athlete_id: str
    created_at: datetime
    label: str | None = None


@dataclass(frozen=True)
class Clip:
    id: str
    athlete_id: str
    source_path: str
    created_at: datetime
    session_id: str | None = None


@dataclass(frozen=True)
class Artifact:
    id: str
    clip_id: str
    kind: str
    path: str
    created_at: datetime
    meta: dict[str, Any] | None = None
