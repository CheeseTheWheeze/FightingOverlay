from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from core.schema import SCHEMA_VERSION


@dataclass(frozen=True)
class PoseSequence:
    video: dict[str, Any]
    tracks: list[dict[str, Any]]

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "PoseSequence":
        video = payload.get("video", {}) if isinstance(payload, dict) else {}
        tracks = payload.get("tracks", []) if isinstance(payload, dict) else []
        if not isinstance(video, dict):
            raise ValueError("PoseSequence payload missing video metadata")
        if not isinstance(tracks, list):
            raise ValueError("PoseSequence payload missing tracks list")
        return cls(video=video, tracks=tracks)

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "video": self.video,
            "tracks": self.tracks,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_payload(), indent=2)

    @classmethod
    def from_json(cls, data: str) -> "PoseSequence":
        payload = json.loads(data)
        return cls.from_payload(payload)
