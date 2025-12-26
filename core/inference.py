from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from core.constants import OUTPUTS_ROOT


MOCK_KEYPOINTS = [
    {"name": "head", "x": 0.5, "y": 0.2, "confidence": 0.9},
    {"name": "hip", "x": 0.5, "y": 0.6, "confidence": 0.85},
]


def run_inference(frames: Iterable[int]) -> dict:
    pose_tracks = {
        "schema_version": "1.0",
        "tracks": [
            {
                "track_id": 1,
                "frames": [
                    {
                        "frame_index": int(frame),
                        "keypoints": MOCK_KEYPOINTS,
                    }
                    for frame in frames
                ],
            }
        ],
    }
    return pose_tracks


def write_pose_tracks(frames: Iterable[int], output_path: Path | None = None) -> Path:
    OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)
    output = output_path or OUTPUTS_ROOT / "pose_tracks.json"
    payload = run_inference(frames)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output
