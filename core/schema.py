from __future__ import annotations

from typing import Any

SCHEMA_VERSION = "1.0"

KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


def validate_pose_tracks_schema(payload: dict[str, Any]) -> tuple[bool, str]:
    if not isinstance(payload, dict):
        return False, "Payload must be an object."
    if payload.get("schema_version") != SCHEMA_VERSION:
        return False, "schema_version missing or unsupported."
    video = payload.get("video")
    if not isinstance(video, dict):
        return False, "video must be an object."
    for key in ("path", "fps", "width", "height"):
        if key not in video:
            return False, f"video missing '{key}'."
    tracks = payload.get("tracks")
    if not isinstance(tracks, list):
        return False, "tracks must be a list."
    for track in tracks:
        if not isinstance(track, dict):
            return False, "track entries must be objects."
        for key in ("track_id", "person_index", "is_foreground", "source", "frames"):
            if key not in track:
                return False, f"track missing '{key}'."
        frames = track.get("frames")
        if not isinstance(frames, list):
            return False, "frames must be a list."
        for frame in frames[:3]:
            if not isinstance(frame, dict):
                return False, "frame entries must be objects."
            for key in ("frame_index", "timestamp_ms", "bbox_xywh", "keypoints_2d", "confidence"):
                if key not in frame:
                    return False, f"frame missing '{key}'."
    return True, "pose_tracks schema is valid."
