from __future__ import annotations

from pathlib import Path
from typing import Iterable

from core.pipeline import json_dumps
from fightai.combat.config import CombatOverlayConfig


def derive_combat_overlay(
    payload: dict[str, object],
    output_path: Path,
    config: CombatOverlayConfig | None = None,
) -> Path:
    config = config or CombatOverlayConfig()
    tracks = payload.get("tracks", [])
    overlay = {
        "schema_version": "1.0",
        "video": payload.get("video", {}),
        "tracks": [],
    }
    for track in tracks:
        track_id = str(track.get("track_id", "unknown"))
        frames_out = []
        for frame in track.get("frames", []):
            frame_index = int(frame.get("frame_index", -1))
            keypoints = frame.get("keypoints_2d", [])
            if not isinstance(keypoints, list):
                continue
            points = {str(item.get("name")): item for item in keypoints if isinstance(item, dict)}
            derived = _derive_from_points(points, config)
            frames_out.append(
                {
                    "frame_index": frame_index,
                    "timestamp_ms": frame.get("timestamp_ms"),
                    "combat_points": derived["points"],
                    "combat_zones": derived["zones"],
                }
            )
        overlay["tracks"].append({"track_id": track_id, "frames": frames_out})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json_dumps(overlay), encoding="utf-8")
    return output_path


def _derive_from_points(
    points: dict[str, dict[str, float]],
    config: CombatOverlayConfig | None = None,
) -> dict[str, list[dict[str, object]]]:
    config = config or CombatOverlayConfig()
    def pt(name: str) -> tuple[float, float, float] | None:
        item = points.get(name)
        if not item:
            return None
        return float(item.get("x", 0.0)), float(item.get("y", 0.0)), float(item.get("c", 0.0))

    derived_points = []
    derived_zones = []

    for label, name in [
        ("left_fist", "left_wrist"),
        ("right_fist", "right_wrist"),
        ("left_elbow", "left_elbow"),
        ("right_elbow", "right_elbow"),
        ("left_knee", "left_knee"),
        ("right_knee", "right_knee"),
        ("left_foot", "left_ankle"),
        ("right_foot", "right_ankle"),
        ("head_strike", "nose"),
    ]:
        data = pt(name)
        if data:
            x, y, c = data
            derived_points.append({"name": label, "x": x, "y": y, "c": c})

    center = _avg_point([pt("left_shoulder"), pt("right_shoulder")])
    if center:
        x, y, c = center
        derived_zones.append(_zone("chin_jaw", x, y - 12.0, c, config))
        derived_zones.append(_zone("throat", x, y + 12.0, c, config))
        derived_zones.append(_zone("sternum", x, y + 30.0, c, config))

    hips = _avg_point([pt("left_hip"), pt("right_hip")])
    if hips:
        x, y, c = hips
        derived_zones.append(_zone("groin", x, y + 18.0, c, config))
        derived_zones.append(_zone("liver", x - 20.0, y, c, config))
        derived_zones.append(_zone("ribs", x + 20.0, y, c, config))

    knees = _avg_point([pt("left_knee"), pt("right_knee")])
    if knees:
        x, y, c = knees
        derived_zones.append(_zone("knees", x, y, c, config))

    ankles = _avg_point([pt("left_ankle"), pt("right_ankle")])
    if ankles:
        x, y, c = ankles
        derived_zones.append(_zone("ankles", x, y, c, config))

    temples = _avg_point([pt("left_eye"), pt("right_eye")])
    if temples:
        x, y, c = temples
        derived_zones.append(_zone("temples", x, y, c, config))

    return {"points": derived_points, "zones": derived_zones}


def _avg_point(values: Iterable[tuple[float, float, float] | None]) -> tuple[float, float, float] | None:
    xs = []
    ys = []
    cs = []
    for item in values:
        if not item:
            continue
        xs.append(item[0])
        ys.append(item[1])
        cs.append(item[2])
    if not xs:
        return None
    return sum(xs) / len(xs), sum(ys) / len(ys), sum(cs) / len(cs)


def _zone(name: str, x: float, y: float, conf: float, config: CombatOverlayConfig) -> dict[str, object]:
    radius = config.base_zone_radius_px + config.uncertainty_scale * (1.0 - conf)
    return {"name": name, "x": x, "y": y, "radius": float(radius), "confidence": conf}
