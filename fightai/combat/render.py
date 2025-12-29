from __future__ import annotations

from typing import Iterable

from core.pipeline import cv2


def render_combat_overlay(
    frame,
    combat_points: Iterable[dict[str, object]],
    combat_zones: Iterable[dict[str, object]],
) -> None:
    if cv2 is None:
        return
    for point in combat_points:
        x = int(round(float(point.get("x", 0.0))))
        y = int(round(float(point.get("y", 0.0))))
        cv2.circle(frame, (x, y), 4, (0, 255, 255), -1)
    for zone in combat_zones:
        x = int(round(float(zone.get("x", 0.0))))
        y = int(round(float(zone.get("y", 0.0))))
        radius = int(round(float(zone.get("radius", 0.0))))
        cv2.circle(frame, (x, y), radius, (255, 0, 255), 2)
