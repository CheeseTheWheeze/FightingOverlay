from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CombatOverlayConfig:
    """Configuration for derived combat overlay zones."""

    # Base radius in pixels for vulnerable zone circles.
    base_zone_radius_px: float = 18.0
    # Multiplier applied based on keypoint confidence (lower conf -> larger radius).
    uncertainty_scale: float = 20.0
    # Extra radius from motion (pixels).
    motion_scale: float = 0.35
