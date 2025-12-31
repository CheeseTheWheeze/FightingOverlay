from __future__ import annotations

import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from core.paths import (
    get_data_root,
    get_outputs_root,
    get_profiles_root,
    get_clip_root,
)


@dataclass(frozen=True)
class LegacyOutputInfo:
    root: Path
    pose_path: Path | None
    overlay_path: Path | None


def generate_athlete_id() -> str:
    return f"ath_{uuid.uuid4().hex[:12]}"


def generate_clip_id() -> str:
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"clip_{stamp}_{uuid.uuid4().hex[:6]}"


def ensure_profile_dirs() -> None:
    get_profiles_root().mkdir(parents=True, exist_ok=True)


def ensure_clip_dir(athlete_id: str, clip_id: str) -> Path:
    clip_dir = get_clip_root(athlete_id, clip_id)
    clip_dir.mkdir(parents=True, exist_ok=True)
    return clip_dir


def copy_source_to_clip(clip_dir: Path, source_path: Path) -> Path:
    destination = clip_dir / f"source{source_path.suffix}"
    if source_path.resolve() == destination.resolve():
        return destination
    shutil.copy2(source_path, destination)
    return destination


def detect_legacy_outputs() -> LegacyOutputInfo | None:
    output_root = get_outputs_root()
    pose_path = output_root / "pose_tracks.json"
    overlay_path = output_root / "overlay.mp4"
    if pose_path.exists() or overlay_path.exists():
        return LegacyOutputInfo(
            root=output_root,
            pose_path=pose_path if pose_path.exists() else None,
            overlay_path=overlay_path if overlay_path.exists() else None,
        )
    return None


def ensure_data_layout() -> None:
    get_data_root().mkdir(parents=True, exist_ok=True)
    ensure_profile_dirs()
