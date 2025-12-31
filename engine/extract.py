from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.pipeline import ProcessingOptions, run_inference
from engine.pose import PoseSequence


@dataclass(frozen=True)
class PoseExtractionConfig:
    tracking_backend: str = "Motion (fast)"
    foreground_mode: str = "Auto (closest/most active)"
    manual_track_ids: list[str] | None = None
    save_background_tracks: bool = True


def extract_pose(video_path: str, *, config: PoseExtractionConfig) -> PoseSequence:
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    options = ProcessingOptions(
        tracking_backend=config.tracking_backend,
        foreground_mode=config.foreground_mode,
        manual_track_ids=config.manual_track_ids,
        save_background_tracks=config.save_background_tracks,
    )
    payload = run_inference(path, options)
    return PoseSequence.from_payload(payload)


class PoseExtractor:
    def extract(self, video_path: str, *, config: PoseExtractionConfig | None = None) -> PoseSequence:
        return extract_pose(video_path, config=config or PoseExtractionConfig())
