from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.pipeline import ProcessingOptions, run_pipeline


@dataclass(frozen=True)
class ClipProcessingConfig:
    export_overlay_video: bool = True
    save_pose_json: bool = True
    save_thumbnails: bool = False
    save_combat_overlay: bool = False
    save_background_tracks: bool = True
    foreground_mode: str = "Auto (closest/most active)"
    debug_overlay: bool = False
    draw_all_tracks: bool = False
    smoothing_alpha: float = 0.7
    smoothing_enabled: bool = True
    min_keypoint_confidence: float = 0.3
    overlay_mode: str = "skeleton"
    max_tracks: int = 3
    track_sort: str = "confidence"
    live_preview: bool = True
    run_evaluation: bool = False
    run_judge: bool = False
    tracking_backend: str = "Motion (fast)"
    manual_track_ids: list[str] | None = None


def process_clip(
    video_path: Path | None,
    *,
    output_dir: Path,
    config: ClipProcessingConfig,
    cancel_event: object | None = None,
    status_callback=None,
    info_callback=None,
) -> Path:
    options = ProcessingOptions(
        export_overlay_video=config.export_overlay_video,
        save_pose_json=config.save_pose_json,
        save_thumbnails=config.save_thumbnails,
        save_combat_overlay=config.save_combat_overlay,
        save_background_tracks=config.save_background_tracks,
        foreground_mode=config.foreground_mode,
        debug_overlay=config.debug_overlay,
        draw_all_tracks=config.draw_all_tracks,
        smoothing_alpha=config.smoothing_alpha,
        smoothing_enabled=config.smoothing_enabled,
        min_keypoint_confidence=config.min_keypoint_confidence,
        overlay_mode=config.overlay_mode,
        max_tracks=config.max_tracks,
        track_sort=config.track_sort,
        live_preview=config.live_preview,
        run_evaluation=config.run_evaluation,
        run_judge=config.run_judge,
        tracking_backend=config.tracking_backend,
        manual_track_ids=config.manual_track_ids,
    )
    return run_pipeline(
        video_path,
        options,
        cancel_event,
        status_callback,
        info_callback,
        output_dir=output_dir,
    )
