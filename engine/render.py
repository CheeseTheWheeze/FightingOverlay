from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.pipeline import export_overlay_video
from engine.pose import PoseSequence


@dataclass(frozen=True)
class OverlayRenderConfig:
    debug_overlay: bool = False
    draw_all_tracks: bool = False
    smoothing_alpha: float = 0.7
    smoothing_enabled: bool = True
    min_keypoint_confidence: float = 0.3
    overlay_mode: str = "skeleton"
    max_tracks: int = 3
    track_sort: str = "confidence"
    live_preview: bool = True
    draw_bboxes: bool = False


class OverlayRenderer:
    def render(
        self,
        video_path: str,
        pose: PoseSequence,
        out_path: str,
        *,
        config: OverlayRenderConfig | None = None,
        cancel_event: object | None = None,
        status_callback=None,
        info_callback=None,
    ) -> str:
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        output_path = Path(out_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        render_config = config or OverlayRenderConfig()
        video_meta = pose.video
        export_overlay_video(
            path,
            output_path,
            pose.tracks,
            float(video_meta.get("fps", 30.0)),
            int(video_meta.get("width", 1280)),
            int(video_meta.get("height", 720)),
            debug_overlay=render_config.debug_overlay,
            draw_all_tracks=render_config.draw_all_tracks,
            smoothing_alpha=render_config.smoothing_alpha,
            smoothing_enabled=render_config.smoothing_enabled,
            min_keypoint_confidence=render_config.min_keypoint_confidence,
            overlay_mode=render_config.overlay_mode,
            max_tracks=render_config.max_tracks,
            track_sort=render_config.track_sort,
            live_preview=render_config.live_preview,
            draw_bboxes=render_config.draw_bboxes,
            cancel_event=cancel_event,
            status_callback=status_callback,
            info_callback=info_callback,
        )
        return str(output_path)
