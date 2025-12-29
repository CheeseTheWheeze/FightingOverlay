from __future__ import annotations

from pathlib import Path

from core.pipeline import cv2
from core.pipeline import json_dumps
from fightai.judge.metrics import FrameMetrics


def write_debug_bundle(
    output_dir: Path,
    metrics: list[FrameMetrics],
    raw_video: Path | None,
    overlay_video: Path | None,
    worst_frame_count: int,
    clip_padding_s: float,
    fps: float,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    worst_frames_dir = output_dir / "worst_frames"
    worst_frames_dir.mkdir(parents=True, exist_ok=True)
    worst_clips_dir = output_dir / "worst_clips"
    worst_clips_dir.mkdir(parents=True, exist_ok=True)

    sorted_metrics = sorted(metrics, key=lambda m: (m.in_mask_ratio, -m.mean_distance_to_mask_px))
    top = sorted_metrics[:worst_frame_count]

    captured_frames: list[dict[str, str | int]] = []
    if raw_video and cv2 is not None:
        cap = cv2.VideoCapture(str(raw_video))
        for entry in top:
            frame_index = entry.frame_index
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                continue
            raw_path = worst_frames_dir / f"raw_f{frame_index:05d}_t{entry.track_id}.jpg"
            cv2.imwrite(str(raw_path), frame)
            captured_frames.append({"frame_index": frame_index, "raw": raw_path.name, "track_id": entry.track_id})
        cap.release()

    if overlay_video and cv2 is not None:
        cap = cv2.VideoCapture(str(overlay_video))
        for entry in top:
            frame_index = entry.frame_index
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                continue
            overlay_path = worst_frames_dir / f"overlay_f{frame_index:05d}_t{entry.track_id}.jpg"
            cv2.imwrite(str(overlay_path), frame)
            for record in captured_frames:
                if record["frame_index"] == frame_index and record["track_id"] == entry.track_id:
                    record["overlay"] = overlay_path.name
        cap.release()

    clip_manifest = []
    if raw_video and cv2 is not None:
        duration_padding = int(clip_padding_s * fps)
        for entry in top:
            start = max(0, entry.frame_index - duration_padding)
            end = entry.frame_index + duration_padding
            clip_path = worst_clips_dir / f"clip_f{entry.frame_index:05d}_t{entry.track_id}.mp4"
            _export_clip(raw_video, clip_path, start, end)
            clip_manifest.append({"frame_index": entry.frame_index, "clip": clip_path.name, "track_id": entry.track_id})

    manifest = {
        "frames": captured_frames,
        "clips": clip_manifest,
    }
    manifest_path = output_dir / "bundle_manifest.json"
    manifest_path.write_text(json_dumps(manifest), encoding="utf-8")

    report_path = output_dir / "report.html"
    report_path.write_text(_build_report_html(captured_frames, clip_manifest), encoding="utf-8")

    return {
        "report_html": str(report_path),
        "bundle_manifest": str(manifest_path),
    }


def _export_clip(video_path: Path, output_path: Path, start_frame: int, end_frame: int) -> None:
    if cv2 is None:
        return
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_index = start_frame
    while frame_index <= end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        frame_index += 1
    writer.release()
    cap.release()


def _build_report_html(frames: list[dict[str, str | int]], clips: list[dict[str, str | int]]) -> str:
    frames_html = "".join(
        f"<div class='frame-card'><h4>Frame {frame['frame_index']} (track {frame['track_id']})</h4>"
        + (f"<img src='worst_frames/{frame['raw']}' alt='raw frame'>" if "raw" in frame else "")
        + (f"<img src='worst_frames/{frame['overlay']}' alt='overlay frame'>" if "overlay" in frame else "")
        + "</div>"
        for frame in frames
    )
    clips_html = "".join(
        f"<li>Frame {clip['frame_index']} (track {clip['track_id']}) - "
        f"<a href='worst_clips/{clip['clip']}'>clip</a></li>"
        for clip in clips
    )
    return (
        "<!doctype html><html><head><meta charset='utf-8'/>"
        "<style>body{font-family:Arial,sans-serif;background:#0f1115;color:#e0e0e0;}"
        ".frame-card{background:#1c2026;padding:12px;margin:12px;border-radius:8px;}"
        ".frame-card img{max-width:100%;display:block;margin-top:8px;border-radius:6px;}"
        "a{color:#8ab4f8}</style></head><body>"
        "<h1>Overlay Alignment Judge Report</h1>"
        "<h2>Worst Frames</h2>"
        f"{frames_html}"
        "<h2>Clips</h2><ul>"
        f"{clips_html}</ul>"
        "</body></html>"
    )
