from __future__ import annotations

import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from core.paths import get_outputs_root
from core.schema import KEYPOINT_NAMES, SCHEMA_VERSION, validate_pose_tracks_schema

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None

StatusCallback = Callable[[str, float | None], None]


@dataclass
class ProcessingOptions:
    export_overlay_video: bool = True
    save_pose_json: bool = True
    save_thumbnails: bool = False
    save_background_tracks: bool = True
    foreground_mode: str = "Auto (closest/most active)"


class ProcessingCancelled(RuntimeError):
    pass


def _now_ms() -> int:
    return int(time.time() * 1000)


def _update_status(callback: StatusCallback | None, message: str, progress: float | None = None) -> None:
    if callback:
        callback(message, progress)


def load_video_metadata(video_path: Path | None) -> dict[str, float | int | str]:
    if video_path and cv2 is not None:
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.release()
            return {
                "path": str(video_path),
                "fps": float(fps),
                "width": width,
                "height": height,
                "frame_count": frame_count if frame_count > 0 else int(fps * 2),
            }
        cap.release()
    return {
        "path": str(video_path) if video_path else "",
        "fps": 30.0,
        "width": 1280,
        "height": 720,
        "frame_count": 60,
    }


def _synthetic_bbox(frame_index: int, track_index: int, width: int, height: int) -> list[float]:
    base_x = 0.1 + 0.2 * track_index
    base_y = 0.2 + 0.1 * track_index
    wiggle = 0.05 * math.sin(frame_index / 5 + track_index)
    x = (base_x + wiggle) * width
    y = (base_y + wiggle) * height
    w = width * (0.18 + 0.04 * track_index)
    h = height * (0.28 + 0.03 * track_index)
    return [round(x, 2), round(y, 2), round(w, 2), round(h, 2)]


def _make_keypoints(bbox: list[float]) -> list[dict[str, float | str]]:
    x, y, w, h = bbox
    points = []
    for idx, name in enumerate(KEYPOINT_NAMES):
        offset_x = (idx % 3) / 4
        offset_y = (idx % 5) / 6
        points.append(
            {
                "name": name,
                "x": round(x + w * (0.2 + offset_x), 2),
                "y": round(y + h * (0.2 + offset_y), 2),
                "c": 0.6,
            }
        )
    return points


def generate_synthetic_tracks(
    frame_count: int,
    fps: float,
    width: int,
    height: int,
    cancel_event: object | None = None,
) -> list[dict[str, object]]:
    tracks: list[dict[str, object]] = []
    for track_index in range(3):
        frames = []
        for frame_index in range(frame_count):
            if cancel_event is not None and getattr(cancel_event, "is_set")():
                raise ProcessingCancelled("Processing cancelled")
            bbox = _synthetic_bbox(frame_index, track_index, width, height)
            frames.append(
                {
                    "frame_index": frame_index,
                    "timestamp_ms": int(frame_index / fps * 1000),
                    "bbox_xywh": bbox,
                    "keypoints_2d": _make_keypoints(bbox),
                    "confidence": 0.55 + 0.1 * track_index,
                }
            )
        tracks.append(
            {
                "track_id": f"t{track_index + 1}",
                "person_index": track_index,
                "is_foreground": False,
                "source": {"backend": "synthetic", "keypoint_format": "name-2d"},
                "frames": frames,
            }
        )
    return tracks


def _score_track(frames: list[dict[str, object]]) -> float:
    if not frames:
        return 0.0
    centers = []
    areas = []
    for frame in frames:
        bbox = frame["bbox_xywh"]
        x, y, w, h = bbox
        centers.append((x + w / 2, y + h / 2))
        areas.append(w * h)
    avg_area = sum(areas) / len(areas)
    motion = 0.0
    for idx in range(1, len(centers)):
        dx = centers[idx][0] - centers[idx - 1][0]
        dy = centers[idx][1] - centers[idx - 1][1]
        motion += math.hypot(dx, dy)
    motion_energy = motion / max(1, len(centers) - 1)
    return avg_area * (1 + motion_energy / 100)


def select_foreground_tracks(tracks: list[dict[str, object]], foreground_count: int = 2) -> None:
    scored = []
    for track in tracks:
        frames = track.get("frames", [])
        score = _score_track(frames)
        scored.append((score, track))
    scored.sort(key=lambda item: item[0], reverse=True)
    for index, (_, track) in enumerate(scored):
        track["is_foreground"] = index < foreground_count


def build_pose_payload(
    video_meta: dict[str, float | int | str],
    tracks: list[dict[str, object]],
) -> dict[str, object]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "video": {
            "path": video_meta["path"],
            "fps": video_meta["fps"],
            "width": video_meta["width"],
            "height": video_meta["height"],
        },
        "tracks": tracks,
    }
    return payload


def run_inference(
    video_path: Path | None,
    options: ProcessingOptions,
    cancel_event: object | None = None,
    status_callback: StatusCallback | None = None,
) -> dict[str, object]:
    _update_status(status_callback, "Loading video metadata...", 5)
    video_meta = load_video_metadata(video_path)
    frame_count = int(video_meta["frame_count"])
    fps = float(video_meta["fps"])
    width = int(video_meta["width"])
    height = int(video_meta["height"])

    _update_status(status_callback, "Generating pose tracks (synthetic fallback)...", 15)
    tracks = generate_synthetic_tracks(frame_count, fps, width, height, cancel_event)
    select_foreground_tracks(tracks)

    if not options.save_background_tracks:
        tracks = [track for track in tracks if track.get("is_foreground")]

    payload = build_pose_payload(video_meta, tracks)
    ok, message = validate_pose_tracks_schema(payload)
    if not ok:
        raise RuntimeError(f"Generated payload invalid: {message}")
    _update_status(status_callback, "Pose tracks ready.", 35)
    return payload


def write_pose_tracks(output_dir: Path, payload: dict[str, object]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "pose_tracks.json"
    output_path.write_text(
        json_dumps(payload),
        encoding="utf-8",
    )
    return output_path


def json_dumps(payload: dict[str, object]) -> str:
    import json

    return json.dumps(payload, indent=2)


def export_overlay_video(
    video_path: Path,
    output_path: Path,
    tracks: list[dict[str, object]],
    fps: float,
    width: int,
    height: int,
    cancel_event: object | None = None,
    status_callback: StatusCallback | None = None,
) -> None:
    if cv2 is None:
        shutil.copyfile(video_path, output_path)
        _update_status(status_callback, "Overlay export fallback: copied source video.", 90)
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Unable to open video for overlay export.")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_map: dict[int, list[list[float]]] = {}
    for track in tracks:
        for frame in track.get("frames", []):
            bbox = frame.get("bbox_xywh")
            frame_map.setdefault(int(frame["frame_index"]), []).append(bbox)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
    frame_index = 0
    while True:
        if cancel_event is not None and getattr(cancel_event, "is_set")():
            cap.release()
            writer.release()
            raise ProcessingCancelled("Processing cancelled")
        ret, frame = cap.read()
        if not ret:
            break
        for bbox in frame_map.get(frame_index, []):
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 255), 2)
        writer.write(frame)
        frame_index += 1
        if frame_index % 10 == 0:
            progress = 35 + (frame_index / max(1, total)) * 50
            _update_status(status_callback, "Exporting overlay video...", progress)
    cap.release()
    writer.release()
    _update_status(status_callback, "Overlay video exported.", 90)


def save_thumbnails(
    video_path: Path,
    output_dir: Path,
    cancel_event: object | None = None,
    status_callback: StatusCallback | None = None,
) -> None:
    if cv2 is None:
        _update_status(status_callback, "Thumbnails skipped (OpenCV unavailable).", 85)
        return
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Unable to open video for thumbnails.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    output_dir.mkdir(parents=True, exist_ok=True)
    next_capture = 0.0
    frame_index = 0
    while True:
        if cancel_event is not None and getattr(cancel_event, "is_set")():
            cap.release()
            raise ProcessingCancelled("Processing cancelled")
        ret, frame = cap.read()
        if not ret:
            break
        timestamp = frame_index / fps
        if timestamp >= next_capture:
            file_path = output_dir / f"thumb_{frame_index:05d}.jpg"
            cv2.imwrite(str(file_path), frame)
            next_capture += 1.0
        frame_index += 1
    cap.release()
    _update_status(status_callback, "Thumbnails saved.", 85)


def run_pipeline(
    video_path: Path | None,
    options: ProcessingOptions,
    cancel_event: object | None = None,
    status_callback: StatusCallback | None = None,
) -> Path:
    output_dir = get_outputs_root()
    payload = run_inference(video_path, options, cancel_event, status_callback)

    if options.save_pose_json:
        pose_path = write_pose_tracks(output_dir, payload)
    else:
        pose_path = output_dir / "pose_tracks.json"

    if video_path and options.export_overlay_video:
        export_overlay_video(
            video_path,
            output_dir / "overlay.mp4",
            payload["tracks"],
            float(payload["video"]["fps"]),
            int(payload["video"]["width"]),
            int(payload["video"]["height"]),
            cancel_event,
            status_callback,
        )

    if video_path and options.save_thumbnails:
        save_thumbnails(video_path, output_dir / "thumbnails", cancel_event, status_callback)

    _update_status(status_callback, "Processing complete.", 100)
    return pose_path


__all__ = [
    "ProcessingOptions",
    "ProcessingCancelled",
    "run_pipeline",
    "run_inference",
    "build_pose_payload",
]
