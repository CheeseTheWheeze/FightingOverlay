from __future__ import annotations

import logging
import math
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
InfoCallback = Callable[[dict[str, object]], None]


@dataclass
class ProcessingOptions:
    export_overlay_video: bool = True
    save_pose_json: bool = True
    save_thumbnails: bool = False
    save_background_tracks: bool = True
    foreground_mode: str = "Auto (closest/most active)"
    debug_overlay: bool = True
    draw_all_tracks: bool = False
    smoothing_alpha: float = 0.7


class ProcessingCancelled(RuntimeError):
    pass


def _now_ms() -> int:
    return int(time.time() * 1000)


def _update_status(callback: StatusCallback | None, message: str, progress: float | None = None) -> None:
    if callback:
        callback(message, progress)


def _update_info(callback: InfoCallback | None, info: dict[str, object]) -> None:
    if callback:
        callback(info)


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
    info_callback: InfoCallback | None = None,
) -> list[dict[str, object]]:
    tracks: list[dict[str, object]] = []
    for track_index in range(3):
        frames = []
        for frame_index in range(frame_count):
            if cancel_event is not None and getattr(cancel_event, "is_set")():
                raise ProcessingCancelled("Processing cancelled")
            if frame_index % 5 == 0 or frame_index == frame_count - 1:
                _update_info(
                    info_callback,
                    {
                        "stage": "Running inference",
                        "frame_index": frame_index + 1,
                        "total_frames": frame_count,
                        "people": 3,
                        "effective_fps": fps,
                    },
                )
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
                "source": {
                    "backend": "synthetic",
                    "keypoint_format": "name-2d",
                    "coord_space": "inference_pixel",
                    "input_width": width,
                    "input_height": height,
                    "pad_x": 0.0,
                    "pad_y": 0.0,
                },
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
    info_callback: InfoCallback | None = None,
) -> dict[str, object]:
    _update_status(status_callback, "Loading video...", 5)
    _update_info(info_callback, {"stage": "Loading video"})
    video_meta = load_video_metadata(video_path)
    frame_count = int(video_meta["frame_count"])
    fps = float(video_meta["fps"])
    width = int(video_meta["width"])
    height = int(video_meta["height"])

    _update_status(status_callback, "Running inference...", 15)
    _update_info(info_callback, {"stage": "Running inference"})
    tracks = generate_synthetic_tracks(frame_count, fps, width, height, cancel_event, info_callback)
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


_SKELETON_CONNECTIONS = [
    ("nose", "left_eye"),
    ("nose", "right_eye"),
    ("left_eye", "left_shoulder"),
    ("right_eye", "right_shoulder"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("right_shoulder", "right_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("right_hip", "right_knee"),
    ("left_knee", "left_ankle"),
    ("right_knee", "right_ankle"),
]


def _normalize_keypoints(keypoints: list[object]) -> list[dict[str, float | str]]:
    """Normalize keypoints into name/x/y/c dicts.

    Coordinate convention:
      - keypoints are expected in *inference frame* pixel coords by default.
      - If coord_space == "normalized", values are normalized [0..1] relative to inference size.
      - All coordinates are converted to ORIGINAL frame pixel coords before drawing.
    """
    if not isinstance(keypoints, list):
        raise ValueError("keypoints must be a list.")
    if not keypoints:
        return []
    if all(isinstance(item, dict) for item in keypoints):
        normalized = []
        for idx, item in enumerate(keypoints):
            fallback = KEYPOINT_NAMES[idx] if idx < len(KEYPOINT_NAMES) else f"k{idx}"
            name = str(item.get("name") or fallback)
            x = float(item.get("x", 0.0))
            y = float(item.get("y", 0.0))
            c = float(item.get("c", 1.0))
            normalized.append({"name": name, "x": x, "y": y, "c": c})
        return normalized
    if all(isinstance(item, (int, float)) for item in keypoints):
        flat = [float(item) for item in keypoints]
        if len(flat) % 3 == 0:
            step = 3
        elif len(flat) % 2 == 0:
            step = 2
        else:
            raise ValueError("Flat keypoint list must be divisible by 2 or 3.")
        normalized = []
        for idx in range(0, len(flat), step):
            kp_index = idx // step
            name = KEYPOINT_NAMES[kp_index] if kp_index < len(KEYPOINT_NAMES) else f"k{kp_index}"
            x = flat[idx]
            y = flat[idx + 1]
            c = flat[idx + 2] if step == 3 else 1.0
            normalized.append({"name": name, "x": x, "y": y, "c": c})
        return normalized
    if all(isinstance(item, (list, tuple)) for item in keypoints):
        lengths = {len(item) for item in keypoints}
        if lengths - {2, 3}:
            raise ValueError("Keypoint rows must have length 2 or 3.")
        if len(lengths) != 1:
            raise ValueError("Keypoint rows must be consistently length 2 or 3.")
        row_len = next(iter(lengths))
        normalized = []
        for idx, row in enumerate(keypoints):
            name = KEYPOINT_NAMES[idx] if idx < len(KEYPOINT_NAMES) else f"k{idx}"
            x = float(row[0])
            y = float(row[1])
            c = float(row[2]) if row_len == 3 else 1.0
            normalized.append({"name": name, "x": x, "y": y, "c": c})
        return normalized
    raise ValueError("Unsupported keypoint format.")


def _build_transform(
    track: dict[str, object],
    frame_width: int,
    frame_height: int,
) -> dict[str, float | str]:
    source = track.get("source", {})
    if not isinstance(source, dict):
        source = {}
    coord_space = str(source.get("coord_space", "inference_pixel"))
    infer_w = int(source.get("input_width", frame_width) or frame_width)
    infer_h = int(source.get("input_height", frame_height) or frame_height)
    pad_x = float(source.get("pad_x", 0.0))
    pad_y = float(source.get("pad_y", 0.0))
    scale_x = frame_width / max(1.0, float(infer_w))
    scale_y = frame_height / max(1.0, float(infer_h))
    return {
        "coord_space": coord_space,
        "infer_w": float(infer_w),
        "infer_h": float(infer_h),
        "pad_x": pad_x,
        "pad_y": pad_y,
        "scale_x": scale_x,
        "scale_y": scale_y,
    }


def _convert_xy(x: float, y: float, transform: dict[str, float | str]) -> tuple[float, float]:
    coord_space = str(transform["coord_space"])
    infer_w = float(transform["infer_w"])
    infer_h = float(transform["infer_h"])
    pad_x = float(transform["pad_x"])
    pad_y = float(transform["pad_y"])
    scale_x = float(transform["scale_x"])
    scale_y = float(transform["scale_y"])
    if coord_space == "normalized":
        x *= infer_w
        y *= infer_h
        coord_space = "inference_pixel"
    if coord_space == "inference_pixel":
        x = (x - pad_x) * scale_x
        y = (y - pad_y) * scale_y
    return x, y


def _convert_keypoints(
    keypoints: list[object],
    transform: dict[str, float | str],
) -> dict[str, tuple[int, int, float]]:
    normalized = _normalize_keypoints(keypoints)
    mapped: dict[str, tuple[int, int, float]] = {}
    for kp in normalized:
        x, y = _convert_xy(float(kp["x"]), float(kp["y"]), transform)
        mapped[str(kp["name"])] = (int(round(x)), int(round(y)), float(kp["c"]))
    return mapped


def _convert_bbox(
    bbox: list[object] | tuple[object, object, object, object],
    transform: dict[str, float | str],
) -> tuple[int, int, int, int] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    x, y, w, h = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    x1, y1 = _convert_xy(x, y, transform)
    x2, y2 = _convert_xy(x + w, y + h, transform)
    return int(round(x1)), int(round(y1)), int(round(x2 - x1)), int(round(y2 - y1))


def _select_label_position(mapped: dict[str, tuple[int, int, float]]) -> tuple[int, int]:
    if "nose" in mapped:
        return mapped["nose"][0], mapped["nose"][1]
    if "left_shoulder" in mapped and "right_shoulder" in mapped:
        left = mapped["left_shoulder"]
        right = mapped["right_shoulder"]
        return int((left[0] + right[0]) / 2), int((left[1] + right[1]) / 2)
    if "left_hip" in mapped and "right_hip" in mapped:
        left = mapped["left_hip"]
        right = mapped["right_hip"]
        return int((left[0] + right[0]) / 2), int((left[1] + right[1]) / 2)
    if mapped:
        _, (x, y, _) = next(iter(mapped.items()))
        return x, y
    return 12, 12


def _select_primary_track(
    tracks: list[dict[str, object]],
    frame_width: int,
    frame_height: int,
) -> str | None:
    if not tracks:
        return None
    center_x = frame_width / 2.0
    center_y = frame_height / 2.0
    candidates: list[tuple[float, float, str]] = []
    for track in tracks:
        transform = _build_transform(track, frame_width, frame_height)
        frames = track.get("frames", [])
        if not isinstance(frames, list):
            continue
        areas = []
        distances = []
        for frame in frames:
            bbox = frame.get("bbox_xywh")
            converted = _convert_bbox(bbox, transform) if bbox is not None else None
            if not converted:
                continue
            x, y, w, h = (float(converted[0]), float(converted[1]), float(converted[2]), float(converted[3]))
            areas.append(max(0.0, w * h))
            distances.append(math.hypot((x + w / 2) - center_x, (y + h / 2) - center_y))
        if not areas:
            continue
        avg_area = sum(areas) / len(areas)
        avg_dist = sum(distances) / max(1, len(distances))
        track_id = str(track.get("track_id", ""))
        candidates.append((avg_area, avg_dist, track_id))
    if not candidates:
        return str(tracks[0].get("track_id", ""))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    return candidates[0][2]


def _draw_debug_corners(frame, frame_width: int, frame_height: int) -> None:
    if cv2 is None:
        return
    color = (0, 255, 255)
    size = 14
    points = [
        (0, 0, 1, 1),
        (frame_width - 1, 0, -1, 1),
        (0, frame_height - 1, 1, -1),
        (frame_width - 1, frame_height - 1, -1, -1),
    ]
    for x, y, dx, dy in points:
        end_x = min(frame_width - 1, max(0, x + dx * size))
        end_y = min(frame_height - 1, max(0, y + dy * size))
        cv2.line(frame, (x, y), (end_x, y), color, 2)
        cv2.line(frame, (x, y), (x, end_y), color, 2)


def _draw_bbox(frame, bbox_xywh: tuple[int, int, int, int], color: tuple[int, int, int]) -> None:
    if cv2 is None:
        return
    x, y, w, h = bbox_xywh
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)


def _draw_pose_overlay(frame, mapped: dict[str, tuple[int, int, float]], color: tuple[int, int, int]) -> None:
    if cv2 is None:
        return
    for a, b in _SKELETON_CONNECTIONS:
        if a in mapped and b in mapped:
            ax, ay, ac = mapped[a]
            bx, by, bc = mapped[b]
            if ac > 0.1 and bc > 0.1:
                cv2.line(frame, (ax, ay), (bx, by), color, 2)
    for _, (x, y, conf) in mapped.items():
        if conf > 0.1:
            cv2.circle(frame, (x, y), 3, color, -1)


def _draw_watermark(frame, frame_index: int, fps: float) -> None:
    if cv2 is None:
        return
    timestamp_ms = int(frame_index / max(1.0, fps) * 1000)
    watermark = f"FightingOverlay | frame {frame_index} | {timestamp_ms}ms"
    cv2.putText(
        frame,
        watermark,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def _draw_no_pose(frame) -> None:
    if cv2 is None:
        return
    cv2.putText(
        frame,
        "NO POSE",
        (10, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )


def export_overlay_video(
    video_path: Path,
    output_path: Path,
    tracks: list[dict[str, object]],
    fps: float,
    width: int,
    height: int,
    debug_overlay: bool = True,
    draw_all_tracks: bool = False,
    smoothing_alpha: float = 0.7,
    cancel_event: object | None = None,
    status_callback: StatusCallback | None = None,
    info_callback: InfoCallback | None = None,
) -> None:
    if cv2 is None:
        _update_status(status_callback, "Overlay export failed: OpenCV unavailable.", 90)
        raise RuntimeError("OpenCV is required to export overlay video.")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Unable to open video for overlay export.")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_map: dict[int, list[dict[str, object]]] = {}
    track_transforms: dict[str, dict[str, float | str]] = {}
    for track in tracks:
        track_id = str(track.get("track_id", "unknown"))
        transform = _build_transform(track, width, height)
        track_transforms[track_id] = transform
        logging.info(
            "Overlay mapping track=%s orig=(%s,%s) infer=(%s,%s) pad=(%.2f,%.2f) scale=(%.4f,%.4f) coord=%s",
            track_id,
            width,
            height,
            int(transform["infer_w"]),
            int(transform["infer_h"]),
            float(transform["pad_x"]),
            float(transform["pad_y"]),
            float(transform["scale_x"]),
            float(transform["scale_y"]),
            transform["coord_space"],
        )
        for frame in track.get("frames", []):
            frame_map.setdefault(int(frame["frame_index"]), []).append(
                {
                    "track_id": track_id,
                    "keypoints_2d": frame.get("keypoints_2d", []),
                    "bbox_xywh": frame.get("bbox_xywh"),
                    "timestamp_ms": frame.get("timestamp_ms", 0),
                }
            )

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
    frame_index = 0
    frames_with_pose = 0
    frames_with_multiple = 0
    start_time = time.time()
    _update_info(info_callback, {"stage": "Rendering overlay"})
    primary_track_id = _select_primary_track(tracks, width, height)
    smoothing_alpha = min(max(smoothing_alpha, 0.0), 1.0)
    smoothed_cache: dict[str, dict[str, tuple[float, float, float]]] = {}
    logged_keypoints = False
    while True:
        if cancel_event is not None and getattr(cancel_event, "is_set")():
            cap.release()
            writer.release()
            raise ProcessingCancelled("Processing cancelled")
        ret, frame = cap.read()
        if not ret:
            break
        annotated = frame.copy()
        frame_entries = frame_map.get(frame_index, [])
        pose_count = len(frame_entries)
        if pose_count >= 1:
            frames_with_pose += 1
        if pose_count >= 2:
            frames_with_multiple += 1
        _draw_watermark(annotated, frame_index, fps)
        if debug_overlay:
            _draw_debug_corners(annotated, width, height)
        if not frame_entries:
            _draw_no_pose(annotated)
        else:
            colors = [(0, 200, 255), (255, 200, 0), (180, 255, 120), (255, 120, 180)]
            for idx, entry in enumerate(frame_entries):
                track_id = str(entry.get("track_id", "track"))
                transform = track_transforms.get(track_id, _build_transform({"source": {}}, width, height))
                bbox = entry.get("bbox_xywh")
                converted_bbox = _convert_bbox(bbox, transform) if bbox is not None else None
                color = colors[idx % len(colors)]
                if converted_bbox:
                    _draw_bbox(annotated, converted_bbox, color)
                if not draw_all_tracks and primary_track_id and track_id != primary_track_id:
                    continue
                keypoints = entry.get("keypoints_2d", [])
                try:
                    mapped = _convert_keypoints(keypoints, transform)
                except ValueError as exc:
                    logging.warning("Skipping invalid keypoints for track %s: %s", track_id, exc)
                    continue
                if not logged_keypoints and frame_index == 0:
                    sample = list(mapped.items())[:3]
                    logging.info("First frame keypoints (converted) track=%s sample=%s", track_id, sample)
                    logged_keypoints = True
                smoothed_track = smoothed_cache.setdefault(track_id, {})
                smoothed_mapped: dict[str, tuple[int, int, float]] = {}
                for name, (x, y, conf) in mapped.items():
                    if name in smoothed_track:
                        prev_x, prev_y, prev_c = smoothed_track[name]
                        x = smoothing_alpha * x + (1 - smoothing_alpha) * prev_x
                        y = smoothing_alpha * y + (1 - smoothing_alpha) * prev_y
                        conf = smoothing_alpha * conf + (1 - smoothing_alpha) * prev_c
                    smoothed_track[name] = (x, y, conf)
                    smoothed_mapped[name] = (int(round(x)), int(round(y)), conf)
                _draw_pose_overlay(annotated, smoothed_mapped, color)
                label_x, label_y = _select_label_position(smoothed_mapped)
                cv2.putText(
                    annotated,
                    track_id,
                    (label_x + 6, max(12, label_y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )
        writer.write(annotated)
        frame_index += 1
        if frame_index % 5 == 0 or frame_index == total:
            progress = 35 + (frame_index / max(1, total)) * 50
            elapsed = max(0.001, time.time() - start_time)
            effective_fps = frame_index / elapsed
            _update_status(
                status_callback,
                f"Rendering overlay frame {frame_index}/{total}...",
                progress,
            )
            _update_info(
                info_callback,
                {
                    "frame_index": frame_index,
                    "total_frames": total,
                    "people": pose_count,
                    "effective_fps": effective_fps,
                },
            )
    cap.release()
    writer.release()
    model_backend = "unknown"
    if tracks:
        source = tracks[0].get("source", {})
        if isinstance(source, dict):
            model_backend = str(source.get("backend", model_backend))
    logging.info(
        "Overlay export stats: total_frames=%s frames_with_pose>=1=%s frames_with_pose>=2=%s input_path=%s output_path=%s model_backend=%s",
        frame_index,
        frames_with_pose,
        frames_with_multiple,
        video_path,
        output_path,
        model_backend,
    )
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
    info_callback: InfoCallback | None = None,
) -> Path:
    output_dir = get_outputs_root()
    payload = run_inference(video_path, options, cancel_event, status_callback, info_callback)

    if options.save_pose_json:
        _update_status(status_callback, "Writing JSON...", 80)
        _update_info(info_callback, {"stage": "Writing JSON"})
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
            debug_overlay=options.debug_overlay,
            draw_all_tracks=options.draw_all_tracks,
            smoothing_alpha=options.smoothing_alpha,
            cancel_event=cancel_event,
            status_callback=status_callback,
            info_callback=info_callback,
        )

    if video_path and options.save_thumbnails:
        save_thumbnails(video_path, output_dir / "thumbnails", cancel_event, status_callback)

    _update_status(status_callback, "Done.", 100)
    _update_info(info_callback, {"stage": "Done"})
    return pose_path


__all__ = [
    "ProcessingOptions",
    "ProcessingCancelled",
    "run_pipeline",
    "run_inference",
    "build_pose_payload",
]
