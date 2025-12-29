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
    tracking_backend: str = "Motion (fast)"
    manual_track_ids: list[str] | None = None


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


def _record_problem(callback: InfoCallback | None, code: str, message: str) -> None:
    logging.warning("%s: %s", code, message)
    _update_info(
        callback,
        {
            "problem": {
                "code": code,
                "message": message,
            }
        },
    )


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
                    "coord_space": "pixels_in_original",
                    "transform_kind": "NONE",
                    "infer_width": width,
                    "infer_height": height,
                    "resized_width": width,
                    "resized_height": height,
                    "pad_left": 0.0,
                    "pad_right": 0.0,
                    "pad_top": 0.0,
                    "pad_bottom": 0.0,
                    "crop_x": 0.0,
                    "crop_y": 0.0,
                    "crop_w": width,
                    "crop_h": height,
                },
                "frames": frames,
            }
        )
    return tracks


def _build_track_source(width: int, height: int, backend: str) -> dict[str, object]:
    return {
        "backend": backend,
        "keypoint_format": "name-2d",
        "coord_space": "pixels_in_original",
        "transform_kind": "NONE",
        "infer_width": width,
        "infer_height": height,
        "resized_width": width,
        "resized_height": height,
        "pad_left": 0.0,
        "pad_right": 0.0,
        "pad_top": 0.0,
        "pad_bottom": 0.0,
        "crop_x": 0.0,
        "crop_y": 0.0,
        "crop_w": width,
        "crop_h": height,
    }


def run_motion_tracking(
    video_path: Path,
    fps: float,
    width: int,
    height: int,
    cancel_event: object | None = None,
    info_callback: InfoCallback | None = None,
) -> list[dict[str, object]]:
    if cv2 is None:
        raise RuntimeError("OpenCV is required for motion tracking.")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Unable to open video for motion tracking.")
    subtractor = cv2.createBackgroundSubtractorMOG2(history=120, varThreshold=24, detectShadows=True)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    next_track_id = 1
    tracks: dict[str, dict[str, object]] = {}
    active: dict[str, dict[str, float]] = {}
    max_distance = max(60.0, min(width, height) * 0.08)
    frame_index = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    while True:
        if cancel_event is not None and getattr(cancel_event, "is_set")():
            cap.release()
            raise ProcessingCancelled("Processing cancelled")
        ret, frame = cap.read()
        if not ret:
            break
        mask = subtractor.apply(frame)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=2)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        contour_result = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contour_result[0] if len(contour_result) == 2 else contour_result[1]
        detections: list[tuple[int, int, int, int, float]] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < (width * height) * 0.003:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            confidence = min(1.0, area / max(1.0, (width * height) * 0.05))
            detections.append((x, y, w, h, confidence))
        matched: set[str] = set()
        for x, y, w, h, confidence in detections:
            cx = x + w / 2
            cy = y + h / 2
            best_id = None
            best_dist = max_distance
            for track_id, state in active.items():
                dx = cx - state["cx"]
                dy = cy - state["cy"]
                dist = math.hypot(dx, dy)
                if dist < best_dist:
                    best_dist = dist
                    best_id = track_id
            if best_id is None:
                track_id = f"t{next_track_id}"
                next_track_id += 1
                tracks[track_id] = {
                    "track_id": track_id,
                    "person_index": int(track_id[1:]) - 1,
                    "is_foreground": False,
                    "source": _build_track_source(width, height, "motion"),
                    "frames": [],
                }
            else:
                track_id = best_id
            active[track_id] = {"cx": cx, "cy": cy, "last_frame": float(frame_index)}
            matched.add(track_id)
            bbox = [float(x), float(y), float(w), float(h)]
            tracks[track_id]["frames"].append(
                {
                    "frame_index": frame_index,
                    "timestamp_ms": int(frame_index / fps * 1000),
                    "bbox_xywh": bbox,
                    "keypoints_2d": _make_keypoints(bbox),
                    "confidence": confidence,
                }
            )
        inactive = [track_id for track_id in active if track_id not in matched]
        for track_id in inactive:
            if frame_index - int(active[track_id]["last_frame"]) > int(fps):
                active.pop(track_id, None)
        if frame_index % 5 == 0 or frame_index == total_frames - 1:
            _update_info(
                info_callback,
                {
                    "stage": "Running inference",
                    "frame_index": frame_index + 1,
                    "total_frames": total_frames or frame_index + 1,
                    "people": len(detections),
                    "effective_fps": fps,
                },
            )
        frame_index += 1
    cap.release()
    return list(tracks.values())


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


def _average_track_confidence(frames: list[dict[str, object]]) -> float:
    confidences = [float(frame.get("confidence", 0.0)) for frame in frames if frame.get("confidence") is not None]
    return sum(confidences) / len(confidences) if confidences else 0.0


def _track_continuity(frames: list[dict[str, object]]) -> float:
    return float(len(frames))


def select_foreground_tracks(
    tracks: list[dict[str, object]],
    foreground_count: int = 2,
    mode: str = "Auto (closest/most active)",
    manual_track_ids: list[str] | None = None,
) -> None:
    for track in tracks:
        track["is_foreground"] = False
    if mode == "Manual pick":
        if manual_track_ids:
            manual_set = {track_id.strip() for track_id in manual_track_ids if track_id.strip()}
            for track in tracks:
                if str(track.get("track_id", "")) in manual_set:
                    track["is_foreground"] = True
        return
    if mode == "Foreground=Top2 largest":
        scored = []
        for track in tracks:
            frames = track.get("frames", [])
            areas = []
            for frame in frames:
                bbox = frame.get("bbox_xywh")
                if not bbox or len(bbox) != 4:
                    continue
                _, _, w, h = bbox
                areas.append(float(w) * float(h))
            avg_area = sum(areas) / len(areas) if areas else 0.0
            scored.append((avg_area, track))
    else:
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
            "frame_count": video_meta.get("frame_count"),
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
    duration_s = frame_count / max(1.0, fps)
    _update_info(
        info_callback,
        {
            "video_width": width,
            "video_height": height,
            "video_fps": fps,
            "video_frames": frame_count,
            "video_duration_s": duration_s,
        },
    )

    _update_status(status_callback, "Running inference...", 15)
    _update_info(info_callback, {"stage": "Running inference"})
    if options.tracking_backend == "Synthetic (demo)" or video_path is None:
        tracks = generate_synthetic_tracks(frame_count, fps, width, height, cancel_event, info_callback)
    else:
        tracks = run_motion_tracking(video_path, fps, width, height, cancel_event, info_callback)
    select_foreground_tracks(
        tracks,
        mode=options.foreground_mode,
        manual_track_ids=options.manual_track_ids,
    )

    if tracks:
        source = tracks[0].get("source", {})
        if isinstance(source, dict):
            _update_info(
                info_callback,
                {
                    "coord_space": source.get("coord_space"),
                    "transform_kind": source.get("transform_kind"),
                    "infer_width": source.get("infer_width"),
                    "infer_height": source.get("infer_height"),
                    "resized_width": source.get("resized_width"),
                    "resized_height": source.get("resized_height"),
                    "pad_left": source.get("pad_left"),
                    "pad_right": source.get("pad_right"),
                    "pad_top": source.get("pad_top"),
                    "pad_bottom": source.get("pad_bottom"),
                    "crop_x": source.get("crop_x"),
                    "crop_y": source.get("crop_y"),
                    "crop_w": source.get("crop_w"),
                    "crop_h": source.get("crop_h"),
                },
            )

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


_SKELETON_15_NAMES = KEYPOINT_NAMES
_SKELETON_15_CONNECTIONS = [
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

_COCO_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

_COCO_SKELETON_CONNECTIONS = [
    ("nose", "left_eye"),
    ("nose", "right_eye"),
    ("left_eye", "left_ear"),
    ("right_eye", "right_ear"),
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

_KEYPOINT_LAYOUTS = {
    "overlay15": {"count": len(_SKELETON_15_NAMES), "names": _SKELETON_15_NAMES, "skeleton": _SKELETON_15_CONNECTIONS},
    "coco17": {"count": len(_COCO_KEYPOINT_NAMES), "names": _COCO_KEYPOINT_NAMES, "skeleton": _COCO_SKELETON_CONNECTIONS},
}


def _inspect_keypoints(keypoints: list[object]) -> dict[str, object]:
    if not isinstance(keypoints, list):
        raise ValueError("keypoints must be a list.")
    if not keypoints:
        return {"count": 0, "stride": 0, "shape": "(0,)", "raw_samples": [], "names_in_data": []}
    if all(isinstance(item, dict) for item in keypoints):
        samples = []
        names = []
        for idx, item in enumerate(keypoints[:3]):
            samples.append(
                {
                    "idx": idx,
                    "name": item.get("name"),
                    "x": item.get("x"),
                    "y": item.get("y"),
                    "c": item.get("c", 1.0),
                }
            )
        for item in keypoints:
            if "name" in item and item.get("name") is not None:
                names.append(str(item.get("name")))
        return {
            "count": len(keypoints),
            "stride": 3,
            "shape": f"({len(keypoints)}, dict)",
            "raw_samples": samples,
            "names_in_data": names,
        }
    if all(isinstance(item, (int, float)) for item in keypoints):
        flat = [float(item) for item in keypoints]
        if len(flat) % 3 == 0:
            stride = 3
        elif len(flat) % 2 == 0:
            stride = 2
        else:
            raise ValueError("Flat keypoint list must be divisible by 2 or 3.")
        samples = []
        for idx in range(0, min(len(flat), stride * 3), stride):
            kp_index = idx // stride
            x = flat[idx]
            y = flat[idx + 1]
            c = flat[idx + 2] if stride == 3 else 1.0
            samples.append({"idx": kp_index, "x": x, "y": y, "c": c})
        return {
            "count": len(flat) // stride,
            "stride": stride,
            "shape": f"({len(flat)},)",
            "raw_samples": samples,
            "names_in_data": [],
        }
    if all(isinstance(item, (list, tuple)) for item in keypoints):
        lengths = {len(item) for item in keypoints}
        if lengths - {2, 3}:
            raise ValueError("Keypoint rows must have length 2 or 3.")
        if len(lengths) != 1:
            raise ValueError("Keypoint rows must be consistently length 2 or 3.")
        row_len = next(iter(lengths))
        samples = []
        for idx, row in enumerate(keypoints[:3]):
            c = float(row[2]) if row_len == 3 else 1.0
            samples.append({"idx": idx, "x": row[0], "y": row[1], "c": c})
        return {
            "count": len(keypoints),
            "stride": row_len,
            "shape": f"({len(keypoints)}, {row_len})",
            "raw_samples": samples,
            "names_in_data": [],
        }
    raise ValueError("Unsupported keypoint format.")


def _resolve_keypoint_layout(keypoint_info: dict[str, object]) -> tuple[str | None, list[str], list[tuple[str, str]]]:
    count = int(keypoint_info.get("count", 0))
    names_in_data = keypoint_info.get("names_in_data", [])
    if (
        count == _KEYPOINT_LAYOUTS["coco17"]["count"]
        and names_in_data
        and (
            names_in_data == _COCO_KEYPOINT_NAMES
            or set(names_in_data) == set(_COCO_KEYPOINT_NAMES)
        )
    ):
        layout = _KEYPOINT_LAYOUTS["coco17"]
        return "coco17", list(layout["names"]), list(layout["skeleton"])
    if count == _KEYPOINT_LAYOUTS["overlay15"]["count"]:
        if not names_in_data or (
            len(names_in_data) == len(_SKELETON_15_NAMES) and set(names_in_data) == set(_SKELETON_15_NAMES)
        ):
            layout = _KEYPOINT_LAYOUTS["overlay15"]
            return "overlay15", list(layout["names"]), list(layout["skeleton"])
    return None, [f"k{idx}" for idx in range(count)], []


def _normalize_keypoints(
    keypoints: list[object],
    names: list[str] | None = None,
) -> list[dict[str, float | str]]:
    """Normalize keypoints into name/x/y/c dicts.

    Coordinate convention:
      - keypoints are expected in the coord_space declared in track metadata.
      - normalized coordinates are normalized [0..1] relative to the declared space.
      - All coordinates are converted to ORIGINAL frame pixel coords before drawing.
    """
    if not isinstance(keypoints, list):
        raise ValueError("keypoints must be a list.")
    if not keypoints:
        return []
    if all(isinstance(item, dict) for item in keypoints):
        normalized = []
        for idx, item in enumerate(keypoints):
            fallback = names[idx] if names and idx < len(names) else f"k{idx}"
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
            name = names[kp_index] if names and kp_index < len(names) else f"k{kp_index}"
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
            name = names[idx] if names and idx < len(names) else f"k{idx}"
            x = float(row[0])
            y = float(row[1])
            c = float(row[2]) if row_len == 3 else 1.0
            normalized.append({"name": name, "x": x, "y": y, "c": c})
        return normalized
    raise ValueError("Unsupported keypoint format.")


_REQUIRED_SOURCE_FIELDS = (
    "coord_space",
    "transform_kind",
    "infer_width",
    "infer_height",
    "resized_width",
    "resized_height",
    "pad_left",
    "pad_right",
    "pad_top",
    "pad_bottom",
    "crop_x",
    "crop_y",
    "crop_w",
    "crop_h",
)


def _build_transform(
    track: dict[str, object],
    frame_width: int,
    frame_height: int,
    info_callback: InfoCallback | None = None,
) -> dict[str, float | str]:
    source = track.get("source", {})
    if not isinstance(source, dict):
        source = {}
    track_id = str(track.get("track_id", "unknown"))
    missing = [field for field in _REQUIRED_SOURCE_FIELDS if source.get(field) is None]
    if missing:
        message = f"Track {track_id} missing required mapping metadata: {', '.join(missing)}"
        _record_problem(info_callback, "MAPPING_METADATA_INCOMPLETE", message)
        raise ValueError(message)
    coord_space = str(source["coord_space"])
    transform_kind = str(source["transform_kind"]).upper()
    infer_w = int(source["infer_width"] or frame_width)
    infer_h = int(source["infer_height"] or frame_height)
    resized_w = int(source["resized_width"] or infer_w)
    resized_h = int(source["resized_height"] or infer_h)
    pad_left = float(source["pad_left"])
    pad_right = float(source["pad_right"])
    pad_top = float(source["pad_top"])
    pad_bottom = float(source["pad_bottom"])
    crop_x = float(source["crop_x"])
    crop_y = float(source["crop_y"])
    crop_w = float(source["crop_w"])
    crop_h = float(source["crop_h"])
    if transform_kind == "LETTERBOX":
        inferred_w = pad_left + float(resized_w) + pad_right
        inferred_h = pad_top + float(resized_h) + pad_bottom
        if abs(inferred_w - infer_w) > 1.0 or abs(inferred_h - infer_h) > 1.0:
            logging.warning(
                "LETTERBOX pad mismatch: infer=(%s,%s) pad+resized=(%.1f,%.1f)",
                infer_w,
                infer_h,
                inferred_w,
                inferred_h,
            )
    if transform_kind == "DIRECT_RESIZE" and any(value > 0.0 for value in (pad_left, pad_right, pad_top, pad_bottom)):
        logging.warning("DIRECT_RESIZE received non-zero padding values.")
    if transform_kind in {"CROP", "CENTER_CROP"} and (crop_w <= 0 or crop_h <= 0):
        logging.warning("CROP transform missing crop bounds (crop_w/crop_h).")
    return {
        "coord_space": coord_space,
        "transform_kind": transform_kind,
        "orig_w": float(frame_width),
        "orig_h": float(frame_height),
        "infer_w": float(infer_w),
        "infer_h": float(infer_h),
        "resized_w": float(resized_w),
        "resized_h": float(resized_h),
        "pad_left": pad_left,
        "pad_right": pad_right,
        "pad_top": pad_top,
        "pad_bottom": pad_bottom,
        "crop_x": crop_x,
        "crop_y": crop_y,
        "crop_w": crop_w,
        "crop_h": crop_h,
    }


def _convert_xy(x: float, y: float, transform: dict[str, float | str]) -> tuple[float, float]:
    """Map keypoints into original frame pixels using stored preprocess metadata.

    Worked example (LETTERBOX):
      - orig=1920x1080, infer=1280x1280, resized=1280x720, pad_top=280, pad_bottom=280
      - keypoint in infer canvas at (640,640) -> resized (640,360)
      - scale=1280/1920=0.6667 -> original (960,540)
    """
    coord_space = str(transform["coord_space"])
    transform_kind = str(transform["transform_kind"]).upper()
    infer_w = float(transform["infer_w"])
    infer_h = float(transform["infer_h"])
    resized_w = float(transform["resized_w"])
    resized_h = float(transform["resized_h"])
    pad_left = float(transform["pad_left"])
    pad_top = float(transform["pad_top"])
    crop_x = float(transform["crop_x"])
    crop_y = float(transform["crop_y"])
    crop_w = float(transform["crop_w"])
    crop_h = float(transform["crop_h"])

    if coord_space == "pixels_in_original":
        return x, y

    if coord_space in {"normalized", "normalized_to_infer_canvas"}:
        x = x * infer_w
        y = y * infer_h
        coord_space = "pixels_in_infer_canvas"
    elif coord_space == "normalized_to_resized_content":
        x = x * resized_w
        y = y * resized_h
        coord_space = "pixels_in_resized_content"

    if coord_space in {"inference_pixel", "pixels_in_infer_canvas"}:
        x = x - pad_left
        y = y - pad_top
        coord_space = "pixels_in_resized_content"

    if coord_space != "pixels_in_resized_content":
        logging.warning("Unexpected coord_space=%s; returning raw values.", coord_space)
        return x, y

    if transform_kind == "DIRECT_RESIZE":
        scale_x = resized_w / max(1.0, float(transform.get("orig_w", resized_w)))
        scale_y = resized_h / max(1.0, float(transform.get("orig_h", resized_h)))
        return x / max(1e-6, scale_x), y / max(1e-6, scale_y)
    if transform_kind == "LETTERBOX":
        scale_x = resized_w / max(1.0, float(transform.get("orig_w", resized_w)))
        scale_y = resized_h / max(1.0, float(transform.get("orig_h", resized_h)))
        if abs(scale_x - scale_y) > 0.001:
            logging.warning("LETTERBOX scale mismatch: scale_x=%.6f scale_y=%.6f", scale_x, scale_y)
        scale = scale_x if scale_x > 0 else 1.0
        return x / max(1e-6, scale), y / max(1e-6, scale)
    if transform_kind in {"CROP", "CENTER_CROP"}:
        scale_x = crop_w / max(1.0, resized_w)
        scale_y = crop_h / max(1.0, resized_h)
        x_crop = x * scale_x
        y_crop = y * scale_y
        return x_crop + crop_x, y_crop + crop_y
    if transform_kind == "NONE":
        return x, y
    logging.warning("Unknown transform_kind=%s; returning raw values.", transform_kind)
    return x, y


def _convert_keypoints(
    keypoints: list[object],
    transform: dict[str, float | str],
) -> tuple[dict[str, tuple[int, int, float]], dict[str, object], str | None, list[tuple[str, str]], list[dict[str, float | str]]]:
    keypoint_info = _inspect_keypoints(keypoints)
    layout_name, layout_names, skeleton = _resolve_keypoint_layout(keypoint_info)
    normalized = _normalize_keypoints(keypoints, layout_names)
    mapped: dict[str, tuple[int, int, float]] = {}
    for kp in normalized:
        x, y = _convert_xy(float(kp["x"]), float(kp["y"]), transform)
        mapped[str(kp["name"])] = (int(round(x)), int(round(y)), float(kp["c"]))
    return mapped, keypoint_info, layout_name, skeleton, normalized


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
        try:
            transform = _build_transform(track, frame_width, frame_height)
        except ValueError:
            continue
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
        (0, 0, 1, 1, "(0,0)"),
        (frame_width - 1, 0, -1, 1, f"({frame_width - 1},0)"),
        (0, frame_height - 1, 1, -1, f"(0,{frame_height - 1})"),
        (frame_width - 1, frame_height - 1, -1, -1, f"({frame_width - 1},{frame_height - 1})"),
    ]
    for x, y, dx, dy, label in points:
        end_x = min(frame_width - 1, max(0, x + dx * size))
        end_y = min(frame_height - 1, max(0, y + dy * size))
        cv2.line(frame, (x, y), (end_x, y), color, 2)
        cv2.line(frame, (x, y), (x, end_y), color, 2)
        text_x = min(frame_width - 1, max(0, x + dx * (size + 4)))
        text_y = min(frame_height - 1, max(0, y + dy * (size + 4)))
        cv2.putText(
            frame,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )


def _convert_xy_from_space(
    x: float,
    y: float,
    coord_space: str,
    transform: dict[str, float | str],
) -> tuple[float, float]:
    local = dict(transform)
    local["coord_space"] = coord_space
    return _convert_xy(x, y, local)


def _draw_debug_regions(frame, transform: dict[str, float | str], frame_width: int, frame_height: int) -> None:
    if cv2 is None:
        return
    frame_color = (0, 255, 0)
    content_color = (255, 0, 255)
    cv2.rectangle(frame, (0, 0), (frame_width - 1, frame_height - 1), frame_color, 2)

    transform_kind = str(transform["transform_kind"]).upper()
    if transform_kind in {"DIRECT_RESIZE", "NONE"}:
        cv2.rectangle(frame, (0, 0), (frame_width - 1, frame_height - 1), content_color, 2)
        return
    if transform_kind in {"CROP", "CENTER_CROP"}:
        crop_x = int(round(float(transform["crop_x"])))
        crop_y = int(round(float(transform["crop_y"])))
        crop_w = int(round(float(transform["crop_w"])))
        crop_h = int(round(float(transform["crop_h"])))
        cv2.rectangle(frame, (crop_x, crop_y), (crop_x + crop_w, crop_y + crop_h), content_color, 2)
        return
    if transform_kind == "LETTERBOX":
        pad_left = float(transform["pad_left"])
        pad_top = float(transform["pad_top"])
        resized_w = float(transform["resized_w"])
        resized_h = float(transform["resized_h"])
        x1, y1 = _convert_xy_from_space(pad_left, pad_top, "pixels_in_infer_canvas", transform)
        x2, y2 = _convert_xy_from_space(
            pad_left + resized_w,
            pad_top + resized_h,
            "pixels_in_infer_canvas",
            transform,
        )
        cv2.rectangle(
            frame,
            (int(round(x1)), int(round(y1))),
            (int(round(x2)), int(round(y2))),
            content_color,
            2,
        )


def _draw_keypoint_labels(
    frame,
    mapped: dict[str, tuple[int, int, float]],
    color: tuple[int, int, int],
    max_labels: int = 5,
) -> None:
    if cv2 is None or not mapped:
        return
    preferred = ["nose", "left_hip", "right_hip", "left_shoulder", "right_shoulder"]
    drawn = 0
    for name in preferred:
        if name in mapped and drawn < max_labels:
            x, y, _ = mapped[name]
            cv2.putText(
                frame,
                f"{name} ({x},{y})",
                (x + 6, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
            drawn += 1
    if drawn >= max_labels:
        return
    for name, (x, y, _) in mapped.items():
        if name in preferred:
            continue
        cv2.putText(
            frame,
            f"{name} ({x},{y})",
            (x + 6, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
        drawn += 1
        if drawn >= max_labels:
            break


def _draw_bbox(frame, bbox_xywh: tuple[int, int, int, int], color: tuple[int, int, int]) -> None:
    if cv2 is None:
        return
    x, y, w, h = bbox_xywh
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)


def _draw_pose_overlay(
    frame,
    mapped: dict[str, tuple[int, int, float]],
    color: tuple[int, int, int],
    skeleton_connections: list[tuple[str, str]],
    min_confidence: float = 0.1,
    draw_lines: bool = True,
) -> None:
    if cv2 is None:
        return
    if draw_lines:
        for a, b in skeleton_connections:
            if a in mapped and b in mapped:
                ax, ay, ac = mapped[a]
                bx, by, bc = mapped[b]
                if ac >= min_confidence and bc >= min_confidence:
                    cv2.line(frame, (ax, ay), (bx, by), color, 2)
    for _, (x, y, conf) in mapped.items():
        if conf >= min_confidence:
            cv2.circle(frame, (x, y), 3, color, -1)


def _center_of_mass(mapped: dict[str, tuple[int, int, float]], min_confidence: float) -> tuple[int, int] | None:
    if not mapped:
        return None
    xs = [x for x, _, conf in mapped.values() if conf >= min_confidence]
    ys = [y for _, y, conf in mapped.values() if conf >= min_confidence]
    if not xs or not ys:
        return None
    return int(round(sum(xs) / len(xs))), int(round(sum(ys) / len(ys)))


def _draw_balance_overlay(
    frame,
    mapped: dict[str, tuple[int, int, float]],
    min_confidence: float,
    color: tuple[int, int, int],
) -> None:
    if cv2 is None:
        return
    com = _center_of_mass(mapped, min_confidence)
    left_ankle = mapped.get("left_ankle")
    right_ankle = mapped.get("right_ankle")
    if left_ankle and left_ankle[2] >= min_confidence and right_ankle and right_ankle[2] >= min_confidence:
        cv2.line(frame, (left_ankle[0], left_ankle[1]), (right_ankle[0], right_ankle[1]), color, 2)
        cv2.circle(frame, (left_ankle[0], left_ankle[1]), 4, color, -1)
        cv2.circle(frame, (right_ankle[0], right_ankle[1]), 4, color, -1)
    elif left_ankle and left_ankle[2] >= min_confidence:
        cv2.circle(frame, (left_ankle[0], left_ankle[1]), 4, color, -1)
    elif right_ankle and right_ankle[2] >= min_confidence:
        cv2.circle(frame, (right_ankle[0], right_ankle[1]), 4, color, -1)
    if com:
        cv2.circle(frame, com, 6, (0, 255, 0), -1)
        cv2.putText(
            frame,
            "COM",
            (com[0] + 6, com[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )


def _count_joints_above(normalized: list[dict[str, float | str]], min_confidence: float) -> int:
    return sum(1 for kp in normalized if float(kp.get("c", 0.0)) >= min_confidence)


def _average_confidence(normalized: list[dict[str, float | str]]) -> float:
    confidences = [float(kp.get("c", 0.0)) for kp in normalized]
    return sum(confidences) / len(confidences) if confidences else 0.0


def _score_primary_candidate(
    joints_above: int,
    avg_conf: float,
    bbox_area: float,
    continuity: float,
) -> float:
    return joints_above * 3.0 + avg_conf * 5.0 + bbox_area * 0.001 + continuity * 0.1


def _similar_bbox(area_a: float | None, area_b: float | None) -> bool:
    if not area_a or not area_b:
        return False
    ratio = area_a / area_b if area_b > 0 else 0.0
    return 0.6 <= ratio <= 1.4
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
    smoothing_enabled: bool = True,
    min_keypoint_confidence: float = 0.1,
    overlay_mode: str = "skeleton",
    max_tracks: int = 3,
    track_sort: str = "confidence",
    live_preview: bool = True,
    draw_bboxes: bool = False,
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
    warned_frame_sync: set[str] = set()
    valid_tracks: list[dict[str, object]] = []
    for track in tracks:
        track_id = str(track.get("track_id", "unknown"))
        try:
            transform = _build_transform(track, width, height, info_callback)
        except ValueError:
            continue
        track_transforms[track_id] = transform
        valid_tracks.append(track)
        logging.info(
            (
                "Overlay mapping track=%s orig=(%s,%s) infer=(%s,%s) resized=(%s,%s) "
                "pad=(l=%.2f,r=%.2f,t=%.2f,b=%.2f) crop=(%.1f,%.1f,%.1f,%.1f) "
                "transform=%s coord=%s"
            ),
            track_id,
            width,
            height,
            int(transform["infer_w"]),
            int(transform["infer_h"]),
            int(transform["resized_w"]),
            int(transform["resized_h"]),
            float(transform["pad_left"]),
            float(transform["pad_right"]),
            float(transform["pad_top"]),
            float(transform["pad_bottom"]),
            float(transform["crop_x"]),
            float(transform["crop_y"]),
            float(transform["crop_w"]),
            float(transform["crop_h"]),
            transform["transform_kind"],
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
    primary_track_id = None
    smoothing_alpha = min(max(smoothing_alpha, 0.0), 1.0)
    smoothed_cache: dict[str, dict[str, tuple[float, float, float]]] = {}
    logged_keypoints = False
    logged_mapping_warning = False
    overlay_mode_normalized = overlay_mode.strip().lower()
    mode_debug = "debug" in overlay_mode_normalized
    mode_balance = "balance" in overlay_mode_normalized
    mode_joints = "joints" in overlay_mode_normalized or "dots" in overlay_mode_normalized
    mode_skeleton = "skeleton" in overlay_mode_normalized or "lines" in overlay_mode_normalized
    draw_lines = mode_skeleton or mode_debug
    draw_joints = mode_skeleton or mode_joints or mode_debug
    draw_balance = mode_balance
    show_labels = mode_debug
    show_debug_regions = mode_debug
    draw_bboxes = draw_bboxes or mode_debug
    if mode_debug:
        logging.info("Debug overlay enabled: capture a screenshot to validate mapping primitives.")

    max_tracks = max(1, min(int(max_tracks), 50))
    track_stats: dict[str, dict[str, float]] = {}
    for track in valid_tracks:
        track_id = str(track.get("track_id", ""))
        frames = track.get("frames", [])
        if not isinstance(frames, list) or not frames:
            continue
        joints_above_total = 0.0
        avg_conf_total = 0.0
        bbox_area_total = 0.0
        frame_count = 0.0
        transform = track_transforms.get(track_id)
        for frame in frames:
            keypoints = frame.get("keypoints_2d", [])
            try:
                keypoint_info = _inspect_keypoints(keypoints)
                _, layout_names, _ = _resolve_keypoint_layout(keypoint_info)
                normalized = _normalize_keypoints(keypoints, layout_names)
            except ValueError:
                continue
            joints_above_total += _count_joints_above(normalized, min_keypoint_confidence)
            avg_conf_total += _average_confidence(normalized)
            bbox = frame.get("bbox_xywh")
            if bbox is not None and transform is not None:
                converted = _convert_bbox(bbox, transform)
                if converted:
                    _, _, w, h = converted
                    bbox_area_total += max(0.0, float(w) * float(h))
            frame_count += 1.0
        if frame_count <= 0:
            continue
        track_stats[track_id] = {
            "avg_joints_above": joints_above_total / frame_count,
            "avg_conf": avg_conf_total / frame_count,
            "avg_area": bbox_area_total / frame_count,
            "continuity": float(len(frames)),
        }

    if track_stats:
        best_score = None
        for track_id, stats in track_stats.items():
            score = _score_primary_candidate(
                stats.get("avg_joints_above", 0.0),
                stats.get("avg_conf", 0.0),
                stats.get("avg_area", 0.0),
                stats.get("continuity", 0.0),
            )
            if best_score is None or score > best_score:
                best_score = score
                primary_track_id = track_id

    debug_transform = None
    if primary_track_id and primary_track_id in track_transforms:
        debug_transform = track_transforms[primary_track_id]
    elif track_transforms:
        debug_transform = next(iter(track_transforms.values()))
    else:
        message = "No valid tracks available for overlay (missing mapping metadata)."
        _record_problem(info_callback, "MAPPING_METADATA_INCOMPLETE", message)
        debug_transform = _build_transform(
            {
                "track_id": "debug",
                "source": {
                    "coord_space": "pixels_in_original",
                    "transform_kind": "NONE",
                    "infer_width": width,
                    "infer_height": height,
                    "resized_width": width,
                    "resized_height": height,
                    "pad_left": 0.0,
                    "pad_right": 0.0,
                    "pad_top": 0.0,
                    "pad_bottom": 0.0,
                    "crop_x": 0.0,
                    "crop_y": 0.0,
                    "crop_w": width,
                    "crop_h": height,
                },
            },
            width,
            height,
        )

    ranked_tracks: list[str] = []
    if draw_all_tracks:
        scored: list[tuple[float, str]] = []
        for track_id, stats in track_stats.items():
            score = _score_primary_candidate(
                stats.get("avg_joints_above", 0.0),
                stats.get("avg_conf", 0.0),
                stats.get("avg_area", 0.0),
                stats.get("continuity", 0.0),
            )
            scored.append((score, track_id))
        scored.sort(key=lambda item: item[0], reverse=True)
        ranked_tracks = [track_id for _, track_id in scored]

    allowed_track_ids: set[str] = set()
    sticky_gap = 10
    last_primary_id: str | None = primary_track_id
    last_primary_seen = -1
    last_primary_com: tuple[int, int] | None = None
    last_primary_area: float | None = None
    last_preview_time = 0.0
    preview_interval_s = 0.15
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
        if show_debug_regions:
            _draw_debug_regions(annotated, debug_transform, width, height)
            _draw_debug_corners(annotated, width, height)
        if not frame_entries:
            _draw_no_pose(annotated)
        else:
            colors = [(0, 200, 255), (255, 200, 0), (180, 255, 120), (255, 120, 180)]
            frame_candidates: list[dict[str, object]] = []
            for entry in frame_entries:
                track_id = str(entry.get("track_id", "track"))
                transform = track_transforms.get(track_id)
                if transform is None:
                    continue
                timestamp_ms = float(entry.get("timestamp_ms", 0.0))
                expected_ts = frame_index / max(1.0, fps) * 1000.0
                if track_id not in warned_frame_sync and abs(timestamp_ms - expected_ts) > (1000.0 / max(1.0, fps)) * 1.5:
                    message = (
                        f"Frame sync mismatch track={track_id} frame={frame_index} "
                        f"timestamp_ms={timestamp_ms:.1f} expected={expected_ts:.1f}"
                    )
                    _record_problem(info_callback, "FRAME_SYNC_DRIFT", message)
                    warned_frame_sync.add(track_id)
                bbox = entry.get("bbox_xywh")
                converted_bbox = _convert_bbox(bbox, transform) if bbox is not None else None
                keypoints = entry.get("keypoints_2d", [])
                try:
                    mapped, keypoint_info, layout_name, skeleton, normalized = _convert_keypoints(keypoints, transform)
                except ValueError as exc:
                    _record_problem(info_callback, "KEYPOINT_SHAPE_MISMATCH", f"Track {track_id}: {exc}")
                    continue
                if layout_name is None and not logged_keypoints:
                    message = (
                        "Unknown keypoint layout "
                        f"K={keypoint_info.get('count')} shape={keypoint_info.get('shape')} "
                        f"stride={keypoint_info.get('stride')}; drawing dots only."
                    )
                    _record_problem(info_callback, "KEYPOINT_LAYOUT_UNKNOWN", message)
                if mapped:
                    total_kps = len(mapped)
                    out_of_bounds = sum(
                        1
                        for _, (x, y, _) in mapped.items()
                        if x < 0 or y < 0 or x >= width or y >= height
                    )
                    if total_kps > 0 and out_of_bounds / total_kps >= 0.6 and not logged_mapping_warning:
                        message = (
                            f"MAPPING BROKEN: {out_of_bounds}/{total_kps} keypoints out of bounds "
                            f"on frame {frame_index} (track={track_id})"
                        )
                        _record_problem(info_callback, "MAPPING_BROKEN", message)
                        _update_info(info_callback, {"mapping_warning": True})
                        logged_mapping_warning = True
                if not logged_keypoints and frame_index == 0:
                    raw_samples = []
                    for idx, kp in enumerate(normalized[:3]):
                        name = kp.get("name", f"k{idx}")
                        mapped_sample = mapped.get(str(name))
                        raw_samples.append(
                            {
                                "idx": idx,
                                "name": name,
                                "raw": (kp.get("x"), kp.get("y"), kp.get("c")),
                                "mapped": mapped_sample,
                            }
                        )
                    logging.info(
                        (
                            "Diagnostic keypoints frame=%s track=%s K=%s shape=%s stride=%s "
                            "layout=%s raw_mapped_samples=%s"
                        ),
                        frame_index,
                        track_id,
                        keypoint_info.get("count"),
                        keypoint_info.get("shape"),
                        keypoint_info.get("stride"),
                        layout_name,
                        raw_samples,
                    )
                    logged_keypoints = True
                joints_above = _count_joints_above(normalized, min_keypoint_confidence)
                avg_conf = _average_confidence(normalized)
                bbox_area = 0.0
                if converted_bbox:
                    _, _, w, h = converted_bbox
                    bbox_area = max(0.0, float(w) * float(h))
                frame_candidates.append(
                    {
                        "track_id": track_id,
                        "mapped": mapped,
                        "layout_name": layout_name,
                        "skeleton": skeleton,
                        "normalized": normalized,
                        "bbox": converted_bbox,
                        "joints_above": joints_above,
                        "avg_conf": avg_conf,
                        "bbox_area": bbox_area,
                        "com": _center_of_mass(mapped, min_keypoint_confidence),
                    }
                )

            current_primary_id = None
            current_primary_com: tuple[int, int] | None = None
            current_primary_area: float | None = None
            if last_primary_id:
                match = next(
                    (candidate for candidate in frame_candidates if candidate["track_id"] == last_primary_id),
                    None,
                )
                if match:
                    current_primary_id = last_primary_id
                    current_primary_com = match.get("com")
                    current_primary_area = float(match.get("bbox_area", 0.0) or 0.0)
            if (
                current_primary_id is None
                and last_primary_id
                and last_primary_seen >= 0
                and frame_index - last_primary_seen <= sticky_gap
                and last_primary_com
            ):
                nearest = None
                nearest_dist = None
                for candidate in frame_candidates:
                    candidate_com = candidate.get("com")
                    candidate_area = float(candidate.get("bbox_area", 0.0) or 0.0)
                    if candidate_com is None or not _similar_bbox(last_primary_area, candidate_area):
                        continue
                    dist = math.hypot(candidate_com[0] - last_primary_com[0], candidate_com[1] - last_primary_com[1])
                    if nearest_dist is None or dist < nearest_dist:
                        nearest_dist = dist
                        nearest = candidate
                if nearest:
                    current_primary_id = str(nearest.get("track_id", ""))
                    current_primary_com = nearest.get("com")
                    current_primary_area = float(nearest.get("bbox_area", 0.0) or 0.0)
            if current_primary_id is None and frame_candidates:
                best_score = None
                best_candidate = None
                for candidate in frame_candidates:
                    track_id = str(candidate.get("track_id", ""))
                    stats = track_stats.get(track_id, {})
                    score = _score_primary_candidate(
                        float(candidate.get("joints_above", 0.0)),
                        float(candidate.get("avg_conf", 0.0)),
                        float(candidate.get("bbox_area", 0.0)),
                        float(stats.get("continuity", 0.0)),
                    )
                    if best_score is None or score > best_score:
                        best_score = score
                        best_candidate = candidate
                if best_candidate:
                    current_primary_id = str(best_candidate.get("track_id", ""))
                    current_primary_com = best_candidate.get("com")
                    current_primary_area = float(best_candidate.get("bbox_area", 0.0) or 0.0)

            if current_primary_id:
                last_primary_id = current_primary_id
                last_primary_seen = frame_index
                last_primary_com = current_primary_com
                last_primary_area = current_primary_area

            if draw_all_tracks and ranked_tracks:
                limit = max_tracks if max_tracks > 0 else len(ranked_tracks)
                allowed_track_ids = set(ranked_tracks[:limit])
                if current_primary_id:
                    allowed_track_ids.add(current_primary_id)
            elif current_primary_id:
                allowed_track_ids = {current_primary_id}
            else:
                allowed_track_ids = set()

            for idx, candidate in enumerate(frame_candidates):
                track_id = str(candidate.get("track_id", "track"))
                if allowed_track_ids and track_id not in allowed_track_ids:
                    continue
                color = colors[idx % len(colors)]
                mapped = candidate["mapped"]
                layout_name = candidate["layout_name"]
                skeleton = candidate["skeleton"]
                smoothed_track = smoothed_cache.setdefault(track_id, {})
                smoothed_mapped: dict[str, tuple[int, int, float]] = {}
                if smoothing_enabled:
                    for name, (x, y, conf) in mapped.items():
                        if conf < min_keypoint_confidence:
                            continue
                        if name in smoothed_track:
                            prev_x, prev_y, prev_c = smoothed_track[name]
                            x = smoothing_alpha * x + (1 - smoothing_alpha) * prev_x
                            y = smoothing_alpha * y + (1 - smoothing_alpha) * prev_y
                            conf = smoothing_alpha * conf + (1 - smoothing_alpha) * prev_c
                        smoothed_track[name] = (x, y, conf)
                        smoothed_mapped[name] = (int(round(x)), int(round(y)), conf)
                else:
                    for name, (x, y, conf) in mapped.items():
                        if conf < min_keypoint_confidence:
                            continue
                        smoothed_mapped[name] = (int(round(x)), int(round(y)), conf)

                if draw_bboxes:
                    bbox = candidate.get("bbox")
                    if bbox:
                        _draw_bbox(annotated, bbox, color)
                if show_labels:
                    _draw_keypoint_labels(annotated, mapped, color)
                if draw_balance:
                    _draw_balance_overlay(annotated, smoothed_mapped, min_keypoint_confidence, color)
                if draw_joints:
                    _draw_pose_overlay(
                        annotated,
                        smoothed_mapped,
                        color,
                        skeleton,
                        min_keypoint_confidence,
                        draw_lines=layout_name is not None and draw_lines,
                    )
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
        if live_preview and info_callback is not None:
            now = time.time()
            if now - last_preview_time >= preview_interval_s:
                preview_frame = annotated
                max_width = 640
                if width > max_width:
                    scale = max_width / float(width)
                    preview_frame = cv2.resize(
                        preview_frame,
                        (int(round(width * scale)), int(round(height * scale))),
                        interpolation=cv2.INTER_AREA,
                    )
                ok, buffer = cv2.imencode(".png", preview_frame)
                if ok:
                    _update_info(
                        info_callback,
                        {
                            "preview_image": buffer.tobytes(),
                            "preview_frame_index": frame_index + 1,
                            "preview_total_frames": total,
                        },
                    )
                last_preview_time = now
        writer.write(annotated)
        frame_index += 1
        if frame_index % 5 == 0 or frame_index == total:
            progress = 35 + (frame_index / max(1, total)) * 50
            elapsed = max(0.001, time.time() - start_time)
            effective_fps = frame_index / elapsed
            realtime_ratio = effective_fps / max(1.0, fps)
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
                    "realtime_ratio": realtime_ratio,
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
        overlay_variants = [
            ("overlay_skeleton.mp4", "skeleton", False),
            ("overlay_joints.mp4", "joints", False),
            ("overlay_balance.mp4", "balance", False),
        ]
        if options.debug_overlay:
            overlay_variants.append(("overlay_debug.mp4", "debug", True))
        preview_mode = options.overlay_mode
        known_modes = {variant[1] for variant in overlay_variants}
        if preview_mode not in known_modes:
            preview_mode = "skeleton"
        for filename, mode, draw_bboxes in overlay_variants:
            export_overlay_video(
                video_path,
                output_dir / filename,
                payload["tracks"],
                float(payload["video"]["fps"]),
                int(payload["video"]["width"]),
                int(payload["video"]["height"]),
                debug_overlay=options.debug_overlay,
                draw_all_tracks=options.draw_all_tracks,
                smoothing_alpha=options.smoothing_alpha,
                smoothing_enabled=options.smoothing_enabled,
                min_keypoint_confidence=options.min_keypoint_confidence,
                overlay_mode=mode,
                max_tracks=options.max_tracks,
                track_sort=options.track_sort,
                live_preview=options.live_preview and mode == preview_mode,
                draw_bboxes=draw_bboxes,
                cancel_event=cancel_event,
                status_callback=status_callback,
                info_callback=info_callback,
            )
        skeleton_path = output_dir / "overlay_skeleton.mp4"
        legacy_path = output_dir / "overlay.mp4"
        if skeleton_path.exists():
            try:
                import shutil

                shutil.copy2(skeleton_path, legacy_path)
            except OSError as exc:
                logging.warning("Failed to write legacy overlay.mp4: %s", exc)

    if video_path and options.save_thumbnails:
        save_thumbnails(video_path, output_dir / "thumbnails", cancel_event, status_callback)

    if options.run_evaluation:
        from core.evaluation import write_evaluation_report

        _update_status(status_callback, "Evaluating tracking quality...", 95)
        _update_info(info_callback, {"stage": "Evaluating tracks"})
        json_path, text_path, results = write_evaluation_report(payload)
        summary = (
            f"Tracks={results['tracks']['count']} "
            f"AvgLen={results['tracks']['avg_track_length']:.1f} "
            f"Frames>=1={results['frames']['with_pose']}/{results['frames']['total']} "
            f"Frames>=2={results['frames']['with_multiple']}/{results['frames']['total']} "
            f"Conf={results['keypoints']['avg_confidence']:.2f} "
            f"COMJitter={results['jitter']['center_of_mass_avg_px']:.1f}px"
        )
        _update_info(
            info_callback,
            {
                "evaluation_summary": summary,
                "evaluation_json": str(json_path),
                "evaluation_text": str(text_path),
                "evaluation_stats": results,
            },
        )
        pose_ratio = float(results.get("frames", {}).get("pose_ratio", 0.0) or 0.0)
        if pose_ratio < 0.4:
            _update_info(
                info_callback,
                {
                    "evaluation_warning": (
                        "Low pose coverage detected. Try lowering the confidence threshold "
                        "or enable diagnostic mode for troubleshooting."
                    )
                },
            )

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
