from __future__ import annotations

import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any

from core.paths import get_log_root
from core.pipeline import _build_transform, _convert_keypoints


def _center_of_mass(mapped: dict[str, tuple[int, int, float]]) -> tuple[float, float] | None:
    if not mapped:
        return None
    xs = [point[0] for point in mapped.values()]
    ys = [point[1] for point in mapped.values()]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def evaluate_pose_payload(payload: dict[str, Any]) -> dict[str, Any]:
    video = payload.get("video", {})
    width = int(video.get("width", 0) or 0)
    height = int(video.get("height", 0) or 0)
    total_frames = int(video.get("frame_count", 0) or 0)
    tracks = payload.get("tracks", [])
    total_keypoints = 0
    out_of_bounds = 0
    confidence_sum = 0.0
    confidence_count = 0
    track_lengths = []
    jitter_values = []
    invalid_tracks = []
    frame_presence: dict[int, int] = {}

    for track in tracks:
        frames = track.get("frames", [])
        if not isinstance(frames, list):
            continue
        try:
            transform = _build_transform(track, width, height)
        except ValueError as exc:
            invalid_tracks.append({"track_id": track.get("track_id"), "reason": str(exc)})
            continue
        track_lengths.append(len(frames))
        prev_center = None
        for frame in frames:
            keypoints = frame.get("keypoints_2d", [])
            frame_index = int(frame.get("frame_index", -1))
            if frame_index >= 0:
                frame_presence[frame_index] = frame_presence.get(frame_index, 0) + 1
            try:
                mapped, _, _, _, normalized = _convert_keypoints(keypoints, transform)
            except ValueError as exc:
                logging.warning("Skipping invalid keypoints during evaluation: %s", exc)
                continue
            for kp in normalized:
                confidence_sum += float(kp.get("c", 0.0))
                confidence_count += 1
            for _, (x, y, _) in mapped.items():
                total_keypoints += 1
                if x < 0 or y < 0 or x >= width or y >= height:
                    out_of_bounds += 1
            center = _center_of_mass(mapped)
            if center is not None and prev_center is not None:
                jitter_values.append(math.hypot(center[0] - prev_center[0], center[1] - prev_center[1]))
            if center is not None:
                prev_center = center

    avg_confidence = confidence_sum / confidence_count if confidence_count else 0.0
    avg_track_length = sum(track_lengths) / len(track_lengths) if track_lengths else 0.0
    jitter = sum(jitter_values) / len(jitter_values) if jitter_values else 0.0
    out_of_bounds_pct = (out_of_bounds / total_keypoints * 100.0) if total_keypoints else 0.0
    if total_frames <= 0 and frame_presence:
        total_frames = max(frame_presence.keys()) + 1
    frames_with_pose = sum(1 for count in frame_presence.values() if count >= 1)
    frames_with_multiple = sum(1 for count in frame_presence.values() if count >= 2)
    pose_ratio = frames_with_pose / total_frames if total_frames > 0 else 0.0

    return {
        "video": {
            "path": str(video.get("path", "")),
            "width": width,
            "height": height,
            "fps": float(video.get("fps", 0.0) or 0.0),
            "frame_count": total_frames,
        },
        "tracks": {
            "count": len(tracks),
            "invalid_tracks": invalid_tracks,
            "avg_track_length": avg_track_length,
        },
        "frames": {
            "total": total_frames,
            "with_pose": frames_with_pose,
            "with_multiple": frames_with_multiple,
            "pose_ratio": pose_ratio,
        },
        "keypoints": {
            "total": total_keypoints,
            "out_of_bounds_pct": out_of_bounds_pct,
            "avg_confidence": avg_confidence,
        },
        "jitter": {
            "center_of_mass_avg_px": jitter,
        },
    }


def write_evaluation_report(payload: dict[str, Any]) -> tuple[Path, Path, dict[str, Any]]:
    results = evaluate_pose_payload(payload)
    log_root = get_log_root()
    log_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = log_root / f"evaluation_{timestamp}.json"
    text_path = log_root / f"evaluation_{timestamp}.txt"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    summary = (
        "Evaluation Summary\n"
        f"- Tracks: {results['tracks']['count']}\n"
        f"- Avg track length: {results['tracks']['avg_track_length']:.1f} frames\n"
        f"- Frames with pose >=1: {results['frames']['with_pose']}/{results['frames']['total']}\n"
        f"- Frames with pose >=2: {results['frames']['with_multiple']}/{results['frames']['total']}\n"
        f"- Out-of-bounds keypoints: {results['keypoints']['out_of_bounds_pct']:.1f}%\n"
        f"- Avg keypoint confidence: {results['keypoints']['avg_confidence']:.2f}\n"
        f"- Center-of-mass jitter: {results['jitter']['center_of_mass_avg_px']:.2f}px\n"
    )
    text_path.write_text(summary, encoding="utf-8")
    logging.info("Evaluation written to %s", json_path)
    return json_path, text_path, results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate pose tracks quality metrics.")
    parser.add_argument("pose_json", type=Path, help="Path to pose_tracks.json")
    args = parser.parse_args()

    payload = json.loads(args.pose_json.read_text(encoding="utf-8"))
    json_path, text_path, results = write_evaluation_report(payload)
    print(f"Saved evaluation JSON: {json_path}")
    print(f"Saved evaluation summary: {text_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
