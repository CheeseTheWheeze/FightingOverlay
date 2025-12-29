from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from fightai.judge.events import JudgeEvent
from fightai.judge.metrics import FrameMetrics


def write_metrics_csv(output_path: Path, metrics: Iterable[FrameMetrics]) -> None:
    import csv

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame_index",
                "track_id",
                "in_mask_ratio",
                "mean_distance_to_mask_px",
                "com_x",
                "com_y",
                "com_jump_px",
                "limb_length_px",
                "limb_length_ratio",
                "confident_joints",
                "total_joints",
                "swap_candidate",
            ]
        )
        for entry in metrics:
            writer.writerow(
                [
                    entry.frame_index,
                    entry.track_id,
                    f"{entry.in_mask_ratio:.4f}",
                    f"{entry.mean_distance_to_mask_px:.2f}",
                    "" if entry.com_x is None else f"{entry.com_x:.2f}",
                    "" if entry.com_y is None else f"{entry.com_y:.2f}",
                    "" if entry.com_jump_px is None else f"{entry.com_jump_px:.2f}",
                    "" if entry.limb_length_px is None else f"{entry.limb_length_px:.2f}",
                    "" if entry.limb_length_ratio is None else f"{entry.limb_length_ratio:.2f}",
                    entry.confident_joints,
                    entry.total_joints,
                    int(entry.swap_candidate),
                ]
            )


def write_alignment_report(
    output_path: Path,
    metrics: Iterable[FrameMetrics],
    events: Iterable[JudgeEvent],
    video_meta: dict[str, object],
    track_summary: dict[str, dict[str, object]],
) -> None:
    events_list = [asdict(event) for event in events]
    recommended = sorted({fix for event in events for fix in event.recommended_fixes})
    summary = {
        "tracks": len(track_summary),
        "events": len(events_list),
        "event_types": _count_event_types(events_list),
        "recommended_fixes": recommended,
    }
    payload = {
        "video": video_meta,
        "summary": summary,
        "tracks": track_summary,
        "events": events_list,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def summarize_tracks(metrics: Iterable[FrameMetrics]) -> dict[str, dict[str, object]]:
    summary: dict[str, dict[str, object]] = {}
    for entry in metrics:
        track = summary.setdefault(
            entry.track_id,
            {
                "frames": 0,
                "avg_in_mask_ratio": 0.0,
                "avg_mean_distance_px": 0.0,
                "swap_candidates": 0,
                "avg_com_jump_px": 0.0,
            },
        )
        track["frames"] += 1
        track["avg_in_mask_ratio"] += entry.in_mask_ratio
        track["avg_mean_distance_px"] += entry.mean_distance_to_mask_px
        if entry.com_jump_px is not None:
            track["avg_com_jump_px"] += entry.com_jump_px
        if entry.swap_candidate:
            track["swap_candidates"] += 1
    for track_id, data in summary.items():
        frames = max(1, int(data["frames"]))
        data["avg_in_mask_ratio"] = data["avg_in_mask_ratio"] / frames
        data["avg_mean_distance_px"] = data["avg_mean_distance_px"] / frames
        data["avg_com_jump_px"] = data["avg_com_jump_px"] / frames
    return summary


def _count_event_types(events: list[dict[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in events:
        event_type = str(event.get("event_type", "unknown"))
        counts[event_type] = counts.get(event_type, 0) + 1
    return counts
