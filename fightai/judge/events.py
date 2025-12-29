from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from fightai.judge.config import JudgeConfig
from fightai.judge.metrics import FrameMetrics


@dataclass(frozen=True)
class JudgeEvent:
    event_type: str
    start_frame: int
    end_frame: int
    track_id: str | None
    severity: str
    details: dict[str, object]
    recommended_fixes: list[str]


def detect_events(metrics: Iterable[FrameMetrics], config: JudgeConfig) -> list[JudgeEvent]:
    events: list[JudgeEvent] = []
    metrics_by_track: dict[str, list[FrameMetrics]] = {}
    for entry in metrics:
        metrics_by_track.setdefault(entry.track_id, []).append(entry)

    for track_id, entries in metrics_by_track.items():
        entries.sort(key=lambda item: item.frame_index)
        events.extend(_detect_pose_dropout(entries, config))
        events.extend(_detect_projection_bug(entries, config))
        events.extend(_detect_skeleton_explosion(entries, config))
        events.extend(_detect_jitter(entries, config))
        events.extend(_detect_identity_swap(entries, config))
    return events


def _segments_from_flags(entries: list[FrameMetrics], flag: callable, min_len: int) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start = None
    last_frame = None
    for entry in entries:
        if flag(entry):
            if start is None:
                start = entry.frame_index
            last_frame = entry.frame_index
        else:
            if start is not None and last_frame is not None:
                if last_frame - start + 1 >= min_len:
                    segments.append((start, last_frame))
                start = None
                last_frame = None
    if start is not None and last_frame is not None and last_frame - start + 1 >= min_len:
        segments.append((start, last_frame))
    return segments


def _detect_pose_dropout(entries: list[FrameMetrics], config: JudgeConfig) -> list[JudgeEvent]:
    segments = _segments_from_flags(
        entries,
        lambda e: e.confident_joints < config.min_confident_joints or e.in_mask_ratio < config.in_mask_ratio_min,
        config.min_event_length,
    )
    return [
        JudgeEvent(
            event_type="pose_dropout",
            start_frame=start,
            end_frame=end,
            track_id=entries[0].track_id if entries else None,
            severity="warning",
            details={"reason": "low_confidence_or_outside_region"},
            recommended_fixes=["increase_reid_weight_during_overlap", "add_overlap_gate_keep_id_if_motion_consistent"],
        )
        for start, end in segments
    ]


def _detect_projection_bug(entries: list[FrameMetrics], config: JudgeConfig) -> list[JudgeEvent]:
    segments = _segments_from_flags(
        entries,
        lambda e: e.mean_distance_to_mask_px >= config.mean_distance_bug_px and e.in_mask_ratio < config.in_mask_ratio_min,
        config.min_event_length,
    )
    return [
        JudgeEvent(
            event_type="projection_misalignment",
            start_frame=start,
            end_frame=end,
            track_id=entries[0].track_id if entries else None,
            severity="error",
            details={"mean_distance_px": config.mean_distance_bug_px},
            recommended_fixes=["verify_crop_to_fullframe_transform", "check_letterbox_math"],
        )
        for start, end in segments
    ]


def _detect_skeleton_explosion(entries: list[FrameMetrics], config: JudgeConfig) -> list[JudgeEvent]:
    segments = _segments_from_flags(
        entries,
        lambda e: e.limb_length_ratio is not None and e.limb_length_ratio > config.skeleton_explosion_ratio,
        config.min_event_length,
    )
    segments += _segments_from_flags(
        entries,
        lambda e: e.limb_length_ratio is not None and e.limb_length_ratio < config.skeleton_compression_ratio,
        config.min_event_length,
    )
    events = []
    for start, end in segments:
        events.append(
            JudgeEvent(
                event_type="skeleton_explosion",
                start_frame=start,
                end_frame=end,
                track_id=entries[0].track_id if entries else None,
                severity="error",
                details={"ratio_threshold": config.skeleton_explosion_ratio},
                recommended_fixes=["verify_crop_to_fullframe_transform", "check_letterbox_math"],
            )
        )
    return events


def _detect_jitter(entries: list[FrameMetrics], config: JudgeConfig) -> list[JudgeEvent]:
    segments = _segments_from_flags(
        entries,
        lambda e: e.com_jump_px is not None and e.com_jump_px >= config.jitter_com_jump_px,
        config.min_event_length,
    )
    return [
        JudgeEvent(
            event_type="jitter",
            start_frame=start,
            end_frame=end,
            track_id=entries[0].track_id if entries else None,
            severity="warning",
            details={"com_jump_px": config.jitter_com_jump_px},
            recommended_fixes=["smoothing_causing_jitter"],
        )
        for start, end in segments
    ]


def _detect_identity_swap(entries: list[FrameMetrics], config: JudgeConfig) -> list[JudgeEvent]:
    segments = _segments_from_flags(
        entries,
        lambda e: e.swap_candidate,
        config.min_event_length,
    )
    return [
        JudgeEvent(
            event_type="identity_swap",
            start_frame=start,
            end_frame=end,
            track_id=entries[0].track_id if entries else None,
            severity="error",
            details={"swap_margin": config.swap_margin},
            recommended_fixes=["increase_reid_weight_during_overlap", "add_overlap_gate_keep_id_if_motion_consistent"],
        )
        for start, end in segments
    ]
