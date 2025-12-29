from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median as stats_median
from typing import Iterable

from core.schema import KEYPOINT_NAMES
from fightai.judge.config import JudgeConfig
from fightai.judge.person_region import PersonRegion


@dataclass(frozen=True)
class FrameMetrics:
    frame_index: int
    track_id: str
    in_mask_ratio: float
    mean_distance_to_mask_px: float
    com_x: float | None
    com_y: float | None
    com_jump_px: float | None
    limb_length_px: float | None
    limb_length_ratio: float | None
    confident_joints: int
    total_joints: int
    swap_candidate: bool


_LIMB_PAIRS = [
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("left_shoulder", "right_shoulder"),
    ("left_hip", "right_hip"),
]


def _normalize_keypoints(
    keypoints: list[object],
    min_confidence: float,
) -> tuple[dict[str, tuple[float, float, float]], list[tuple[float, float, float]]]:
    named: dict[str, tuple[float, float, float]] = {}
    points: list[tuple[float, float, float]] = []
    if all(isinstance(item, dict) for item in keypoints):
        for item in keypoints:
            name = str(item.get("name", ""))
            x = float(item.get("x", 0.0))
            y = float(item.get("y", 0.0))
            c = float(item.get("c", 0.0))
            if name:
                named[name] = (x, y, c)
            points.append((x, y, c))
    elif all(isinstance(item, (list, tuple)) for item in keypoints):
        for idx, item in enumerate(keypoints):
            values = list(item)
            if len(values) < 2:
                continue
            x = float(values[0])
            y = float(values[1])
            c = float(values[2]) if len(values) > 2 else 1.0
            name = KEYPOINT_NAMES[idx] if idx < len(KEYPOINT_NAMES) else ""
            if name:
                named[name] = (x, y, c)
            points.append((x, y, c))
    return named, [(x, y, c) for x, y, c in points if c >= min_confidence]


def _center_of_mass(points: Iterable[tuple[float, float, float]]) -> tuple[float, float] | None:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    if not xs or not ys:
        return None
    return sum(xs) / len(xs), sum(ys) / len(ys)


def _point_inside_bbox(point: tuple[float, float], bbox: tuple[float, float, float, float]) -> bool:
    x, y = point
    bx, by, bw, bh = bbox
    return bx <= x <= bx + bw and by <= y <= by + bh


def _mean_distance_to_bbox(points: Iterable[tuple[float, float]], bbox: tuple[float, float, float, float]) -> float:
    bx, by, bw, bh = bbox
    if bw <= 0 or bh <= 0:
        return 0.0
    distances = []
    for x, y in points:
        if _point_inside_bbox((x, y), bbox):
            distances.append(0.0)
        else:
            dx = max(bx - x, 0.0, x - (bx + bw))
            dy = max(by - y, 0.0, y - (by + bh))
            distances.append(math.hypot(dx, dy))
    return sum(distances) / len(distances) if distances else 0.0


def _mean_distance_to_mask(points: Iterable[tuple[float, float]], mask: list[list[int]] | None) -> float:
    if mask is None:
        return 0.0
    height = len(mask)
    width = len(mask[0]) if height else 0
    distances = []
    for x, y in points:
        xi = min(max(int(round(x)), 0), width - 1)
        yi = min(max(int(round(y)), 0), height - 1)
        if mask[yi][xi] > 0:
            distances.append(0.0)
        else:
            distances.append(0.0)
    return sum(distances) / len(distances) if distances else 0.0


def _limb_length_ratio(named: dict[str, tuple[float, float, float]], min_confidence: float) -> float | None:
    lengths = []
    for left, right in _LIMB_PAIRS:
        left_pt = named.get(left)
        right_pt = named.get(right)
        if not left_pt or not right_pt:
            continue
        if left_pt[2] < min_confidence or right_pt[2] < min_confidence:
            continue
        lengths.append(math.hypot(left_pt[0] - right_pt[0], left_pt[1] - right_pt[1]))
    if not lengths:
        return None
    median_value = float(stats_median(lengths))
    if median_value <= 0:
        return None
    return median_value


def compute_metrics(
    tracks: list[dict[str, object]],
    regions: dict[int, dict[str, PersonRegion]],
    config: JudgeConfig,
) -> list[FrameMetrics]:
    metrics: list[FrameMetrics] = []
    last_com: dict[str, tuple[float, float] | None] = {}
    limb_medians: dict[str, float] = {}
    limb_values: dict[str, list[float]] = {}

    for track in tracks:
        track_id = str(track.get("track_id", "unknown"))
        for frame in track.get("frames", []):
            frame_index = int(frame.get("frame_index", -1))
            if frame_index < 0:
                continue
            keypoints = frame.get("keypoints_2d", [])
            named, confident = _normalize_keypoints(keypoints, config.min_keypoint_confidence)
            total_points = len(keypoints) if isinstance(keypoints, list) else 0
            region = regions.get(frame_index, {}).get(track_id)
            bbox = region.bbox_xywh if region else (0.0, 0.0, 0.0, 0.0)
            inside = 0
            for x, y, _ in confident:
                if _point_inside_bbox((x, y), bbox):
                    inside += 1
            in_mask_ratio = inside / len(confident) if confident else 0.0
            points_xy = [(x, y) for x, y, _ in confident]
            mean_distance = (
                _mean_distance_to_mask(points_xy, region.mask) if region and region.mask is not None else _mean_distance_to_bbox(points_xy, bbox)
            )
            com = _center_of_mass(confident)
            prev_com = last_com.get(track_id)
            com_jump = math.hypot(com[0] - prev_com[0], com[1] - prev_com[1]) if com and prev_com else None
            if com:
                last_com[track_id] = com
            limb_len = _limb_length_ratio(named, config.min_keypoint_confidence)
            if limb_len is not None:
                limb_values.setdefault(track_id, []).append(limb_len)
            metrics.append(
                FrameMetrics(
                    frame_index=frame_index,
                    track_id=track_id,
                    in_mask_ratio=in_mask_ratio,
                    mean_distance_to_mask_px=mean_distance,
                    com_x=com[0] if com else None,
                    com_y=com[1] if com else None,
                    com_jump_px=com_jump,
                    limb_length_px=limb_len,
                    limb_length_ratio=None,
                    confident_joints=len(confident),
                    total_joints=total_points,
                    swap_candidate=False,
                )
            )

    for track_id, values in limb_values.items():
        if values:
            limb_medians[track_id] = float(stats_median(values))

    updated: list[FrameMetrics] = []
    for entry in metrics:
        limb_median = limb_medians.get(entry.track_id)
        limb_ratio = None
        if limb_median and limb_median > 0 and entry.limb_length_px is not None:
            limb_ratio = entry.limb_length_px / limb_median
        updated.append(
            FrameMetrics(
                frame_index=entry.frame_index,
                track_id=entry.track_id,
                in_mask_ratio=entry.in_mask_ratio,
                mean_distance_to_mask_px=entry.mean_distance_to_mask_px,
                com_x=entry.com_x,
                com_y=entry.com_y,
                com_jump_px=entry.com_jump_px,
                limb_length_px=entry.limb_length_px,
                limb_length_ratio=limb_ratio,
                confident_joints=entry.confident_joints,
                total_joints=entry.total_joints,
                swap_candidate=entry.swap_candidate,
            )
        )
    return updated


def compute_swap_candidates(
    metrics: list[FrameMetrics],
    regions: dict[int, dict[str, PersonRegion]],
    config: JudgeConfig,
) -> list[FrameMetrics]:
    metrics_by_frame: dict[int, list[FrameMetrics]] = {}
    for entry in metrics:
        metrics_by_frame.setdefault(entry.frame_index, []).append(entry)
    updated: list[FrameMetrics] = []
    for frame_index, entries in metrics_by_frame.items():
        frame_regions = regions.get(frame_index, {})
        for entry in entries:
            swap = False
            track_region = frame_regions.get(entry.track_id)
            if track_region and len(frame_regions) > 1:
                for other_id, other_region in frame_regions.items():
                    if other_id == entry.track_id:
                        continue
                    if _iou(track_region.bbox_xywh, other_region.bbox_xywh) < config.overlap_iou_threshold:
                        continue
                    if entry.in_mask_ratio + config.swap_margin < _in_bbox_ratio_for_track(
                        entries,
                        other_id,
                    ):
                        swap = True
                        break
            updated.append(
                FrameMetrics(
                    frame_index=entry.frame_index,
                    track_id=entry.track_id,
                    in_mask_ratio=entry.in_mask_ratio,
                    mean_distance_to_mask_px=entry.mean_distance_to_mask_px,
                    com_x=entry.com_x,
                    com_y=entry.com_y,
                    com_jump_px=entry.com_jump_px,
                    limb_length_px=entry.limb_length_px,
                    limb_length_ratio=entry.limb_length_ratio,
                    confident_joints=entry.confident_joints,
                    total_joints=entry.total_joints,
                    swap_candidate=swap,
                )
            )
    return updated


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    union_area = max(0.0, aw * ah + bw * bh - inter_area)
    return inter_area / union_area if union_area > 0 else 0.0


def _in_bbox_ratio_for_track(entries: list[FrameMetrics], track_id: str) -> float:
    for entry in entries:
        if entry.track_id == track_id:
            return entry.in_mask_ratio
    return 0.0
