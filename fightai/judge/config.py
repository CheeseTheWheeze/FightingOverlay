from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class JudgeConfig:
    """Thresholds and tuning parameters for overlay alignment judge.

    Values are intentionally centralized to avoid "mystery constants" in the
    metrics/event logic.
    """

    # Minimum keypoint confidence to include a joint in alignment scoring.
    min_keypoint_confidence: float = 0.3
    # Minimum fraction of confident joints that must be inside a mask/bbox before
    # flagging a dropout/misalignment event.
    in_mask_ratio_min: float = 0.4
    # Mean distance (pixels) to the person region beyond which a projection bug is suspected.
    mean_distance_bug_px: float = 30.0
    # IOU threshold to treat two tracks as overlapping enough to check for swaps.
    overlap_iou_threshold: float = 0.1
    # Margin by which another person's region must outperform the current region to flag a swap.
    swap_margin: float = 0.2
    # Ratio above median limb length to flag a skeleton explosion.
    skeleton_explosion_ratio: float = 1.7
    # Ratio below median limb length to flag a compressed skeleton.
    skeleton_compression_ratio: float = 0.6
    # Consecutive frame count to consider a segment an event.
    min_event_length: int = 5
    # Center-of-mass jump threshold for jitter detection.
    jitter_com_jump_px: float = 20.0
    # Minimum confident joints for pose dropout detection.
    min_confident_joints: int = 4
    # Number of worst frames to export.
    worst_frame_count: int = 12
    # Clip padding around an event window (seconds).
    clip_padding_s: float = 0.5
