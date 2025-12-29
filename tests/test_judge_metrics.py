import unittest

from fightai.judge.config import JudgeConfig
from fightai.judge.metrics import FrameMetrics, compute_metrics, compute_swap_candidates
from fightai.judge.person_region import PersonRegion


class JudgeMetricsTests(unittest.TestCase):
    def test_in_mask_ratio_bbox(self) -> None:
        tracks = [
            {
                "track_id": "t1",
                "frames": [
                    {
                        "frame_index": 0,
                        "keypoints_2d": [
                            {"name": "nose", "x": 5, "y": 5, "c": 0.9},
                            {"name": "left_eye", "x": 15, "y": 5, "c": 0.9},
                        ],
                    }
                ],
            }
        ]
        regions = {
            0: {
                "t1": PersonRegion(frame_index=0, track_id="t1", bbox_xywh=(0.0, 0.0, 10.0, 10.0), mask=None)
            }
        }
        metrics = compute_metrics(tracks, regions, JudgeConfig(min_keypoint_confidence=0.1))
        self.assertEqual(len(metrics), 1)
        self.assertAlmostEqual(metrics[0].in_mask_ratio, 0.5, delta=0.01)

    def test_swap_candidate_detection(self) -> None:
        metrics = [
            FrameMetrics(
                frame_index=0,
                track_id="a",
                in_mask_ratio=0.1,
                mean_distance_to_mask_px=0.0,
                com_x=None,
                com_y=None,
                com_jump_px=None,
                limb_length_px=None,
                limb_length_ratio=None,
                confident_joints=5,
                total_joints=5,
                swap_candidate=False,
            ),
            FrameMetrics(
                frame_index=0,
                track_id="b",
                in_mask_ratio=0.6,
                mean_distance_to_mask_px=0.0,
                com_x=None,
                com_y=None,
                com_jump_px=None,
                limb_length_px=None,
                limb_length_ratio=None,
                confident_joints=5,
                total_joints=5,
                swap_candidate=False,
            ),
        ]
        regions = {
            0: {
                "a": PersonRegion(frame_index=0, track_id="a", bbox_xywh=(0, 0, 10, 10), mask=None),
                "b": PersonRegion(frame_index=0, track_id="b", bbox_xywh=(5, 5, 10, 10), mask=None),
            }
        }
        config = JudgeConfig(overlap_iou_threshold=0.1, swap_margin=0.2)
        updated = compute_swap_candidates(metrics, regions, config)
        swap_flags = {entry.track_id: entry.swap_candidate for entry in updated}
        self.assertTrue(swap_flags["a"])
        self.assertFalse(swap_flags["b"])


if __name__ == "__main__":
    unittest.main()
