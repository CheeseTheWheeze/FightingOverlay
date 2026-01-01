import os
import tempfile
import unittest

from engine.pose import PoseSequence
from engine.extract import PoseExtractionConfig, PoseExtractor


class PoseSequenceTests(unittest.TestCase):
    def test_round_trip_json(self) -> None:
        pose = PoseSequence(
            video={"fps": 30.0, "width": 1280, "height": 720, "path": "demo.mp4"},
            tracks=[{"track_id": "t1", "frames": []}],
        )
        encoded = pose.to_json()
        decoded = PoseSequence.from_json(encoded)
        self.assertEqual(decoded.video["fps"], 30.0)
        self.assertEqual(decoded.tracks[0]["track_id"], "t1")

    def test_pose_extractor_synthetic(self) -> None:
        extractor = PoseExtractor()
        with tempfile.TemporaryDirectory() as tmp_dir:
            dummy_path = os.path.join(tmp_dir, "demo.mp4")
            with open(dummy_path, "wb"):
                pass
            pose = extractor.extract(
                dummy_path,
                config=PoseExtractionConfig(tracking_backend="Synthetic (demo)"),
            )
        self.assertTrue(pose.tracks)


if __name__ == "__main__":
    unittest.main()
