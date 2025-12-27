import unittest

from core.pipeline import build_pose_payload, generate_synthetic_tracks
from core.schema import SCHEMA_VERSION, validate_pose_tracks_schema


class PoseSchemaTests(unittest.TestCase):
    def test_schema_validation(self) -> None:
        video_meta = {"path": "sample.mp4", "fps": 30.0, "width": 1280, "height": 720}
        tracks = generate_synthetic_tracks(frame_count=10, fps=30.0, width=1280, height=720)
        payload = build_pose_payload(video_meta, tracks)
        self.assertEqual(payload["schema_version"], SCHEMA_VERSION)
        ok, message = validate_pose_tracks_schema(payload)
        self.assertTrue(ok, message)


if __name__ == "__main__":
    unittest.main()
