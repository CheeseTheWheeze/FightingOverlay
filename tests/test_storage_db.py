import os
import tempfile
import unittest
from pathlib import Path

from core.paths import get_clip_root, get_profiles_root
from core.storage import copy_source_to_clip, ensure_clip_dir
from db.index import add_artifact, create_athlete, create_clip, init_db, list_athletes


class StorageDbTests(unittest.TestCase):
    def test_profile_and_clip_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            os.environ["LOCALAPPDATA"] = tmp_dir
            os.environ["APPDATA"] = tmp_dir
            profiles_root = get_profiles_root()
            self.assertTrue(str(profiles_root).endswith("data/profiles"))
            clip_root = get_clip_root("ath_123", "clip_456")
            self.assertTrue(str(clip_root).endswith("profiles/ath_123/clips/clip_456"))

    def test_db_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            os.environ["LOCALAPPDATA"] = tmp_dir
            os.environ["APPDATA"] = tmp_dir
            init_db()
            athlete = create_athlete("Ada")
            clip = create_clip(athlete.id, source_path="demo.mp4")
            add_artifact(clip.id, "pose_json", "pose_tracks.json")
            athletes = list_athletes()
            self.assertEqual(len(athletes), 1)
            self.assertEqual(athletes[0].name, "Ada")

    def test_copy_source_to_clip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            os.environ["LOCALAPPDATA"] = tmp_dir
            os.environ["APPDATA"] = tmp_dir
            clip_dir = ensure_clip_dir("ath_x", "clip_y")
            source_path = os.path.join(tmp_dir, "source.mp4")
            with open(source_path, "wb") as handle:
                handle.write(b"demo")
            copied = copy_source_to_clip(clip_dir, Path(source_path))
            self.assertTrue(copied.exists())


if __name__ == "__main__":
    unittest.main()
