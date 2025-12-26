from pathlib import Path

from core.schema import validate_pose_tracks


def test_validate_missing_file(tmp_path: Path) -> None:
    ok, message = validate_pose_tracks(tmp_path / "missing.json")
    assert not ok
    assert "File not found" in message
