from __future__ import annotations

import json
from pathlib import Path


REQUIRED_KEYS = {"schema_version", "tracks"}


def validate_pose_tracks(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, f"File not found: {path}"

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return False, f"Invalid JSON: {exc}"

    missing = REQUIRED_KEYS - payload.keys()
    if missing:
        return False, f"Missing keys: {', '.join(sorted(missing))}"

    if not isinstance(payload.get("tracks"), list):
        return False, "tracks must be a list"

    return True, "Schema OK"
