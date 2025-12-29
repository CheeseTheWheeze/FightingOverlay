from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from core.paths import get_settings_path


def _ensure_settings_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_settings(defaults: dict[str, Any]) -> dict[str, Any]:
    path = get_settings_path()
    if not path.exists():
        return dict(defaults)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logging.warning("Settings file unreadable; using defaults.")
        return dict(defaults)
    if not isinstance(payload, dict):
        logging.warning("Settings file invalid; using defaults.")
        return dict(defaults)
    merged = dict(defaults)
    for key, value in payload.items():
        if key in defaults:
            merged[key] = value
    return merged


def save_settings(settings: dict[str, Any]) -> None:
    path = get_settings_path()
    _ensure_settings_dir(path)
    try:
        path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    except OSError as exc:
        logging.warning("Failed to save settings: %s", exc)
