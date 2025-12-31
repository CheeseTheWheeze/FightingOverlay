from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Callable

from core.paths import get_settings_path


def _ensure_settings_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _split_key_value(value: object) -> tuple[str | None, object]:
    if (
        isinstance(value, tuple)
        and len(value) == 2
        and isinstance(value[0], str)
    ):
        return value[0], value[1]
    return None, value


def _log_invalid_numeric(key: str | None, value: object, default: object) -> None:
    if key:
        logging.warning("Invalid numeric setting %s=%r; using default %s.", key, value, default)
    else:
        logging.warning("Invalid numeric setting value %r; using default %s.", value, default)


def _clamp_numeric(value: float, min_value: float | None, max_value: float | None) -> float:
    if min_value is not None and value < min_value:
        return min_value
    if max_value is not None and value > max_value:
        return max_value
    return value


def safe_float(
    value: object,
    default: float,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    key, raw = _split_key_value(value)
    parsed: float | None = None
    if isinstance(raw, str):
        stripped = raw.strip()
        if stripped:
            try:
                parsed = float(stripped)
            except ValueError:
                parsed = None
    elif isinstance(raw, (int, float)):
        parsed = float(raw)

    if parsed is None or not math.isfinite(parsed):
        _log_invalid_numeric(key, raw, default)
        return default

    parsed = _clamp_numeric(parsed, min_value, max_value)
    return float(parsed)


def safe_int(
    value: object,
    default: int,
    *,
    min_value: int | None = None,
    max_value: int | None = None,
) -> int:
    key, raw = _split_key_value(value)
    parsed: int | None = None
    if isinstance(raw, str):
        stripped = raw.strip()
        if stripped:
            try:
                parsed = int(stripped)
            except ValueError:
                parsed = None
    elif isinstance(raw, int):
        parsed = raw
    elif isinstance(raw, float):
        if math.isfinite(raw) and raw.is_integer():
            parsed = int(raw)

    if parsed is None:
        _log_invalid_numeric(key, raw, default)
        return default

    if min_value is not None and parsed < min_value:
        return min_value
    if max_value is not None and parsed > max_value:
        return max_value
    return int(parsed)


def apply_setting_change(
    value: object,
    *,
    key: str,
    cast: Callable[[object], object] | None,
    last_good: object,
    save: Callable[[object], None],
) -> tuple[object, bool]:
    if cast is None:
        cast = lambda v: v
    try:
        new_value = cast(value)
    except Exception:
        logging.warning("Invalid setting %s=%r; keeping %r.", key, value, last_good)
        return last_good, False

    save(new_value)
    return new_value, True


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
