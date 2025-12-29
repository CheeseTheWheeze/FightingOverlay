from __future__ import annotations

import os
from pathlib import Path

APP_NAME = "FightingOverlay"


def _require_local_appdata() -> Path:
    local_appdata = os.environ.get("LOCALAPPDATA")
    if not local_appdata:
        raise RuntimeError("LOCALAPPDATA is not set. This app requires Windows.")
    return Path(local_appdata)


def _require_roaming_appdata() -> Path:
    roaming_appdata = os.environ.get("APPDATA")
    if not roaming_appdata:
        raise RuntimeError("APPDATA is not set. This app requires Windows.")
    return Path(roaming_appdata)


def get_app_root() -> Path:
    return _require_local_appdata() / APP_NAME / "app"


def get_data_root() -> Path:
    return _require_local_appdata() / APP_NAME / "data"


def get_log_root() -> Path:
    return _require_local_appdata() / APP_NAME / "logs"


def get_outputs_root() -> Path:
    return get_data_root() / "outputs"


def get_models_root() -> Path:
    return get_data_root() / "models"


def get_versions_root() -> Path:
    return get_app_root() / "versions"


def get_current_pointer() -> Path:
    return get_app_root() / "current.txt"


def get_last_update_path() -> Path:
    return get_app_root() / "last_update.json"


def get_bootstrap_root() -> Path:
    return _require_local_appdata() / APP_NAME / "bootstrap"


def get_bootstrapper_path() -> Path:
    return get_bootstrap_root() / "FightingOverlayBootstrap.exe"


def get_settings_path() -> Path:
    return _require_roaming_appdata() / APP_NAME / "settings.json"
