from __future__ import annotations

import os
from pathlib import Path

APP_NAME = "GrapplingOverlay"
APP_ROOT = Path(os.environ.get("LOCALAPPDATA", Path.home())) / APP_NAME / "app"
DATA_ROOT = Path(os.environ.get("LOCALAPPDATA", Path.home())) / APP_NAME / "data"
LOG_ROOT = Path(os.environ.get("LOCALAPPDATA", Path.home())) / APP_NAME / "logs"
OUTPUTS_ROOT = DATA_ROOT / "outputs"
MODELS_ROOT = DATA_ROOT / "models"

CURRENT_POINTER = APP_ROOT / "current.txt"
LAST_UPDATE_STATUS = APP_ROOT / "last_update.json"
BOOTSTRAPPER_ROOT = APP_ROOT / "bootstrapper"
BOOTSTRAPPER_EXE_NAME = "GrapplingOverlayBootstrap.exe"

FULL_PACKAGE_ASSET = "GrapplingOverlay-Full-Windows.zip"


def ensure_directories() -> None:
    for path in [APP_ROOT, DATA_ROOT, LOG_ROOT, OUTPUTS_ROOT, MODELS_ROOT, BOOTSTRAPPER_ROOT]:
        path.mkdir(parents=True, exist_ok=True)
