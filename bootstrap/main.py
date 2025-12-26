from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from subprocess import run, Popen

from core.paths import (
    get_app_root,
    get_current_pointer,
    get_data_root,
    get_last_update_path,
    get_log_root,
    get_models_root,
    get_outputs_root,
    get_versions_root,
)

RELEASES_URL = "https://api.github.com/repos/CheeseTheWheeze/FightingOverlay/releases/latest"
ASSET_NAME = "FightingOverlay-Full-Windows.zip"
BOOTSTRAP_COPY_NAME = "FightingOverlayBootstrap.exe"


def setup_logging() -> Path:
    log_root = get_log_root()
    log_root.mkdir(parents=True, exist_ok=True)
    log_path = log_root / "bootstrap.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8")],
    )
    return log_path


def ensure_directories() -> None:
    get_app_root().mkdir(parents=True, exist_ok=True)
    get_data_root().mkdir(parents=True, exist_ok=True)
    get_log_root().mkdir(parents=True, exist_ok=True)
    get_outputs_root().mkdir(parents=True, exist_ok=True)
    get_models_root().mkdir(parents=True, exist_ok=True)
    get_versions_root().mkdir(parents=True, exist_ok=True)


def show_message(title: str, message: str) -> None:
    try:
        import ctypes

        ctypes.windll.user32.MessageBoxW(0, message, title, 0x00000000)
    except Exception:
        logging.info("Message: %s - %s", title, message)


def fetch_latest_release() -> dict:
    request = urllib.request.Request(
        RELEASES_URL,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "FightingOverlayBootstrap"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = response.read().decode("utf-8")
    return json.loads(payload)


def download_asset(url: str, destination: Path) -> None:
    logging.info("Downloading asset from %s", url)
    with urllib.request.urlopen(url, timeout=60) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def extract_zip(zip_path: Path, destination: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(destination)


def atomic_write(path: Path, content: str) -> None:
    temp_path = path.with_suffix(".tmp")
    temp_path.write_text(content, encoding="utf-8")
    os.replace(temp_path, path)


def record_last_update(status: str, version: str | None, message: str | None = None) -> None:
    payload = {
        "status": status,
        "version": version,
        "message": message,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    get_last_update_path().write_text(json.dumps(payload, indent=2), encoding="utf-8")


def clean_old_versions(current_version: str, previous_version: str | None) -> None:
    versions_root = get_versions_root()
    keep = {current_version}
    if previous_version:
        keep.add(previous_version)
    for entry in versions_root.iterdir():
        if entry.is_dir() and entry.name not in keep:
            logging.info("Removing old version %s", entry)
            shutil.rmtree(entry, ignore_errors=True)


def update_shortcut(target: Path) -> None:
    desktop = Path(os.path.expandvars(r"%USERPROFILE%")) / "Desktop"
    shortcut_path = desktop / "FightingOverlay.lnk"
    script = (
        "$WshShell = New-Object -ComObject WScript.Shell;"
        f"$Shortcut = $WshShell.CreateShortcut('{shortcut_path}');"
        f"$Shortcut.TargetPath = '{target}';"
        f"$Shortcut.WorkingDirectory = '{target.parent}';"
        "$Shortcut.Save();"
    )
    run(["powershell", "-NoProfile", "-Command", script], check=False)


def copy_bootstrapper() -> Path:
    app_root = get_app_root()
    app_root.mkdir(parents=True, exist_ok=True)
    destination = app_root / BOOTSTRAP_COPY_NAME
    source = Path(sys.executable)
    if getattr(sys, "frozen", False) and source.exists() and source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    return destination


def install_release(release: dict) -> Path:
    assets = release.get("assets", [])
    asset = next((item for item in assets if item.get("name") == ASSET_NAME), None)
    if not asset:
        raise RuntimeError(f"Release is missing asset {ASSET_NAME}")

    version = release.get("tag_name") or release.get("name") or "unknown"
    temp_dir = Path(tempfile.mkdtemp(prefix="fightingoverlay_"))
    zip_path = temp_dir / ASSET_NAME
    extract_dir = temp_dir / "extract"
    extract_dir.mkdir(parents=True, exist_ok=True)

    download_asset(asset["browser_download_url"], zip_path)
    extract_zip(zip_path, extract_dir)

    versions_root = get_versions_root()
    target_dir = versions_root / version
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.move(str(extract_dir), str(target_dir))

    return target_dir


def launch_control_center(target_dir: Path) -> None:
    executable = target_dir / "ControlCenter.exe"
    if not executable.exists():
        raise RuntimeError("ControlCenter.exe not found in installed version")
    Popen([str(executable)], cwd=str(target_dir))


def run_self_test() -> int:
    ensure_directories()
    setup_logging()
    logging.info("Self-test completed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="FightingOverlay Bootstrapper")
    parser.add_argument("--self-test", action="store_true", help="Run bootstrapper self-test")
    parser.add_argument("--update", action="store_true", help="Run update mode")
    args = parser.parse_args()

    if args.self_test:
        return run_self_test()

    ensure_directories()
    log_path = setup_logging()
    logging.info("Bootstrap started")
    logging.info("Log path: %s", log_path)

    previous_version = None
    current_pointer = get_current_pointer()
    if current_pointer.exists():
        previous_version = Path(current_pointer.read_text(encoding="utf-8").strip()).name

    try:
        release = fetch_latest_release()
    except urllib.error.HTTPError as error:
        if error.code == 404:
            record_last_update("no_release", None, "No release published yet")
            show_message("FightingOverlay", "No release published yet")
            return 0
        record_last_update("error", None, f"HTTP error {error.code}")
        raise

    version = release.get("tag_name") or release.get("name") or "unknown"
    try:
        target_dir = install_release(release)
        atomic_write(get_current_pointer(), str(target_dir.resolve()))
        update_shortcut(target_dir / "ControlCenter.exe")
        copy_bootstrapper()
        record_last_update("success", version, "Installed successfully")
        clean_old_versions(version, previous_version)
        launch_control_center(target_dir)
    except Exception as exc:
        logging.exception("Bootstrap failed")
        record_last_update("error", version, str(exc))
        show_message("FightingOverlay", f"Install/update failed: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
