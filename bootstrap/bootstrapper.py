from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

from core.constants import (
    APP_ROOT,
    BOOTSTRAPPER_EXE_NAME,
    BOOTSTRAPPER_ROOT,
    CURRENT_POINTER,
    FULL_PACKAGE_ASSET,
    LAST_UPDATE_STATUS,
    LOG_ROOT,
    OUTPUTS_ROOT,
    DATA_ROOT,
    MODELS_ROOT,
    ensure_directories,
)

DEFAULT_REPO = os.environ.get("GITHUB_REPO", "FightingOverlay/GrapplingOverlay")


def configure_logging() -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / "bootstrap.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
    )


def write_status(status: dict[str, Any]) -> None:
    APP_ROOT.mkdir(parents=True, exist_ok=True)
    status["timestamp"] = datetime.utcnow().isoformat() + "Z"
    LAST_UPDATE_STATUS.write_text(json.dumps(status, indent=2), encoding="utf-8")


def show_message(message: str) -> None:
    try:
        import tkinter  # noqa: WPS433
        from tkinter import messagebox  # noqa: WPS433

        root = tkinter.Tk()
        root.withdraw()
        messagebox.showinfo("GrapplingOverlay", message)
        root.destroy()
    except Exception:  # pragma: no cover - best effort
        print(message)


def get_current_version_path() -> Path | None:
    if not CURRENT_POINTER.exists():
        return None
    content = CURRENT_POINTER.read_text(encoding="utf-8").strip()
    if not content:
        return None
    return Path(content)


def read_current_version() -> str | None:
    path = get_current_version_path()
    return path.name if path else None


def github_latest_release(repo: str) -> dict[str, Any] | None:
    url = f"https://api.github.com/repos/{repo}/releases/latest"
    response = requests.get(url, timeout=30)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.json()


def find_asset(release: dict[str, Any], asset_name: str) -> dict[str, Any] | None:
    for asset in release.get("assets", []):
        if asset.get("name") == asset_name:
            return asset
    return None


def download_asset(url: str, destination: Path) -> None:
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)


def extract_zip(zip_path: Path, destination: Path) -> None:
    import zipfile

    with zipfile.ZipFile(zip_path) as zip_file:
        zip_file.extractall(destination)


def install_version(version: str, extracted_dir: Path) -> Path:
    versions_root = APP_ROOT / "versions"
    versions_root.mkdir(parents=True, exist_ok=True)
    target = versions_root / version
    if target.exists():
        shutil.rmtree(target)
    shutil.move(str(extracted_dir), str(target))
    return target


def update_current_pointer(target: Path) -> None:
    tmp = CURRENT_POINTER.with_suffix(".tmp")
    tmp.write_text(str(target), encoding="utf-8")
    tmp.replace(CURRENT_POINTER)


def cleanup_versions(keep_latest: Path) -> None:
    versions_root = APP_ROOT / "versions"
    if not versions_root.exists():
        return
    versions = sorted(versions_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    keep = {keep_latest}
    for version in versions:
        if version in keep:
            continue
        if len(keep) < 2:
            keep.add(version)
            continue
        shutil.rmtree(version, ignore_errors=True)


def desktop_shortcut_path() -> Path:
    desktop = Path(os.environ.get("USERPROFILE", str(Path.home()))) / "Desktop"
    return desktop / "GrapplingOverlay.lnk"


def create_shortcut(target: Path) -> None:
    shortcut = desktop_shortcut_path()
    powershell = (
        "$WScriptShell = New-Object -ComObject WScript.Shell;"
        f"$Shortcut = $WScriptShell.CreateShortcut('{shortcut}');"
        f"$Shortcut.TargetPath = '{target}';"
        "$Shortcut.WorkingDirectory = (Split-Path $Shortcut.TargetPath);"
        "$Shortcut.Save();"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", powershell], check=False)


def copy_bootstrapper() -> None:
    try:
        source = Path(sys.executable)
        if source.name.lower() != BOOTSTRAPPER_EXE_NAME.lower():
            return
        BOOTSTRAPPER_ROOT.mkdir(parents=True, exist_ok=True)
        destination = BOOTSTRAPPER_ROOT / BOOTSTRAPPER_EXE_NAME
        if destination.exists() and destination.read_bytes() == source.read_bytes():
            return
        shutil.copy2(source, destination)
    except Exception as exc:  # pragma: no cover - best effort
        logging.warning("Failed to copy bootstrapper: %s", exc)


def launch_control_center(version_path: Path) -> None:
    exe = version_path / "ControlCenter.exe"
    if not exe.exists():
        raise FileNotFoundError(f"ControlCenter.exe not found in {version_path}")
    subprocess.Popen([str(exe)], cwd=str(version_path))


def run_self_test() -> int:
    ensure_directories()
    status = {
        "status": "ok",
        "message": "Self-test completed",
    }
    write_status(status)
    logging.info("Self-test ok")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="GrapplingOverlay Bootstrapper")
    parser.add_argument("--self-test", action="store_true", help="Run a local self-test")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="GitHub repo in owner/name format")
    parser.add_argument("--update", action="store_true", help="Force update check")
    args = parser.parse_args()

    configure_logging()
    ensure_directories()

    if args.self_test:
        return run_self_test()

    try:
        release = github_latest_release(args.repo)
        if release is None:
            message = "No release published yet"
            logging.info(message)
            write_status({"status": "no_release", "message": message})
            show_message(message)
            return 0

        latest_version = release.get("tag_name") or release.get("name") or "unknown"
        current_version = read_current_version()

        asset = find_asset(release, FULL_PACKAGE_ASSET)
        if asset is None:
            raise RuntimeError(f"Asset {FULL_PACKAGE_ASSET} not found in release {latest_version}")

        if current_version == latest_version and not args.update:
            logging.info("Already on latest version %s", current_version)
            version_path = get_current_version_path()
            if version_path:
                create_shortcut(version_path / "ControlCenter.exe")
                copy_bootstrapper()
                launch_control_center(version_path)
                return 0

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            zip_path = temp_dir_path / FULL_PACKAGE_ASSET
            download_asset(asset["browser_download_url"], zip_path)
            extract_dir = temp_dir_path / "extracted"
            extract_dir.mkdir(parents=True, exist_ok=True)
            extract_zip(zip_path, extract_dir)

            extracted_contents = list(extract_dir.iterdir())
            if len(extracted_contents) == 1 and extracted_contents[0].is_dir():
                extracted_root = extracted_contents[0]
            else:
                extracted_root = extract_dir

            version_path = install_version(latest_version, extracted_root)

        update_current_pointer(version_path)
        cleanup_versions(version_path)
        create_shortcut(version_path / "ControlCenter.exe")
        copy_bootstrapper()

        status = {
            "status": "updated",
            "message": f"Updated to {latest_version}",
            "version": latest_version,
            "install_path": str(version_path),
        }
        write_status(status)
        launch_control_center(version_path)
        return 0
    except Exception as exc:
        logging.exception("Bootstrapper failed")
        write_status({"status": "error", "message": str(exc)})
        show_message(str(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
