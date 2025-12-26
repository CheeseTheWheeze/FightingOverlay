from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from tkinter import BOTH, LEFT, RIGHT, Button, Frame, Label, StringVar, Tk, messagebox

import requests

from core.constants import (
    APP_ROOT,
    BOOTSTRAPPER_EXE_NAME,
    BOOTSTRAPPER_ROOT,
    CURRENT_POINTER,
    DATA_ROOT,
    FULL_PACKAGE_ASSET,
    LAST_UPDATE_STATUS,
    LOG_ROOT,
    OUTPUTS_ROOT,
    ensure_directories,
)
from core.inference import write_pose_tracks
from core.schema import validate_pose_tracks

DEFAULT_REPO = os.environ.get("GITHUB_REPO", "FightingOverlay/GrapplingOverlay")


def read_current_version() -> str:
    if not CURRENT_POINTER.exists():
        return "unknown"
    path = Path(CURRENT_POINTER.read_text(encoding="utf-8").strip())
    return path.name


def read_last_update() -> str:
    if not LAST_UPDATE_STATUS.exists():
        return "No updates yet"
    try:
        payload = json.loads(LAST_UPDATE_STATUS.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "Update status unreadable"
    message = payload.get("message", "Unknown")
    return message


def github_latest_release(repo: str) -> dict | None:
    url = f"https://api.github.com/repos/{repo}/releases/latest"
    response = requests.get(url, timeout=30)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.json()


def open_folder(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    os.startfile(path)  # type: ignore[attr-defined]


def run_synthetic_test() -> None:
    output = write_pose_tracks(range(5))
    messagebox.showinfo("Synthetic Test", f"Wrote synthetic output to {output}")


def open_video_and_run() -> None:
    messagebox.showinfo("Overlay", "Inference for video is not available yet.")


def validate_output_schema() -> None:
    output_path = OUTPUTS_ROOT / "pose_tracks.json"
    ok, message = validate_pose_tracks(output_path)
    title = "Schema Validation"
    if ok:
        messagebox.showinfo(title, message)
    else:
        messagebox.showwarning(title, message)


def check_for_updates(repo: str, diagnostics: StringVar) -> None:
    release = github_latest_release(repo)
    if release is None:
        messagebox.showinfo("Updates", "No release published yet")
        return
    latest_version = release.get("tag_name") or release.get("name") or "unknown"
    current_version = read_current_version()
    if latest_version == current_version:
        messagebox.showinfo("Updates", f"You are on the latest version ({current_version})")
    else:
        messagebox.showinfo("Updates", f"Update available: {latest_version}")
    diagnostics.set(build_diagnostics())


def find_bootstrapper() -> Path | None:
    candidate = BOOTSTRAPPER_ROOT / BOOTSTRAPPER_EXE_NAME
    if candidate.exists():
        return candidate
    return None


def update_now(repo: str) -> None:
    bootstrapper = find_bootstrapper()
    if not bootstrapper:
        messagebox.showwarning(
            "Update",
            "Bootstrapper not found. Please re-run the GrapplingOverlayBootstrap.exe you downloaded.",
        )
        return
    subprocess.Popen([str(bootstrapper), "--update", "--repo", repo])
    sys.exit(0)


def build_diagnostics() -> str:
    current_version = read_current_version()
    last_update = read_last_update()
    return (
        f"Current version: {current_version}\n"
        f"Install path: {APP_ROOT}\n"
        f"Data path: {DATA_ROOT}\n"
        f"Last update: {last_update}"
    )


def build_ui(repo: str) -> None:
    ensure_directories()
    root = Tk()
    root.title("GrapplingOverlay Control Center")
    root.geometry("640x420")

    main_frame = Frame(root)
    main_frame.pack(fill=BOTH, expand=True, padx=20, pady=20)

    left_frame = Frame(main_frame)
    left_frame.pack(side=LEFT, fill=BOTH, expand=True)

    right_frame = Frame(main_frame)
    right_frame.pack(side=RIGHT, fill=BOTH, expand=True)

    Button(left_frame, text="Run Synthetic Test", command=run_synthetic_test).pack(fill=BOTH, pady=4)
    Button(left_frame, text="Open Video and Run Overlay", command=open_video_and_run).pack(fill=BOTH, pady=4)
    Button(left_frame, text="Validate Output JSON schema", command=validate_output_schema).pack(fill=BOTH, pady=4)
    Button(left_frame, text="Open Outputs Folder", command=lambda: open_folder(OUTPUTS_ROOT)).pack(fill=BOTH, pady=4)
    Button(left_frame, text="Open Logs Folder", command=lambda: open_folder(LOG_ROOT)).pack(fill=BOTH, pady=4)
    Button(left_frame, text="Check for Updates", command=lambda: check_for_updates(repo, diagnostics_var)).pack(fill=BOTH, pady=4)
    Button(left_frame, text="Update Now", command=lambda: update_now(repo)).pack(fill=BOTH, pady=4)

    diagnostics_var = StringVar(value=build_diagnostics())
    Label(right_frame, text="Diagnostics", font=("Segoe UI", 12, "bold")).pack(anchor="w")
    Label(right_frame, textvariable=diagnostics_var, justify="left").pack(anchor="w")

    root.mainloop()


def run_test_mode() -> int:
    ensure_directories()
    write_pose_tracks(range(2))
    ok, message = validate_pose_tracks(OUTPUTS_ROOT / "pose_tracks.json")
    print("ControlCenter test mode:", message)
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="GrapplingOverlay Control Center")
    parser.add_argument("--test-mode", action="store_true", help="Run a quick smoke test and exit")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="GitHub repo in owner/name format")
    args = parser.parse_args()

    if args.test_mode:
        return run_test_mode()

    build_ui(args.repo)
    return 0


if __name__ == "__main__":
    sys.exit(main())
