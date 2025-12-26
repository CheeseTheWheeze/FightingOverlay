from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from tkinter import Tk, messagebox, ttk

from core.paths import (
    get_app_root,
    get_current_pointer,
    get_data_root,
    get_last_update_path,
    get_log_root,
    get_outputs_root,
)

RELEASES_URL = "https://api.github.com/repos/CheeseTheWheeze/FightingOverlay/releases/latest"


def setup_logging() -> None:
    log_root = get_log_root()
    log_root.mkdir(parents=True, exist_ok=True)
    log_path = log_root / "control_center.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8")],
    )


def get_current_version() -> str:
    pointer = get_current_pointer()
    if not pointer.exists():
        return "unknown"
    return Path(pointer.read_text(encoding="utf-8").strip()).name


def write_synthetic_test() -> Path:
    outputs = get_outputs_root()
    outputs.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tracks": [
            {"frame": 0, "pose": [{"x": 0.1, "y": 0.2}, {"x": 0.3, "y": 0.4}]},
            {"frame": 1, "pose": [{"x": 0.2, "y": 0.3}, {"x": 0.4, "y": 0.5}]},
        ],
    }
    output_path = outputs / "pose_tracks.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def open_folder(path: Path) -> None:
    subprocess.run(["explorer", str(path)], check=False)


def check_updates() -> tuple[bool, str]:
    request = urllib.request.Request(
        RELEASES_URL,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "FightingOverlayControlCenter"},
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        payload = response.read().decode("utf-8")
    data = json.loads(payload)
    latest = data.get("tag_name") or data.get("name") or "unknown"
    current = get_current_version()
    return latest != current, latest


def run_bootstrap_update() -> tuple[bool, str]:
    bootstrap_path = get_app_root() / "FightingOverlayBootstrap.exe"
    if not bootstrap_path.exists():
        return False, "Bootstrapper not found. Please download FightingOverlayBootstrap.exe from Releases."
    subprocess.Popen([str(bootstrap_path), "--update"], cwd=str(get_app_root()))
    return True, "Update started"


def load_last_update() -> str:
    last_update = get_last_update_path()
    if not last_update.exists():
        return "No updates yet"
    try:
        payload = json.loads(last_update.read_text(encoding="utf-8"))
        return f"{payload.get('status')} ({payload.get('version')}) {payload.get('timestamp')}"
    except json.JSONDecodeError:
        return "Update log unreadable"


def validate_output_schema() -> tuple[bool, str]:
    output_path = get_outputs_root() / "pose_tracks.json"
    if not output_path.exists():
        return False, "pose_tracks.json not found in outputs folder."
    try:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False, "pose_tracks.json is not valid JSON."
    if "tracks" not in payload or not isinstance(payload["tracks"], list):
        return False, "pose_tracks.json missing 'tracks' list."
    return True, "pose_tracks.json is valid."


def main() -> None:
    parser = argparse.ArgumentParser(description="FightingOverlay Control Center")
    parser.add_argument("--test-mode", action="store_true", help="Run synthetic test and exit")
    args = parser.parse_args()

    setup_logging()

    if args.test_mode:
        write_synthetic_test()
        logging.info("Control Center test mode completed")
        return

    root = Tk()
    root.title("FightingOverlay Control Center")

    frame = ttk.Frame(root, padding=12)
    frame.grid(row=0, column=0, sticky="nsew")

    def on_synthetic_test() -> None:
        output = write_synthetic_test()
        logging.info("Synthetic test wrote %s", output)
        messagebox.showinfo("Synthetic Test", f"Wrote {output}")

    def on_open_video() -> None:
        logging.info("Open Video + Run Overlay requested but inference unavailable")
        messagebox.showwarning("Overlay", "Inference not available yet.")

    def on_validate() -> None:
        ok, message = validate_output_schema()
        if ok:
            messagebox.showinfo("Validate Output", message)
        else:
            messagebox.showerror("Validate Output", message)

    def on_open_outputs() -> None:
        open_folder(get_outputs_root())

    def on_open_logs() -> None:
        open_folder(get_log_root())

    def on_check_updates() -> None:
        try:
            update_available, latest = check_updates()
        except urllib.error.HTTPError as error:
            messagebox.showerror("Updates", f"Failed to check updates: {error}")
            return
        if update_available:
            messagebox.showinfo("Updates", f"Update available: {latest}")
        else:
            messagebox.showinfo("Updates", "You are up to date.")

    def on_update_now() -> None:
        success, message = run_bootstrap_update()
        if success:
            messagebox.showinfo("Update", message)
            root.destroy()
        else:
            messagebox.showerror("Update", message)

    diagnostics = ttk.LabelFrame(frame, text="Diagnostics", padding=8)
    diagnostics.grid(row=0, column=1, padx=12, sticky="nsew")

    ttk.Label(diagnostics, text=f"Current version: {get_current_version()}").grid(row=0, column=0, sticky="w")
    ttk.Label(diagnostics, text=f"Install path: {get_app_root()}").grid(row=1, column=0, sticky="w")
    ttk.Label(diagnostics, text=f"Data path: {get_data_root()}").grid(row=2, column=0, sticky="w")
    ttk.Label(diagnostics, text=f"Last update: {load_last_update()}").grid(row=3, column=0, sticky="w")

    buttons = [
        ("Run Synthetic Test", on_synthetic_test),
        ("Open Video + Run Overlay", on_open_video),
        ("Validate Output JSON schema", on_validate),
        ("Open Outputs Folder", on_open_outputs),
        ("Open Logs Folder", on_open_logs),
        ("Check Updates", on_check_updates),
        ("Update Now", on_update_now),
    ]

    for index, (label, handler) in enumerate(buttons):
        button = ttk.Button(frame, text=label, command=handler)
        button.grid(row=index, column=0, sticky="ew", pady=2)

    root.columnconfigure(0, weight=1)
    frame.columnconfigure(0, weight=1)

    root.mainloop()


if __name__ == "__main__":
    main()
