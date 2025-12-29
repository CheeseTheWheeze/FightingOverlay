from __future__ import annotations

import argparse
import base64
from collections import deque
import json
import logging
import subprocess
import threading
import urllib.error
import urllib.request
from pathlib import Path
from tkinter import BooleanVar, DoubleVar, IntVar, PhotoImage, StringVar, Text, Tk, filedialog, messagebox, ttk

from core.paths import (
    get_app_root,
    get_bootstrapper_path,
    get_current_pointer,
    get_data_root,
    get_last_update_path,
    get_log_root,
    get_outputs_root,
)
from core.pipeline import ProcessingCancelled, ProcessingOptions, run_pipeline
from core.settings import load_settings, save_settings
from core.schema import validate_pose_tracks_schema

RELEASES_URL = "https://api.github.com/repos/CheeseTheWheeze/FightingOverlay/releases/latest"


class TkTextHandler(logging.Handler):
    def __init__(self, text_widget: Text, root: Tk) -> None:
        super().__init__()
        self.text_widget = text_widget
        self.root = root

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        self.root.after(0, self._append, msg)

    def _append(self, msg: str) -> None:
        self.text_widget.configure(state="normal")
        self.text_widget.insert("end", msg + "\n")
        self.text_widget.see("end")
        self.text_widget.configure(state="disabled")


def setup_logging() -> None:
    log_root = get_log_root()
    log_root.mkdir(parents=True, exist_ok=True)
    log_path = log_root / "control_center.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8")],
    )


def check_opencv(show_dialog: bool) -> bool:
    try:
        import cv2  # type: ignore
    except ImportError:
        message = (
            "OpenCV (cv2) is missing from this build. Please update or reinstall "
            "FightingOverlay using the latest release so OpenCV is bundled."
        )
        logging.error(message)
        if show_dialog:
            root = Tk()
            root.withdraw()
            messagebox.showerror("Missing OpenCV", message, parent=root)
            root.destroy()
        return False
    version = getattr(cv2, "__version__", "unknown")
    logging.info("OpenCV detected (cv2 version %s).", version)
    return True


def get_current_version() -> str:
    pointer = get_current_pointer()
    if not pointer.exists():
        return "unknown"
    return Path(pointer.read_text(encoding="utf-8").strip()).name


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
    bootstrap_path = get_bootstrapper_path()
    if not bootstrap_path.exists():
        return False, "Bootstrapper not found. Please download FightingOverlayBootstrap.exe from Releases."
    subprocess.Popen([str(bootstrap_path), "--update"], cwd=str(bootstrap_path.parent))
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
    return validate_pose_tracks_schema(payload)


def configure_dark_theme(root: Tk) -> None:
    style = ttk.Style()
    style.theme_use("clam")
    background = "#1c1f24"
    surface = "#252a31"
    accent = "#2f6fed"
    text = "#e7e9ee"

    root.configure(background=background)
    style.configure("TFrame", background=background)
    style.configure("Card.TFrame", background=surface)
    style.configure("TLabel", background=background, foreground=text)
    style.configure("Card.TLabel", background=surface, foreground=text)
    style.configure("Header.TLabel", background=background, foreground=text, font=("Segoe UI", 16, "bold"))
    style.configure("Subheader.TLabel", background=background, foreground=text, font=("Segoe UI", 11, "bold"))
    style.configure("TButton", background=surface, foreground=text, padding=6)
    style.map("TButton", background=[("active", "#313740")])
    style.configure("Accent.TButton", background=accent, foreground="#ffffff", font=("Segoe UI", 10, "bold"))
    style.map("Accent.TButton", background=[("active", "#3b7bff")])
    style.configure("TNotebook", background=background, borderwidth=0)
    style.configure("TNotebook.Tab", background=surface, foreground=text, padding=(12, 6))
    style.map("TNotebook.Tab", background=[("selected", "#313740")])
    style.configure("TProgressbar", troughcolor=surface, background=accent)


def format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def main() -> None:
    parser = argparse.ArgumentParser(description="FightingOverlay Control Center")
    parser.add_argument("--test-mode", action="store_true", help="Run synthetic test and exit")
    args = parser.parse_args()

    setup_logging()
    if not check_opencv(show_dialog=not args.test_mode):
        return

    defaults = {
        "debug_overlay": False,
        "draw_all_tracks": False,
        "render_mode": "Skeleton (lines+dots)",
        "confidence_threshold": 0.3,
        "smoothing_enabled": True,
        "smoothing_alpha": 0.7,
        "max_tracks": 3,
        "track_sort": "Top confidence",
        "live_preview": True,
        "run_evaluation": False,
        "export_overlay": True,
        "save_pose_json": True,
        "save_thumbnails": False,
        "save_background_tracks": True,
        "tracking_backend": "Motion (fast)",
        "foreground_mode": "Auto (closest/most active)",
        "manual_track_ids": "",
        "last_video_path": "",
        "last_output_path": str(get_outputs_root()),
    }
    settings = load_settings(defaults)

    if args.test_mode:
        options = ProcessingOptions(
            export_overlay_video=False,
            save_pose_json=True,
            save_thumbnails=False,
            save_background_tracks=True,
        )
        run_pipeline(None, options)
        logging.info("Control Center test mode completed")
        return

    root = Tk()
    configure_dark_theme(root)
    root.title("FightingOverlay Control Center")
    root.geometry("980x720")

    last_video = settings.get("last_video_path") or ""
    selected_video = StringVar(value=last_video if last_video else "No video selected")
    status_var = StringVar(value="Idle")
    run_stage_var = StringVar(value="Idle")
    frame_stats_var = StringVar(value="Frame -/-")
    fps_var = StringVar(value="Processed FPS: --")
    source_fps_var = StringVar(value="Source FPS: --")
    resolution_var = StringVar(value="Resolution: --")
    duration_var = StringVar(value="Duration: --")
    inference_var = StringVar(value="Inference: --")
    people_var = StringVar(value="People: --")
    realtime_var = StringVar(value="Realtime: --")
    error_var = StringVar(value="Last error: None")
    evaluation_var = StringVar(value="Evaluation: --")
    preview_stats_var = StringVar(value="Preview -/-")
    all_tracks_warning_var = StringVar(value="")

    export_overlay_var = BooleanVar(value=bool(settings.get("export_overlay")))
    save_pose_var = BooleanVar(value=bool(settings.get("save_pose_json")))
    save_thumbnails_var = BooleanVar(value=bool(settings.get("save_thumbnails")))
    save_background_var = BooleanVar(value=bool(settings.get("save_background_tracks")))
    debug_overlay_var = BooleanVar(value=bool(settings.get("debug_overlay")))
    draw_all_tracks_var = BooleanVar(value=bool(settings.get("draw_all_tracks")))
    render_mode_var = StringVar(value=str(settings.get("render_mode")))
    max_tracks_var = IntVar(value=int(settings.get("max_tracks") or 3))
    track_sort_var = StringVar(value=str(settings.get("track_sort")))
    live_preview_var = BooleanVar(value=bool(settings.get("live_preview")))
    run_evaluation_var = BooleanVar(value=bool(settings.get("run_evaluation")))
    smoothing_enabled_var = BooleanVar(value=bool(settings.get("smoothing_enabled")))
    foreground_mode_var = StringVar(value=str(settings.get("foreground_mode")))
    manual_tracks_var = StringVar(value=str(settings.get("manual_track_ids")))
    smoothing_alpha_var = DoubleVar(value=float(settings.get("smoothing_alpha") or 0.7))
    min_conf_var = DoubleVar(value=float(settings.get("confidence_threshold") or 0.3))
    tracking_backend_var = StringVar(value=str(settings.get("tracking_backend")))

    cancel_event = threading.Event()

    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    container = ttk.Frame(root, padding=16)
    container.grid(row=0, column=0, sticky="nsew")
    container.columnconfigure(0, weight=1)
    container.rowconfigure(3, weight=1)

    header = ttk.Label(container, text="FightingOverlay Control Center", style="Header.TLabel")
    header.grid(row=0, column=0, sticky="w")

    meta = ttk.Label(
        container,
        text=f"Version {get_current_version()} · Install {get_app_root()}",
    )
    meta.grid(row=1, column=0, sticky="w", pady=(4, 16))

    notebook = ttk.Notebook(container)
    notebook.grid(row=2, column=0, sticky="nsew")

    run_tab = ttk.Frame(notebook, padding=12, style="Card.TFrame")
    data_tab = ttk.Frame(notebook, padding=12, style="Card.TFrame")
    settings_tab = ttk.Frame(notebook, padding=12, style="Card.TFrame")

    notebook.add(run_tab, text="Run / Processing")
    notebook.add(data_tab, text="Data & Storage")
    notebook.add(settings_tab, text="Settings")

    run_tab.columnconfigure(1, weight=1)
    run_tab.rowconfigure(5, weight=1)

    ttk.Label(run_tab, text="Source Video", style="Subheader.TLabel").grid(row=0, column=0, sticky="w")
    ttk.Label(run_tab, textvariable=selected_video, style="Card.TLabel").grid(
        row=1, column=0, columnspan=2, sticky="w", pady=(4, 12)
    )

    def on_open_video() -> None:
        path = filedialog.askopenfilename(
            title="Select MP4 video",
            filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")],
        )
        if path:
            selected_video.set(path)
            settings["last_video_path"] = path
            save_settings(settings)
            logging.info("Selected video: %s", path)

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

    def on_validate() -> None:
        ok, message = validate_output_schema()
        if ok:
            messagebox.showinfo("Validate Output", message)
        else:
            messagebox.showerror("Validate Output", message)

    action_buttons: list[ttk.Button] = []
    settings_controls: list[ttk.Widget] = []

    run_actions = ttk.Frame(run_tab)
    run_actions.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 8))
    run_actions.columnconfigure(0, weight=1)
    run_actions.columnconfigure(1, weight=1)

    open_button = ttk.Button(run_actions, text="Open Video", style="Accent.TButton", command=on_open_video)
    open_button.grid(row=0, column=0, sticky="ew", padx=(0, 8))

    run_button = ttk.Button(run_actions, text="Run Overlay", style="Accent.TButton")
    run_button.grid(row=0, column=1, sticky="ew")

    advanced_visible = BooleanVar(value=False)

    def toggle_advanced() -> None:
        if advanced_visible.get():
            advanced_frame.grid_remove()
            advanced_visible.set(False)
            advanced_toggle.configure(text="Advanced ▸")
        else:
            advanced_frame.grid()
            advanced_visible.set(True)
            advanced_toggle.configure(text="Advanced ▾")

    advanced_toggle = ttk.Button(run_actions, text="Advanced ▸", command=toggle_advanced)
    advanced_toggle.grid(row=1, column=0, sticky="w", pady=(6, 0))

    cancel_button = ttk.Button(run_actions, text="Cancel", command=lambda: cancel_event.set())
    cancel_button.grid(row=1, column=1, sticky="e", pady=(6, 0))

    action_buttons.extend([open_button, run_button, advanced_toggle])

    advanced_frame = ttk.Frame(run_tab, padding=12, style="Card.TFrame")
    advanced_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(0, 12))
    advanced_frame.columnconfigure(0, weight=1)
    advanced_frame.columnconfigure(1, weight=1)
    advanced_frame.grid_remove()

    ttk.Label(advanced_frame, text="Advanced Options", style="Card.TLabel").grid(
        row=0, column=0, columnspan=2, sticky="w", pady=(0, 6)
    )

    export_overlay_check = ttk.Checkbutton(
        advanced_frame,
        text="Export overlay video (MP4)",
        variable=export_overlay_var,
    )
    export_overlay_check.grid(row=1, column=0, sticky="w")
    save_pose_check = ttk.Checkbutton(advanced_frame, text="Save pose JSON", variable=save_pose_var)
    save_pose_check.grid(row=1, column=1, sticky="w")

    save_thumbnails_check = ttk.Checkbutton(
        advanced_frame,
        text="Save thumbnails (1 per second)",
        variable=save_thumbnails_var,
    )
    save_thumbnails_check.grid(row=2, column=0, sticky="w")
    save_background_check = ttk.Checkbutton(
        advanced_frame,
        text="Save background tracks (track everyone)",
        variable=save_background_var,
    )
    save_background_check.grid(row=2, column=1, sticky="w")

    debug_overlay_check = ttk.Checkbutton(
        advanced_frame,
        text="Debug overlay (mapping primitives)",
        variable=debug_overlay_var,
    )
    debug_overlay_check.grid(row=3, column=0, sticky="w")
    draw_all_tracks_check = ttk.Checkbutton(
        advanced_frame,
        text="Draw all tracks (debug)",
        variable=draw_all_tracks_var,
    )
    draw_all_tracks_check.grid(row=3, column=1, sticky="w")

    draw_all_warning = ttk.Label(
        advanced_frame,
        text="All-tracks view can look cluttered; use Dots-only + Top-N filter.",
        style="Card.TLabel",
    )
    draw_all_warning.grid(row=4, column=0, columnspan=2, sticky="w", pady=(2, 6))

    ttk.Label(advanced_frame, text="Render mode", style="Card.TLabel").grid(row=5, column=0, sticky="w")
    render_mode_combo = ttk.Combobox(
        advanced_frame,
        textvariable=render_mode_var,
        values=["Dots only", "Skeleton (lines+dots)"],
        state="readonly",
    )
    render_mode_combo.grid(row=5, column=1, sticky="ew")

    ttk.Label(advanced_frame, text="Confidence threshold", style="Card.TLabel").grid(row=6, column=0, sticky="w")
    conf_scale = ttk.Scale(advanced_frame, from_=0.0, to=1.0, variable=min_conf_var)
    conf_scale.grid(row=6, column=1, sticky="ew")

    smoothing_enabled_check = ttk.Checkbutton(
        advanced_frame,
        text="EMA smoothing",
        variable=smoothing_enabled_var,
    )
    smoothing_enabled_check.grid(row=7, column=0, sticky="w")
    smoothing_scale = ttk.Scale(advanced_frame, from_=0.0, to=1.0, variable=smoothing_alpha_var)
    smoothing_scale.grid(row=7, column=1, sticky="ew")

    ttk.Label(advanced_frame, text="Max tracks (all-tracks)", style="Card.TLabel").grid(row=8, column=0, sticky="w")
    max_tracks_spin = ttk.Spinbox(advanced_frame, from_=1, to=10, textvariable=max_tracks_var, width=5)
    max_tracks_spin.grid(row=8, column=1, sticky="w")

    ttk.Label(advanced_frame, text="Track ranking", style="Card.TLabel").grid(row=9, column=0, sticky="w")
    track_sort_combo = ttk.Combobox(
        advanced_frame,
        textvariable=track_sort_var,
        values=["Top confidence", "Most continuous"],
        state="readonly",
    )
    track_sort_combo.grid(row=9, column=1, sticky="ew")

    ttk.Label(advanced_frame, text="Tracking backend", style="Card.TLabel").grid(row=10, column=0, sticky="w")
    backend_combo = ttk.Combobox(
        advanced_frame,
        textvariable=tracking_backend_var,
        values=[
            "Motion (fast)",
            "Synthetic (demo)",
        ],
        state="readonly",
    )
    backend_combo.grid(row=10, column=1, sticky="ew")

    ttk.Label(advanced_frame, text="Foreground selection mode", style="Card.TLabel").grid(row=11, column=0, sticky="w")
    mode_combo = ttk.Combobox(
        advanced_frame,
        textvariable=foreground_mode_var,
        values=[
            "Auto (closest/most active)",
            "Manual pick",
            "Foreground=Top2 largest",
        ],
        state="readonly",
    )
    mode_combo.grid(row=11, column=1, sticky="ew")

    ttk.Label(advanced_frame, text="Manual track IDs (comma-separated)", style="Card.TLabel").grid(
        row=12, column=0, sticky="w"
    )
    manual_entry = ttk.Entry(advanced_frame, textvariable=manual_tracks_var)
    manual_entry.grid(row=12, column=1, sticky="ew")

    live_preview_check = ttk.Checkbutton(
        advanced_frame,
        text="Live preview while processing",
        variable=live_preview_var,
    )
    live_preview_check.grid(row=13, column=0, sticky="w")
    run_eval_check = ttk.Checkbutton(
        advanced_frame,
        text="Run evaluation after processing",
        variable=run_evaluation_var,
    )
    run_eval_check.grid(row=13, column=1, sticky="w")

    reset_button = ttk.Button(advanced_frame, text="Reset to defaults")
    reset_button.grid(row=14, column=1, sticky="e", pady=(6, 0))

    quick_actions = ttk.Frame(run_tab, padding=8, style="Card.TFrame")
    quick_actions.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(10, 0))
    quick_actions.columnconfigure(0, weight=1)
    quick_actions.columnconfigure(1, weight=1)

    quick_outputs = ttk.Button(quick_actions, text="Open Outputs", command=on_open_outputs)
    quick_logs = ttk.Button(quick_actions, text="Open Logs", command=on_open_logs)
    quick_outputs.grid(row=0, column=0, sticky="ew", padx=(0, 6))
    quick_logs.grid(row=0, column=1, sticky="ew")

    status_frame = ttk.Frame(run_tab, padding=12, style="Card.TFrame")
    status_frame.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(16, 8))
    status_frame.columnconfigure(1, weight=1)

    ttk.Label(status_frame, text="Run Status", style="Subheader.TLabel").grid(row=0, column=0, sticky="w")
    ttk.Label(status_frame, textvariable=run_stage_var, style="Card.TLabel").grid(row=0, column=1, sticky="w")

    ttk.Label(status_frame, text="Details", style="Card.TLabel").grid(row=1, column=0, sticky="w")
    ttk.Label(status_frame, textvariable=status_var, style="Card.TLabel").grid(row=1, column=1, sticky="w")

    warning_label = ttk.Label(status_frame, textvariable=all_tracks_warning_var, style="Card.TLabel", foreground="#f2c94c")
    warning_label.grid(row=2, column=0, columnspan=2, sticky="w", pady=(4, 0))

    progress = ttk.Progressbar(status_frame, mode="determinate", maximum=100)
    progress.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(8, 0))

    stats_frame = ttk.Frame(status_frame, padding=(0, 8), style="Card.TFrame")
    stats_frame.grid(row=4, column=0, columnspan=2, sticky="ew")
    stats_frame.columnconfigure(1, weight=1)

    ttk.Label(stats_frame, text="Frame", style="Card.TLabel").grid(row=0, column=0, sticky="w")
    ttk.Label(stats_frame, textvariable=frame_stats_var, style="Card.TLabel").grid(row=0, column=1, sticky="w")
    ttk.Label(stats_frame, text="Resolution", style="Card.TLabel").grid(row=1, column=0, sticky="w")
    ttk.Label(stats_frame, textvariable=resolution_var, style="Card.TLabel").grid(row=1, column=1, sticky="w")
    ttk.Label(stats_frame, text="Duration", style="Card.TLabel").grid(row=2, column=0, sticky="w")
    ttk.Label(stats_frame, textvariable=duration_var, style="Card.TLabel").grid(row=2, column=1, sticky="w")
    ttk.Label(stats_frame, text="Source FPS", style="Card.TLabel").grid(row=3, column=0, sticky="w")
    ttk.Label(stats_frame, textvariable=source_fps_var, style="Card.TLabel").grid(row=3, column=1, sticky="w")
    ttk.Label(stats_frame, text="Processed FPS", style="Card.TLabel").grid(row=4, column=0, sticky="w")
    ttk.Label(stats_frame, textvariable=fps_var, style="Card.TLabel").grid(row=4, column=1, sticky="w")
    ttk.Label(stats_frame, text="Realtime", style="Card.TLabel").grid(row=5, column=0, sticky="w")
    ttk.Label(stats_frame, textvariable=realtime_var, style="Card.TLabel").grid(row=5, column=1, sticky="w")
    ttk.Label(stats_frame, text="Inference", style="Card.TLabel").grid(row=6, column=0, sticky="w")
    ttk.Label(stats_frame, textvariable=inference_var, style="Card.TLabel").grid(row=6, column=1, sticky="w")
    ttk.Label(stats_frame, text="Last-frame people", style="Card.TLabel").grid(row=7, column=0, sticky="w")
    ttk.Label(stats_frame, textvariable=people_var, style="Card.TLabel").grid(row=7, column=1, sticky="w")
    ttk.Label(stats_frame, text="Last error", style="Card.TLabel").grid(row=8, column=0, sticky="w")
    ttk.Label(stats_frame, textvariable=error_var, style="Card.TLabel").grid(row=8, column=1, sticky="w")
    ttk.Label(stats_frame, text="Evaluation", style="Card.TLabel").grid(row=9, column=0, sticky="w")
    ttk.Label(stats_frame, textvariable=evaluation_var, style="Card.TLabel").grid(row=9, column=1, sticky="w")

    ttk.Label(status_frame, text="Last update: " + load_last_update(), style="Card.TLabel").grid(
        row=5, column=0, columnspan=2, sticky="w", pady=(6, 0)
    )

    preview_frame = ttk.Frame(run_tab, padding=12, style="Card.TFrame")
    preview_frame.grid(row=5, column=0, columnspan=2, sticky="nsew", pady=(0, 8))
    preview_frame.columnconfigure(0, weight=1)

    ttk.Label(preview_frame, text="Live Preview", style="Subheader.TLabel").grid(row=0, column=0, sticky="w")
    ttk.Label(preview_frame, textvariable=preview_stats_var, style="Card.TLabel").grid(row=0, column=1, sticky="e")

    preview_label = ttk.Label(preview_frame, text="Preview will appear while processing", style="Card.TLabel")
    preview_label.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(8, 0))

    problems_frame = ttk.Frame(run_tab, padding=12, style="Card.TFrame")
    problems_frame.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(0, 8))
    problems_frame.columnconfigure(0, weight=1)

    ttk.Label(problems_frame, text="Problems", style="Subheader.TLabel").grid(row=0, column=0, sticky="w")
    problems_list = Text(problems_frame, height=5, bg="#0f1216", fg="#e7e9ee", relief="flat")
    problems_list.configure(state="disabled")
    problems_list.grid(row=1, column=0, sticky="ew", pady=(6, 0))

    data_tab.columnconfigure(0, weight=1)

    data_group = ttk.LabelFrame(data_tab, text="Storage", padding=12)
    data_group.grid(row=0, column=0, sticky="ew")
    ttk.Label(data_group, text=f"Outputs: {get_outputs_root()}").grid(row=0, column=0, sticky="w")
    ttk.Label(data_group, text=f"Logs: {get_log_root()}").grid(row=1, column=0, sticky="w")

    data_actions = ttk.Frame(data_tab, padding=12, style="Card.TFrame")
    data_actions.grid(row=1, column=0, sticky="ew", pady=(12, 0))
    data_actions.columnconfigure(0, weight=1)
    data_actions.columnconfigure(1, weight=1)

    outputs_button = ttk.Button(data_actions, text="Open Outputs", style="Accent.TButton", command=on_open_outputs)
    logs_button = ttk.Button(data_actions, text="Open Logs", command=on_open_logs)
    outputs_button.grid(row=0, column=0, sticky="ew", padx=(0, 8))
    logs_button.grid(row=0, column=1, sticky="ew")

    settings_tab.columnconfigure(0, weight=1)

    update_group = ttk.LabelFrame(settings_tab, text="Updates", padding=12)
    update_group.grid(row=0, column=0, sticky="ew")

    updates_button = ttk.Button(update_group, text="Check Updates", command=on_check_updates)
    update_now_button = ttk.Button(update_group, text="Update Now", style="Accent.TButton", command=on_update_now)
    updates_button.grid(row=0, column=0, sticky="ew", padx=(0, 8))
    update_now_button.grid(row=0, column=1, sticky="ew")

    update_group.columnconfigure(0, weight=1)
    update_group.columnconfigure(1, weight=1)

    log_frame = ttk.LabelFrame(container, text="Logs", padding=8)
    log_frame.grid(row=3, column=0, sticky="nsew", pady=(12, 0))
    log_frame.columnconfigure(0, weight=1)
    log_frame.rowconfigure(0, weight=1)

    log_text = Text(log_frame, height=10, bg="#0f1216", fg="#e7e9ee", relief="flat")
    log_text.configure(state="disabled")
    log_text.grid(row=0, column=0, sticky="nsew")

    log_scroll = ttk.Scrollbar(log_frame, command=log_text.yview)
    log_scroll.grid(row=0, column=1, sticky="ns")
    log_text.configure(yscrollcommand=log_scroll.set)

    handler = TkTextHandler(log_text, root)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(handler)

    problem_buffer: deque[str] = deque(maxlen=6)

    settings_controls.extend(
        [
            save_pose_check,
            save_thumbnails_check,
            save_background_check,
            export_overlay_check,
            debug_overlay_check,
            draw_all_tracks_check,
            render_mode_combo,
            max_tracks_spin,
            track_sort_combo,
            live_preview_check,
            run_eval_check,
            smoothing_enabled_check,
            backend_combo,
            mode_combo,
            manual_entry,
            smoothing_scale,
            conf_scale,
            reset_button,
        ]
    )

    def save_setting(key: str, value: object) -> None:
        settings[key] = value
        save_settings(settings)

    def bind_setting(var, key: str, cast=None) -> None:
        def _on_change(*_args: object) -> None:
            value = var.get()
            if cast:
                value = cast(value)
            save_setting(key, value)

        var.trace_add("write", _on_change)

    bind_setting(export_overlay_var, "export_overlay", bool)
    bind_setting(save_pose_var, "save_pose_json", bool)
    bind_setting(save_thumbnails_var, "save_thumbnails", bool)
    bind_setting(save_background_var, "save_background_tracks", bool)
    bind_setting(debug_overlay_var, "debug_overlay", bool)
    bind_setting(draw_all_tracks_var, "draw_all_tracks", bool)
    bind_setting(render_mode_var, "render_mode", str)
    bind_setting(max_tracks_var, "max_tracks", int)
    bind_setting(track_sort_var, "track_sort", str)
    bind_setting(live_preview_var, "live_preview", bool)
    bind_setting(run_evaluation_var, "run_evaluation", bool)
    bind_setting(smoothing_enabled_var, "smoothing_enabled", bool)
    bind_setting(smoothing_alpha_var, "smoothing_alpha", float)
    bind_setting(min_conf_var, "confidence_threshold", float)
    bind_setting(tracking_backend_var, "tracking_backend", str)
    bind_setting(foreground_mode_var, "foreground_mode", str)
    bind_setting(manual_tracks_var, "manual_track_ids", str)

    def update_draw_all_warning() -> None:
        if draw_all_tracks_var.get():
            draw_all_warning.grid()
            all_tracks_warning_var.set("All-tracks view can look cluttered; use Dots-only + Top-N filter.")
            if render_mode_var.get() != "Dots only":
                render_mode_var.set("Dots only")
        else:
            draw_all_warning.grid_remove()
            all_tracks_warning_var.set("")

    def update_smoothing_state() -> None:
        smoothing_scale.configure(state="normal" if smoothing_enabled_var.get() else "disabled")

    def reset_defaults() -> None:
        export_overlay_var.set(defaults["export_overlay"])
        save_pose_var.set(defaults["save_pose_json"])
        save_thumbnails_var.set(defaults["save_thumbnails"])
        save_background_var.set(defaults["save_background_tracks"])
        debug_overlay_var.set(defaults["debug_overlay"])
        draw_all_tracks_var.set(defaults["draw_all_tracks"])
        render_mode_var.set(defaults["render_mode"])
        max_tracks_var.set(defaults["max_tracks"])
        track_sort_var.set(defaults["track_sort"])
        live_preview_var.set(defaults["live_preview"])
        run_evaluation_var.set(defaults["run_evaluation"])
        smoothing_enabled_var.set(defaults["smoothing_enabled"])
        smoothing_alpha_var.set(defaults["smoothing_alpha"])
        min_conf_var.set(defaults["confidence_threshold"])
        tracking_backend_var.set(defaults["tracking_backend"])
        foreground_mode_var.set(defaults["foreground_mode"])
        manual_tracks_var.set(defaults["manual_track_ids"])

    reset_button.configure(command=reset_defaults)
    update_draw_all_warning()
    update_smoothing_state()
    draw_all_tracks_var.trace_add("write", lambda *_: update_draw_all_warning())
    smoothing_enabled_var.trace_add("write", lambda *_: update_smoothing_state())

    def set_running(running: bool) -> None:
        state = "disabled" if running else "normal"
        for button in action_buttons:
            button.configure(state=state)
        outputs_button.configure(state=state)
        logs_button.configure(state=state)
        updates_button.configure(state=state)
        update_now_button.configure(state=state)
        quick_outputs.configure(state=state)
        quick_logs.configure(state=state)
        for widget in settings_controls:
            if isinstance(widget, ttk.Combobox):
                widget.configure(state="disabled" if running else "readonly")
            else:
                widget.configure(state=state)
        cancel_button.configure(state="normal" if running else "disabled")

    def update_status(message: str, progress_value: float | None = None) -> None:
        status_var.set(message)
        if progress_value is not None:
            progress["value"] = progress_value

    def update_info(info: dict[str, object]) -> None:
        stage = info.get("stage")
        if stage:
            run_stage_var.set(str(stage))
        if "frame_index" in info and "total_frames" in info:
            frame_stats_var.set(f"Frame {info['frame_index']} / {info['total_frames']}")
        if "effective_fps" in info:
            fps_var.set(f"{float(info['effective_fps']):.1f} fps")
        if "realtime_ratio" in info:
            realtime_var.set(f"{float(info['realtime_ratio']):.2f}x realtime")
        if "people" in info:
            people_var.set(f"People: {info['people']}")
        if "video_width" in info and "video_height" in info:
            width = int(info["video_width"])
            height = int(info["video_height"])
            mp = (width * height) / 1_000_000
            resolution_var.set(f"{width} x {height} ({mp:.2f} MP)")
        if "video_duration_s" in info:
            duration_var.set(format_duration(float(info["video_duration_s"])))
        if "video_fps" in info:
            source_fps_var.set(f"{float(info['video_fps']):.1f} fps")
        if "infer_width" in info or "infer_height" in info:
            infer_w = info.get("infer_width")
            infer_h = info.get("infer_height")
            resized_w = info.get("resized_width")
            resized_h = info.get("resized_height")
            pad_left = info.get("pad_left")
            pad_right = info.get("pad_right")
            pad_top = info.get("pad_top")
            pad_bottom = info.get("pad_bottom")
            transform_kind = info.get("transform_kind")
            inference_var.set(
                f"{transform_kind} infer={infer_w}x{infer_h} resized={resized_w}x{resized_h} "
                f"pads L{pad_left} R{pad_right} T{pad_top} B{pad_bottom}"
            )
        if info.get("mapping_warning"):
            error_var.set("Mapping warning: keypoints out of bounds")
        if "error" in info:
            error_var.set(f"Last error: {info['error']}")
        if "evaluation_summary" in info:
            evaluation_var.set(f"Evaluation: {info['evaluation_summary']}")
        if "problem" in info:
            problem = info.get("problem", {})
            if isinstance(problem, dict):
                code = problem.get("code", "Problem")
                message = problem.get("message", "")
                problem_buffer.appendleft(f"[{code}] {message}")
                problems_list.configure(state="normal")
                problems_list.delete("1.0", "end")
                problems_list.insert("end", "\n".join(problem_buffer))
                problems_list.configure(state="disabled")
        if "preview_frame_index" in info and "preview_total_frames" in info:
            preview_stats_var.set(f"Preview {info['preview_frame_index']} / {info['preview_total_frames']}")
        if "preview_image" in info:
            preview_data = info.get("preview_image")
            if isinstance(preview_data, (bytes, bytearray)):
                encoded = base64.b64encode(preview_data).decode("ascii")
                image = PhotoImage(data=encoded)
                preview_label.configure(image=image, text="")
                preview_label.image = image

    def on_run_overlay() -> None:
        if selected_video.get() == "No video selected":
            messagebox.showwarning("Run Overlay", "Please select an MP4 file first.")
            return
        if not Path(selected_video.get()).exists():
            messagebox.showerror("Run Overlay", "Selected video file does not exist.")
            return

        settings["last_video_path"] = selected_video.get()
        settings["last_output_path"] = str(get_outputs_root())
        save_settings(settings)

        cancel_event.clear()
        set_running(True)
        update_status("Starting processing...", 0)
        run_stage_var.set("Loading video")
        frame_stats_var.set("Frame -/-")
        fps_var.set("Processed FPS: --")
        source_fps_var.set("Source FPS: --")
        resolution_var.set("Resolution: --")
        duration_var.set("Duration: --")
        inference_var.set("Inference: --")
        people_var.set("People: --")
        realtime_var.set("Realtime: --")
        error_var.set("Last error: None")
        evaluation_var.set("Evaluation: --")
        preview_stats_var.set("Preview -/-")
        preview_label.configure(image="", text="Preview will appear while processing")
        problem_buffer.clear()
        problems_list.configure(state="normal")
        problems_list.delete("1.0", "end")
        problems_list.configure(state="disabled")

        def status_callback(message: str, progress_value: float | None) -> None:
            root.after(0, update_status, message, progress_value)

        def info_callback(info: dict[str, object]) -> None:
            root.after(0, update_info, info)

        def run_background() -> None:
            try:
                mode = foreground_mode_var.get()
                if mode != "Auto (closest/most active)":
                    logging.info("Foreground mode: %s", mode)
                manual_ids = [chunk.strip() for chunk in manual_tracks_var.get().split(",") if chunk.strip()]
                if mode == "Manual pick" and not manual_ids:
                    root.after(0, messagebox.showwarning, "Run Overlay", "Enter track IDs for Manual pick mode.")
                    root.after(0, set_running, False)
                    return
                render_mode = "dots" if render_mode_var.get().lower().startswith("dots") else "skeleton"
                track_sort = "continuity" if track_sort_var.get().lower().startswith("most") else "confidence"
                options = ProcessingOptions(
                    export_overlay_video=export_overlay_var.get(),
                    save_pose_json=save_pose_var.get(),
                    save_thumbnails=save_thumbnails_var.get(),
                    save_background_tracks=save_background_var.get(),
                    foreground_mode=mode,
                    debug_overlay=debug_overlay_var.get(),
                    draw_all_tracks=draw_all_tracks_var.get(),
                    smoothing_alpha=float(smoothing_alpha_var.get()),
                    smoothing_enabled=smoothing_enabled_var.get(),
                    min_keypoint_confidence=float(min_conf_var.get()),
                    render_mode=render_mode,
                    max_tracks=int(max_tracks_var.get()),
                    track_sort=track_sort,
                    live_preview=live_preview_var.get(),
                    run_evaluation=run_evaluation_var.get(),
                    tracking_backend=tracking_backend_var.get(),
                    manual_track_ids=manual_ids,
                )
                pose_path = run_pipeline(
                    Path(selected_video.get()),
                    options,
                    cancel_event,
                    status_callback,
                    info_callback,
                )
                logging.info("Processing finished. Output: %s", pose_path)
                root.after(0, messagebox.showinfo, "Run Overlay", "Processing complete.")
            except ProcessingCancelled:
                logging.warning("Processing cancelled by user.")
                root.after(0, update_status, "Processing cancelled.", 0)
                root.after(0, update_info, {"stage": "Cancelled"})
            except Exception as exc:
                logging.exception("Processing failed: %s", exc)
                root.after(0, update_info, {"error": exc, "stage": "Failed"})
                root.after(0, messagebox.showerror, "Run Overlay", f"Processing failed: {exc}")
                root.after(0, update_status, "Processing failed.", 0)
            finally:
                root.after(0, set_running, False)

        thread = threading.Thread(target=run_background, daemon=True)
        thread.start()

    run_button.configure(command=on_run_overlay)

    validate_button = ttk.Button(run_tab, text="Validate Output JSON schema", command=on_validate)
    validate_button.grid(row=8, column=0, sticky="w", pady=(6, 0))

    action_buttons.append(validate_button)

    set_running(False)
    logging.info("Control Center UI ready")

    root.mainloop()


if __name__ == "__main__":
    main()
