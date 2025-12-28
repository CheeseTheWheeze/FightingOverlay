from __future__ import annotations

import argparse
import json
import logging
import subprocess
import threading
import urllib.error
import urllib.request
from pathlib import Path
from tkinter import BooleanVar, StringVar, Text, Tk, filedialog, messagebox, ttk

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


def main() -> None:
    parser = argparse.ArgumentParser(description="FightingOverlay Control Center")
    parser.add_argument("--test-mode", action="store_true", help="Run synthetic test and exit")
    args = parser.parse_args()

    setup_logging()
    if not check_opencv(show_dialog=not args.test_mode):
        return

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

    selected_video = StringVar(value="No video selected")
    status_var = StringVar(value="Idle")
    run_stage_var = StringVar(value="Idle")
    frame_stats_var = StringVar(value="Frame -/-")
    fps_var = StringVar(value="FPS: --")
    people_var = StringVar(value="People: --")
    error_var = StringVar(value="Last error: None")

    export_overlay_var = BooleanVar(value=True)
    save_pose_var = BooleanVar(value=True)
    save_thumbnails_var = BooleanVar(value=False)
    save_background_var = BooleanVar(value=True)
    draw_all_tracks_var = BooleanVar(value=False)
    foreground_mode_var = StringVar(value="Auto (closest/most active)")

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

    open_button = ttk.Button(run_tab, text="Open Video", style="Accent.TButton", command=on_open_video)
    open_button.grid(row=2, column=0, sticky="ew", padx=(0, 12))

    run_button = ttk.Button(run_tab, text="Run Overlay", style="Accent.TButton")
    run_button.grid(row=2, column=1, sticky="ew")

    cancel_button = ttk.Button(run_tab, text="Cancel", command=lambda: cancel_event.set())
    cancel_button.grid(row=3, column=1, sticky="e", pady=(10, 0))

    action_buttons.extend([open_button, run_button])

    quick_actions = ttk.Frame(run_tab, padding=8, style="Card.TFrame")
    quick_actions.grid(row=3, column=0, sticky="w", pady=(10, 0))
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

    progress = ttk.Progressbar(status_frame, mode="determinate", maximum=100)
    progress.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))

    stats_frame = ttk.Frame(status_frame, padding=(0, 8), style="Card.TFrame")
    stats_frame.grid(row=3, column=0, columnspan=2, sticky="ew")
    stats_frame.columnconfigure(1, weight=1)

    ttk.Label(stats_frame, text="Frame", style="Card.TLabel").grid(row=0, column=0, sticky="w")
    ttk.Label(stats_frame, textvariable=frame_stats_var, style="Card.TLabel").grid(row=0, column=1, sticky="w")
    ttk.Label(stats_frame, text="Effective FPS", style="Card.TLabel").grid(row=1, column=0, sticky="w")
    ttk.Label(stats_frame, textvariable=fps_var, style="Card.TLabel").grid(row=1, column=1, sticky="w")
    ttk.Label(stats_frame, text="Last-frame people", style="Card.TLabel").grid(row=2, column=0, sticky="w")
    ttk.Label(stats_frame, textvariable=people_var, style="Card.TLabel").grid(row=2, column=1, sticky="w")
    ttk.Label(stats_frame, text="Last error", style="Card.TLabel").grid(row=3, column=0, sticky="w")
    ttk.Label(stats_frame, textvariable=error_var, style="Card.TLabel").grid(row=3, column=1, sticky="w")

    ttk.Label(status_frame, text="Last update: " + load_last_update(), style="Card.TLabel").grid(
        row=4, column=0, columnspan=2, sticky="w", pady=(6, 0)
    )

    data_tab.columnconfigure(0, weight=1)

    data_group = ttk.LabelFrame(data_tab, text="Output Options", padding=12)
    data_group.grid(row=0, column=0, sticky="ew")

    save_pose_check = ttk.Checkbutton(data_group, text="Save pose JSON", variable=save_pose_var)
    save_pose_check.grid(row=0, column=0, sticky="w")
    save_thumbnails_check = ttk.Checkbutton(data_group, text="Save thumbnails (1 per second)", variable=save_thumbnails_var)
    save_thumbnails_check.grid(row=1, column=0, sticky="w")
    save_background_check = ttk.Checkbutton(
        data_group, text="Save background tracks (track everyone)", variable=save_background_var
    )
    save_background_check.grid(row=2, column=0, sticky="w")

    data_actions = ttk.Frame(data_tab, padding=12, style="Card.TFrame")
    data_actions.grid(row=1, column=0, sticky="ew", pady=(12, 0))
    data_actions.columnconfigure(0, weight=1)
    data_actions.columnconfigure(1, weight=1)

    outputs_button = ttk.Button(data_actions, text="Open Outputs", style="Accent.TButton", command=on_open_outputs)
    logs_button = ttk.Button(data_actions, text="Open Logs", command=on_open_logs)
    outputs_button.grid(row=0, column=0, sticky="ew", padx=(0, 8))
    logs_button.grid(row=0, column=1, sticky="ew")

    settings_tab.columnconfigure(0, weight=1)

    settings_group = ttk.LabelFrame(settings_tab, text="Processing", padding=12)
    settings_group.grid(row=0, column=0, sticky="ew")

    export_overlay_check = ttk.Checkbutton(settings_group, text="Export overlay video (MP4)", variable=export_overlay_var)
    export_overlay_check.grid(row=0, column=0, sticky="w")
    draw_all_tracks_check = ttk.Checkbutton(
        settings_group,
        text="Draw all tracks (debug)",
        variable=draw_all_tracks_var,
    )
    draw_all_tracks_check.grid(row=1, column=0, sticky="w")

    ttk.Label(settings_group, text="Foreground selection mode").grid(row=2, column=0, sticky="w", pady=(8, 2))
    mode_combo = ttk.Combobox(
        settings_group,
        textvariable=foreground_mode_var,
        values=[
            "Auto (closest/most active)",
            "Manual pick",
            "Foreground=Top2 largest",
        ],
        state="readonly",
    )
    mode_combo.grid(row=3, column=0, sticky="ew")

    update_group = ttk.LabelFrame(settings_tab, text="Updates", padding=12)
    update_group.grid(row=1, column=0, sticky="ew", pady=(12, 0))

    updates_button = ttk.Button(update_group, text="Check Updates", command=on_check_updates)
    update_now_button = ttk.Button(update_group, text="Update Now", style="Accent.TButton", command=on_update_now)
    updates_button.grid(row=0, column=0, sticky="ew", padx=(0, 8))
    update_now_button.grid(row=0, column=1, sticky="ew")

    settings_group.columnconfigure(0, weight=1)
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

    settings_controls.extend(
        [
            save_pose_check,
            save_thumbnails_check,
            save_background_check,
            export_overlay_check,
            draw_all_tracks_check,
            mode_combo,
        ]
    )

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
            fps_var.set(f"FPS: {float(info['effective_fps']):.1f}")
        if "people" in info:
            people_var.set(f"People: {info['people']}")
        if "error" in info:
            error_var.set(f"Last error: {info['error']}")

    def on_run_overlay() -> None:
        if selected_video.get() == "No video selected":
            messagebox.showwarning("Run Overlay", "Please select an MP4 file first.")
            return
        if not Path(selected_video.get()).exists():
            messagebox.showerror("Run Overlay", "Selected video file does not exist.")
            return

        cancel_event.clear()
        set_running(True)
        update_status("Starting processing...", 0)
        run_stage_var.set("Loading video")
        frame_stats_var.set("Frame -/-")
        fps_var.set("FPS: --")
        people_var.set("People: --")
        error_var.set("Last error: None")

        def status_callback(message: str, progress_value: float | None) -> None:
            root.after(0, update_status, message, progress_value)

        def info_callback(info: dict[str, object]) -> None:
            root.after(0, update_info, info)

        def run_background() -> None:
            try:
                mode = foreground_mode_var.get()
                if mode != "Auto (closest/most active)":
                    logging.info("Foreground mode '%s' not implemented yet; using Auto.", mode)
                options = ProcessingOptions(
                    export_overlay_video=export_overlay_var.get(),
                    save_pose_json=save_pose_var.get(),
                    save_thumbnails=save_thumbnails_var.get(),
                    save_background_tracks=save_background_var.get(),
                    foreground_mode="Auto (closest/most active)",
                    draw_all_tracks=draw_all_tracks_var.get(),
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
    validate_button.grid(row=3, column=0, sticky="w", pady=(10, 0))

    action_buttons.append(validate_button)

    set_running(False)
    logging.info("Control Center UI ready")

    root.mainloop()


if __name__ == "__main__":
    main()
