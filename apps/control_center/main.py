from __future__ import annotations

import argparse
import base64
from collections import deque
import json
import logging
import shutil
import subprocess
import threading
import traceback
import urllib.error
import urllib.request
import webbrowser
from datetime import datetime
from pathlib import Path
from tkinter import (
    BooleanVar,
    Canvas,
    DoubleVar,
    IntVar,
    Listbox,
    PhotoImage,
    StringVar,
    Text,
    Tk,
    filedialog,
    messagebox,
    ttk,
)
import tkinter.font as tkfont

from core.paths import (
    get_app_root,
    get_bootstrapper_path,
    get_codex_packets_root,
    get_current_pointer,
    get_last_update_path,
    get_log_root,
    get_outputs_root,
    get_sent_codex_packets_root,
    get_update_history_path,
)
from core.pipeline import ProcessingCancelled, ProcessingOptions, run_pipeline
from core.settings import load_settings, save_settings
from core.schema import validate_pose_tracks_schema

RELEASES_URL = "https://api.github.com/repos/CheeseTheWheeze/FightingOverlay/releases/latest"

OVERLAY_OPTIONS = {
    "Skeleton overlay": {"slug": "skeleton", "file": "overlay_skeleton.mp4"},
    "Joints-only overlay": {"slug": "joints", "file": "overlay_joints.mp4"},
    "Balance overlay": {"slug": "balance", "file": "overlay_balance.mp4"},
    "Combat overlay": {"slug": "combat", "file": "overlay_combat.mp4"},
    "Debug overlay": {"slug": "debug", "file": "overlay_debug.mp4"},
}

_CV2 = None


def get_cv2():
    global _CV2
    if _CV2 is None:
        import cv2  # type: ignore

        _CV2 = cv2
    return _CV2


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


class FilteredTkTextHandler(logging.Handler):
    def __init__(self, text_widget: Text, root: Tk, filter_var: StringVar) -> None:
        super().__init__()
        self.text_widget = text_widget
        self.root = root
        self.filter_var = filter_var

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        filter_value = self.filter_var.get().strip()
        if filter_value and filter_value.lower() not in msg.lower():
            return
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
        cv2 = get_cv2()
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


def open_path(path: Path) -> None:
    if path.is_dir():
        open_folder(path)
    else:
        subprocess.run(["explorer", str(path)], check=False)


def fetch_latest_release() -> dict:
    logging.info("Update check: GET %s", RELEASES_URL)
    request = urllib.request.Request(
        RELEASES_URL,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "FightingOverlayControlCenter"},
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        payload = response.read().decode("utf-8")
    return json.loads(payload)


def summarize_release_notes(body: str, max_lines: int = 10) -> str:
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    return "\n".join(lines[:max_lines]) if lines else "No release notes provided."


def load_update_history() -> list[dict[str, str]]:
    path = get_update_history_path()
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    return [entry for entry in payload if isinstance(entry, dict)]


def write_update_history(entries: list[dict[str, str]]) -> None:
    path = get_update_history_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def sync_update_history() -> None:
    last_update = get_last_update_path()
    if not last_update.exists():
        return
    try:
        payload = json.loads(last_update.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    if not isinstance(payload, dict):
        return
    history = load_update_history()
    if history and history[0].get("timestamp") == payload.get("timestamp"):
        return
    entry = {
        "previous_version": payload.get("previous_version", "unknown"),
        "new_version": payload.get("version", "unknown"),
        "timestamp": payload.get("timestamp", ""),
        "status": payload.get("status", ""),
        "error": payload.get("message", ""),
    }
    history.insert(0, entry)
    write_update_history(history[:50])


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
        if "previous_version" not in payload:
            payload["previous_version"] = "unknown"
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


def configure_dark_theme(root: Tk, scale: float) -> dict[str, str]:
    style = ttk.Style()
    style.theme_use("clam")
    background = "#1c1f24"
    surface = "#252a31"
    accent = "#2f6fed"
    text = "#e7e9ee"
    base_padding = max(4, int(6 * scale))
    tab_padding = (max(6, int(12 * scale)), max(4, int(6 * scale)))

    root.configure(background=background)
    style.configure("TFrame", background=background)
    style.configure("Card.TFrame", background=surface)
    style.configure("TLabel", background=background, foreground=text)
    style.configure("Card.TLabel", background=surface, foreground=text)
    style.configure(
        "Header.TLabel",
        background=background,
        foreground=text,
        font=("Segoe UI", max(12, int(16 * scale)), "bold"),
    )
    style.configure(
        "Subheader.TLabel",
        background=background,
        foreground=text,
        font=("Segoe UI", max(9, int(11 * scale)), "bold"),
    )
    style.configure("TButton", background=surface, foreground=text, padding=base_padding)
    style.map("TButton", background=[("active", "#313740")])
    style.configure(
        "Accent.TButton",
        background=accent,
        foreground="#ffffff",
        font=("Segoe UI", max(9, int(10 * scale)), "bold"),
    )
    style.map("Accent.TButton", background=[("active", "#3b7bff")])
    style.configure("TNotebook", background=background, borderwidth=0)
    style.configure("TNotebook.Tab", background=surface, foreground=text, padding=tab_padding)
    style.map("TNotebook.Tab", background=[("selected", "#313740")])
    style.configure("TProgressbar", troughcolor=surface, background=accent)
    return {"background": background, "surface": surface, "accent": accent, "text": text}


def get_ui_scale(root: Tk) -> float:
    width = root.winfo_screenwidth()
    height = root.winfo_screenheight()
    scale = min(width / 1280, height / 720)
    return max(0.85, min(scale, 2.5))


def get_reference_scale() -> float:
    return min(1920 / 1280, 1080 / 720)


def create_scrollable_tab(parent: ttk.Notebook, theme: dict[str, str]) -> tuple[ttk.Frame, ttk.Frame]:
    tab = ttk.Frame(parent, style="Card.TFrame")
    tab.rowconfigure(0, weight=1)
    tab.columnconfigure(0, weight=1)

    canvas = Canvas(tab, highlightthickness=0, bd=0, background=theme["surface"])
    scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.grid(row=0, column=0, sticky="nsew")
    scrollbar.grid(row=0, column=1, sticky="ns")

    content = ttk.Frame(canvas, padding=12, style="Card.TFrame")
    window_id = canvas.create_window((0, 0), window=content, anchor="nw")

    def _on_content_configure(_event: object) -> None:
        canvas.configure(scrollregion=canvas.bbox("all"))

    def _on_canvas_configure(event: object) -> None:
        canvas.itemconfigure(window_id, width=event.width)

    def _on_mousewheel(event: object) -> None:
        if getattr(event, "delta", 0):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        elif getattr(event, "num", 0) == 4:
            canvas.yview_scroll(-1, "units")
        elif getattr(event, "num", 0) == 5:
            canvas.yview_scroll(1, "units")

    def _bind_mousewheel(_event: object) -> None:
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        canvas.bind_all("<Button-4>", _on_mousewheel)
        canvas.bind_all("<Button-5>", _on_mousewheel)

    def _unbind_mousewheel(_event: object) -> None:
        canvas.unbind_all("<MouseWheel>")
        canvas.unbind_all("<Button-4>")
        canvas.unbind_all("<Button-5>")

    content.bind("<Configure>", _on_content_configure)
    canvas.bind("<Configure>", _on_canvas_configure)
    canvas.bind("<Enter>", _bind_mousewheel)
    canvas.bind("<Leave>", _unbind_mousewheel)
    content.bind("<Enter>", _bind_mousewheel)
    content.bind("<Leave>", _unbind_mousewheel)

    return tab, content


def create_scrollable_panel(parent: ttk.Frame, theme: dict[str, str]) -> tuple[ttk.Frame, ttk.Frame]:
    panel = ttk.Frame(parent, style="Card.TFrame")
    panel.rowconfigure(0, weight=1)
    panel.columnconfigure(0, weight=1)

    canvas = Canvas(panel, highlightthickness=0, bd=0, background=theme["surface"])
    scrollbar = ttk.Scrollbar(panel, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.grid(row=0, column=0, sticky="nsew")
    scrollbar.grid(row=0, column=1, sticky="ns")

    content = ttk.Frame(canvas, padding=12, style="Card.TFrame")
    window_id = canvas.create_window((0, 0), window=content, anchor="nw")

    def _on_content_configure(_event: object) -> None:
        canvas.configure(scrollregion=canvas.bbox("all"))

    def _on_canvas_configure(event: object) -> None:
        canvas.itemconfigure(window_id, width=event.width)

    content.bind("<Configure>", _on_content_configure)
    canvas.bind("<Configure>", _on_canvas_configure)
    return panel, content


def bind_mousewheel(widget: object, scroll_target: object | None = None) -> None:
    target = scroll_target or widget

    def _on_mousewheel(event: object) -> str:
        delta = getattr(event, "delta", 0)
        if delta:
            target.yview_scroll(int(-1 * (delta / 120)), "units")
        elif getattr(event, "num", 0) == 4:
            target.yview_scroll(-1, "units")
        elif getattr(event, "num", 0) == 5:
            target.yview_scroll(1, "units")
        return "break"

    widget.bind("<MouseWheel>", _on_mousewheel)
    widget.bind("<Button-4>", _on_mousewheel)
    widget.bind("<Button-5>", _on_mousewheel)


def format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def list_packet_paths(root: Path) -> list[Path]:
    root.mkdir(parents=True, exist_ok=True)
    return sorted(root.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)


def load_latest_evaluation_summary(log_root: Path) -> tuple[str | None, Path | None]:
    candidates = sorted(log_root.glob("evaluation_*.txt"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        return None, None
    latest = candidates[0]
    return latest.read_text(encoding="utf-8").strip(), latest


def load_recent_evaluation_json_paths(log_root: Path, limit: int = 5) -> list[str]:
    candidates = sorted(log_root.glob("evaluation_*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    return [str(path) for path in candidates[:limit]]


def load_debug_pack_manifest(output_root: Path) -> dict[str, object]:
    report_path = output_root / "debug_pack_report.json"
    if not report_path.exists():
        return {"status": "missing", "path": str(report_path)}
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"status": "unreadable", "path": str(report_path)}
    payload["path"] = str(report_path)
    return payload


def get_latest_debug_pack_dir(output_root: Path) -> Path | None:
    candidates = sorted(output_root.glob("debug_pack_*"), key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def main() -> None:
    parser = argparse.ArgumentParser(description="FightingOverlay Control Center")
    parser.add_argument("--test-mode", action="store_true", help="Run synthetic test and exit")
    parser.add_argument(
        "--ui-smoke-test",
        action="store_true",
        help="Build the UI, process idle events, and exit (fails on TclError).",
    )
    args = parser.parse_args()

    setup_logging()
    if not check_opencv(show_dialog=not args.test_mode):
        return
    cv2 = get_cv2()

    defaults = {
        "debug_overlay": False,
        "draw_all_tracks": False,
        "overlay_mode": "Skeleton overlay",
        "confidence_threshold": 0.3,
        "smoothing_enabled": True,
        "smoothing_alpha": 0.7,
        "max_tracks": 3,
        "track_sort": "Stability score",
        "live_preview": True,
        "run_evaluation": False,
        "export_overlay": True,
        "save_pose_json": True,
        "save_combat_overlay": False,
        "save_thumbnails": False,
        "save_background_tracks": True,
        "run_judge": False,
        "tracking_backend": "Motion (fast)",
        "foreground_mode": "Auto (closest/most active)",
        "manual_track_ids": "",
        "last_video_path": "",
        "last_output_path": str(get_outputs_root()),
        "update_channel": "Stable",
        "last_update_check": "",
        "last_selected_tab": "Run / Processing",
        "ui_scale_multiplier": 1.0,
        "ui_font_size": 12,
    }
    settings = load_settings(defaults)
    sync_update_history()

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
    scale = get_ui_scale(root) * float(settings.get("ui_scale_multiplier", 1.0))
    reference_scale = get_reference_scale()
    root.tk.call("tk", "scaling", scale)
    base_font = tkfont.nametofont("TkDefaultFont")
    base_font_size = max(9, int(float(settings.get("ui_font_size", 12)) * (scale / reference_scale)))
    base_font.configure(size=base_font_size)
    theme = configure_dark_theme(root, scale)
    root.title("FightingOverlay Control Center")
    root.geometry(f"{int(980 * scale)}x{int(720 * scale)}")
    root.minsize(int(860 * scale), int(620 * scale))

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
    update_status_var = StringVar(value="Up to date")
    update_channel_var = StringVar(value=str(settings.get("update_channel") or "Stable"))
    update_last_checked_var = StringVar(value=str(settings.get("last_update_check") or "Never"))
    update_latest_version_var = StringVar(value="Latest: --")
    update_release_notes_var = StringVar(value="Release notes will appear after a check.")
    update_error_var = StringVar(value="")
    viewer_status_var = StringVar(value="Viewer idle")

    export_overlay_var = BooleanVar(value=bool(settings.get("export_overlay")))
    save_pose_var = BooleanVar(value=bool(settings.get("save_pose_json")))
    save_combat_var = BooleanVar(value=bool(settings.get("save_combat_overlay")))
    save_thumbnails_var = BooleanVar(value=bool(settings.get("save_thumbnails")))
    save_background_var = BooleanVar(value=bool(settings.get("save_background_tracks")))
    debug_overlay_var = BooleanVar(value=bool(settings.get("debug_overlay")))
    draw_all_tracks_var = BooleanVar(value=bool(settings.get("draw_all_tracks")))
    overlay_mode_var = StringVar(value=str(settings.get("overlay_mode")))
    max_tracks_var = IntVar(value=int(settings.get("max_tracks") or 3))
    track_sort_var = StringVar(value=str(settings.get("track_sort")))
    live_preview_var = BooleanVar(value=bool(settings.get("live_preview")))
    run_evaluation_var = BooleanVar(value=bool(settings.get("run_evaluation")))
    run_judge_var = BooleanVar(value=bool(settings.get("run_judge")))
    smoothing_enabled_var = BooleanVar(value=bool(settings.get("smoothing_enabled")))
    foreground_mode_var = StringVar(value=str(settings.get("foreground_mode")))
    manual_tracks_var = StringVar(value=str(settings.get("manual_track_ids")))
    smoothing_alpha_var = DoubleVar(value=float(settings.get("smoothing_alpha") or 0.7))
    min_conf_var = DoubleVar(value=float(settings.get("confidence_threshold") or 0.3))
    tracking_backend_var = StringVar(value=str(settings.get("tracking_backend")))
    ui_scale_var = DoubleVar(value=float(settings.get("ui_scale_multiplier", 1.0)))
    ui_font_size_var = IntVar(value=int(settings.get("ui_font_size", 12)))

    cancel_event = threading.Event()

    if overlay_mode_var.get() not in OVERLAY_OPTIONS:
        overlay_mode_var.set("Skeleton overlay")
    if track_sort_var.get() != "Stability score":
        track_sort_var.set("Stability score")

    pending_count_var = StringVar(value="Pending packets: 0")
    run_state: dict[str, object] = {
        "evaluation_summary": None,
        "evaluation_text_path": None,
        "evaluation_json_path": None,
        "evaluation_stats": None,
        "stack_trace": None,
        "last_error": None,
        "problems": [],
        "judge_output": None,
    }

    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    container = ttk.Frame(root, padding=16)
    container.grid(row=0, column=0, sticky="nsew")
    container.columnconfigure(0, weight=1)
    container.rowconfigure(2, weight=1)

    header = ttk.Label(container, text="FightingOverlay Control Center", style="Header.TLabel")
    header.grid(row=0, column=0, sticky="w")

    meta = ttk.Label(
        container,
        text=f"Version {get_current_version()} · Install {get_app_root()}",
    )
    meta.grid(row=1, column=0, sticky="w", pady=(4, 16))

    notebook = ttk.Notebook(container)
    paned_main = ttk.Panedwindow(container, orient="vertical")
    paned_main.grid(row=2, column=0, sticky="nsew")
    paned_main.add(notebook, weight=4)

    run_tab, run_content = create_scrollable_tab(notebook, theme)
    data_tab, data_content = create_scrollable_tab(notebook, theme)
    settings_tab, settings_content = create_scrollable_tab(notebook, theme)
    dev_tab = ttk.Frame(notebook, style="Card.TFrame")
    dev_tab.rowconfigure(0, weight=1)
    dev_tab.columnconfigure(0, weight=1)

    notebook.add(run_tab, text="Run / Processing")
    notebook.add(data_tab, text="Data & Storage")
    notebook.add(settings_tab, text="Settings")
    notebook.add(dev_tab, text="Dev Mode")

    last_tab = settings.get("last_selected_tab")
    if last_tab:
        for tab_id in notebook.tabs():
            if notebook.tab(tab_id, "text") == last_tab:
                notebook.select(tab_id)
                break

    def on_tab_changed(_event: object) -> None:
        selected = notebook.tab(notebook.select(), "text")
        settings["last_selected_tab"] = selected
        save_settings(settings)

    notebook.bind("<<NotebookTabChanged>>", on_tab_changed)

    run_content.columnconfigure(1, weight=1)
    run_content.rowconfigure(5, weight=1)
    run_content.rowconfigure(6, weight=1)

    ttk.Label(run_content, text="Source Video", style="Subheader.TLabel").grid(row=0, column=0, sticky="w")
    ttk.Label(run_content, textvariable=selected_video, style="Card.TLabel").grid(
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

    dev_log_filter_var = StringVar(value="")
    dev_judge_report_var = StringVar(value="Judge report: --")
    dev_bundle_var = StringVar(value="Bundle: --")
    dev_artifacts_list: Listbox | None = None
    dev_judge_state = {"report_path": None}

    def refresh_artifacts() -> None:
        nonlocal dev_artifacts_list
        if dev_artifacts_list is None:
            return
        output_root = get_outputs_root()
        entries = []
        if output_root.exists():
            entries = sorted([path.name for path in output_root.iterdir()])
        dev_artifacts_list.delete(0, "end")
        for entry in entries:
            dev_artifacts_list.insert("end", entry)

    def open_selected_artifact() -> None:
        if dev_artifacts_list is None:
            return
        selection = dev_artifacts_list.curselection()
        if not selection:
            return
        output_root = get_outputs_root()
        name = dev_artifacts_list.get(selection[0])
        open_path(output_root / name)

    def open_judge_report() -> None:
        report_path = dev_judge_state.get("report_path")
        if report_path:
            open_path(Path(report_path))

    dev_pane = ttk.Panedwindow(dev_tab, orient="horizontal")
    dev_pane.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

    dev_left_panel, dev_left_content = create_scrollable_panel(dev_pane, theme)
    dev_center_panel, dev_center_content = create_scrollable_panel(dev_pane, theme)
    dev_right_panel, dev_right_content = create_scrollable_panel(dev_pane, theme)

    dev_pane.add(dev_left_panel, weight=1)
    dev_pane.add(dev_center_panel, weight=2)
    dev_pane.add(dev_right_panel, weight=1)

    ttk.Label(dev_left_content, text="Inputs & Run", style="Subheader.TLabel").grid(row=0, column=0, sticky="w")
    ttk.Label(dev_left_content, textvariable=selected_video, style="Card.TLabel").grid(
        row=1, column=0, sticky="w", pady=(4, 8)
    )
    dev_left_actions = ttk.Frame(dev_left_content)
    dev_left_actions.grid(row=2, column=0, sticky="ew", pady=(0, 8))
    dev_left_actions.columnconfigure(0, weight=1)
    dev_left_actions.columnconfigure(1, weight=1)
    dev_open_button = ttk.Button(dev_left_actions, text="Open Video", command=on_open_video)
    dev_run_button = ttk.Button(dev_left_actions, text="Run Overlay", command=lambda: on_run_overlay())
    dev_open_button.grid(row=0, column=0, sticky="ew", padx=(0, 6))
    dev_run_button.grid(row=0, column=1, sticky="ew")
    dev_cancel_button = ttk.Button(dev_left_content, text="Cancel", command=lambda: cancel_event.set())
    dev_cancel_button.grid(row=3, column=0, sticky="w", pady=(0, 12))

    ttk.Label(dev_left_content, text="Overlay mode", style="Card.TLabel").grid(row=4, column=0, sticky="w")
    dev_overlay_combo = ttk.Combobox(
        dev_left_content,
        textvariable=overlay_mode_var,
        values=list(OVERLAY_OPTIONS.keys()),
        state="readonly",
    )
    dev_overlay_combo.grid(row=5, column=0, sticky="ew")
    dev_save_combat_check = ttk.Checkbutton(
        dev_left_content,
        text="Save combat overlay JSON",
        variable=save_combat_var,
    )
    dev_save_combat_check.grid(row=6, column=0, sticky="w")
    dev_run_judge_check = ttk.Checkbutton(
        dev_left_content,
        text="Run alignment judge after processing",
        variable=run_judge_var,
    )
    dev_run_judge_check.grid(row=7, column=0, sticky="w")

    ttk.Label(dev_center_content, text="Live Logs", style="Subheader.TLabel").grid(row=0, column=0, sticky="w")
    ttk.Label(dev_center_content, text="Filter", style="Card.TLabel").grid(row=1, column=0, sticky="w")
    dev_filter_entry = ttk.Entry(dev_center_content, textvariable=dev_log_filter_var)
    dev_filter_entry.grid(row=2, column=0, sticky="ew", pady=(0, 8))
    dev_center_content.columnconfigure(0, weight=1)

    dev_log_frame = ttk.Frame(dev_center_content, style="Card.TFrame")
    dev_log_frame.grid(row=3, column=0, sticky="nsew")
    dev_log_frame.columnconfigure(0, weight=1)
    dev_log_frame.rowconfigure(0, weight=1)
    dev_log_text = Text(dev_log_frame, height=18, bg="#0f1216", fg="#e7e9ee", relief="flat", font=base_font)
    dev_log_text.configure(state="disabled")
    dev_log_text.grid(row=0, column=0, sticky="nsew")
    dev_log_scroll = ttk.Scrollbar(dev_log_frame, command=dev_log_text.yview)
    dev_log_scroll.grid(row=0, column=1, sticky="ns")
    dev_log_text.configure(yscrollcommand=dev_log_scroll.set)
    bind_mousewheel(dev_log_text)

    dev_handler = FilteredTkTextHandler(dev_log_text, root, dev_log_filter_var)
    dev_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(dev_handler)

    ttk.Label(dev_right_content, text="Artifacts", style="Subheader.TLabel").grid(row=0, column=0, sticky="w")
    ttk.Label(dev_right_content, textvariable=dev_judge_report_var, style="Card.TLabel").grid(
        row=1, column=0, sticky="w"
    )
    ttk.Label(dev_right_content, textvariable=dev_bundle_var, style="Card.TLabel").grid(row=2, column=0, sticky="w")

    dev_artifacts_frame = ttk.Frame(dev_right_content, style="Card.TFrame")
    dev_artifacts_frame.grid(row=3, column=0, sticky="nsew", pady=(8, 8))
    dev_artifacts_frame.columnconfigure(0, weight=1)
    dev_artifacts_frame.rowconfigure(0, weight=1)
    dev_artifacts_list = Listbox(dev_artifacts_frame, height=12)
    dev_artifacts_list.grid(row=0, column=0, sticky="nsew")
    dev_artifacts_scroll = ttk.Scrollbar(dev_artifacts_frame, command=dev_artifacts_list.yview)
    dev_artifacts_scroll.grid(row=0, column=1, sticky="ns")
    dev_artifacts_list.configure(yscrollcommand=dev_artifacts_scroll.set)
    bind_mousewheel(dev_artifacts_list)

    dev_artifacts_actions = ttk.Frame(dev_right_content)
    dev_artifacts_actions.grid(row=4, column=0, sticky="ew")
    dev_refresh_button = ttk.Button(dev_artifacts_actions, text="Refresh", command=refresh_artifacts)
    dev_open_outputs_button = ttk.Button(dev_artifacts_actions, text="Open Outputs", command=on_open_outputs)
    dev_open_selected_button = ttk.Button(dev_artifacts_actions, text="Open Selected", command=open_selected_artifact)
    dev_open_report_button = ttk.Button(dev_artifacts_actions, text="Open Judge Report", command=open_judge_report)
    dev_refresh_button.grid(row=0, column=0, sticky="ew", padx=(0, 6))
    dev_open_outputs_button.grid(row=0, column=1, sticky="ew", padx=(0, 6))
    dev_open_selected_button.grid(row=0, column=2, sticky="ew", padx=(0, 6))
    dev_open_report_button.grid(row=0, column=3, sticky="ew")
    dev_artifacts_actions.columnconfigure(0, weight=1)
    dev_artifacts_actions.columnconfigure(1, weight=1)
    dev_artifacts_actions.columnconfigure(2, weight=1)
    dev_artifacts_actions.columnconfigure(3, weight=1)

    refresh_artifacts()

    def update_pending_count() -> None:
        pending_paths = list_packet_paths(get_codex_packets_root())
        pending_count_var.set(f"Pending packets: {len(pending_paths)}")

    def build_debug_pack_status() -> dict[str, object]:
        output_root = get_outputs_root()
        manifest = load_debug_pack_manifest(output_root)
        latest_pack = get_latest_debug_pack_dir(output_root)
        file_count = len(list(latest_pack.glob("*"))) if latest_pack else 0
        return {
            "manifest": manifest,
            "latest_pack_dir": str(latest_pack) if latest_pack else None,
            "latest_pack_file_count": file_count,
        }

    def build_codex_packet() -> dict[str, object]:
        log_root = get_log_root()
        summary_block, summary_path = load_latest_evaluation_summary(log_root)
        evaluation_json_paths = load_recent_evaluation_json_paths(log_root)
        debug_pack_status = build_debug_pack_status()
        manifest = debug_pack_status.get("manifest", {})
        overlay_stats = None
        track_summary = None
        if isinstance(manifest, dict):
            overlay_stats = manifest.get("overlay_stats")
            track_summary = manifest.get("track_summary")
        return {
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "evaluation_summary": summary_block or run_state.get("evaluation_summary"),
            "evaluation_summary_path": str(summary_path) if summary_path else run_state.get("evaluation_text_path"),
            "evaluation_json_paths": evaluation_json_paths,
            "evaluation_stats": run_state.get("evaluation_stats"),
            "overlay_stats": overlay_stats,
            "track_summary": track_summary,
            "problems": run_state.get("problems", []),
            "last_error": run_state.get("last_error"),
            "stack_trace": run_state.get("stack_trace"),
            "run_config": {
                "model_backend": (overlay_stats or {}).get("model_backend") or tracking_backend_var.get(),
                "tracking_backend": tracking_backend_var.get(),
                "confidence_threshold": float(min_conf_var.get()),
                "smoothing_enabled": bool(smoothing_enabled_var.get()),
                "smoothing_alpha": float(smoothing_alpha_var.get()),
                "overlay_mode": overlay_mode_var.get(),
                "debug_overlay": bool(debug_overlay_var.get()),
                "run_judge": bool(run_judge_var.get()),
            },
            "debug_pack_status": debug_pack_status,
            "user_note": "fighters remain center-ish in frame; data seems way off",
        }

    def format_codex_packet_text(packet: dict[str, object]) -> str:
        lines = [
            "Codex Fix Packet",
            f"Created: {packet.get('created_at')}",
            "",
            "Evaluation Summary:",
            str(packet.get("evaluation_summary") or "None"),
            "",
            "Evaluation JSON Paths:",
        ]
        for path in packet.get("evaluation_json_paths") or []:
            lines.append(f"- {path}")
        lines.extend(
            [
                "",
                "Run Config:",
                json.dumps(packet.get("run_config", {}), indent=2),
                "",
                "Debug Pack Status:",
                json.dumps(packet.get("debug_pack_status", {}), indent=2),
                "",
                "Track Summary:",
                json.dumps(packet.get("track_summary", {}), indent=2),
                "",
                "Overlay Stats:",
                json.dumps(packet.get("overlay_stats", {}), indent=2),
                "",
                "Problems / Sanity Checks:",
                json.dumps(packet.get("problems", []), indent=2),
                "",
                "Last Error:",
                str(packet.get("last_error") or "None"),
                "",
                "Stack Trace:",
                str(packet.get("stack_trace") or "None"),
                "",
                "User Note:",
                str(packet.get("user_note") or ""),
            ]
        )
        return "\n".join(lines)

    def copy_to_clipboard(text: str) -> None:
        root.clipboard_clear()
        root.clipboard_append(text)

    def save_pending_packet(packet: dict[str, object]) -> Path | None:
        if not packet.get("evaluation_summary") and not packet.get("last_error") and not packet.get("overlay_stats"):
            return None
        pending_root = get_codex_packets_root()
        pending_root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = pending_root / f"codex_packet_{timestamp}.json"
        path.write_text(json.dumps(packet, indent=2), encoding="utf-8")
        return path

    def write_codex_packet_text(packet: dict[str, object]) -> Path:
        output_dir = get_outputs_root()
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = output_dir / f"codex_fix_packet_{timestamp}.txt"
        path.write_text(format_codex_packet_text(packet), encoding="utf-8")
        return path

    def on_copy_eval_warning() -> None:
        warning = (
            "[EVAL] Low pose coverage detected. Try lowering the confidence threshold "
            "or enable diagnostic mode for troubleshooting."
        )
        summary_block, _ = load_latest_evaluation_summary(get_log_root())
        message = warning
        if summary_block:
            message = f"{warning}\n\n{summary_block}"
        copy_to_clipboard(message)
        messagebox.showinfo(
            "Copied",
            f"Evaluation warning and summary copied to clipboard.\n\n{summary_block or 'No summary found.'}",
        )

    def on_generate_codex_packet() -> None:
        packet = build_codex_packet()
        text = format_codex_packet_text(packet)
        copy_to_clipboard(text)
        path = write_codex_packet_text(packet)
        messagebox.showinfo("Codex Fix Packet", f"Codex Fix Packet copied to clipboard.\nSaved: {path}")

    def on_generate_plan_from_pending() -> None:
        pending_paths = list_packet_paths(get_codex_packets_root())
        if not pending_paths:
            messagebox.showinfo("Pending Packets", "No pending Codex packets found.")
            return
        packets = []
        for path in pending_paths:
            try:
                packets.append(json.loads(path.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                packets.append({"error": "Unreadable packet", "path": str(path)})
        prompt_lines = [
            "You are Codex. Review the following pending packets and provide:",
            "1) Ranked root-cause hypotheses",
            "2) A patch plan with file paths + key functions",
            "3) Tests to run",
            "",
            "Pending packets:",
            json.dumps(packets, indent=2),
        ]
        prompt_text = "\n".join(prompt_lines)
        copy_to_clipboard(prompt_text)
        output_dir = get_outputs_root()
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prompt_path = output_dir / f"codex_pending_prompt_{timestamp}.txt"
        prompt_path.write_text(prompt_text, encoding="utf-8")
        sent_root = get_sent_codex_packets_root()
        sent_root.mkdir(parents=True, exist_ok=True)
        for path in pending_paths:
            shutil.move(str(path), sent_root / path.name)
        update_pending_count()
        messagebox.showinfo(
            "Pending Packets Sent",
            f"Prompt copied to clipboard.\nSaved: {prompt_path}\nMoved {len(pending_paths)} packets to sent/.",
        )

    update_pending_count()

    def overlay_slug(label: str) -> str:
        return OVERLAY_OPTIONS.get(label, OVERLAY_OPTIONS["Skeleton overlay"])["slug"]

    def overlay_file(label: str) -> Path:
        filename = OVERLAY_OPTIONS.get(label, OVERLAY_OPTIONS["Skeleton overlay"])["file"]
        return get_outputs_root() / filename

    def refresh_update_history_list() -> None:
        history = load_update_history()
        entries = history[:5]
        update_history_text.configure(state="normal")
        update_history_text.delete("1.0", "end")
        if not entries:
            update_history_text.insert("end", "No updates recorded yet.")
        else:
            for entry in entries:
                status = entry.get("status", "unknown")
                before = entry.get("previous_version", "unknown")
                after = entry.get("new_version", "unknown")
                timestamp = entry.get("timestamp", "")
                error = entry.get("error", "")
                line = f"{timestamp} | {before} -> {after} | {status}"
                if error:
                    line += f" | {error}"
                update_history_text.insert("end", line + "\n")
        update_history_text.configure(state="disabled")

    def set_release_notes(text: str) -> None:
        update_notes_text.configure(state="normal")
        update_notes_text.delete("1.0", "end")
        update_notes_text.insert("end", text)
        update_notes_text.configure(state="disabled")

    def on_check_updates() -> None:
        update_status_var.set("Checking...")
        update_error_var.set("")
        checked_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        update_last_checked_var.set(checked_at)
        save_setting("last_update_check", checked_at)

        def _run_check() -> None:
            try:
                release = fetch_latest_release()
                latest = release.get("tag_name") or release.get("name") or "unknown"
                notes = summarize_release_notes(release.get("body", ""))
                update_url = release.get("html_url", "")
                update_state["latest_version"] = latest
                update_state["release_notes"] = notes
                update_state["release_url"] = update_url
                current = get_current_version()
                logging.info("Update check: current=%s latest=%s url=%s", current, latest, update_url)

                def _apply() -> None:
                    label = latest if str(latest).startswith("v") else f"v{latest}"
                    update_latest_version_var.set(f"Latest version: {label}")
                    set_release_notes(notes)
                    if latest != "unknown" and current != "unknown" and latest != current:
                        update_status_var.set("Update available")
                        update_action_button.configure(state="normal")
                    else:
                        update_status_var.set("Up to date")
                        update_action_button.configure(state="disabled")
                    refresh_update_history_list()

                root.after(0, _apply)
            except urllib.error.HTTPError as error:
                message = f"HTTP error {error.code}"
                if error.code == 403:
                    message += " (rate limit)"
                logging.error("Update check failed: %s", message)

                def _apply_error() -> None:
                    update_status_var.set("Error")
                    update_error_var.set(message)
                    update_action_button.configure(state="disabled")
                    set_release_notes("Release notes unavailable due to error.")

                root.after(0, _apply_error)
            except Exception as exc:
                logging.exception("Update check failed: %s", exc)

                def _apply_exc() -> None:
                    update_status_var.set("Error")
                    update_error_var.set(str(exc))
                    update_action_button.configure(state="disabled")
                    set_release_notes("Release notes unavailable due to error.")

                root.after(0, _apply_exc)

        threading.Thread(target=_run_check, daemon=True).start()

    def show_restart_modal(countdown: int = 3) -> None:
        modal = ttk.Frame(root, padding=16, style="Card.TFrame")
        modal.place(relx=0.5, rely=0.5, anchor="center")
        modal.columnconfigure(0, weight=1)
        label_var = StringVar(value=f"Update installed. Restarting now in {countdown}...")
        ttk.Label(modal, textvariable=label_var, style="Card.TLabel").grid(row=0, column=0, pady=(0, 8))
        update_status_var.set("Restarting...")
        def on_restart_now() -> None:
            root.destroy()
        restart_button = ttk.Button(modal, text="Restart now", style="Accent.TButton", command=on_restart_now)
        restart_button.grid(row=1, column=0, sticky="ew")

        def tick(remaining: int) -> None:
            if remaining <= 0:
                root.destroy()
                return
            label_var.set(f"Update installed. Restarting now in {remaining}...")
            root.after(1000, tick, remaining - 1)

        root.after(1000, tick, countdown - 1)

    def trigger_update(allow_running: bool = False) -> None:
        if processing_state["running"] and not allow_running:
            if messagebox.askyesno(
                "Update",
                "Processing is running. Install after completion?",
            ):
                install_after_job["pending"] = True
                update_status_var.set("Update scheduled after job")
            return
        update_status_var.set("Downloading...")
        update_error_var.set("")
        update_action_button.configure(state="disabled")
        logging.info("Update requested: current=%s latest=%s", get_current_version(), update_state.get("latest_version"))
        success, message = run_bootstrap_update()
        if success:
            update_status_var.set("Applying...")
            show_restart_modal()
        else:
            update_status_var.set("Error")
            update_error_var.set(message)

    def on_update_now() -> None:
        if update_status_var.get() != "Update available":
            messagebox.showinfo("Update", "No update is available right now.")
            return
        trigger_update()

    def on_bootstrap_repair() -> None:
        current = get_current_version()
        latest = update_state.get("latest_version") or current
        if latest == current:
            messagebox.showinfo("Bootstrap / Repair", "No changes needed. You are already on the latest version.")
            return
        messagebox.showinfo("Bootstrap / Repair", f"Updating from v{current} to v{latest}.")
        trigger_update()

    def on_open_release_notes() -> None:
        url = update_state.get("release_url")
        if url:
            webbrowser.open(url)
        else:
            messagebox.showinfo("Updates", "No release notes URL available yet.")

    def on_validate() -> None:
        ok, message = validate_output_schema()
        if ok:
            messagebox.showinfo("Validate Output", message)
        else:
            messagebox.showerror("Validate Output", message)

    action_buttons: list[ttk.Button] = []
    settings_controls: list[ttk.Widget] = []
    update_state: dict[str, str] = {"latest_version": "", "release_notes": "", "release_url": ""}
    install_after_job = {"pending": False}
    processing_state = {"running": False}

    run_actions = ttk.Frame(run_content)
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

    action_buttons.extend([open_button, run_button, advanced_toggle, dev_open_button, dev_run_button])

    advanced_frame = ttk.Frame(run_content, padding=12, style="Card.TFrame")
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

    save_combat_check = ttk.Checkbutton(
        advanced_frame,
        text="Save combat overlay JSON",
        variable=save_combat_var,
    )
    save_combat_check.grid(row=3, column=0, sticky="w")
    run_judge_check = ttk.Checkbutton(
        advanced_frame,
        text="Run alignment judge after processing",
        variable=run_judge_var,
    )
    run_judge_check.grid(row=3, column=1, sticky="w")

    debug_overlay_check = ttk.Checkbutton(
        advanced_frame,
        text="Debug overlay (mapping primitives)",
        variable=debug_overlay_var,
    )
    debug_overlay_check.grid(row=4, column=0, sticky="w")
    draw_all_tracks_check = ttk.Checkbutton(
        advanced_frame,
        text="Draw all tracks (debug)",
        variable=draw_all_tracks_var,
    )
    draw_all_tracks_check.grid(row=4, column=1, sticky="w")

    draw_all_warning = ttk.Label(
        advanced_frame,
        text="All-tracks view can look cluttered; use Joints-only or lower the Top-N cap.",
        style="Card.TLabel",
    )
    draw_all_warning.grid(row=5, column=0, columnspan=2, sticky="w", pady=(2, 6))

    ttk.Label(advanced_frame, text="Overlay mode", style="Card.TLabel").grid(row=6, column=0, sticky="w")
    overlay_mode_combo = ttk.Combobox(
        advanced_frame,
        textvariable=overlay_mode_var,
        values=list(OVERLAY_OPTIONS.keys()),
        state="readonly",
    )
    overlay_mode_combo.grid(row=6, column=1, sticky="ew")

    ttk.Label(advanced_frame, text="Confidence threshold", style="Card.TLabel").grid(row=7, column=0, sticky="w")
    conf_scale = ttk.Scale(advanced_frame, from_=0.0, to=1.0, variable=min_conf_var)
    conf_scale.grid(row=7, column=1, sticky="ew")

    smoothing_enabled_check = ttk.Checkbutton(
        advanced_frame,
        text="EMA smoothing",
        variable=smoothing_enabled_var,
    )
    smoothing_enabled_check.grid(row=8, column=0, sticky="w")
    smoothing_scale = ttk.Scale(advanced_frame, from_=0.0, to=1.0, variable=smoothing_alpha_var)
    smoothing_scale.grid(row=8, column=1, sticky="ew")

    ttk.Label(
        advanced_frame,
        text="Top-N tracks cap (debug limiter)",
        style="Card.TLabel",
    ).grid(row=9, column=0, sticky="w")
    max_tracks_spin = ttk.Spinbox(advanced_frame, from_=1, to=50, textvariable=max_tracks_var, width=5)
    max_tracks_spin.grid(row=9, column=1, sticky="w")

    ttk.Label(advanced_frame, text="Track ranking", style="Card.TLabel").grid(row=10, column=0, sticky="w")
    track_sort_combo = ttk.Combobox(
        advanced_frame,
        textvariable=track_sort_var,
        values=["Stability score"],
        state="readonly",
    )
    track_sort_combo.grid(row=10, column=1, sticky="ew")

    ttk.Label(advanced_frame, text="Tracking backend", style="Card.TLabel").grid(row=11, column=0, sticky="w")
    backend_combo = ttk.Combobox(
        advanced_frame,
        textvariable=tracking_backend_var,
        values=[
            "Motion (fast)",
            "Synthetic (demo)",
        ],
        state="readonly",
    )
    backend_combo.grid(row=11, column=1, sticky="ew")

    ttk.Label(advanced_frame, text="Foreground selection mode", style="Card.TLabel").grid(row=12, column=0, sticky="w")
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
    mode_combo.grid(row=12, column=1, sticky="ew")

    ttk.Label(advanced_frame, text="Manual track IDs (comma-separated)", style="Card.TLabel").grid(
        row=13, column=0, sticky="w"
    )
    manual_entry = ttk.Entry(advanced_frame, textvariable=manual_tracks_var)
    manual_entry.grid(row=13, column=1, sticky="ew")

    live_preview_check = ttk.Checkbutton(
        advanced_frame,
        text="Live preview",
        variable=live_preview_var,
    )
    live_preview_check.grid(row=14, column=0, sticky="w")
    run_eval_check = ttk.Checkbutton(
        advanced_frame,
        text="Run evaluation after processing",
        variable=run_evaluation_var,
    )
    run_eval_check.grid(row=14, column=1, sticky="w")

    reset_button = ttk.Button(advanced_frame, text="Reset to defaults")
    reset_button.grid(row=15, column=1, sticky="e", pady=(6, 0))

    quick_actions = ttk.Frame(run_content, padding=8, style="Card.TFrame")
    quick_actions.grid(row=8, column=0, columnspan=2, sticky="ew", pady=(10, 0))
    quick_actions.columnconfigure(0, weight=1)
    quick_actions.columnconfigure(1, weight=1)

    quick_outputs = ttk.Button(quick_actions, text="Open Outputs", command=on_open_outputs)
    quick_logs = ttk.Button(quick_actions, text="Open Logs", command=on_open_logs)
    quick_outputs.grid(row=0, column=0, sticky="ew", padx=(0, 6))
    quick_logs.grid(row=0, column=1, sticky="ew")

    status_frame = ttk.Frame(run_content, padding=12, style="Card.TFrame")
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

    preview_frame = ttk.Frame(run_content, padding=12, style="Card.TFrame")
    preview_frame.grid(row=5, column=0, columnspan=2, sticky="nsew", pady=(0, 8))
    preview_frame.columnconfigure(0, weight=1)

    ttk.Label(preview_frame, text="Live Preview", style="Subheader.TLabel").grid(row=0, column=0, sticky="w")
    ttk.Label(preview_frame, textvariable=preview_stats_var, style="Card.TLabel").grid(row=0, column=1, sticky="e")

    preview_label = ttk.Label(preview_frame, text="Preview will appear while processing", style="Card.TLabel")
    preview_label.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(8, 0))

    output_frame = ttk.Frame(run_content, padding=12, style="Card.TFrame")
    output_frame.grid(row=6, column=0, columnspan=2, sticky="nsew", pady=(0, 8))
    output_frame.columnconfigure(1, weight=1)

    ttk.Label(output_frame, text="Results Viewer", style="Subheader.TLabel").grid(row=0, column=0, sticky="w")
    ttk.Label(output_frame, textvariable=viewer_status_var, style="Card.TLabel").grid(row=0, column=1, sticky="e")

    ttk.Label(output_frame, text="Overlay", style="Card.TLabel").grid(row=1, column=0, sticky="w")
    viewer_overlay_combo = ttk.Combobox(
        output_frame,
        textvariable=overlay_mode_var,
        values=list(OVERLAY_OPTIONS.keys()),
        state="readonly",
    )
    viewer_overlay_combo.grid(row=1, column=1, sticky="ew")

    viewer_label = ttk.Label(output_frame, text="Output video will appear after processing", style="Card.TLabel")
    viewer_label.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=(8, 0))

    viewer_controls = ttk.Frame(output_frame)
    viewer_controls.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(8, 0))
    viewer_controls.columnconfigure(0, weight=1)
    viewer_controls.columnconfigure(1, weight=1)
    viewer_controls.columnconfigure(2, weight=1)

    viewer_state = {"cap": None, "playing": False, "after_id": None, "fps": 30.0}

    def stop_viewer() -> None:
        if viewer_state["after_id"]:
            root.after_cancel(viewer_state["after_id"])
        viewer_state["after_id"] = None
        viewer_state["playing"] = False
        cap = viewer_state["cap"]
        if cap is not None:
            cap.release()
        viewer_state["cap"] = None

    def show_viewer_frame(frame) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ok, buffer = cv2.imencode(".png", rgb)
        if not ok:
            return
        encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
        image = PhotoImage(data=encoded)
        viewer_label.configure(image=image, text="")
        viewer_label.image = image

    def load_viewer_video() -> None:
        stop_viewer()
        path = overlay_file(overlay_mode_var.get())
        if not path.exists():
            viewer_status_var.set("Viewer idle")
            viewer_label.configure(image="", text=f"No output found: {path.name}")
            return
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            viewer_status_var.set("Viewer idle")
            viewer_label.configure(image="", text="Unable to open output video")
            return
        viewer_state["cap"] = cap
        viewer_state["fps"] = cap.get(cv2.CAP_PROP_FPS) or 30.0
        ret, frame = cap.read()
        if ret:
            show_viewer_frame(frame)
            viewer_status_var.set(f"Loaded {path.name}")
        else:
            viewer_label.configure(image="", text="Output video empty")
            viewer_status_var.set("Viewer idle")

    def play_viewer() -> None:
        if viewer_state["cap"] is None:
            load_viewer_video()
        if viewer_state["cap"] is None:
            return
        viewer_state["playing"] = True

        def _step() -> None:
            cap = viewer_state["cap"]
            if cap is None or not viewer_state["playing"]:
                return
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            if ret:
                show_viewer_frame(frame)
            delay_ms = int(1000 / max(1.0, viewer_state["fps"]))
            viewer_state["after_id"] = root.after(delay_ms, _step)

        _step()
        viewer_status_var.set("Playing")

    def pause_viewer() -> None:
        viewer_state["playing"] = False
        viewer_status_var.set("Paused")

    def restart_viewer() -> None:
        if viewer_state["cap"] is not None:
            viewer_state["cap"].set(cv2.CAP_PROP_POS_FRAMES, 0)
        play_viewer()

    viewer_play_button = ttk.Button(viewer_controls, text="Play", command=play_viewer)
    viewer_pause_button = ttk.Button(viewer_controls, text="Pause", command=pause_viewer)
    viewer_restart_button = ttk.Button(viewer_controls, text="Restart", command=restart_viewer)
    viewer_play_button.grid(row=0, column=0, sticky="ew", padx=(0, 6))
    viewer_pause_button.grid(row=0, column=1, sticky="ew", padx=(0, 6))
    viewer_restart_button.grid(row=0, column=2, sticky="ew")

    overlay_mode_var.trace_add("write", lambda *_: load_viewer_video())

    problems_frame = ttk.Frame(run_content, padding=12, style="Card.TFrame")
    problems_frame.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(0, 8))
    problems_frame.columnconfigure(0, weight=1)

    ttk.Label(problems_frame, text="Problems", style="Subheader.TLabel").grid(row=0, column=0, sticky="w")
    problems_list = Text(problems_frame, height=5, bg="#0f1216", fg="#e7e9ee", relief="flat", font=base_font)
    problems_list.configure(state="disabled")
    problems_list.grid(row=1, column=0, sticky="ew", pady=(6, 0))
    bind_mousewheel(problems_list)

    codex_frame = ttk.Frame(run_content, padding=12, style="Card.TFrame")
    codex_frame.grid(row=8, column=0, columnspan=2, sticky="ew", pady=(0, 8))
    codex_frame.columnconfigure(0, weight=1)
    codex_frame.columnconfigure(1, weight=1)
    codex_frame.columnconfigure(2, weight=1)

    ttk.Label(codex_frame, text="Codex Fix Packets", style="Subheader.TLabel").grid(
        row=0, column=0, sticky="w"
    )
    ttk.Label(codex_frame, textvariable=pending_count_var, style="Card.TLabel").grid(
        row=0, column=1, sticky="w"
    )

    copy_warning_button = ttk.Button(
        codex_frame,
        text="Copy Low Pose Warning + Summary",
        command=on_copy_eval_warning,
    )
    copy_warning_button.grid(row=1, column=0, sticky="ew", pady=(6, 0), padx=(0, 6))

    codex_packet_button = ttk.Button(
        codex_frame,
        text="Generate Codex Fix Packet",
        style="Accent.TButton",
        command=on_generate_codex_packet,
    )
    codex_packet_button.grid(row=1, column=1, sticky="ew", pady=(6, 0), padx=(0, 6))

    codex_pending_button = ttk.Button(
        codex_frame,
        text="Generate Plan from Pending + Send",
        command=on_generate_plan_from_pending,
    )
    codex_pending_button.grid(row=1, column=2, sticky="ew", pady=(6, 0))

    data_content.columnconfigure(0, weight=1)

    data_group = ttk.LabelFrame(data_content, text="Storage", padding=12)
    data_group.grid(row=0, column=0, sticky="ew")
    ttk.Label(data_group, text=f"Outputs: {get_outputs_root()}").grid(row=0, column=0, sticky="w")
    ttk.Label(data_group, text=f"Logs: {get_log_root()}").grid(row=1, column=0, sticky="w")

    data_actions = ttk.Frame(data_content, padding=12, style="Card.TFrame")
    data_actions.grid(row=1, column=0, sticky="ew", pady=(12, 0))
    data_actions.columnconfigure(0, weight=1)
    data_actions.columnconfigure(1, weight=1)

    outputs_button = ttk.Button(data_actions, text="Open Outputs", style="Accent.TButton", command=on_open_outputs)
    logs_button = ttk.Button(data_actions, text="Open Logs", command=on_open_logs)
    outputs_button.grid(row=0, column=0, sticky="ew", padx=(0, 8))
    logs_button.grid(row=0, column=1, sticky="ew")

    settings_content.columnconfigure(0, weight=1)

    update_group = ttk.LabelFrame(settings_content, text="Updates", padding=12)
    update_group.grid(row=0, column=0, sticky="ew")
    update_group.columnconfigure(1, weight=1)

    ttk.Label(update_group, text="Current version", style="Card.TLabel").grid(row=0, column=0, sticky="w")
    ttk.Label(update_group, text=f"v{get_current_version()}", style="Card.TLabel").grid(row=0, column=1, sticky="w")

    ttk.Label(update_group, text="Update channel", style="Card.TLabel").grid(row=1, column=0, sticky="w")
    channel_combo = ttk.Combobox(
        update_group,
        textvariable=update_channel_var,
        values=["Stable", "Beta"],
        state="readonly",
    )
    channel_combo.grid(row=1, column=1, sticky="w")

    ttk.Label(update_group, text="Last checked", style="Card.TLabel").grid(row=2, column=0, sticky="w")
    ttk.Label(update_group, textvariable=update_last_checked_var, style="Card.TLabel").grid(row=2, column=1, sticky="w")

    ttk.Label(update_group, text="Status", style="Card.TLabel").grid(row=3, column=0, sticky="w")
    ttk.Label(update_group, textvariable=update_status_var, style="Card.TLabel").grid(row=3, column=1, sticky="w")

    ttk.Label(update_group, text="Latest version", style="Card.TLabel").grid(row=4, column=0, sticky="w")
    ttk.Label(update_group, textvariable=update_latest_version_var, style="Card.TLabel").grid(row=4, column=1, sticky="w")

    ttk.Label(update_group, text="Release notes", style="Card.TLabel").grid(row=5, column=0, sticky="nw")
    update_notes_text = Text(update_group, height=6, bg="#0f1216", fg="#e7e9ee", relief="flat", font=base_font)
    update_notes_text.grid(row=5, column=1, sticky="ew")
    update_notes_text.configure(state="disabled")
    bind_mousewheel(update_notes_text)

    update_error_label = ttk.Label(update_group, textvariable=update_error_var, style="Card.TLabel", foreground="#f2994a")
    update_error_label.grid(row=6, column=0, columnspan=2, sticky="w", pady=(4, 0))

    update_actions = ttk.Frame(update_group)
    update_actions.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(8, 0))
    update_actions.columnconfigure(0, weight=1)
    update_actions.columnconfigure(1, weight=1)
    update_actions.columnconfigure(2, weight=1)
    update_actions.columnconfigure(3, weight=1)

    update_check_button = ttk.Button(update_actions, text="Check updates", command=on_check_updates)
    update_action_button = ttk.Button(update_actions, text="Download & install", style="Accent.TButton", command=on_update_now)
    open_notes_button = ttk.Button(update_actions, text="Open full notes", command=on_open_release_notes)
    bootstrap_button = ttk.Button(update_actions, text="Bootstrap / Repair", command=on_bootstrap_repair)

    update_check_button.grid(row=0, column=0, sticky="ew", padx=(0, 6))
    update_action_button.grid(row=0, column=1, sticky="ew", padx=(0, 6))
    open_notes_button.grid(row=0, column=2, sticky="ew", padx=(0, 6))
    bootstrap_button.grid(row=0, column=3, sticky="ew")
    update_action_button.configure(state="disabled")

    ttk.Label(update_group, text="Update history (last 5)", style="Card.TLabel").grid(
        row=8, column=0, sticky="nw", pady=(8, 0)
    )
    update_history_text = Text(update_group, height=5, bg="#0f1216", fg="#e7e9ee", relief="flat", font=base_font)
    update_history_text.grid(row=8, column=1, sticky="ew", pady=(8, 0))
    update_history_text.configure(state="disabled")
    bind_mousewheel(update_history_text)

    set_release_notes(update_release_notes_var.get())
    refresh_update_history_list()

    ui_group = ttk.LabelFrame(settings_content, text="UI Preferences", padding=12)
    ui_group.grid(row=1, column=0, sticky="ew", pady=(12, 0))
    ui_group.columnconfigure(1, weight=1)

    ttk.Label(ui_group, text="UI scale multiplier", style="Card.TLabel").grid(row=0, column=0, sticky="w")
    ui_scale_slider = ttk.Scale(ui_group, from_=0.8, to=1.6, variable=ui_scale_var)
    ui_scale_slider.grid(row=0, column=1, sticky="ew")
    ui_scale_value = ttk.Label(ui_group, textvariable=ui_scale_var, style="Card.TLabel")
    ui_scale_value.grid(row=0, column=2, sticky="w", padx=(6, 0))

    ttk.Label(ui_group, text="Base font size (1080p)", style="Card.TLabel").grid(row=1, column=0, sticky="w")
    ui_font_spin = ttk.Spinbox(ui_group, from_=9, to=32, textvariable=ui_font_size_var, width=6)
    ui_font_spin.grid(row=1, column=1, sticky="w")

    ttk.Label(ui_group, text="Restart to apply UI changes.", style="Card.TLabel").grid(
        row=2,
        column=0,
        columnspan=3,
        sticky="w",
        pady=(6, 0),
    )

    log_frame = ttk.LabelFrame(container, text="Logs", padding=8)
    log_frame.columnconfigure(0, weight=1)
    log_frame.rowconfigure(0, weight=1)

    log_text = Text(log_frame, height=10, bg="#0f1216", fg="#e7e9ee", relief="flat", font=base_font)
    log_text.configure(state="disabled")
    log_text.grid(row=0, column=0, sticky="nsew")

    log_scroll = ttk.Scrollbar(log_frame, command=log_text.yview)
    log_scroll.grid(row=0, column=1, sticky="ns")
    log_text.configure(yscrollcommand=log_scroll.set)
    bind_mousewheel(log_text)
    paned_main.add(log_frame, weight=1)
    try:
        paned_main.pane(log_frame, minsize=0)
    except TclError as exc:
        logging.exception(
            "Failed to configure paned window for logs (widget=%s, options=%s): %s",
            paned_main.winfo_class(),
            {"minsize": 0},
            exc,
        )

    handler = TkTextHandler(log_text, root)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(handler)

    problem_buffer: deque[str] = deque(maxlen=6)

    settings_controls.extend(
        [
            save_pose_check,
            save_combat_check,
            save_thumbnails_check,
            save_background_check,
            export_overlay_check,
            debug_overlay_check,
            draw_all_tracks_check,
            overlay_mode_combo,
            dev_overlay_combo,
            dev_save_combat_check,
            dev_run_judge_check,
            max_tracks_spin,
            track_sort_combo,
            live_preview_check,
            run_eval_check,
            run_judge_check,
            smoothing_enabled_check,
            backend_combo,
            mode_combo,
            manual_entry,
            smoothing_scale,
            conf_scale,
            reset_button,
            ui_scale_slider,
            ui_font_spin,
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
    bind_setting(save_combat_var, "save_combat_overlay", bool)
    bind_setting(save_thumbnails_var, "save_thumbnails", bool)
    bind_setting(save_background_var, "save_background_tracks", bool)
    bind_setting(debug_overlay_var, "debug_overlay", bool)
    bind_setting(draw_all_tracks_var, "draw_all_tracks", bool)
    bind_setting(overlay_mode_var, "overlay_mode", str)
    bind_setting(max_tracks_var, "max_tracks", int)
    bind_setting(track_sort_var, "track_sort", str)
    bind_setting(live_preview_var, "live_preview", bool)
    bind_setting(run_evaluation_var, "run_evaluation", bool)
    bind_setting(run_judge_var, "run_judge", bool)
    bind_setting(smoothing_enabled_var, "smoothing_enabled", bool)
    bind_setting(smoothing_alpha_var, "smoothing_alpha", float)
    bind_setting(min_conf_var, "confidence_threshold", float)
    bind_setting(tracking_backend_var, "tracking_backend", str)
    bind_setting(foreground_mode_var, "foreground_mode", str)
    bind_setting(manual_tracks_var, "manual_track_ids", str)
    bind_setting(update_channel_var, "update_channel", str)
    bind_setting(ui_scale_var, "ui_scale_multiplier", float)
    bind_setting(ui_font_size_var, "ui_font_size", int)

    def clamp_max_tracks() -> None:
        try:
            value = int(max_tracks_var.get())
        except Exception:
            value = defaults["max_tracks"]
        if value < 1:
            max_tracks_var.set(1)
        elif value > 50:
            max_tracks_var.set(50)

    max_tracks_var.trace_add("write", lambda *_: clamp_max_tracks())


    def update_draw_all_warning() -> None:
        if draw_all_tracks_var.get():
            draw_all_warning.grid()
            all_tracks_warning_var.set(
                "All-tracks view can look cluttered; consider Joints-only or reduce Top-N tracks."
            )
        else:
            draw_all_warning.grid_remove()
            all_tracks_warning_var.set("")

    def update_smoothing_state() -> None:
        smoothing_scale.configure(state="normal" if smoothing_enabled_var.get() else "disabled")

    def reset_defaults() -> None:
        export_overlay_var.set(defaults["export_overlay"])
        save_pose_var.set(defaults["save_pose_json"])
        save_combat_var.set(defaults["save_combat_overlay"])
        save_thumbnails_var.set(defaults["save_thumbnails"])
        save_background_var.set(defaults["save_background_tracks"])
        debug_overlay_var.set(defaults["debug_overlay"])
        draw_all_tracks_var.set(defaults["draw_all_tracks"])
        overlay_mode_var.set(defaults["overlay_mode"])
        max_tracks_var.set(defaults["max_tracks"])
        track_sort_var.set(defaults["track_sort"])
        live_preview_var.set(defaults["live_preview"])
        run_evaluation_var.set(defaults["run_evaluation"])
        run_judge_var.set(defaults["run_judge"])
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
        processing_state["running"] = running
        for button in action_buttons:
            button.configure(state=state)
        outputs_button.configure(state=state)
        logs_button.configure(state=state)
        update_check_button.configure(state="normal")
        update_action_button.configure(state="normal" if update_status_var.get() == "Update available" else "disabled")
        bootstrap_button.configure(state="normal")
        quick_outputs.configure(state=state)
        quick_logs.configure(state=state)
        for widget in settings_controls:
            if isinstance(widget, ttk.Combobox):
                widget.configure(state="disabled" if running else "readonly")
            else:
                widget.configure(state=state)
        cancel_button.configure(state="normal" if running else "disabled")
        dev_cancel_button.configure(state="normal" if running else "disabled")

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
            run_state["last_error"] = str(info["error"])
        if "evaluation_summary" in info:
            evaluation_var.set(f"Evaluation: {info['evaluation_summary']}")
            run_state["evaluation_summary"] = info.get("evaluation_summary")
        if "evaluation_stats" in info:
            run_state["evaluation_stats"] = info.get("evaluation_stats")
        if "evaluation_json" in info:
            run_state["evaluation_json_path"] = info.get("evaluation_json")
        if "evaluation_text" in info:
            run_state["evaluation_text_path"] = info.get("evaluation_text")
        if "evaluation_warning" in info:
            problem_buffer.appendleft(f"[EVAL] {info['evaluation_warning']}")
            problems_list.configure(state="normal")
            problems_list.delete("1.0", "end")
            problems_list.insert("end", "\n".join(problem_buffer))
            problems_list.configure(state="disabled")
        if "judge_output" in info:
            judge_output = info.get("judge_output", {})
            if isinstance(judge_output, dict):
                report_path = judge_output.get("report_html")
                bundle_dir = judge_output.get("bundle_dir")
                if report_path:
                    dev_judge_report_var.set(f"Judge report: {report_path}")
                    dev_judge_state["report_path"] = str(report_path)
                if bundle_dir:
                    dev_bundle_var.set(f"Bundle: {bundle_dir}")
                refresh_artifacts()
                run_state["judge_output"] = judge_output
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
                run_state.setdefault("problems", []).append(problem)
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
        run_state["evaluation_summary"] = None
        run_state["evaluation_text_path"] = None
        run_state["evaluation_json_path"] = None
        run_state["evaluation_stats"] = None
        run_state["stack_trace"] = None
        run_state["last_error"] = None
        run_state["problems"] = []
        run_state["judge_output"] = None

        def status_callback(message: str, progress_value: float | None) -> None:
            root.after(0, update_status, message, progress_value)

        def info_callback(info: dict[str, object]) -> None:
            root.after(0, update_info, info)

        def run_background() -> None:
            run_status = "success"
            try:
                mode = foreground_mode_var.get()
                if mode != "Auto (closest/most active)":
                    logging.info("Foreground mode: %s", mode)
                manual_ids = [chunk.strip() for chunk in manual_tracks_var.get().split(",") if chunk.strip()]
                if mode == "Manual pick" and not manual_ids:
                    root.after(0, messagebox.showwarning, "Run Overlay", "Enter track IDs for Manual pick mode.")
                    root.after(0, set_running, False)
                    return
                overlay_mode = overlay_slug(overlay_mode_var.get())
                track_sort = "stability"
                options = ProcessingOptions(
                    export_overlay_video=export_overlay_var.get(),
                    save_pose_json=save_pose_var.get(),
                    save_combat_overlay=save_combat_var.get(),
                    save_thumbnails=save_thumbnails_var.get(),
                    save_background_tracks=save_background_var.get(),
                    foreground_mode=mode,
                    debug_overlay=debug_overlay_var.get(),
                    draw_all_tracks=draw_all_tracks_var.get(),
                    smoothing_alpha=float(smoothing_alpha_var.get()),
                    smoothing_enabled=smoothing_enabled_var.get(),
                    min_keypoint_confidence=float(min_conf_var.get()),
                    overlay_mode=overlay_mode,
                    max_tracks=max(1, min(int(max_tracks_var.get()), 50)),
                    track_sort=track_sort,
                    live_preview=live_preview_var.get(),
                    run_evaluation=run_evaluation_var.get(),
                    run_judge=run_judge_var.get(),
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
                root.after(0, load_viewer_video)
                root.after(0, messagebox.showinfo, "Run Overlay", "Processing complete.")
            except ProcessingCancelled:
                logging.warning("Processing cancelled by user.")
                run_status = "cancelled"
                root.after(0, update_status, "Processing cancelled.", 0)
                root.after(0, update_info, {"stage": "Cancelled"})
            except Exception as exc:
                logging.exception("Processing failed: %s", exc)
                run_status = "failed"
                run_state["stack_trace"] = traceback.format_exc()
                root.after(0, update_info, {"error": exc, "stage": "Failed"})
                root.after(0, messagebox.showerror, "Run Overlay", f"Processing failed: {exc}")
                root.after(0, update_status, "Processing failed.", 0)
            finally:
                if run_status in {"success", "failed"}:
                    packet = build_codex_packet()
                    path = save_pending_packet(packet)
                    if path:
                        logging.info("Saved pending Codex packet: %s", path)
                    root.after(0, update_pending_count)
                def _finish() -> None:
                    set_running(False)
                    if install_after_job["pending"]:
                        install_after_job["pending"] = False
                        trigger_update(allow_running=True)

                root.after(0, _finish)

        thread = threading.Thread(target=run_background, daemon=True)
        thread.start()

    run_button.configure(command=on_run_overlay)

    validate_button = ttk.Button(run_content, text="Validate Output JSON schema", command=on_validate)
    validate_button.grid(row=9, column=0, sticky="w", pady=(6, 0))

    action_buttons.append(validate_button)

    set_running(False)
    logging.info("Control Center UI ready")
    if not args.ui_smoke_test:
        load_viewer_video()

    if args.ui_smoke_test:
        try:
            root.update_idletasks()
            root.update()
        except TclError as exc:
            logging.exception("UI smoke test failed: %s", exc)
            raise SystemExit(1) from exc
        finally:
            root.destroy()
        logging.info("UI smoke test completed")
        return

    root.mainloop()


if __name__ == "__main__":
    main()
