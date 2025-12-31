from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import shutil
import socket
import ssl
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from subprocess import Popen, run
import subprocess
from typing import Any, Dict, Optional

from core.paths import (
    get_app_root,
    get_bootstrap_root,
    get_bootstrapper_path,
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
BOOTSTRAP_CACHE_NAME = "bootstrap_cache.json"
BOOTSTRAP_ERROR_TITLE = "FightingOverlay failed to start"
MAX_NETWORK_RETRIES = 2
BOOTSTRAP_POINTER_NAME = "current_bootstrapper.txt"
BOOTSTRAP_KEEP_VERSIONS = 3
CONTROL_CENTER_STARTUP_WAIT_S = 1.0
CONTROL_CENTER_LOG_LIMIT = 8192
DESKTOP_ERROR_LOG_NAME = "FightingOverlayBootstrap_errorlog.txt"


class CaptivePortalError(RuntimeError):
    pass


def resolve_log_root() -> Path:
    try:
        log_root = get_log_root()
        log_root.mkdir(parents=True, exist_ok=True)
        return log_root
    except Exception:
        fallback = Path(tempfile.gettempdir()) / "FightingOverlay" / "logs"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def setup_logging(log_root: Path) -> Path:
    log_path = log_root / "bootstrap.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8")],
    )
    bootstrap_version = os.environ.get("FIGHTINGOVERLAY_BOOTSTRAP_VERSION", "unknown")
    local_appdata = os.environ.get("LOCALAPPDATA", "unset")
    logging.info(
        "BOOTSTRAP_START frozen=%s bootstrap_version=%s python=%s os=%s executable=%s cwd=%s local_appdata=%s",
        getattr(sys, "frozen", False),
        bootstrap_version,
        sys.version.replace("\n", " "),
        platform.platform(),
        sys.executable,
        os.getcwd(),
        local_appdata,
    )
    logging.info("BOOTSTRAP_START argv=%s sys.path.head=%s", sys.argv, sys.path[:6])
    return log_path


def resolve_desktop_error_log_path(log_root: Path) -> Path:
    candidates = []
    userprofile = os.environ.get("USERPROFILE")
    if userprofile:
        candidates.append(Path(userprofile) / "Desktop")
    try:
        candidates.append(Path.home() / "Desktop")
    except Exception:
        logging.exception("Failed resolving home directory for Desktop path")
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate / DESKTOP_ERROR_LOG_NAME
    return log_root / DESKTOP_ERROR_LOG_NAME


def write_error_payload(path: Path, payload: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(path, payload)
    except Exception:
        try:
            logging.exception("Failed to write error log payload: %s", path)
        except Exception:
            pass


def log_step(state: Dict[str, Any], step: str, **fields: Any) -> None:
    state["step"] = step
    extras = " ".join(f"{key}={value}" for key, value in fields.items() if value is not None)
    if extras:
        logging.info("BOOTSTRAP_STEP=%s %s", step, extras)
    else:
        logging.info("BOOTSTRAP_STEP=%s", step)


def ensure_directories() -> None:
    get_app_root().mkdir(parents=True, exist_ok=True)
    get_data_root().mkdir(parents=True, exist_ok=True)
    get_log_root().mkdir(parents=True, exist_ok=True)
    get_outputs_root().mkdir(parents=True, exist_ok=True)
    get_models_root().mkdir(parents=True, exist_ok=True)
    get_versions_root().mkdir(parents=True, exist_ok=True)
    get_bootstrap_root().mkdir(parents=True, exist_ok=True)


def get_bootstrapper_pointer_path() -> Path:
    return get_bootstrap_root() / BOOTSTRAP_POINTER_NAME


def resolve_bootstrapper_pointer() -> Optional[Path]:
    pointer = get_bootstrapper_pointer_path()
    if not pointer.exists():
        return None
    try:
        target = Path(pointer.read_text(encoding="utf-8").strip())
    except OSError:
        logging.exception("Failed reading bootstrapper pointer")
        return None
    if not target.exists():
        logging.warning("Bootstrapper pointer target missing: %s", target)
        return None
    return target


def _sample_payload(payload: str, limit: int = 200) -> str:
    sample = payload[:limit]
    sample = " ".join(sample.split())
    return sample


def _captive_portal_suspected(payload: str, content_type: str) -> bool:
    if "application/json" in content_type.lower():
        return False
    lowered = payload[:200].lower()
    markers = ("<html", "<!doctype", "</html", "http-equiv=\"refresh\"", "captive portal")
    return any(marker in lowered for marker in markers)


def fetch_latest_release(state: Dict[str, Any]) -> dict:
    log_step(state, "FETCH_LATEST_RELEASE", url=RELEASES_URL)
    request = urllib.request.Request(
        RELEASES_URL,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "FightingOverlayBootstrap"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = response.read().decode("utf-8")
        content_type = response.headers.get("Content-Type", "")
    if _captive_portal_suspected(payload, content_type):
        sample = _sample_payload(payload)
        logging.warning("CAPTIVE_PORTAL_SUSPECTED content_type=%s sample=%s", content_type, sample)
        raise CaptivePortalError("Update server returned HTML (possible captive portal).")
    if "application/json" not in content_type.lower():
        sample = _sample_payload(payload)
        logging.warning("UNEXPECTED_CONTENT_TYPE content_type=%s sample=%s", content_type, sample)
        raise RuntimeError("Update server returned unexpected content.")
    log_step(state, "PARSE_RELEASE_METADATA")
    release = json.loads(payload)
    log_step(
        state,
        "RESOLVED_RELEASE",
        tag=release.get("tag_name"),
        name=release.get("name"),
    )
    return release


def download_asset(state: Dict[str, Any], url: str, destination: Path) -> None:
    log_step(state, "DOWNLOAD_ASSET", url=url, destination=destination)
    temp_path = destination.with_name(f"{destination.name}.partial")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, temp_path.open("wb") as handle:
            shutil.copyfileobj(response, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, destination)
    except Exception:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                logging.exception("Failed to clean up partial download: %s", temp_path)
        raise


def extract_zip(state: Dict[str, Any], zip_path: Path, destination: Path) -> None:
    log_step(state, "EXTRACT_ASSET", zip_path=zip_path, destination=destination)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(destination)


def is_retryable_fetch_error(exc: Exception) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in {408, 429} or 500 <= exc.code <= 599
    if isinstance(
        exc,
        (
            urllib.error.URLError,
            socket.gaierror,
            socket.timeout,
            TimeoutError,
            ConnectionResetError,
            ConnectionAbortedError,
        ),
    ):
        return True
    return False


def is_retryable_install_error(exc: Exception) -> bool:
    if isinstance(exc, zipfile.BadZipFile):
        return True
    if isinstance(exc, PermissionError):
        return False
    return isinstance(exc, OSError)


def atomic_write(path: Path, content: str) -> None:
    temp_path = path.with_suffix(".tmp")
    temp_path.write_text(content, encoding="utf-8")
    os.replace(temp_path, path)


def record_last_update(
    status: str,
    version: Optional[str],
    message: Optional[str] = None,
    previous_version: Optional[str] = None,
) -> None:
    payload = {
        "status": status,
        "version": version,
        "message": message,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "previous_version": previous_version,
    }
    get_last_update_path().write_text(json.dumps(payload, indent=2), encoding="utf-8")


def clean_old_versions(current_version: str, previous_version: Optional[str]) -> None:
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


def _files_match(source: Path, destination: Path) -> bool:
    try:
        source_stat = source.stat()
        destination_stat = destination.stat()
    except FileNotFoundError:
        return False
    return (
        source_stat.st_size == destination_stat.st_size
        and source_stat.st_mtime_ns == destination_stat.st_mtime_ns
    )


def _safe_version_label(version: str) -> str:
    cleaned = "".join(char for char in version if char.isalnum() or char in ("-", "_", "."))
    return cleaned or "unknown"


def copy_bootstrapper(version: str) -> Path:
    bootstrap_root = get_bootstrap_root()
    bootstrap_root.mkdir(parents=True, exist_ok=True)
    destination = get_bootstrapper_path()
    source = Path(sys.executable)
    if not getattr(sys, "frozen", False) or not source.exists():
        return destination
    if source.resolve() == destination.resolve():
        return destination
    if destination.exists() and _files_match(source, destination):
        logging.info("Bootstrapper already up to date at %s", destination)
        return destination

    version_label = _safe_version_label(version)
    versioned_destination = bootstrap_root / f"FightingOverlayBootstrap_{version_label}.exe"
    if not (versioned_destination.exists() and _files_match(source, versioned_destination)):
        try:
            shutil.copy2(source, versioned_destination)
        except PermissionError as exc:
            logging.warning("Bootstrapper copy failed: %s", exc)
            return destination
    try:
        atomic_write(get_bootstrapper_pointer_path(), str(versioned_destination.resolve()))
        logging.info("Updated bootstrapper pointer to %s", versioned_destination)
    except Exception:
        logging.exception("Failed to update bootstrapper pointer")

    if not destination.exists():
        try:
            shutil.copy2(versioned_destination, destination)
        except PermissionError as exc:
            logging.warning("Bootstrapper staging copy failed: %s", exc)

    try:
        versioned_files = sorted(
            bootstrap_root.glob("FightingOverlayBootstrap_*.exe"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for old_path in versioned_files[BOOTSTRAP_KEEP_VERSIONS:]:
            logging.info("Pruning old bootstrapper %s", old_path)
            old_path.unlink(missing_ok=True)
    except OSError:
        logging.exception("Failed pruning old bootstrapper versions")

    return versioned_destination


def _remove_path(path: Path) -> None:
    try:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.exists():
            path.unlink()
    except OSError:
        logging.exception("Failed to remove path: %s", path)


def install_release(state: Dict[str, Any], release: dict, simulate_bad_zip: bool) -> Path:
    log_step(state, "INSTALL_RELEASE")
    assets = release.get("assets", [])
    asset = next((item for item in assets if item.get("name") == ASSET_NAME), None)
    if not asset:
        raise RuntimeError(f"Release is missing asset {ASSET_NAME}")

    version = release.get("tag_name") or release.get("name") or "unknown"
    log_step(
        state,
        "SELECT_RELEASE_ASSET",
        tag=version,
        asset=asset.get("name"),
        url=asset.get("browser_download_url"),
    )
    temp_dir = Path(tempfile.mkdtemp(prefix="fightingoverlay_"))
    zip_path = temp_dir / ASSET_NAME
    extract_dir = temp_dir / "extract"
    extract_dir.mkdir(parents=True, exist_ok=True)

    if simulate_bad_zip:
        raise zipfile.BadZipFile("Simulated bad zip")

    download_asset(state, asset["browser_download_url"], zip_path)
    log_step(state, "VERIFY_ASSET", zip_path=zip_path)
    if not zipfile.is_zipfile(zip_path):
        _remove_path(zip_path)
        raise zipfile.BadZipFile("Downloaded asset is not a valid zip")

    try:
        extract_zip(state, zip_path, extract_dir)
    except (zipfile.BadZipFile, OSError) as exc:
        _remove_path(zip_path)
        _remove_path(extract_dir)
        raise exc

    versions_root = get_versions_root()
    target_dir = versions_root / version
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.move(str(extract_dir), str(target_dir))

    return target_dir


def find_offline_executable(target_dir: Path) -> Optional[Path]:
    candidates = [
        target_dir / "ControlCenter.exe",
        target_dir / "ControlCenter" / "ControlCenter.exe",
        target_dir / "bin" / "ControlCenter.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            logging.info("Offline executable found at %s", candidate)
            return candidate
    try:
        entries = [entry.name for entry in list(target_dir.iterdir())[:30]]
        logging.warning("Offline executable not found. Top-level entries: %s", entries)
    except OSError:
        logging.exception("Failed to list offline target directory: %s", target_dir)
    return None


def _read_log_snippet(path: Path, limit: int = CONTROL_CENTER_LOG_LIMIT) -> str:
    if not path.exists():
        return ""
    try:
        data = path.read_bytes()
    except OSError:
        logging.exception("Failed reading ControlCenter log snippet: %s", path)
        return ""
    snippet = data[-limit:].decode("utf-8", errors="replace")
    return _sample_payload(snippet, limit=200)


def write_controlcenter_crash(
    log_root: Path,
    returncode: int,
    snippet: str,
    launch_log: Path,
) -> None:
    classification = "controlcenter_crash"
    summary = "ControlCenter crashed on startup."
    hint = "Review the ControlCenter launch log and bootstrap log for details."
    payload = (
        f"timestamp_utc={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n"
        f"classification={classification}\n"
        f"summary={summary}\n"
        f"hint={hint}\n"
        f"returncode={returncode}\n"
        f"snippet={snippet}\n"
        f"controlcenter_log_path={launch_log}\n"
        f"log_path={log_root / 'bootstrap.log'}\n"
    )
    desktop_path = resolve_desktop_error_log_path(log_root)
    write_error_payload(desktop_path, payload)
    for candidate_root in (
        log_root,
        Path(tempfile.gettempdir()) / "FightingOverlay" / "logs",
    ):
        try:
            candidate_root.mkdir(parents=True, exist_ok=True)
            path = candidate_root / "bootstrap_last_error.txt"
            atomic_write(path, payload)
            return
        except Exception:
            try:
                logging.exception("Failed to write bootstrap_last_error.txt in %s", candidate_root)
            except Exception:
                pass


def show_controlcenter_crash_dialog(log_root: Path, launch_log: Path) -> None:
    try:
        import ctypes

        response = ctypes.windll.user32.MessageBoxW(
            0,
            f"ControlCenter crashed on startup.\n\nLog file:\n{launch_log}\n\nOpen logs folder?",
            BOOTSTRAP_ERROR_TITLE,
            0x00000004,
        )
        if response == 6:  # IDYES
            safe_open_logs(log_root)
    except Exception:
        logging.exception("Failed to show ControlCenter crash dialog")


def launch_control_center(target_dir: Path, log_root: Path) -> bool:
    executable = find_offline_executable(target_dir)
    if not executable:
        raise RuntimeError("ControlCenter.exe not found in installed version")

    log_name = f"controlcenter_launch_{time.strftime('%Y%m%d_%H%M%S')}.log"
    launch_log = log_root / log_name
    hide_console = os.environ.get("FIGHTINGOVERLAY_SHOW_CONSOLE") != "1"
    if hide_console and os.name == "nt":
        creationflags = subprocess.CREATE_NO_WINDOW
    else:
        creationflags = 0
        logging.info("ControlCenter console window enabled (debug).")
    with launch_log.open("wb") as output_handle:
        proc = Popen(
            [str(executable)],
            cwd=str(executable.parent),
            stdout=output_handle,
            stderr=output_handle,
            creationflags=creationflags,
        )
        try:
            returncode = proc.wait(timeout=CONTROL_CENTER_STARTUP_WAIT_S)
        except subprocess.TimeoutExpired:
            returncode = None
        output_handle.flush()

    if returncode is None:
        logging.info("ControlCenter launched (pid=%s)", proc.pid)
        return True

    snippet = _read_log_snippet(launch_log)
    logging.error(
        "ControlCenter exited early returncode=%s log=%s snippet=%s",
        returncode,
        launch_log,
        snippet,
    )
    write_controlcenter_crash(log_root, returncode, snippet, launch_log)
    show_controlcenter_crash_dialog(log_root, launch_log)
    return False


def maybe_handoff_to_current_bootstrapper(argv: list[str]) -> bool:
    if os.environ.get("FIGHTINGOVERLAY_BOOTSTRAP_SKIP_HANDOFF") == "1":
        return False
    target = resolve_bootstrapper_pointer()
    if not target:
        return False
    try:
        if Path(sys.executable).resolve() == target.resolve():
            return False
    except OSError:
        logging.exception("Failed comparing bootstrapper paths")
        return False

    env = os.environ.copy()
    env["FIGHTINGOVERLAY_BOOTSTRAP_SKIP_HANDOFF"] = "1"
    logging.info("Handing off to bootstrapper %s", target)
    Popen([str(target), *argv[1:]], cwd=str(target.parent), env=env)
    return True


def get_cache_path() -> Path:
    return get_app_root() / BOOTSTRAP_CACHE_NAME


def read_bootstrap_cache() -> Optional[Dict[str, Any]]:
    try:
        cache_path = get_cache_path()
    except Exception:
        logging.exception("Unable to resolve cache path")
        return None
    if not cache_path.exists():
        return None
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        corrupt_path = cache_path.with_name(f"{cache_path.name}.corrupt.{timestamp}")
        try:
            os.replace(cache_path, corrupt_path)
            logging.warning("Cached install metadata corrupt; renamed to %s", corrupt_path)
        except OSError:
            logging.exception("Failed to quarantine corrupt cache file")
        return None


def write_bootstrap_cache(payload: Dict[str, Any]) -> None:
    try:
        cache_path = get_cache_path()
    except Exception:
        logging.exception("Unable to resolve cache path")
        return
    atomic_write(cache_path, json.dumps(payload, indent=2))


def update_cache_success(version: str, target_dir: Path) -> None:
    payload = read_bootstrap_cache() or {}
    payload.update(
        {
            "last_good_version": version,
            "last_good_path": str(target_dir.resolve()),
            "last_success_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    )
    write_bootstrap_cache(payload)


def update_cache_attempt(version: Optional[str], status: str) -> None:
    payload = read_bootstrap_cache() or {}
    payload.update(
        {
            "last_attempt_version": version,
            "last_attempt_status": status,
        }
    )
    write_bootstrap_cache(payload)


def validate_offline_target(target_dir: Path) -> bool:
    if not target_dir.exists():
        logging.warning("Offline target missing: %s", target_dir)
        return False
    executable = find_offline_executable(target_dir)
    if not executable:
        logging.warning("Offline target missing ControlCenter.exe under: %s", target_dir)
        return False
    return True


def resolve_offline_target(state: Dict[str, Any]) -> Optional[Path]:
    log_step(state, "RESOLVE_OFFLINE_TARGET")
    cache = read_bootstrap_cache()
    if cache:
        cached_path = Path(cache.get("last_good_path", ""))
        if cached_path and validate_offline_target(cached_path):
            state["cached_version"] = cache.get("last_good_version")
            return cached_path
    try:
        current_pointer = get_current_pointer()
    except Exception:
        logging.exception("Unable to resolve current install pointer")
        return None
    if current_pointer.exists():
        try:
            target = Path(current_pointer.read_text(encoding="utf-8").strip())
        except OSError:
            logging.exception("Failed reading current pointer")
            return None
        if validate_offline_target(target):
            return target
    return None


def safe_open_logs(log_root: Path) -> None:
    try:
        os.startfile(log_root)  # type: ignore[attr-defined]
    except Exception:
        logging.exception("Failed to open logs folder: %s", log_root)


def classify_exception(exc: Exception) -> tuple[str, str, str]:
    if isinstance(exc, CaptivePortalError):
        return (
            "Couldn’t check for updates because the network requires sign-in.",
            "Open a browser to complete any Wi-Fi sign-in page, then retry.",
            "captive_portal_suspected",
        )
    if isinstance(exc, urllib.error.HTTPError):
        if exc.code in {403, 429}:
            return (
                "Couldn’t check for updates because GitHub blocked the request.",
                "GitHub may be blocked or rate-limiting. Try again later or use another network.",
                "http_blocked",
            )
        if exc.code == 404:
            return (
                "No release is available yet.",
                "Try again later. If this persists, contact support.",
                "http_not_found",
            )
        return (
            "The update server returned an error.",
            f"HTTP error {exc.code}. Check your internet connection, VPN, or firewall.",
            "http_error",
        )
    if isinstance(exc, ssl.SSLError):
        return (
            "Secure connection to the update server failed.",
            "Corporate proxies or antivirus HTTPS inspection can cause this. Try another network.",
            "tls_error",
        )
    if isinstance(exc, socket.gaierror):
        return (
            "Couldn’t resolve the update server address.",
            "DNS lookup failed. Try switching networks or disabling VPN/proxy.",
            "dns_failure",
        )
    if isinstance(exc, urllib.error.URLError):
        return (
            "Couldn’t reach the update server.",
            "Check your internet connection, captive portal sign-in, VPN/proxy, or firewall.",
            "network_unreachable",
        )
    if isinstance(exc, (TimeoutError, socket.timeout, ConnectionResetError, ConnectionAbortedError)):
        return (
            "The network connection was interrupted.",
            "Try again after checking your internet connection or firewall.",
            "network_interrupted",
        )
    if isinstance(exc, json.JSONDecodeError):
        return (
            "Couldn’t read update metadata.",
            "A captive portal or proxy may be intercepting the connection.",
            "metadata_parse_error",
        )
    if isinstance(exc, (zipfile.BadZipFile, OSError)):
        return (
            "The update download was corrupt or incomplete.",
            "Try again. If it keeps failing, check your disk space or antivirus settings.",
            "asset_error",
        )
    return (
        "FightingOverlay couldn’t start due to an unexpected error.",
        "Try again or check the logs for details.",
        "unexpected_error",
    )


def build_user_message(summary: str, hint: str, log_root: Path) -> str:
    desktop_log_path = resolve_desktop_error_log_path(log_root)
    return (
        f"{summary}\n\n"
        "Common causes:\n"
        "- Offline or unstable internet\n"
        "- DNS issues\n"
        "- VPN/proxy/firewall blocking GitHub\n"
        "- Captive portal (sign-in Wi-Fi page)\n"
        "- Antivirus HTTPS inspection\n\n"
        f"Next steps: {hint}\n\n"
        f"Log file: {log_root / 'bootstrap.log'}\n"
        f"Desktop log: {desktop_log_path}\n"
        "Choose Retry to try again or Cancel for more options."
    )


def write_last_error(log_root: Path, classification: str, summary: str, hint: str) -> None:
    payload = (
        f"timestamp_utc={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n"
        f"classification={classification}\n"
        f"summary={summary}\n"
        f"hint={hint}\n"
        f"log_path={log_root / 'bootstrap.log'}\n"
    )
    desktop_path = resolve_desktop_error_log_path(log_root)
    write_error_payload(desktop_path, payload)
    for candidate_root in (
        log_root,
        Path(tempfile.gettempdir()) / "FightingOverlay" / "logs",
    ):
        try:
            candidate_root.mkdir(parents=True, exist_ok=True)
            path = candidate_root / "bootstrap_last_error.txt"
            atomic_write(path, payload)
            return
        except Exception:
            try:
                logging.exception("Failed to write bootstrap_last_error.txt in %s", candidate_root)
            except Exception:
                pass


def show_startup_dialog(
    state: Dict[str, Any],
    summary: str,
    hint: str,
    log_root: Path,
    offline_available: bool,
) -> str:
    try:
        import ctypes

        message = build_user_message(summary, hint, log_root)
        response = ctypes.windll.user32.MessageBoxW(0, message, BOOTSTRAP_ERROR_TITLE, 0x00000005)
        if response == 4:  # IDRETRY
            return "retry"

        if offline_available:
            response = ctypes.windll.user32.MessageBoxW(
                0,
                "Run the last installed version offline?\nYes = Run Offline\nNo = Skip",
                BOOTSTRAP_ERROR_TITLE,
                0x00000004,
            )
            if response == 6:  # IDYES
                return "offline"

        response = ctypes.windll.user32.MessageBoxW(
            0,
            "Open logs folder?\nYes = Open Logs\nNo = Exit",
            BOOTSTRAP_ERROR_TITLE,
            0x00000004,
        )
        if response == 6:  # IDYES
            safe_open_logs(log_root)
            return "open_logs"
        return "exit"
    except Exception:
        logging.exception("Failed to show startup dialog")
        return "exit"


def run_self_test() -> int:
    ensure_directories()
    logging.info("Self-test completed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="FightingOverlay Bootstrapper")
    parser.add_argument("--self-test", action="store_true", help="Run bootstrapper self-test")
    parser.add_argument("--update", action="store_true", help="Run update mode")
    parser.add_argument(
        "--simulate-offline",
        action="store_true",
        help="Simulate offline failure without network access",
    )
    parser.add_argument(
        "--simulate-captive-portal",
        action="store_true",
        help="Simulate captive portal response",
    )
    parser.add_argument(
        "--simulate-bad-zip",
        action="store_true",
        help="Simulate a corrupt release archive",
    )
    args = parser.parse_args()

    log_root = resolve_log_root()
    log_path = setup_logging(log_root)
    logging.info("BOOTSTRAP_LOG_PATH=%s", log_path)

    if maybe_handoff_to_current_bootstrapper(sys.argv):
        logging.info("Bootstrapper handoff started; exiting current process")
        return 0

    if args.self_test:
        return run_self_test()

    user_retry_count = 0
    previous_version = None
    state: Dict[str, Any] = {"step": "INIT"}
    while True:
        try:
            log_step(state, "ENSURE_DIRECTORIES")
            ensure_directories()

            log_step(state, "READ_CURRENT_POINTER")
            current_pointer = get_current_pointer()
            if current_pointer.exists():
                previous_version = Path(current_pointer.read_text(encoding="utf-8").strip()).name

            if args.simulate_offline:
                raise urllib.error.URLError("Simulated offline mode")
            if args.simulate_captive_portal:
                raise CaptivePortalError("Simulated captive portal")

            if args.simulate_bad_zip:
                release = {
                    "tag_name": "simulated-bad-zip",
                    "assets": [
                        {
                            "name": ASSET_NAME,
                            "browser_download_url": "https://example.invalid/simulated.zip",
                        }
                    ],
                }
            else:
                release = None
                for attempt in range(MAX_NETWORK_RETRIES + 1):
                    try:
                        release = fetch_latest_release(state)
                        break
                    except Exception as exc:
                        logging.exception("FETCH_LATEST_RELEASE failed (attempt %s)", attempt + 1)
                        if not is_retryable_fetch_error(exc) or attempt >= MAX_NETWORK_RETRIES:
                            raise
                        backoff = 1 if attempt == 0 else 3
                        logging.info("Retrying after %ss backoff", backoff)
                        time.sleep(backoff)

                if release is None:
                    raise RuntimeError("Failed to fetch latest release.")

            version = release.get("tag_name") or release.get("name") or "unknown"
            state["target_version"] = version
            target_dir = None
            for attempt in range(MAX_NETWORK_RETRIES + 1):
                try:
                    target_dir = install_release(state, release, args.simulate_bad_zip)
                    break
                except (zipfile.BadZipFile, OSError, RuntimeError) as exc:
                    logging.exception("INSTALL_RELEASE failed (attempt %s)", attempt + 1)
                    if not is_retryable_install_error(exc) or attempt >= MAX_NETWORK_RETRIES:
                        raise exc
                    backoff = 1 if attempt == 0 else 3
                    logging.info("Retrying after %ss backoff", backoff)
                    time.sleep(backoff)

            if target_dir is None:
                raise RuntimeError("Failed to install update.")

            log_step(state, "LAUNCH_GUI", target_dir=target_dir)
            launch_ok = launch_control_center(target_dir, log_root)
            if not launch_ok:
                record_last_update(
                    "error",
                    version,
                    "ControlCenter crashed on startup",
                    previous_version,
                )
                update_cache_attempt(version, "controlcenter_crash")
                log_step(state, "CONTROL_CENTER_CRASH")
                return 1

            atomic_write(get_current_pointer(), str(target_dir.resolve()))
            update_shortcut(target_dir / "ControlCenter.exe")
            try:
                copy_bootstrapper(version)
            except PermissionError as exc:
                logging.warning("Bootstrapper update skipped due to permission error: %s", exc)
            record_last_update("success", version, "Installed successfully", previous_version)
            update_cache_success(version, target_dir)
            clean_old_versions(version, previous_version)
            log_step(state, "BOOTSTRAP_SUCCESS")
            return 0
        except Exception as exc:
            summary, hint, classification = classify_exception(exc)
            logging.exception(
                "BOOTSTRAP_FAILURE step=%s classification=%s", state.get("step"), classification
            )
            try:
                record_last_update("error", state.get("target_version"), str(exc), previous_version)
            except Exception:
                logging.exception("Failed to record last update")
            update_cache_attempt(state.get("target_version"), classification)
            cached_target = resolve_offline_target(state)
            offline_available = cached_target is not None
            log_step(state, "SHOW_DIALOG", classification=classification, offline_available=offline_available)
            write_last_error(log_root, classification, summary, hint)
            action = show_startup_dialog(state, summary, hint, log_root, offline_available)
            if action == "retry":
                user_retry_count += 1
                if user_retry_count > MAX_NETWORK_RETRIES:
                    logging.info("User retry limit reached")
                    return 1
                continue
            if action == "offline" and cached_target:
                try:
                    log_step(state, "LAUNCH_GUI_OFFLINE", target_dir=cached_target)
                    if launch_control_center(cached_target, log_root):
                        return 0
                    return 1
                except Exception:
                    logging.exception("Offline launch failed")
                    return 1
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
