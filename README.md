# FightingOverlay

FightingOverlay is a Windows desktop app for running a single Control Center UI that manages installs, updates, and outputs for the FightingOverlay toolset via a one-click bootstrapper.

## HARD REQUIREMENTS (Acceptance Checklist)

- **Must** provide a single download: `FightingOverlayBootstrap.exe`.
- **Must** run on a fresh Windows machine with no Python installed (double-clickable EXE).
- **Must** install under `%LOCALAPPDATA%\FightingOverlay\` (not Desktop/Downloads).
- **Must** create exactly one Desktop shortcut named **FightingOverlay**.
- **Must** have only one UI app: **Control Center** (`ControlCenter.exe`).
- **Must** not require any user ZIP handling after running the bootstrapper.
- **Must** keep at most two installed versions (current + previous) and auto-delete older versions.
- **Must** never delete user data on update; data lives under `%LOCALAPPDATA%\FightingOverlay\data`.
- **Must** fetch updates from the GitHub Releases API (not Actions artifacts).
- **Must** publish release assets with exact names:
  - `FightingOverlayBootstrap.exe`
  - `FightingOverlay-Full-Windows.zip`

## Quickstart (No Terminal)

1. Download `FightingOverlayBootstrap.exe` from GitHub Releases.
2. Double-click the EXE to install/update and launch Control Center.
3. Use the Desktop shortcut **FightingOverlay** for future launches.

**Outputs:** `%LOCALAPPDATA%\FightingOverlay\data\outputs`  
**Logs:** `%LOCALAPPDATA%\FightingOverlay\logs`

**Updating:** Open Control Center → **Update Now**.

## Troubleshooting

**Logs live here:** `%LOCALAPPDATA%\\FightingOverlay\\logs` (look for `control_center.log` and `controlcenter_launch_*.log`).

**UI smoke test (Windows):**
```powershell
$env:FIGHTINGOVERLAY_UI_SMOKE = "1"
python apps/control_center/main.py --ui-smoke-test
```

**If OpenCV (cv2) is missing:** reinstall the latest FightingOverlay release so the bundled dependencies are restored. The Control Center will surface the error and point to the log file for details.

**Debug verbosity / diagnostics:**
- Launch Control Center with `--debug-console` to show a Diagnostics tab and verbose logs.
- Set `FIGHTINGOVERLAY_SHOW_CONSOLE=1` before running the bootstrapper to disable `CREATE_NO_WINDOW` during GUI launch.

## Developer build (optional)

**CI / Release:**
- Push to any branch to run the CI build workflow in `.github/workflows/ci.yml`.
- Tag a release with `vX.Y.Z` to publish a GitHub Release and upload both assets.

**Versioning:**
- The release version comes from the git tag `vX.Y.Z` in GitHub Releases.

**Asset name changes:**
- If you must rename assets, update both the workflow and bootstrapper:
  - `.github/workflows/ci.yml` (zip filename and uploaded asset names)
  - `bootstrap/main.py` (`ASSET_NAME` and `BOOTSTRAP_COPY_NAME`)

## Acceptance Tests

> Copy/paste from a Windows PowerShell prompt.

**Verify install paths exist:**
```powershell
Test-Path "$env:LOCALAPPDATA\FightingOverlay\app"
Test-Path "$env:LOCALAPPDATA\FightingOverlay\data"
Test-Path "$env:LOCALAPPDATA\FightingOverlay\logs"
```

**Verify only one UI entrypoint exists:**
```powershell
Get-ChildItem "$env:LOCALAPPDATA\FightingOverlay\app\versions" -Recurse -Filter "*.exe" | Select-Object -ExpandProperty Name | Sort-Object -Unique
```

**Verify updates do not delete data:**
```powershell
$sentinel = "$env:LOCALAPPDATA\FightingOverlay\data\sentinel.txt"
"keep" | Set-Content $sentinel
# Run update via Control Center, then:
Get-Content $sentinel
```

**Verify release zip contains ControlCenter.exe:**
```powershell
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = [System.IO.Compression.ZipFile]::OpenRead("FightingOverlay-Full-Windows.zip")
$zip.Entries | Where-Object { $_.FullName -like "*ControlCenter.exe" }
$zip.Dispose()
```

## Bootstrapper Manual Tests (Windows)

> Run these by double-clicking `FightingOverlayBootstrap.exe` and observing dialogs/logs.
> Dialog sequence: Stage 1 (Retry/Cancel) → Stage 2 (Run Offline? if available) → Stage 3 (Open Logs?).
> Log entries to confirm: `BOOTSTRAP_START`, `BOOTSTRAP_STEP=FETCH_LATEST_RELEASE`, `BOOTSTRAP_FAILURE`, `BOOTSTRAP_STEP=SHOW_DIALOG`.
> Breadcrumb file: `%LOCALAPPDATA%\\FightingOverlay\\logs\\bootstrap_last_error.txt` (or temp fallback if logs are redirected).

- **Offline mode:** disable network adapter, launch, confirm dialog appears with log path and Stage 1/2/3 flow.
- **DNS failure:** set invalid DNS (e.g., `127.0.0.1`), launch, confirm DNS hint appears.
- **Captive portal:** use public Wi-Fi with sign-in page, launch, confirm captive portal hint appears.
- **TLS intercept:** enable corporate proxy/HTTPS inspection (if available), launch, confirm TLS hint appears.
- **GitHub blocked/rate-limited:** block `api.github.com` or hit rate limit, launch, confirm blocked/rate-limit hint appears.
- **Corrupt cached download:** truncate the cached zip (or delete it), launch, confirm retry + cleanup messaging and successful re-download.
- **Permissions/AV lock:** simulate file lock on `%LOCALAPPDATA%\FightingOverlay\app`, launch, confirm permissions hint appears.
- **Simulated failures (PowerShell):**
  - `FightingOverlayBootstrap.exe --simulate-offline`
  - `FightingOverlayBootstrap.exe --simulate-captive-portal`
  - `FightingOverlayBootstrap.exe --simulate-bad-zip`

## Known Limitations / Roadmap

- **Mock today:** "Open Video + Run Overlay" is a stub and will show a message until inference is implemented.
- **Planned:** full inference pipeline integration with model download and caching.
- **Planned:** API service (`apps/api`) and Android remote update agent for device-triggered capture.

## Codex Prompt

You are Codex working on the project contained in the provided zip (repo). Your job is to **rebuild the project from the ground up into a clean, scalable architecture** while also delivering an immediate, working MVP: a desktop GUI that can select a video, extract skeleton data, and export an overlay video.

### Non-negotiable immediate MVP requirements (must work first)

1. Desktop GUI launches on Windows 11 reliably (including virtualized/streamed environments like Shadow PC).
2. User selects a video clip (MP4).
3. System extracts skeleton/pose keypoints over time.
4. System writes:

   * `pose_tracks.json` (or `.parquet`) with frame timestamps + keypoints + confidences
   * an overlay MP4 showing skeleton over the video
5. No silent failures: if any error occurs (missing runtime, model, codec, GPU), show a visible error UI and write logs.
6. Packageable for Windows (one-click run):

   * Provide a build step (PyInstaller or equivalent) that produces `ControlCenter.exe`
   * Ensure paths work when launched from shortcuts or different working dirs

### Long-term product goal (architectural rebuild)

This project will become an end-to-end system for a **network of fighting/grappling video clips**:

* clients use mobile apps to upload training/competition clips
* server ingests clips, extracts movement data, stores canonical features
* creates athlete profiles + session timelines + skill evaluation
* supports universal striking/grappling sports: boxing/kickboxing/MMA/bjj/wrestling/judo etc.
* becomes a growing “training database” similar to Duolingo: system learns from many clips and coaching labels, improves evaluation and recommendations over time

### Constraints & environment

* Current processing will run locally on a Windows 11 Shadow PC for now.
* Must keep a clean path to future cloud deployment.
* Avoid heavy complexity until the MVP works.
* Use best language/tool for each layer:

  * GUI: keep Python for now (fast iteration), but design boundaries so GUI can later be replaced by Flutter/React Native desktop if needed.
  * Processing engine: Python is acceptable initially (OpenCV + pose model), but design a modular “engine” that can later be rewritten in Rust/C++ for performance without changing APIs.
  * Server: Python FastAPI (initial) is fine; later can move to Go/Rust if needed.
  * Mobile: plan for Flutter or React Native (just scaffold and API contracts for now; do not build full mobile apps yet).

---

# Phase 0: Repo audit and “what is actually runnable”

1. Inspect the zip contents and determine:

   * whether there is an existing GUI entrypoint
   * whether there is a packaged executable expected (`ControlCenter.exe`)
   * how paths and logs are handled
2. If the zip does not contain `ControlCenter.exe` (likely), implement a reliable “run from source” workflow and a “build exe” workflow.

Deliverable: a top-level `RUN_LOCAL_WINDOWS.md` with exact commands for:

* creating venv
* installing deps
* running GUI
* building exe
* where logs go

---

# Phase 1: Make the GUI unbreakable (Windows 11)

Implement these design requirements:

## 1.1 Strong error handling and diagnostics

* Create `main_safe()` wrapper for the GUI entrypoint.
* Hook Tk callback exception handling (or equivalent) so exceptions in callbacks don’t leave a blank window.
* Always show either:

  * the real UI, or
  * a fallback error view with:

    * error summary
    * “Copy details”
    * “Open logs folder”
    * “Exit”
* Always log:

  * sys.executable, sys.version
  * cwd, argv
  * platform, OS build
  * ffmpeg availability or OpenCV codec support
  * model path resolution and whether model files exist

## 1.2 No early returns that produce empty UI

Audit all startup checks. Replace “return” with:

* visible error message + exit non-zero if critical
* or render a “Setup required” panel if recoverable

## 1.3 Fix relative path issues

All file resolution must use an `APP_BASE_DIR`:

* if frozen (PyInstaller): use `sys._MEIPASS` / executable dir correctly
* if running from source: use `Path(__file__).resolve()`

---

# Phase 2: Pose extraction engine (clean boundaries)

You must implement a clean engine interface:

## 2.1 Core interfaces

Create `engine/` package with:

* `PoseExtractor` interface:

  * `extract(video_path) -> PoseSequence`
* `OverlayRenderer`:

  * `render(video_path, pose_sequence, out_path)`

PoseSequence schema must include:

* frame_index
* timestamp_ms (or time_sec)
* persons: list
* keypoints: Nx( x, y, confidence )
* optional: bbox, track_id

## 2.2 Implementation choice (MVP)

Implement MVP using a known pose estimator:

* Prefer MediaPipe Pose (fast & easy) OR OpenPose-like model if already in repo.
* Must run on CPU reliably on Shadow PC.
* Must not require GPU.
* Keep model downloads explicit and cached under an app data folder.

## 2.3 Output formats

Write:

* `pose_tracks.json` (easy)
* also `pose_tracks.parquet` (better for analytics) if feasible

---

# Phase 3: GUI features (MVP workflow)

GUI must provide a single “happy path”:

1. Select video
2. Select output folder (default under LocalAppData FightingOverlay\\data\\outputs)
3. Run Extraction
4. Run Overlay Render
5. Show progress + logs panel
6. Open output folder button
7. Show last run summary (frames processed, avg FPS, failures)

Additional:

* Cancel button
* Reset state if previous run crashed
* Make it impossible to start run without selecting a valid file

---

# Phase 4: Project rebuild into scalable product architecture

Restructure the repo to match a future client/server system while keeping MVP intact.

## 4.1 Suggested monorepo layout

* `apps/desktop/` (Python GUI)
* `engine/` (pose extraction + overlay + tracking; pure logic, no GUI dependencies)
* `server/` (FastAPI ingestion API; not fully deployed yet)
* `mobile/` (Flutter/React Native scaffold only; no heavy build)
* `shared/` (schemas, canonical models, validators)
* `infra/` (future Docker/compose scripts)
* `docs/`

## 4.2 Canonical schemas and DB plan

Define a canonical “movement profile” schema:

* athlete profile
* session (training/competition)
* clip metadata
* extracted pose track
* derived features (stance, guard height, step frequency, sprawl reaction time, etc.)
* labels (coach tags) with provenance and confidence

For now: implement local storage as:

* filesystem for videos and outputs
* SQLite (or DuckDB) for metadata + indexing
* Parquet for pose tracks/features

## 4.3 Future upload pipeline (scaffold only)

Implement server endpoints (FastAPI):

* POST /clips (upload metadata)
* PUT /clips/{id}/file (upload clip file)
* POST /clips/{id}/process (enqueue processing)
* GET /clips/{id}/status
* GET /athletes/{id}/profile

You do NOT need a full auth system yet, but design for it:

* API keys for now
* later OAuth/JWT

---

# Phase 5: Packaging and release automation

Provide:

* `scripts/build_windows.ps1` builds `ControlCenter.exe`
* include all required model files or implement first-run model download with checksum
* write logs to `%LOCALAPPDATA%\\FightingOverlay\\logs`
* ensure a clean “portable” run mode is possible too

Add a `--ui-smoke-test` option:

* creates a hidden root window
* initializes UI + engine
* returns 0 if okay, non-zero if not

---

# Implementation rules (important)

* Keep dependencies minimal and well-documented.
* Do not hide exceptions.
* Prefer explicit error messages over guessing.
* Every module must have clear responsibility.
* All I/O should go through a `paths.py` that supports frozen + source modes.
* Include type hints.

---

# Deliverables checklist

1. Working GUI from source on Windows 11
2. `ControlCenter.exe` build script + documentation
3. Pose extraction + overlay output verified
4. Structured project layout ready for future mobile/server
5. FastAPI scaffold with canonical schemas for future uploads
6. Smoke tests and basic unit tests for engine components
7. Clear docs: RUN_LOCAL_WINDOWS.md + TROUBLESHOOTING.md

Now implement. Start by making the MVP run end-to-end, then refactor into the new structure without breaking it.

---

## A couple of specifics I’d want you (Jackson) to answer (only if you know; not required to proceed)

If you can answer these, it will improve the “from the ground up” choices, but Codex can still proceed without them:

1. Do you want **single-person pose only** for now, or **multi-person tracking** in the same clip?
2. Are most clips **phone portrait**, **landscape**, or mixed?
3. Do you prefer **MediaPipe** (fast CPU) or are you committed to a specific model already in the repo?

If you don’t answer, Codex should default to: **single-person + mixed orientation + MediaPipe Pose CPU** for MVP.

---

If you paste Codex’s output (summary + file tree + key diffs + run logs), I’ll grade it hard on:

* does the GUI launch reliably?
* does it actually produce pose_tracks + overlay?
* is the architecture clean enough to scale without a rewrite?
