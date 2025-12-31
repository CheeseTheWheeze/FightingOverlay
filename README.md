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
