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

**Verify release zip does not contain itself:**
```powershell
Add-Type -AssemblyName System.IO.Compression.FileSystem
[System.IO.Compression.ZipFile]::OpenRead("FightingOverlay-Full-Windows.zip").Entries | Where-Object { $_.Name -eq "FightingOverlay-Full-Windows.zip" }
```

## Known Limitations / Roadmap

- **Mock today:** "Open Video + Run Overlay" is a stub and will show a message until inference is implemented.
- **Planned:** full inference pipeline integration with model download and caching.
- **Planned:** API service (`apps/api`) and Android remote update agent for device-triggered capture.
