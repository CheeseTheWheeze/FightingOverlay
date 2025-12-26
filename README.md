# FightingOverlay

FightingOverlay is a Windows app delivered by a single bootstrapper EXE that installs/updates one GUI (Control Center) from GitHub Releases into deterministic local app/data/log folders.

## Hard Requirements (Acceptance Checklist)
- **Must** ship a single download: `FightingOverlayBootstrap.exe`.
- **Must** run by double-click on a fresh Windows machine with **no Python installed**.
- **Must** install into `%LOCALAPPDATA%\FightingOverlay\...`.
- **Must** create **one** Desktop shortcut named **“FightingOverlay”**.
- **Must** expose **only one UI app**: **Control Center**.
- **Must** never require the user to handle zip files after the bootstrapper runs.
- **Must** keep **at most two versions** installed (current + previous), and auto-clean older versions.
- **Must** preserve data across updates; data lives under `%LOCALAPPDATA%\FightingOverlay\data`.
- **Must** update from the **GitHub Releases API** (not Actions artifacts).
- **Must** publish release assets with **exact names**:
  - `FightingOverlayBootstrap.exe`
  - `FightingOverlay-Full-Windows.zip`

## Quickstart (No Terminal)
1. Download `FightingOverlayBootstrap.exe` from the latest GitHub Release.
2. Double-click it to install/update.
3. Use the **FightingOverlay** desktop shortcut for all future launches.

**Outputs:** `%LOCALAPPDATA%\FightingOverlay\data\outputs`

**Logs:** `%LOCALAPPDATA%\FightingOverlay\logs`

**Update:** Open Control Center → click **Check for Updates** or **Update Now**.

## Developer build (optional)
### Release workflow
- Tag a release to trigger publishing:
  ```bash
  git tag vX.Y.Z
  git push origin vX.Y.Z
  ```
- GitHub Actions builds and publishes both required assets to the Release.

### Change the version
- The bootstrapper reads the latest version from the GitHub Release tag (e.g., `vX.Y.Z`).
- Update the tag when cutting a new release.

### Rename asset names safely
- Update the constants used by the bootstrapper and Control Center:
  - `core/constants.py` → `FULL_PACKAGE_ASSET`
  - `bootstrap/bootstrapper.py` references the asset by name
  - `.github/workflows/build.yml` sets output names

## Acceptance Tests
> Run on a Windows machine with PowerShell.

### Verify installed paths exist
```powershell
$root = "$env:LOCALAPPDATA\FightingOverlay"
Test-Path "$root\app"; Test-Path "$root\data"; Test-Path "$root\logs"; Test-Path "$root\data\outputs"; Test-Path "$root\data\models"
```

### Verify only one UI entrypoint exists
```powershell
Get-ChildItem "$env:LOCALAPPDATA\FightingOverlay\app\versions" -Recurse -Filter "*.exe" | Select-String -Pattern "ControlCenter.exe"
```

### Verify updates do not delete data
```powershell
$testFile = "$env:LOCALAPPDATA\FightingOverlay\data\outputs\_update_preserve_test.txt"
"keep" | Out-File $testFile
# run update from Control Center
Test-Path $testFile
```

### Verify release zip does not contain itself
```powershell
Expand-Archive -Path .\FightingOverlay-Full-Windows.zip -DestinationPath .\_zipcheck -Force
Get-ChildItem .\_zipcheck -Recurse -Filter "FightingOverlay-Full-Windows.zip"
```

## Known Limitations / Roadmap
- **Today:** inference is a mock pipeline that writes a sample `pose_tracks.json`.
- **Planned:** real inference pipeline, video overlay, and model management.
- **Planned:** API service stub under `apps/api` will become a FastAPI service.
- **Planned:** Android remote update agent for managing installs/updates.
