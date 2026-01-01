# FightingOverlay Troubleshooting

## Where logs live

* `%LOCALAPPDATA%\FightingOverlay\logs\`
* Crash logs are named `crash_YYYYMMDD_HHMMSS.log`.

## Common startup issues

### 1) Missing VC++ runtime
Symptoms:
* App closes immediately or Windows reports missing DLLs.

Fix:
* Install the latest “Microsoft Visual C++ Redistributable for Visual Studio 2015-2022”.

### 2) Antivirus / SmartScreen blocks the app
Symptoms:
* Download blocked, executable quarantined, or a warning dialog.

Fix:
* Allow the app in your AV or SmartScreen prompt.
* Re-download the release zip after whitelisting.

### 3) Blocked GitHub access
Symptoms:
* Bootstrapper fails to update with network errors.

Fix:
* Verify GitHub access from the machine.
* Use a VPN or whitelist `github.com` and `api.github.com`.

### 4) Missing ffmpeg
Symptoms:
* Overlay export fails with errors during rendering.

Fix:
* Ensure ffmpeg is available on PATH or bundled with the build.

### 5) OpenCV import failure
Symptoms:
* Fatal error dialog: “OpenCV (cv2) failed to import”.

Fix:
* Reinstall FightingOverlay from the latest release bundle.
* Verify your antivirus has not removed `cv2` binaries.

## UI is blank or closes immediately

* Open the logs folder from the Control Center.
* Look for the latest `crash_*.log` file and the `control_center.log` file.
* Share those logs with support to diagnose startup failures.
