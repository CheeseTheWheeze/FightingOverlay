# Running FightingOverlay Locally (Windows)

## From source (Windows 11)

1. Install Python 3.11.
2. Create and activate a venv:
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```
3. Install dependencies:
   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements/ci.txt
   ```
4. Run the Control Center:
   ```powershell
   python -m apps.control_center.main
   ```
5. UI smoke test (quick sanity check):
   ```powershell
   python -m apps.control_center.main --ui-smoke-test
   ```

## From built artifact (zip)

1. Unzip `FightingOverlay-Full-Windows.zip`.
2. Launch `ControlCenter.exe`.
3. UI smoke test:
   ```powershell
   .\ControlCenter.exe --ui-smoke-test
   ```

## Logs

Crash logs and runtime logs live in:
`%LOCALAPPDATA%\FightingOverlay\logs\`

Use the “Open Logs” button inside the Control Center if you need to open the folder quickly.
