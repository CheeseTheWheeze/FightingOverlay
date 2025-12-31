# Dev Notes

## Current Data Flow

1. **GUI** (`apps/control_center/main.py`)
   - Runs `core.pipeline.run_pipeline` with user-selected options.
   - Outputs artifacts to the outputs root (`core.paths.get_outputs_root`).

2. **Tracking + Overlay** (`core/pipeline.py`)
   - `run_inference` produces a pose payload with per-frame keypoints and bboxes.
   - `write_pose_tracks` writes `pose_tracks.json` in the outputs folder.
   - `export_overlay_video` renders overlay MP4 variants.
   - Optional postprocess steps now include:
     - `combat_overlay.json` via `fightai.combat.derive.derive_combat_overlay`.
     - Alignment judge bundle via `fightai.judge.runner.run_judge`.

3. **Evaluation** (`core/evaluation.py`)
   - Generates tracking quality metrics for the payload (optional via GUI).

4. **Overlay Alignment Judge** (`fightai/judge`)
   - Uses `pose_tracks.json` + raw video to compute alignment metrics.
   - Emits `alignment_report.json`, `metrics.csv`, `report.html`, and debug bundles.

## Key Outputs

- `pose_tracks.json`: per-frame keypoints and bbox tracks.
- `overlay_*.mp4`: overlay render variants.
- `combat_overlay.json`: derived combat zones and points.
- `judge_*/alignment_report.json`: alignment events + recommended fixes.
- `judge_*/report.html`: debug bundle HTML for review.

## Troubleshooting

- Settings live at `%APPDATA%\FightingOverlay\settings.json`. Invalid numeric values are auto-sanitized on launch; delete the file to reset to defaults.
- UI smoke test (requires a display): `FIGHTINGOVERLAY_UI_SMOKE=1 python apps/control_center/main.py --ui-smoke-test`

## Pre-release GUI verification (Windows)

1. Rebuild `ControlCenter.exe` via the release pipeline or PyInstaller.
2. Run `%LOCALAPPDATA%\\FightingOverlay\\app\\versions\\latest\\ControlCenter.exe`.
3. Confirm the window appears with no TclError.
4. Check `%LOCALAPPDATA%\\FightingOverlay\\logs\\controlcenter_launch_*.log` for the absence of UI_BUILD pane errors.
5. Run the UI smoke test locally before cutting a Windows release.
