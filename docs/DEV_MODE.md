# Dev Mode Guide

## Run the Control Center

```bash
python apps/control_center/main.py
```

## Dev Mode Layout

- **Left panel**: inputs, overlay mode, and run toggles (judge/combat).
- **Center panel**: live log tailing with filter.
- **Right panel**: artifacts browser (outputs + judge bundle).

## Alignment Judge

CLI usage:

```bash
python -m fightai.cli judge --raw /path/to/raw.mp4 --tracks /path/to/pose_tracks.json --overlay /path/to/overlay_skeleton.mp4 --out /path/to/outputs
```

Pipeline + judge in one command:

```bash
python -m fightai.cli run --video /path/to/raw.mp4 --export-overlay --save-combat --run-judge
```

Outputs:
- `alignment_report.json`
- `metrics.csv`
- `report.html`
- `worst_frames/` and `worst_clips/`

### `alignment_report.json`

Key fields:
- `events`: list of event objects with `event_type`, `frame` ranges, and `recommended_fixes` tags.
- `tracks`: per-track summaries for alignment scores.

## Combat Overlay

Enable **Save combat overlay JSON** to emit `combat_overlay.json`.

## Adding New Event Detectors

1. Extend `fightai/judge/events.py` with new detection logic.
2. Add new thresholds to `fightai/judge/config.py`.
3. Update tests in `tests/test_judge_metrics.py` with synthetic sequences.
