# FightingOverlay Architecture Notes

## Now vs Next milestones (summary)

**Milestone 1**: Reliable Windows GUI startup + fatal error visibility.

**Milestone 2**: Extract a stable `engine/` module for pose extraction and overlay rendering while keeping GUI behavior unchanged.

**Milestone 3**: Athlete Profiles + per-clip artifact storage + minimal SQLite index for discoverability.

**Milestone 4 (scaffold)**: Feature extraction/corpus export placeholders.

## Engine module

The `engine/` package provides stable processing interfaces used by the GUI:

* `PoseExtractor.extract(video_path, config=...) -> PoseSequence`
* `OverlayRenderer.render(video_path, pose, out_path, config=...) -> str`
* `process_clip(...)` wraps the existing pipeline to keep GUI behavior unchanged while isolating processing.

## Data model + storage layout

* Athlete Profile
* Session (planned)
* Clip
* Artifacts

On disk (under `%LOCALAPPDATA%\FightingOverlay\data\`):

```
profiles/<athlete_id>/clips/<clip_id>/
  source.<ext>
  pose_tracks.json
  overlay.mp4
  overlay_skeleton.mp4
  ...
```

SQLite index lives at:

```
%LOCALAPPDATA%\FightingOverlay\data\db\index.sqlite
```

Tables include `athletes`, `clips`, `artifacts`, plus a minimal `corpus_exports` table for future metrics export.
