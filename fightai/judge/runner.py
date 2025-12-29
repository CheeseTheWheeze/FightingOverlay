from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from core.schema import validate_pose_tracks_schema
from fightai.judge.bundle import write_debug_bundle
from fightai.judge.config import JudgeConfig
from fightai.judge.events import detect_events
from fightai.judge.metrics import compute_metrics, compute_swap_candidates
from fightai.judge.person_region import PersonRegionExtractor
from fightai.judge.report import summarize_tracks, write_alignment_report, write_metrics_csv


def run_judge(
    raw_video: Path,
    tracks_json: Path,
    output_dir: Path,
    overlay_video: Path | None = None,
    config: JudgeConfig | None = None,
) -> dict[str, str]:
    config = config or JudgeConfig()
    payload = json.loads(tracks_json.read_text(encoding="utf-8"))
    ok, message = validate_pose_tracks_schema(payload)
    if not ok:
        raise ValueError(f"pose_tracks.json invalid: {message}")

    video_meta = payload.get("video", {})
    frame_width = int(video_meta.get("width", 0) or 0)
    frame_height = int(video_meta.get("height", 0) or 0)
    fps = float(video_meta.get("fps", 30.0) or 30.0)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir / f"judge_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    region_cache = output_dir / "region_cache"
    extractor = PersonRegionExtractor(region_cache)
    tracks = payload.get("tracks", [])
    regions = extractor.extract_regions(frame_width, frame_height, tracks)
    extractor.write_region_manifest(output_dir / "regions.json", regions)

    metrics = compute_metrics(tracks, regions, config)
    metrics = compute_swap_candidates(metrics, regions, config)

    events = detect_events(metrics, config)
    track_summary = summarize_tracks(metrics)

    metrics_path = output_dir / "metrics.csv"
    write_metrics_csv(metrics_path, metrics)

    report_path = output_dir / "alignment_report.json"
    write_alignment_report(report_path, metrics, events, video_meta, track_summary)

    bundle = write_debug_bundle(
        output_dir,
        metrics,
        raw_video,
        overlay_video,
        worst_frame_count=config.worst_frame_count,
        clip_padding_s=config.clip_padding_s,
        fps=fps,
    )

    return {
        "alignment_report": str(report_path),
        "metrics_csv": str(metrics_path),
        "report_html": bundle.get("report_html", ""),
        "bundle_dir": str(output_dir),
    }
