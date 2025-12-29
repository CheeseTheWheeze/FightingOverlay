from __future__ import annotations

import argparse
from pathlib import Path

from core.pipeline import ProcessingOptions, run_pipeline
from fightai.judge.runner import run_judge


def main() -> None:
    parser = argparse.ArgumentParser(description="FightAI CLI tools")
    sub = parser.add_subparsers(dest="command", required=True)

    judge_parser = sub.add_parser("judge", help="Run overlay alignment judge")
    judge_parser.add_argument("--raw", type=Path, required=True, help="Path to raw input video")
    judge_parser.add_argument("--tracks", type=Path, required=True, help="Path to pose_tracks.json")
    judge_parser.add_argument("--overlay", type=Path, help="Optional overlay video path")
    judge_parser.add_argument("--out", type=Path, required=True, help="Output directory for debug bundle")

    run_parser = sub.add_parser("run", help="Run overlay pipeline (optionally with judge)")
    run_parser.add_argument("--video", type=Path, required=True, help="Path to input video")
    run_parser.add_argument("--export-overlay", action="store_true", help="Export overlay MP4s")
    run_parser.add_argument("--save-combat", action="store_true", help="Write combat_overlay.json")
    run_parser.add_argument("--run-judge", action="store_true", help="Run alignment judge after overlay")

    args = parser.parse_args()

    if args.command == "judge":
        results = run_judge(args.raw, args.tracks, args.out, overlay_video=args.overlay)
        print("Judge complete:")
        for key, value in results.items():
            print(f"- {key}: {value}")
    if args.command == "run":
        options = ProcessingOptions(
            export_overlay_video=args.export_overlay,
            save_pose_json=True,
            save_combat_overlay=args.save_combat,
            run_judge=args.run_judge,
        )
        pose_path = run_pipeline(args.video, options)
        print(f"Overlay pipeline complete: {pose_path}")


if __name__ == "__main__":
    main()
