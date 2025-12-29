from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from core.pipeline import json_dumps


@dataclass(frozen=True)
class PersonRegion:
    frame_index: int
    track_id: str
    bbox_xywh: tuple[float, float, float, float]
    mask: list[list[int]] | None


class PersonRegionExtractor:
    """Extract person regions (mask or bbox) from tracks.

    Current implementation uses bbox-only regions with optional cached masks.
    """

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def extract_regions(
        self,
        frame_width: int,
        frame_height: int,
        tracks: Iterable[dict[str, object]],
    ) -> dict[int, dict[str, PersonRegion]]:
        regions: dict[int, dict[str, PersonRegion]] = {}
        for track in tracks:
            track_id = str(track.get("track_id", "unknown"))
            frames = track.get("frames", [])
            if not isinstance(frames, list):
                continue
            for frame in frames:
                frame_index = int(frame.get("frame_index", -1))
                if frame_index < 0:
                    continue
                bbox = frame.get("bbox_xywh") or [0.0, 0.0, 0.0, 0.0]
                if not isinstance(bbox, list) or len(bbox) != 4:
                    bbox = [0.0, 0.0, 0.0, 0.0]
                bbox_tuple = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
                mask = self._load_or_build_mask(frame_index, track_id, bbox_tuple, frame_width, frame_height)
                regions.setdefault(frame_index, {})[track_id] = PersonRegion(
                    frame_index=frame_index,
                    track_id=track_id,
                    bbox_xywh=bbox_tuple,
                    mask=mask,
                )
        return regions

    def _load_or_build_mask(
        self,
        frame_index: int,
        track_id: str,
        bbox_xywh: tuple[float, float, float, float],
        frame_width: int,
        frame_height: int,
    ) -> list[list[int]] | None:
        x, y, w, h = bbox_xywh
        if w <= 0 or h <= 0 or frame_width <= 0 or frame_height <= 0:
            return None
        # Bbox-only fallback: mask generation is optional and can be added later.
        return None

    def write_region_manifest(self, output_path: Path, regions: dict[int, dict[str, PersonRegion]]) -> None:
        manifest = {
            "frames": {
                str(frame_index): {
                    track_id: {
                        "bbox_xywh": region.bbox_xywh,
                        "mask_cached": region.mask is not None,
                    }
                    for track_id, region in frame_regions.items()
                }
                for frame_index, frame_regions in regions.items()
            }
        }
        output_path.write_text(json_dumps(manifest), encoding="utf-8")
