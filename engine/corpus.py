from __future__ import annotations

import json
from pathlib import Path

from core.paths import get_corpus_root
from db.index import record_corpus_export
from engine.features import FeatureSet


def export_to_corpus(athlete_id: str, clip_id: str, feature_set: FeatureSet) -> Path:
    corpus_root = get_corpus_root()
    feature_dir = corpus_root / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    output_path = feature_dir / f"{athlete_id}_{clip_id}.json"
    payload = {
        "athlete_id": athlete_id,
        "clip_id": clip_id,
        "features": {
            "stance": feature_set.stance,
            "tempo": feature_set.tempo,
            "quality_flags": feature_set.quality_flags,
        },
        "todo": "Replace stub export with anonymized corpus pipeline",
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    record_corpus_export(athlete_id, clip_id, "stub_features", str(output_path))
    return output_path
