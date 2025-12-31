from __future__ import annotations

from dataclasses import dataclass

from engine.pose import PoseSequence


@dataclass(frozen=True)
class FeatureSet:
    stance: str | None
    tempo: float | None
    quality_flags: list[str]


# Feature scaffolds live in engine so they can evolve alongside pose extraction/rendering.
# TODO: replace stubs with real heuristics when modeling work begins.

def detect_stance(_pose: PoseSequence) -> str | None:
    return None


def estimate_tempo(_pose: PoseSequence) -> float | None:
    return None


def quality_flags(_pose: PoseSequence) -> list[str]:
    return []


def extract_features(pose: PoseSequence) -> FeatureSet:
    return FeatureSet(
        stance=detect_stance(pose),
        tempo=estimate_tempo(pose),
        quality_flags=quality_flags(pose),
    )
