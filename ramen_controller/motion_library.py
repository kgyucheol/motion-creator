"""MotionPort backed by authored MotionCreator bundles, with a replaceable grasp planner.

Segments come from bundle folders listed in a JSON config. The grasp is the one stage that depends on the
observed object pose, so it sits behind `GraspPlanner`; `FixedGraspPlanner` is a prototype that replays
one authored grasp and refuses poses it was not authored for. A real planner (MotionCreator's grasp
generation, run offline or as a service) replaces it without touching the FSM.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol

import numpy as np

from .contracts import MotionKind, MotionSegment, ObjectPose
from .fsm import GraspPlanError
from .motion_bundle import load_segment


class GraspPlanner(Protocol):
    def plan(self, pose: ObjectPose) -> MotionSegment:
        """Grasp segment for an object pose in the base frame; raise GraspPlanError if infeasible."""


@dataclass(frozen=True)
class MotionEntry:
    bundle: Path
    start_s: float = 0.0
    end_s: float | None = None


@dataclass
class FixedGraspPlanner:
    """Prototype: replays one authored grasp if the object is where it was authored."""
    segment: MotionSegment
    authored_position: np.ndarray                 # object position (base frame) the grasp was authored for
    tolerance_m: float = 0.03

    def plan(self, pose: ObjectPose) -> MotionSegment:
        if pose.frame_id != "base":
            raise GraspPlanError(f"pose must be in the base frame, got {pose.frame_id!r}")
        offset = float(np.linalg.norm(pose.position - np.asarray(self.authored_position)))
        if offset > self.tolerance_m:
            raise GraspPlanError(f"object is {offset * 1000:.0f} mm from the authored grasp position "
                                 f"(limit {self.tolerance_m * 1000:.0f} mm)")
        return self.segment


class MotionLibrary:
    """MotionPort: fixed stages from bundles, grasp from the planner."""

    def __init__(self, entries: Mapping[MotionKind, MotionEntry], grasp_planner: GraspPlanner):
        missing = {MotionKind.READY, MotionKind.TRANSFER, MotionKind.PLACE, MotionKind.RETURN} - set(entries)
        if missing:
            raise ValueError(f"motion library lacks entries for {sorted(k.value for k in missing)}")
        self.entries, self.grasp_planner = dict(entries), grasp_planner
        self._cache: dict[MotionKind, MotionSegment] = {}

    def segment(self, kind: MotionKind) -> MotionSegment:
        if kind not in self._cache:
            entry = self.entries[kind]
            self._cache[kind] = load_segment(entry.bundle, kind, entry.start_s, entry.end_s)
        return self._cache[kind]

    def plan_grasp(self, pose: ObjectPose) -> MotionSegment:
        return self.grasp_planner.plan(pose)


def load_entries(config_path) -> dict[MotionKind, MotionEntry]:
    """Read {"ready": {"bundle": "...", "start_s": 0, "end_s": 3.2}, ...}; bundle paths are relative to the file."""
    config_path = Path(config_path)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    entries = {}
    for name, item in raw.items():
        try:
            kind = MotionKind(name)
        except ValueError as error:
            raise ValueError(f"unknown motion kind {name!r} in {config_path}") from error
        entries[kind] = MotionEntry(bundle=(config_path.parent / item["bundle"]).resolve(),
                                    start_s=float(item.get("start_s", 0.0)),
                                    end_s=None if item.get("end_s") is None else float(item["end_s"]))
    return entries
