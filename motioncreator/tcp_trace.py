"""Planned-versus-actual tool-tip trace for the task controller's debug view.

For every motion segment the FSM plays, the *planned* tool path is the forward kinematics of the segment's arm and
waist angles on the pelvis and legs measured when the segment started (the plan assumes the lower body stays put).
The *actual* path is the simulated tool tip. Their gap is what the whole-body controller's lower-body motion and
waist lag cost; clearances to the target and to the nearest neighbour show whether the tools can reach the target.

Composition layer: it reads the editor simulation and the FSM, so it lives beside `controller_session`.
"""
from __future__ import annotations

import threading
from collections import deque

import mujoco
import numpy as np

from ramen_controller.contracts import ARMS_SLICE, WAIST_SLICE, MotionSegment

HANDS = {"left": "left_hand", "right": "right_hand"}        # `Robot` handles: the scoop TCP of each arm
PLAN_STEP_S = 0.1
TRAIL_POINTS = 400
CLEARANCE_RANGE_M = 0.3                                     # farther than this is reported as "far"
NEIGHBOUR_RADIUS_M = 0.5


class TcpTrace:
    def __init__(self, session):
        self.session = session
        self.simulation = session.simulation
        self.robot = session.simulation.robot
        self.lock = threading.Lock()
        self._segment: MotionSegment | None = None
        self.path_id = 0
        self.label = ""
        self.planned: dict[str, list] = {side: [] for side in HANDS}
        self.actual = {side: deque(maxlen=TRAIL_POINTS) for side in HANDS}
        self.base_q: np.ndarray | None = None
        self._ticks = 0

    # -- kinematics ---------------------------------------------------------------------------------------

    def _qpos(self) -> np.ndarray:
        with self.simulation.lock:
            return self.simulation.data.qpos[:self.robot.model.nq].copy()

    def _tips(self, q: np.ndarray) -> dict[str, np.ndarray]:
        data = self.robot.data(q)
        return {side: np.asarray(self.robot.point(data, hand)[0], dtype=float) for side, hand in HANDS.items()}

    def _planned_q(self, segment: MotionSegment, t: float) -> np.ndarray:
        arms, waist = segment.sample(float(np.clip(t, 0.0, segment.duration)))
        q = self.base_q.copy()
        joints = q[7:]
        joints[ARMS_SLICE], joints[WAIST_SLICE] = arms, waist
        return q

    # -- recording (session thread, once per FSM tick) ----------------------------------------------------

    def update(self) -> None:
        progress = self.session.fsm.playback_progress()
        if progress is None:
            return
        segment, t, blending = progress
        if segment is not self._segment:
            q = self._qpos()
            with self.lock:
                self._segment, self.base_q = segment, q
                self.path_id += 1
                self.label = segment.name or segment.kind.value
                steps = np.unique(np.append(np.arange(0.0, segment.duration, PLAN_STEP_S), segment.duration))
                paths = [self._tips(self._planned_q(segment, float(s))) for s in steps]
                self.planned = {side: [p[side].round(4).tolist() for p in paths] for side in HANDS}
                for trail in self.actual.values():
                    trail.clear()
        self._ticks += 1
        if self._ticks % 3 == 0 and not blending:
            tips = self._tips(self._qpos())
            with self.lock:
                for side in HANDS:
                    self.actual[side].append(tips[side].round(4).tolist())

    # -- view (request thread) ----------------------------------------------------------------------------

    def _clearances(self, tips: dict[str, np.ndarray]) -> dict:
        simulation, model = self.simulation, self.simulation.model
        with simulation.lock:
            data = mujoco.MjData(model)
            mujoco.mj_copyData(data, model, simulation.data)
        monitor = self.session.contact
        names = {item["id"]: item["name"] for item in simulation.scene_objects}
        target_id = self.session.target["id"]
        fromto = np.zeros(6)
        report = {}
        for side, bodies in monitor.side_bodies.items():
            tool_geoms = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in bodies and model.geom_contype[g]]
            candidates = {identifier: geoms for identifier, geoms in simulation.object_geoms.items()
                          if identifier == target_id or any(
                              np.linalg.norm(data.geom_xpos[g] - tips[side]) < NEIGHBOUR_RADIUS_M for g in geoms)}
            distance = {}
            for identifier, geoms in candidates.items():
                distance[identifier] = min(
                    (mujoco.mj_geomDistance(model, data, a, b, CLEARANCE_RANGE_M, fromto) for a in tool_geoms for b in geoms),
                    default=CLEARANCE_RANGE_M)
            others = sorted((value, identifier) for identifier, value in distance.items() if identifier != target_id)
            nearest = ({"name": names[others[0][1]], "mm": round(1000 * others[0][0], 1)} if others else None)
            report[side] = {"target_mm": round(1000 * distance.get(target_id, CLEARANCE_RANGE_M), 1), "nearest": nearest}
        return report

    def report(self) -> dict:
        progress = self.session.fsm.playback_progress()
        tips = self._tips(self._qpos())
        with self.lock:
            out = {"path_id": self.path_id, "segment": self.label, "planned": self.planned,
                   "actual": {side: list(trail) for side, trail in self.actual.items()},
                   "now": {side: tip.round(4).tolist() for side, tip in tips.items()},
                   "planned_now": None, "error_mm": None, "t": None, "blending": False, "active": False}
            segment, base_q = self._segment, self.base_q
        if progress is not None and segment is not None and progress[0] is segment:
            _, t, blending = progress
            planned = self._tips(self._planned_q(segment, t))
            out.update(t=round(t, 2), blending=blending, active=t <= segment.duration + 1.0,
                       planned_now={side: tip.round(4).tolist() for side, tip in planned.items()},
                       error_mm={side: round(1000 * float(np.linalg.norm(planned[side] - tips[side])), 1) for side in HANDS})
        out["clearance"] = self._clearances(tips)
        return out
