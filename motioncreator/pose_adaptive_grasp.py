"""Pose-adaptive grasp planning by object-relative retargeting of an authored grasp.

An authored grasp (for example one drawn in the Cylinder Lab) already knows how to thread the tools between the
neighbouring objects. Its tool-centre-point paths are re-expressed relative to the object that was authored,
re-applied to the object pose actually observed, and solved with whole-body IK from the robot's current posture
(pelvis and feet pinned: the lower body belongs to the whole-body controller). Only arm and waist joints of the
result are used.

Composition layer: it implements the runtime's `GraspPlanner` using the authoring package's `Robot` and IK.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from ramen_controller.contracts import ARMS_SLICE, WAIST_SLICE, MotionKind, MotionSegment, ObjectPose, RobotState
from ramen_controller.fsm import GraspPlanError

from .robot import FEET, Robot

HANDS = ("left_hand", "right_hand")
RELAXATIONS = ((.8, 180), (.3, 280), (.08, 400))      # posture weight, IK evaluations: tighten, then relax the preference


def _rotation(quaternion_xyzw) -> Rotation:
    return Rotation.from_quat(np.asarray(quaternion_xyzw, dtype=float))


class RetargetedGraspPlanner:
    def __init__(self, robot: Robot, authored_qpos: np.ndarray, fps: float, authored_object_position,
                 authored_object_quaternion_xyzw, sample_period_s: float = 0.2, pins=(*FEET, "pelvis"),
                 max_object_shift_m: float = 0.25, lock_waist: bool = False):
        authored_qpos = np.asarray(authored_qpos, dtype=float)
        if authored_qpos.ndim != 2 or authored_qpos.shape[1] != robot.model.nq or len(authored_qpos) < 2:
            raise ValueError(f"authored_qpos must be (T>=2, {robot.model.nq})")
        self.robot, self.fps, self.pins = robot, float(fps), tuple(pins)
        self.max_object_shift_m = max_object_shift_m
        # The whole-body controller realises the waist only approximately (it receives a torso request), so a plan
        # that leans the torso hands the error to the tool tips. Locking the waist makes the arms do the reaching.
        self.angle_pins = ["pelvis", "waist"] if lock_waist else ["pelvis"]
        authored_center = np.asarray(authored_object_position, dtype=float)
        authored_frame = _rotation(authored_object_quaternion_xyzw)
        duration = (len(authored_qpos) - 1) / fps
        times = np.unique(np.append(np.arange(0.0, duration, sample_period_s), duration))
        self.sample_times = times
        # tool paths relative to the authored object: (position in object frame, rotation in object frame)
        self.relative_paths = []
        for time in times:
            data = robot.data(authored_qpos[min(int(round(time * fps)), len(authored_qpos) - 1)])
            row = {}
            for hand in HANDS:
                position, matrix = robot.point(data, hand)
                row[hand] = (authored_frame.inv().apply(np.asarray(position) - authored_center),
                             authored_frame.inv() * Rotation.from_matrix(matrix))
            self.relative_paths.append(row)
        self.authored_center = authored_center

    def object_pose_in_world(self, pose: ObjectPose, state: RobotState) -> tuple[np.ndarray, Rotation]:
        if pose.frame_id != "base":
            raise GraspPlanError(f"pose must be in the base frame, got {pose.frame_id!r}")
        w, x, y, z = state.root_quaternion_wxyz
        base = Rotation.from_quat([x, y, z, w])
        return base.apply(pose.position) + state.root_position, base * _rotation(pose.quaternion_xyzw)

    def plan(self, pose: ObjectPose, state: RobotState | None = None) -> MotionSegment:
        if state is None:
            raise GraspPlanError("a pose-adaptive grasp needs the robot's current state")
        center, frame = self.object_pose_in_world(pose, state)
        shift = float(np.linalg.norm(center - self.authored_center))
        if shift > self.max_object_shift_m:
            raise GraspPlanError(f"object is {shift * 1000:.0f} mm from where the grasp was authored "
                                 f"(limit {self.max_object_shift_m * 1000:.0f} mm)")
        robot = self.robot
        start = np.concatenate([state.root_position, state.root_quaternion_wxyz, state.joint_positions])
        seed, solved = robot.validate_q(start), []
        for time, row in zip(self.sample_times, self.relative_paths):
            targets = {hand: frame.apply(row[hand][0]) + center for hand in HANDS}
            orientations = {hand: (frame * row[hand][1]).as_quat() for hand in HANDS}
            candidate, info = self._solve(seed, start, targets, orientations)
            if not info["converged"] or info.get("rejected"):
                errors = " · ".join(f"{hand.split('_')[0]} {error:.1f} mm"
                                    for hand, error in sorted(info.get("target_errors_mm", {}).items()))
                raise GraspPlanError(f"retargeted grasp unreachable at t={time:.1f} s: {errors or 'IK failed'} · "
                                     f"orientation {info['angle_error_deg']:.1f} deg")
            solved.append(candidate)
            seed = candidate
        solved = np.stack(solved)
        # densify to the authored rate by interpolating between the solved samples
        dense_times = np.arange(0.0, self.sample_times[-1] + 1e-9, 1.0 / self.fps)
        joints = np.stack([np.interp(dense_times, self.sample_times, solved[:, 7 + column])
                           for column in range(29)], axis=1)
        return MotionSegment(kind=MotionKind.GRASP, fps=self.fps, arm_targets=joints[:, ARMS_SLICE],
                             waist_ypr=joints[:, WAIST_SLICE], name="retargeted")

    def _solve(self, seed, start, targets, orientations):
        best = None
        initial = seed
        for weight, evaluations in RELAXATIONS:
            solved, info = self.robot.solve(
                initial, start, pins=self.pins, angle_pins=self.angle_pins, mode="free", resistance=0.,
                selected_targets={hand: targets[hand] for hand in HANDS},
                orientation_targets={hand: orientations[hand] for hand in HANDS},
                posture_reference=seed, posture_weight=weight, max_nfev=evaluations)
            if best is None or (info["converged"] and not info.get("rejected")) or \
                    info["target_error_mm"] < best[1]["target_error_mm"]:
                best = (solved, info)
            if info["converged"] and not info.get("rejected"):
                break
            initial = solved if not info.get("rejected") else seed
        return best
