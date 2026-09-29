"""MuJoCo FK and bounded, weighted whole-body IK. No dynamics stepping."""
from pathlib import Path
import hashlib
import os
import copy
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .grip_geometry import grip_pad_center, grip_pad_half_size, grip_pad_rotation

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / 'assets/g1/g1.xml'
HANDLES = {
    'pelvis': ('pelvis', (0, 0, 0), '골반'),
    'left_knee': ('left_knee_link', (0, 0, 0), '왼 무릎'),
    'right_knee': ('right_knee_link', (0, 0, 0), '오른 무릎'),
    'left_foot': ('left_ankle_roll_link', (.035, 0, -.035), '왼발'),
    'right_foot': ('right_ankle_roll_link', (.035, 0, -.035), '오른발'),
    'left_hand': ('left_wrist_yaw_link', (.055, 0, 0), '왼손'),
    'right_hand': ('right_wrist_yaw_link', (.055, 0, 0), '오른손'),
    'left_elbow': ('left_elbow_link', (0, 0, 0), '왼 팔꿈치'),
    'right_elbow': ('right_elbow_link', (0, 0, 0), '오른 팔꿈치'),
    'left_shoulder': ('left_shoulder_pitch_link', (0, 0, 0), '왼 어깨'),
    'right_shoulder': ('right_shoulder_pitch_link', (0, 0, 0), '오른 어깨'),
}
FEET = ('left_foot', 'right_foot')
ROTATABLE = ('pelvis', 'left_hand', 'right_hand', *FEET)
BASIC_ROTATABLE = ROTATABLE
HINGES = {key: key + '_joint' for key in ('left_elbow', 'right_elbow', 'left_knee', 'right_knee')}
BASIC_HANDLES = tuple(HANDLES)
PART_LABELS = {'hip_pitch': '고관절 피치', 'hip_roll': '고관절 롤', 'hip_yaw': '고관절 요',
               'knee': '무릎', 'ankle_pitch': '발목 피치', 'ankle_roll': '발목 롤',
               'shoulder_pitch': '어깨 피치', 'shoulder_roll': '어깨 롤', 'shoulder_yaw': '어깨 요',
               'elbow': '팔꿈치', 'wrist_roll': '손목 롤', 'wrist_pitch': '손목 피치', 'wrist_yaw': '손목 요',
               'waist_yaw': '허리 요', 'waist_roll': '허리 롤', 'waist_pitch': '허리 피치'}
# Derive anchor locations and body bindings from the same MJCF as FK/IK.
JOINT_HANDLES = {}
for body in ET.parse(MODEL_PATH).iter('body'):
    for joint in body.findall('joint'):
        name = joint.get('name', '')
        if not name.endswith('_joint') or joint.get('type', 'hinge') != 'hinge':
            continue
        part = name.removesuffix('_joint')
        side = '왼 ' if part.startswith('left_') else '오른 ' if part.startswith('right_') else ''
        part = part.removeprefix('left_').removeprefix('right_')
        JOINT_HANDLES[name] = (body.get('name'), tuple(float(x) for x in joint.get('pos', '0 0 0').split()), side + PART_LABELS[part])
HANDLES.update(JOINT_HANDLES)
HINGES.update({name: name for name in JOINT_HANDLES})
HIP_HANDLES = ('left_hip', 'right_hip')
for side, key in zip(('left', 'right'), HIP_HANDLES):
    body, offset, _ = JOINT_HANDLES[f'{side}_hip_roll_joint']
    HANDLES[key] = (body, offset, ('왼' if side == 'left' else '오른') + ' 고관절')
ROTATABLE = (*ROTATABLE, *HIP_HANDLES)
for key, joint, label in [('waist', 'waist_roll_joint', '허리'),
                          ('left_ankle', 'left_ankle_roll_joint', '왼 발목'),
                          ('right_ankle', 'right_ankle_roll_joint', '오른 발목')]:
    body, offset, _ = JOINT_HANDLES[joint]
    HANDLES[key] = (body, offset, label)
# Ankles expose motor-angle targets, not an arbitrary 3-D orientation target.
ROTATABLE = (*ROTATABLE, 'waist')
ANGLE_LOCKABLE = tuple(dict.fromkeys((*ROTATABLE, *HINGES)))


def skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def right_jacobian(v):
    theta = np.linalg.norm(v)
    k = skew(v)
    if theta < 1e-5:
        return np.eye(3) - .5*k + k@k/6
    return np.eye(3) - (1-np.cos(theta))/theta**2*k + (theta-np.sin(theta))/theta**3*(k@k)


def left_jacobian_inverse(v):
    theta = np.linalg.norm(v)
    k = skew(v)
    coefficient = 1/12 if theta < 1e-5 else (1 - .5*theta/np.tan(theta/2))/theta**2
    return np.eye(3) - .5*k + coefficient*(k@k)


def quat_matrix(wxyz):
    return Rotation.from_quat(np.asarray(wxyz)[[1, 2, 3, 0]]).as_matrix()


def matrix_quat(matrix):
    return Rotation.from_matrix(matrix).as_quat()[[3, 0, 1, 2]]


class Robot:
    def __init__(self, model_id=None):
        self.model_id = model_id or os.environ.get('MOTIONCREATOR_MODEL', 'g1')
        self.handles = dict(HANDLES)
        if self.model_id == 'g1-tools':
            from .tool_model import tool_model_xml
            self.xml, tool_handles = tool_model_xml()
            self.handles.update(tool_handles)
            self.model = mujoco.MjModel.from_xml_string(ET.tostring(self.xml, encoding='unicode'))
        elif self.model_id == 'g1':
            self.xml = ET.parse(MODEL_PATH).getroot()
            self.xml.find('compiler').set('meshdir', str(MODEL_PATH.parent / 'meshes'))
            self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        else:
            raise ValueError('Unknown robot model')
        m = self.model
        self.names = [m.joint(i).name for i in range(1, m.njnt)]
        self.ids = {k: m.body(v[0]).id for k, v in self.handles.items()}
        # Hip translation/pivot follows the roll anchor; orientation includes all three hip axes.
        self.orientation_ids = {**self.ids, **{f'{side}_hip': m.body(f'{side}_hip_yaw_link').id for side in ('left', 'right')}, 'waist': m.body('torso_link').id}
        self.q_indices = np.r_[0:3, 7:m.nq]
        self.v_indices = np.r_[0:3, 6:m.nv]
        self.lower = np.r_[[-4, -4, .20], m.jnt_range[1:, 0]]
        self.upper = np.r_[[4, 4, 1.5], m.jnt_range[1:, 1]]
        self.home = m.qpos0.copy()
        for side in ('left', 'right'):
            for name, value in [('hip_pitch', -.12), ('knee', .24), ('ankle_pitch', -.12), ('elbow', .15)]:
                self.home[m.joint(f'{side}_{name}_joint').qposadr[0]] = value
        d = self.data(self.home)
        # The model uses radius-5 mm contact spheres at z=-30 mm.
        self.home[2] -= min(self.point(d, k)[0][2] for k in FEET)
        self.visual_ids = [i for i in range(m.ngeom) if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_group[i] == 1]
        if self.model_id == 'g1':
            fingerprint_source = MODEL_PATH.read_bytes()
        else:
            # mj_saveLastXML may keep or omit the inferred content_type attribute
            # depending on MuJoCo's process-local mesh cache.  It has no effect on
            # the model, so exclude it from the project identity.  Otherwise the
            # editor and the physics worker can reject the same gripper URDF as
            # two different robots.
            fingerprint_xml = copy.deepcopy(self.xml)
            for element in fingerprint_xml.iter():
                element.attrib.pop('content_type', None)
            fingerprint_source = ET.tostring(fingerprint_xml)
        self.fingerprint = hashlib.sha256(fingerprint_source).hexdigest()
        self.compatible_fingerprints = {self.fingerprint}
        if self.model_id == 'g1-tools':
            # The gripper model changes only fixed end-effector links. Its 29
            # actuated joints, order and qpos layout are identical to base G1,
            # so authored base-model motion references are directly portable.
            self.compatible_fingerprints.add(hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest())
            # The right support TCP was aligned with the mirrored left scoop
            # TCP. The 29 joints and physical geoms did not change, so saved
            # projects using the earlier TCP calibration remain editable.
            self.compatible_fingerprints.update({
                'd270539dbd3c592d46a7721e623c4b71d9be7b73ad7437a5531ec3161519aa45',
                'bde0de43b90b2ff14ebf6641393c1347bd6b1e82620ffcc90d155cfa6ff1163a',
            })
            # Projects created by the first gripper-model load in older builds
            # may contain MuJoCo's inferred STL content_type attributes.
            legacy_xml = copy.deepcopy(fingerprint_xml)
            for mesh in legacy_xml.findall('./asset/mesh'):
                mesh.set('content_type', 'model/stl')
            self.compatible_fingerprints.add(hashlib.sha256(ET.tostring(legacy_xml)).hexdigest())

    def xml_root(self):
        return copy.deepcopy(self.xml)

    def validate_q(self, value):
        q = np.array(value, dtype=float, copy=True)
        if q.shape != (self.model.nq,) or not np.isfinite(q).all():
            raise ValueError(f'qpos must contain {self.model.nq} finite numbers')
        norm = np.linalg.norm(q[3:7])
        if abs(norm - 1.) > 1e-4:
            raise ValueError('Root quaternion must be normalized (wxyz)')
        q[3:7] /= norm
        x = q[self.q_indices]
        if np.any(x < self.lower - 1e-7) or np.any(x > self.upper + 1e-7):
            raise ValueError('Pose exceeds model joint or workspace limits')
        return q.copy()

    def data(self, q):
        d = mujoco.MjData(self.model)
        d.qpos[:] = q
        mujoco.mj_forward(self.model, d)
        return d

    def point(self, d, key):
        b = self.ids[key]
        r = d.xmat[b].reshape(3, 3)
        return d.xpos[b] + r @ np.array(self.handles[key][1]), d.xmat[self.orientation_ids[key]].reshape(3, 3)

    def distance(self, a, b):
        def ancestors(n):
            out = []
            while n:
                out.append(n)
                n = int(self.model.body_parentid[n])
            return out + [0]
        aa, bb = ancestors(self.ids[a]), ancestors(self.ids[b])
        return min(i + bb.index(n) for i, n in enumerate(aa) if n in bb)

    def solve(self, q, anchor, focus=None, target=None, pins=FEET, resistance=1., mode='elastic', targets=None, max_nfev=50,
              posture_reference=None, posture_weight=.055, selected_targets=None, orientation_targets=None,
              joint_targets=None, angle_pins=()):
        q, anchor = self.validate_q(q), self.validate_q(anchor)
        selected_targets = dict(selected_targets or {})
        orientation_targets = dict(orientation_targets or {})
        joint_targets = dict(joint_targets or {})
        angle_pins = tuple(dict.fromkeys(angle_pins or ()))
        if focus is not None:
            selected_targets[focus] = target
        if any(k not in HANDLES for k in [*pins, *selected_targets, *orientation_targets]):
            raise ValueError('Unknown handle')
        if set(selected_targets) & set(pins):
            raise ValueError('선택한 부위에 고정된 부위가 있습니다. 이동하려면 먼저 고정을 해제하세요.')
        if any(k not in ROTATABLE for k in orientation_targets):
            raise ValueError('방향 회전은 골반·손·발·통합 고관절·허리에서 지원합니다. 개별 관절과 통합 발목은 관절각 목표를 사용하세요.')
        if any(k not in ANGLE_LOCKABLE for k in angle_pins):
            raise ValueError('각도 고정을 지원하지 않는 부위입니다.')
        if set(orientation_targets) & set(angle_pins):
            raise ValueError('각도가 고정된 부위를 회전하려면 각도 고정을 해제하세요.')
        if set(orientation_targets) & set(pins) & set(FEET):
            raise ValueError('발 방향이 고정되어 있습니다. 회전하려면 발 고정을 해제하세요.')
        if any(k not in self.names for k in joint_targets):
            raise ValueError('Unknown joint')
        for key, value in joint_targets.items():
            i = self.names.index(key)
            if not np.isfinite(value) or not self.lower[3+i] <= value <= self.upper[3+i]:
                raise ValueError('Joint target exceeds joint limits')
        ad = self.data(anchor)
        base_targets = {k: self.point(ad, k) for k in HANDLES}
        orientation_locks = {key: base_targets[key][1] for key in angle_pins if key in ROTATABLE}
        joint_locks = {HINGES[key]: float(anchor[self.model.joint(HINGES[key]).qposadr[0]])
                       for key in angle_pins if key in HINGES}
        if set(joint_targets) & set(joint_locks):
            raise ValueError('각도가 고정된 관절을 조정하려면 각도 고정을 해제하세요.')
        desired = {k: (p.copy(), r.copy()) for k, (p, r) in base_targets.items()}
        if targets:
            desired.update(targets)
        for key, value in selected_targets.items():
            t = np.asarray(value, dtype=float)
            if t.shape != (3,) or not np.isfinite(t).all() or np.max(np.abs(t)) > 5:
                raise ValueError('Target must be a finite XYZ point within 5 m')
            desired[key] = (t, desired[key][1])
        for key, value in orientation_targets.items():
            quat = np.asarray(value, dtype=float)
            if quat.shape != (4,) or not np.isfinite(quat).all() or abs(np.linalg.norm(quat)-1) > 1e-4:
                raise ValueError('Orientation target must be a normalized xyzw quaternion')
            desired[key] = (desired[key][0], Rotation.from_quat(quat).as_matrix())
        for key, rotation in orientation_locks.items():
            desired[key] = (desired[key][0], rotation)
        active = set(selected_targets) | set(orientation_targets) | set(joint_targets) | set(angle_pins)
        # Extra selectable anchors must not add passive resistance everywhere.
        solve_handles = tuple(dict.fromkeys([*BASIC_HANDLES, *pins, *selected_targets,
                                             *orientation_targets, *orientation_locks]))
        weights = {}
        for k in solve_handles:
            if k in pins:
                weights[k] = 180.
                desired[k] = (base_targets[k][0], desired[k][1] if k in orientation_targets else base_targets[k][1])
            elif k in selected_targets or k in orientation_targets:
                weights[k] = 28.
            elif k in orientation_locks:
                # An angle-only lock must not resist translation of the handle.
                weights[k] = 0.
            elif targets:
                weights[k] = 4.
            elif mode == 'elastic':
                distance = min(self.distance(key, k) for key in (active or {'pelvis'}))
                weights[k] = resistance * (.12 + 3.0 * min(distance / 10, 1) ** 2)
            else:
                weights[k] = .015
        d = self.data(q)
        rotate_base = 'pelvis' in orientation_targets
        n_basic = len(self.q_indices)
        n = n_basic + (3 if rotate_base else 0)
        base_rotation = quat_matrix(q[3:7])
        posture_q = anchor if posture_reference is None else self.validate_q(posture_reference)
        posture_x = posture_q[self.q_indices]
        if rotate_base:
            posture_x = np.r_[posture_x, Rotation.from_matrix(base_rotation.T @ quat_matrix(posture_q[3:7])).as_rotvec()]
        jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        rotation_map = np.eye(3)

        def jacobian(point, body, orientation_body=None):
            mujoco.mj_jac(self.model, d, jp, jr, point, body)
            if orientation_body is not None and orientation_body != body:
                mujoco.mj_jacBody(self.model, d, None, jr, orientation_body)
            p, r = jp[:, self.v_indices].copy(), jr[:, self.v_indices].copy()
            if rotate_base:
                p = np.column_stack([p, jp[:, 3:6] @ rotation_map])
                r = np.column_stack([r, jr[:, 3:6] @ rotation_map])
            return p, r

        def evaluate(x, jac=False):
            nonlocal rotation_map
            d.qpos[:] = q
            d.qpos[self.q_indices] = x[:n_basic]
            if rotate_base:
                d.qpos[3:7] = matrix_quat(base_rotation @ Rotation.from_rotvec(x[-3:]).as_matrix())
                rotation_map = right_jacobian(x[-3:])
            mujoco.mj_kinematics(self.model, d)
            mujoco.mj_comPos(self.model, d)
            residuals, matrices = [], []
            for k in solve_handles:
                p, r = self.point(d, k)
                tp, tr = desired[k]
                jpos = jrot = None
                if weights[k]:
                    residuals.append(weights[k] * (p-tp))
                    if jac:
                        jpos, jrot = jacobian(p, self.ids[k], self.orientation_ids[k])
                        matrices.append(weights[k] * jpos)
                if k in orientation_targets or k in orientation_locks or (k in pins and k in FEET):
                    w = 90. if k in orientation_locks or (k in pins and k in FEET) else 18.
                    err = Rotation.from_matrix(r @ tr.T).as_rotvec()
                    residuals.append(w * err)
                    if jac:
                        if jrot is None:
                            _, jrot = jacobian(p, self.ids[k], self.orientation_ids[k])
                        matrices.append(w * left_jacobian_inverse(err) @ jrot)
                if k in FEET:
                    for offset in ((-.085, -.03, 0), (-.085, .03, 0), (.085, -.03, 0), (.085, .03, 0)):
                        corner = p + r @ offset
                        residuals.append(np.array([120 * min(0., corner[2])]))
                        if jac:
                            cj, _ = jacobian(corner, self.ids[k])
                            matrices.append(120*cj[2:3] if corner[2] < 0 else np.zeros((1, n)))
            for key, value in joint_targets.items():
                index = 3 + self.names.index(key)
                residuals.append(np.array([24.*(x[index]-value)]))
                if jac:
                    row = np.zeros((1, n)); row[0, index] = 24.; matrices.append(row)
            for key, value in joint_locks.items():
                index = 3 + self.names.index(key)
                residuals.append(np.array([120.*(x[index]-value)]))
                if jac:
                    row = np.zeros((1, n)); row[0, index] = 120.; matrices.append(row)
            residuals.append(posture_weight*(x-posture_x))
            if jac:
                matrices.append(posture_weight*np.eye(n))
                return np.vstack(matrices)
            return np.concatenate(residuals)

        initial = q[self.q_indices]
        lower, upper = self.lower, self.upper
        if rotate_base:
            initial = np.r_[initial, np.zeros(3)]
            lower, upper = np.r_[lower, [-np.pi]*3], np.r_[upper, [np.pi]*3]
        result = least_squares(evaluate, np.clip(initial, lower+1e-9, upper-1e-9),
                               jac=lambda x: evaluate(x, True), bounds=(lower, upper),
                               max_nfev=max_nfev, ftol=1e-5, xtol=1e-6, gtol=1e-5)
        answer = q.copy()
        answer[self.q_indices] = result.x[:n_basic]
        if rotate_base:
            answer[3:7] = matrix_quat(base_rotation @ Rotation.from_rotvec(result.x[-3:]).as_matrix())
        rd = self.data(answer)
        pin_error = max((np.linalg.norm(self.point(rd, k)[0]-base_targets[k][0]) for k in pins), default=0.)
        angle_error = max((np.linalg.norm(Rotation.from_matrix(self.point(rd, k)[1] @ base_targets[k][1].T).as_rotvec()) for k in pins if k in FEET), default=0.)
        orientation_pin_error = max((np.linalg.norm(Rotation.from_matrix(self.point(rd, k)[1] @ base_targets[k][1].T).as_rotvec())
                                     for k in orientation_locks), default=0.)
        joint_pin_error = max((abs(answer[self.model.joint(name).qposadr[0]] - value)
                               for name, value in joint_locks.items()), default=0.)
        angle_pin_error = max(orientation_pin_error, joint_pin_error)
        rejected = bool(pin_error > .003 or angle_error > .015 or angle_pin_error > np.deg2rad(.5))
        if rejected:
            answer = q; rd = self.data(answer)
        errors = {key: float(np.linalg.norm(self.point(rd, key)[0]-desired[key][0]))*1000
                  for key in set(selected_targets) | set(orientation_targets)}
        rotation_errors = {key: float(Rotation.from_matrix(self.point(rd, key)[1] @ desired[key][1].T).magnitude()*180/np.pi) for key in orientation_targets}
        joint_errors = {key: float(abs(answer[7+self.names.index(key)]-value)*180/np.pi) for key, value in joint_targets.items()}
        angular_error = max([*rotation_errors.values(), *joint_errors.values()], default=0.)
        error = max(errors.values(), default=0.)
        return answer, {'target_error_mm': error, 'pin_error_mm': float(pin_error*1000),
                        'rejected': rejected, 'converged': error < 10 and angular_error < 2 and not rejected,
                        'evaluations': result.nfev, 'target_errors_mm': errors,
                        'angle_error_deg': angular_error, 'rotation_errors_deg': rotation_errors,
                        'joint_errors_deg': joint_errors,
                        'angle_pin_error_deg': float(np.rad2deg(angle_pin_error))}

    def state(self, q):
        d = self.data(q)
        handles = {k: {'position': p.tolist(), 'quaternion': Rotation.from_matrix(r).as_quat().tolist(), 'label': self.handles[k][2]}
                   for k in HANDLES for p, r in [self.point(d, k)]}
        geoms = {str(i): {'position': d.geom_xpos[i].tolist(), 'quaternion': Rotation.from_matrix(d.geom_xmat[i].reshape(3, 3)).as_quat().tolist()}
                 for i in self.visual_ids}
        grip_pads = {}
        for side in (() if self.model_id == 'g1-tools' else ('left', 'right')):
            body = self.model.body(f'{side}_wrist_yaw_link').id
            rotation = d.xmat[body].reshape(3, 3)
            grip_pads[side] = {
                'position': (d.xpos[body] + rotation @ grip_pad_center(side)).tolist(),
                'quaternion': Rotation.from_matrix(rotation @ grip_pad_rotation(side)).as_quat().tolist(),
                'size': (2 * grip_pad_half_size()).tolist(),
            }
        cameras = {}
        if self.model_id == 'g1-tools':
            from .tool_model import head_camera_spec
            camera = head_camera_spec()
            body = self.model.body(camera['link']).id
            cameras['head'] = {
                'position': d.xpos[body].tolist(),
                'quaternion': Rotation.from_matrix(d.xmat[body].reshape(3, 3)).as_quat().tolist(),
                'label': 'Head Cam · D435',
                'calibrated_projection': False,
            }
        floor_min = min(self.point(d, k)[0][2] + (self.point(d, k)[1] @ np.array(o))[2]
                        for k in FEET for o in ((-.085, -.03, 0), (-.085, .03, 0), (.085, -.03, 0), (.085, .03, 0)))
        return {'model_id': self.model_id, 'qpos': np.asarray(q).tolist(), 'handles': handles, 'geoms': geoms,
                'grip_pads': grip_pads, 'cameras': cameras,
                'com': d.subtree_com[self.ids['pelvis']].tolist(), 'floor_min_mm': float(floor_min * 1000),
                'hinges': {k: {'joint_name': name, 'angle': float(q[self.model.joint(name).qposadr[0]]),
                              'limits': self.model.joint(name).range.tolist(),
                              'axis_world': d.xaxis[self.model.joint(name).id].tolist(),
                              'position': d.xanchor[self.model.joint(name).id].tolist()} for k, name in HINGES.items()}}

    def export_visual(self, path):
        import trimesh
        scene = trimesh.Scene()
        m = self.model
        for i in self.visual_ids:
            mid = m.geom_dataid[i]
            va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            fa, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
            mesh = trimesh.Trimesh(m.mesh_vert[va:va+vn].copy(), m.mesh_face[fa:fa+fn].copy(), process=False)
            mesh.visual = trimesh.visual.ColorVisuals(mesh, face_colors=np.tile((m.geom_rgba[i] * 255).astype(np.uint8), (len(mesh.faces), 1)))
            scene.add_geometry(mesh, node_name=f'geom_{i}', geom_name=f'geom_{i}')
        Path(path).write_bytes(scene.export(file_type='glb'))
