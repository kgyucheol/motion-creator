#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_gripper_urdf.py
Unitree G1 29DoF rev1.0 공식 URDF에서 더미손(rubber hand)을 빼고, 그 자리(손목 플랜지)에
커스텀 그리퍼 두 개(V 주걱 scoop + ㄴ 끝단 받침 end_support)를 붙인 URDF 패키지를 만든다.

  python make_gripper_urdf.py --src <unitree_ros/robots/g1_description> --out <출력 폴더>
        [--scoop-side right|left] [--groove-depth-mm 0.65] [--groove-width-mm 7]
        [--body-density 2700] [--adapter-density 2700]

필요 패키지: numpy scipy trimesh shapely lxml manifold3d  (본인 전용 venv에만 설치)
"""
import argparse, hashlib, json, os, re, shutil
import numpy as np, trimesh
from scipy.interpolate import CubicSpline
from shapely.geometry import Polygon, LineString, box as sbox
from shapely.ops import unary_union
from lxml import etree

SRC_URDF = 'g1_29dof_rev_1_0.urdf'
FLANGE_X = 0.0415          # 공식 *_hand_palm_joint 원점 x = 손목 yaw 링크 메시 앞면(플랜지면)

def T(x=0.0, y=0.0, z=0.0): return trimesh.transformations.translation_matrix((x, y, z))
def union(ms): return trimesh.boolean.union(ms, engine='manifold') if len(ms) > 1 else ms[0]

def grid_solid(U, S, xy, top, bot):
    """(u, s) 격자 위 윗면·아랫면·옆면을 이어 붙인 닫힌 솔리드"""
    nu, ns = len(U), len(S)
    Pt = np.array([[(*xy(u, s), top(u, s)) for s in S] for u in U]).reshape(-1, 3)
    Pb = np.array([[(*xy(u, s), bot(u, s)) for s in S] for u in U]).reshape(-1, 3)
    it = lambda i, j: i*ns + j; ib = lambda i, j: nu*ns + i*ns + j
    F = []
    for i in range(nu - 1):
        for j in range(ns - 1):
            F += [[it(i, j), it(i+1, j), it(i+1, j+1)], [it(i, j), it(i+1, j+1), it(i, j+1)]]
            F += [[ib(i, j), ib(i+1, j+1), ib(i+1, j)], [ib(i, j), ib(i, j+1), ib(i+1, j+1)]]
        for j in (0, ns - 1):
            F += [[it(i, j), ib(i, j), ib(i+1, j)], [it(i, j), ib(i+1, j), it(i+1, j)]]
    for i in (0, nu - 1):
        for j in range(ns - 1):
            F += [[it(i, j), ib(i, j+1), ib(i, j)], [it(i, j), it(i, j+1), ib(i, j+1)]]
    m = trimesh.Trimesh(np.vstack([Pt, Pb]), np.array(F), process=True); m.fix_normals(); return m

# ---------------- 공통: 플랜지 어댑터 (원판 Ø58×12 + 클램프 블록 30×44 + 볼트머리 2개) ----------------
def adapter(z0, z1):
    disk = trimesh.creation.cylinder(radius=0.029, height=0.012, sections=64)
    disk.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [0, 1, 0])); disk.apply_translation([0.006, 0, 0])
    blk = trimesh.creation.box((0.030, 0.044, z1 - z0), T(0.027, 0, (z0 + z1)/2))
    bolts = [trimesh.creation.cylinder(radius=0.0035, height=0.003, sections=24, transform=T(0.027, yy, z1 + 0.0015)) for yy in (-0.011, 0.011)]
    prims = [('cylinder', dict(radius=0.029, length=0.012), (0.006, 0, 0), (0, np.pi/2, 0)),
             ('box', dict(size=(0.030, 0.044, z1 - z0)), (0.027, 0, (z0 + z1)/2), (0, 0, 0))]
    return union([disk, blk] + bolts), prims

# ---------------- V 주걱 (scoop) : 링크 원점 = 플랜지면 중심, x = 그리퍼 방향, z = 윗면 법선 ----------------
X0, LP = 0.060 - FLANGE_X, 0.230                                          # 목 시작 x, 전체 길이
FZ = CubicSpline([0, 0.03, 0.07, 0.12, 0.165, 0.20, 0.23], [0.004, 0.002, -0.010, -0.030, -0.038, -0.030, -0.014])  # 옆에서 본 V
DISH = 1.8                                                                # 폭 방향 오목 (가장자리 +11.5 mm)
def hw(u):                                                                # 위에서 본 밥주걱 반폭 (목 32 → 최대 160 mm)
    if u < 0.05: return 0.016
    if u < 0.19:
        t = (u - 0.05)/0.14; t = t*t*(3 - 2*t); return 0.016 + 0.064*t
    return 0.080*np.sqrt(max(0.0, 1 - ((u - 0.19)/0.04)**2))
def th(u): return 0.008 - 0.005*u/LP                                      # 두께 8 → 3 mm
def s_top(u, s): return FZ(u) + DISH*(s*hw(u))**2
def scoop_patch(ua, ub, sa, sb, nu, ns):
    return grid_solid(np.linspace(ua, ub, nu), np.linspace(sa, sb, ns), lambda u, s: (X0 + u, s*hw(u)), s_top, lambda u, s: s_top(u, s) - th(u))
def make_scoop(sign):
    ad, prims = adapter(-0.012, 0.012)
    body = trimesh.boolean.difference([scoop_patch(0, LP - 2e-4, -1, 1, 140, 37), ad], engine='manifold')
    # 충돌체: 볼록 조각 13 × 6 = 78개. 곡률이 큰 둥근 끝(u 190~230)과 폭 가장자리에 조각을 몰아 배치
    #   목(u 0~50, 폭 32 mm): 길이 3 × 폭 2 = 6조각 — 목이 아래로 꺾이는 구간이라 길이 방향만 잘게
    #   날·끝(u 50~230): 길이 12 × 폭 6 = 72조각 — 둥근 끝(u 190~230)과 폭 가장자리에 조밀하게
    un = np.array([0.0, 0.0235, 0.037, 0.05]); sn = np.array([-1.0, 0.0, 1.0])
    ub = np.r_[np.linspace(0.05, 0.19, 8), np.linspace(0.19, LP - 2e-4, 6)[1:]]; sb = np.sin(np.linspace(-np.pi/2, np.pi/2, 7))
    cols = [scoop_patch(ue[i], ue[i+1], se[j], se[j+1], 7, 7).convex_hull
            for ue, se in ((un, sn), (ub, sb)) for i in range(len(ue) - 1) for j in range(len(se) - 1)]
    u = np.linspace(0, LP, 4001); us = u[np.argmin(FZ(u))]
    tcp = (X0 + us, 0.0, float(FZ(us)))                                  # 오목면 최저점 (묶음이 앉는 곳)
    return body, ad, cols, prims, tcp

# ---------------- ㄴ 끝단 받침 (end_support) : 평판 180×90×4 + 윗면 홈 3줄 + 바깥쪽 옆벽 ----------------
PT, PB, WALL_H = -0.030, -0.034, 0.036
PX0, PX1, PW = 0.0535 - FLANGE_X, 0.2335 - FLANGE_X, 0.090
WX_FULL = 0.160 - FLANGE_X                                                # 옆벽이 최대 높이가 되는 x
def make_end_support(sign, groove_depth, groove_width):
    outline = sbox(PX0, -PW/2, PX1, PW/2).buffer(-0.005).buffer(0.005)
    grooves = unary_union([LineString([(0.100 - FLANGE_X, yc), (0.200 - FLANGE_X, yc)]).buffer(groove_width/2) for yc in (-0.022, 0.0, 0.022)])
    base = trimesh.creation.extrude_polygon(outline, height=(PT - PB) - groove_depth, transform=T(0, 0, PB))
    toplayer = trimesh.creation.extrude_polygon(outline.difference(grooves), height=groove_depth, transform=T(0, 0, PT - groove_depth))
    wp = Polygon([(PX0, PB), (PX1, PB), (PX1, PB + WALL_H), (WX_FULL, PB + WALL_H), (PX0, PB + 0.012)])
    wall = trimesh.creation.extrude_polygon(wp, height=0.004); v = wall.vertices.copy()
    y0 = 0.041 if sign > 0 else -0.045                                    # 옆벽 = 몸 바깥쪽
    wall.vertices = np.c_[v[:, 0], y0 + v[:, 2], v[:, 1]]; wall.invert(); wall.fix_normals()
    body = union([base, toplayer, wall])
    ad, prims = adapter(PT, 0.014)
    prims = prims + [('box', dict(size=(PX1 - PX0, PW, PT - PB)), ((PX0 + PX1)/2, 0, (PT + PB)/2), (0, 0, 0))]   # 판 (홈은 충돌체에서 생략)
    tcp = (0.180305, 0.0, -0.0380589785)  # 주걱 TCP와 손목 기준 좌우 대칭인 제어점
    return body, ad, [wall.convex_hull], prims, tcp

# ---------------- 질량·관성 ----------------
def inertial(parts):
    props = []
    for m, rho in parts:
        m = m.copy(); m.density = rho; props.append((m.mass, m.center_mass, m.moment_inertia))
    M = sum(p[0] for p in props); c = sum(p[0]*p[1] for p in props)/M; I = np.zeros((3, 3))
    for mass, com, Ic in props:
        d = com - c; I += Ic + mass*((d @ d)*np.eye(3) - np.outer(d, d))
    return M, c, I

# ---------------- URDF 조각 ----------------
f = lambda v: f'{v:.9g}'
xyz = lambda p: ' '.join(f(v) for v in p)
def link_xml(name, M, c, I, meshes, cols, prims):
    s = [f'<link name="{name}">',
         f'  <inertial>\n    <origin xyz="{xyz(c)}" rpy="0 0 0"/>\n    <mass value="{f(M)}"/>',
         f'    <inertia ixx="{f(I[0,0])}" ixy="{f(I[0,1])}" ixz="{f(I[0,2])}" iyy="{f(I[1,1])}" iyz="{f(I[1,2])}" izz="{f(I[2,2])}"/>\n  </inertial>']
    for fn, matname in meshes:
        s.append(f'  <visual>\n    <origin xyz="0 0 0" rpy="0 0 0"/>\n    <geometry>\n      <mesh filename="meshes/{fn}"/>\n    </geometry>\n    <material name="{matname}"/>\n  </visual>')
    for fn in cols:
        s.append(f'  <collision>\n    <origin xyz="0 0 0" rpy="0 0 0"/>\n    <geometry>\n      <mesh filename="meshes/{fn}"/>\n    </geometry>\n  </collision>')
    for kind, a, o, r in prims:
        g = f'<cylinder radius="{f(a["radius"])}" length="{f(a["length"])}"/>' if kind == 'cylinder' else f'<box size="{xyz(a["size"])}"/>'
        s.append(f'  <collision>\n    <origin xyz="{xyz(o)}" rpy="{xyz(r)}"/>\n    <geometry>\n      {g}\n    </geometry>\n  </collision>')
    s.append('</link>'); return '\n'.join(s)
def fixed_joint(name, parent, child, o):
    return f'<joint name="{name}" type="fixed">\n  <origin xyz="{xyz(o)}" rpy="0 0 0"/>\n  <parent link="{parent}"/>\n  <child link="{child}"/>\n</joint>'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--scoop-side', choices=['right', 'left'], default='right')
    ap.add_argument('--groove-depth-mm', type=float, default=0.65); ap.add_argument('--groove-width-mm', type=float, default=7.0)
    ap.add_argument('--body-density', type=float, default=2700.0, help='그리퍼 본체 밀도 kg/m3 (알루미늄 6061=2700, PLA≈1240)')
    ap.add_argument('--adapter-density', type=float, default=2700.0)
    a = ap.parse_args()
    assert 0.3 <= a.groove_depth_mm <= 2.0, 'groove depth out of range'
    os.makedirs(os.path.join(a.out, 'meshes'), exist_ok=True)
    tree = etree.parse(os.path.join(a.src, SRC_URDF)); root = tree.getroot()
    sides = {a.scoop_side: 'scoop', ('left' if a.scoop_side == 'right' else 'right'): 'end_support'}
    report = {'source': SRC_URDF, 'scoop_side': a.scoop_side, 'groove_depth_mm': a.groove_depth_mm, 'groove_width_mm': a.groove_width_mm,
              'body_density': a.body_density, 'adapter_density': a.adapter_density, 'links': {}}
    snippet = []
    for side, kind in sides.items():
        sign = +1 if side == 'left' else -1
        body, ad, cols, prims, tcp = make_scoop(sign) if kind == 'scoop' else make_end_support(sign, a.groove_depth_mm/1000, a.groove_width_mm/1000)
        assert body.is_watertight and ad.is_watertight, f'{side} {kind}: mesh not watertight'
        base = f'{side}_{kind}'
        body.export(os.path.join(a.out, 'meshes', f'{base}_body.STL')); ad.export(os.path.join(a.out, 'meshes', f'{base}_adapter.STL'))
        colnames = []
        for k, cm in enumerate(cols):
            fn = f'{base}_col_{k:02d}.STL'; cm.export(os.path.join(a.out, 'meshes', fn)); colnames.append(fn)
        M, c, I = inertial([(body, a.body_density), (ad, a.adapter_density)])
        ev = np.linalg.eigvalsh(I); assert ev.min() > 0 and ev[2] <= ev[0] + ev[1] + 1e-12, 'bad inertia'
        link = link_xml(f'{base}_link', M, c, I, [(f'{base}_body.STL', 'gripper_white'), (f'{base}_adapter.STL', 'adapter_dark')], colnames, prims)
        j1 = fixed_joint(f'{base}_joint', f'{side}_wrist_yaw_link', f'{base}_link', (FLANGE_X, 0, 0))
        tl = f'<link name="{base}_tcp"/>'; j2 = fixed_joint(f'{base}_tcp_joint', f'{base}_link', f'{base}_tcp', tcp)
        # 공식 더미손 링크·관절을 같은 자리에서 교체
        old_link = root.find(f"link[@name='{side}_rubber_hand']"); old_joint = root.find(f"joint[@name='{side}_hand_palm_joint']")
        idx = list(root).index(old_link); tail = old_link.tail
        root.remove(old_link); root.remove(old_joint)
        els = [etree.Comment(f' custom gripper ({kind}) replaces {side}_rubber_hand — generated by make_gripper_urdf.py ')] + [etree.fromstring(x) for x in (link, j1, tl, j2)]
        for k, e in enumerate(els):
            e.tail = tail; root.insert(idx + k, e)
        snippet += [link, j1, tl, j2]
        report['links'][f'{base}_link'] = dict(mass_kg=round(M, 4), com_m=[round(v, 5) for v in c], tcp_m=[round(v, 5) for v in tcp],
                                               body_volume_cm3=round(body.volume*1e6, 1), adapter_volume_cm3=round(ad.volume*1e6, 1),
                                               n_collision_meshes=len(colnames), n_collision_primitives=len(prims),
                                               bounds_mm=(body.bounds*1000).round(1).tolist())
    # 재질 2개 추가 (맨 위, 기존 재질 뒤)
    mats = [etree.fromstring('<material name="gripper_white"><color rgba="0.95 0.95 0.93 1"/></material>'),
            etree.fromstring('<material name="adapter_dark"><color rgba="0.30 0.31 0.33 1"/></material>')]
    last_mat = [e for e in root if e.tag == 'material'][-1]; pos = list(root).index(last_mat)
    for k, m in enumerate(mats): m.tail = last_mat.tail; root.insert(pos + 1 + k, m)
    comp = root.find('mujoco/compiler')                                    # MuJoCo 버전별 경로 처리 차이 제거: 파일명만 쓰고 meshdir에서 찾기
    if comp is not None: comp.set('strippath', 'true')
    root.set('name', root.get('name') + '_scoop_endsupport')
    root.insert(0, etree.Comment(' Based on unitreerobotics/unitree_ros robots/g1_description/g1_29dof_rev_1_0.urdf (BSD-3-Clause, see LICENSE_unitree_ros). '
                                 'Rubber hands removed; custom grippers attached at the wrist flange (x=0.0415 m). Everything else unchanged. '))
    out_urdf = os.path.join(a.out, 'g1_29dof_rev_1_0_scoop_endsupport.urdf')
    tree.write(out_urdf, xml_declaration=True, encoding='utf-8')
    # 공식 메시 복사 (더미손 메시는 제외)
    used = sorted(set(re.findall(r'filename="meshes/([^"]+)"', open(out_urdf).read())))
    for fn in used:
        sp = os.path.join(a.src, 'meshes', fn)
        if os.path.exists(sp): shutil.copy2(sp, os.path.join(a.out, 'meshes', fn))
    missing = [fn for fn in used if not os.path.exists(os.path.join(a.out, 'meshes', fn))]
    assert not missing, f'missing meshes: {missing}'
    lic = os.path.join(a.src, '..', '..', 'LICENSE')
    if os.path.exists(lic): shutil.copy2(lic, os.path.join(a.out, 'LICENSE_unitree_ros'))
    with open(os.path.join(a.out, 'gripper_links_snippet.urdf.xml'), 'w') as fp:
        fp.write('<!-- 다른 G1 URDF에 붙일 때: 기존 손 링크·관절을 지우고 아래를 <robot> 안에 붙여 넣기. 재질 2개도 함께 -->\n')
        fp.write('<material name="gripper_white"><color rgba="0.95 0.95 0.93 1"/></material>\n<material name="adapter_dark"><color rgba="0.30 0.31 0.33 1"/></material>\n')
        fp.write('\n'.join(snippet) + '\n')
    report['urdf_sha256'] = hashlib.sha256(open(out_urdf, 'rb').read()).hexdigest()[:16]
    json.dump(report, open(os.path.join(a.out, 'build_report.json'), 'w'), indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()
