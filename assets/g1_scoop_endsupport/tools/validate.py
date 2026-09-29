#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""validate.py <패키지 폴더> [원본 g1_description 폴더]
 ① yourdfpy 파싱·메시 로딩·관절 수 ② MuJoCo 컴파일·질량 ③ TCP 위치 ④ 주걱 충돌체 오차(광선 검사) ⑤ 홈 깊이 실측"""
import sys, os, numpy as np, trimesh, yourdfpy, mujoco
PKG = sys.argv[1]; SRC = sys.argv[2] if len(sys.argv) > 2 else None
U = os.path.join(PKG, 'g1_29dof_rev_1_0_scoop_endsupport.urdf')
ok = True
def check(cond, msg):
    global ok; ok &= bool(cond); print(('PASS ' if cond else 'FAIL ') + msg)
r = yourdfpy.URDF.load(U, build_collision_scene_graph=True, load_collision_meshes=True)
check(len(r.actuated_joint_names) == 29, f'① 구동 관절 {len(r.actuated_joint_names)}개 (원본과 같은 29)')
new = [l for l in r.link_map if 'scoop' in l or 'end_support' in l]; check(len(new) == 4, f'① 새 링크 {new}')
check(not any('rubber_hand' in l for l in r.link_map), '① 더미손 링크 제거됨')
check(r.validate(), '① yourdfpy validate()')
m = mujoco.MjModel.from_xml_path(U); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
tot = mujoco.mj_getTotalmass(m)
if SRC:
    xml0 = open(os.path.join(SRC, 'g1_29dof_rev_1_0.urdf')).read().replace('discardvisual="false"', 'discardvisual="false" strippath="true"')
    cwd = os.getcwd(); os.chdir(SRC); m0 = mujoco.MjModel.from_xml_string(xml0); os.chdir(cwd); tot0 = mujoco.mj_getTotalmass(m0)
    print(f'② MuJoCo 총질량 원본 {tot0:.3f} kg → 새 모델 {tot:.3f} kg (차이 {tot - tot0:+.3f})')
check(np.isfinite(d.qacc).all(), f'② MuJoCo 컴파일·순방향 계산 OK (body {m.nbody}, geom {m.ngeom}, mesh {m.nmesh})')
for s in ['left', 'right']:
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f'{s}_wrist_yaw_link'); print(f'② {s}_wrist_yaw_link(그리퍼 병합) 질량 {m.body_mass[b]:.4f} kg')
r.update_cfg({n: 0.0 for n in r.actuated_joint_names})
for s, k in [('right', 'scoop'), ('left', 'end_support')]:
    Tw = r.get_transform(frame_to=f'{s}_wrist_yaw_link', frame_from='pelvis'); Tt = r.get_transform(frame_to=f'{s}_{k}_tcp', frame_from='pelvis')
    print(f'③ {s}_{k}_tcp: 손목 기준', np.round((np.linalg.inv(Tw) @ Tt)[:3, 3]*1000, 2), 'mm')
# ④ 주걱: 시각 메시 윗면·아랫면 vs 충돌 조각 합집합 (x > 45 mm, 어댑터 밖)
ln = r.link_map['right_scoop_link']
vis = trimesh.load(os.path.join(PKG, 'meshes', 'right_scoop_body.STL'))
hull = trimesh.util.concatenate([trimesh.load(os.path.join(PKG, c.geometry.mesh.filename)) for c in ln.collisions if c.geometry.mesh is not None])
xs, ys = np.meshgrid(np.linspace(0.047, 0.245, 90), np.linspace(-0.079, 0.079, 60)); P = np.c_[xs.ravel(), ys.ravel()]
def hits(mesh, P, down=True):
    o = np.c_[P, np.full(len(P), 1.0 if down else -1.0)]; dvec = np.tile([0, 0, -1.0 if down else 1.0], (len(P), 1))
    loc, idx, _ = mesh.ray.intersects_location(o, dvec, multiple_hits=False); z = np.full(len(P), np.nan); z[idx] = loc[:, 2]; return z
zt_v, zt_h = hits(vis, P), hits(hull, P); zb_v, zb_h = hits(vis, P, False), hits(hull, P, False)
mk = ~np.isnan(zt_v) & ~np.isnan(zt_h)
dt = (zt_h - zt_v)[mk]*1000; db = (zb_v - zb_h)[mk]*1000
check(np.nanmax(np.abs(dt)) < 1.5 and np.nanmax(np.abs(db)) < 1.5, f'④ 주걱 충돌체 오차: 윗면 최대 {np.nanmax(np.abs(dt)):.2f} mm(평균 {np.nanmean(np.abs(dt)):.2f}), 아랫면 최대 {np.nanmax(np.abs(db)):.2f} mm — {mk.sum()}점')
cov = (~np.isnan(zt_h) & ~np.isnan(zt_v)).sum()/(~np.isnan(zt_v)).sum(); check(cov > 0.995, f'④ 충돌체가 주걱 윗면을 덮는 비율 {cov*100:.1f}%')
# ⑤ 홈 깊이: 홈 중심선 vs 홈 사이 판 윗면
es = trimesh.load(os.path.join(PKG, 'meshes', 'left_end_support_body.STL'))
g = hits(es, np.array([[0.108, 0.0], [0.108, 0.022], [0.108, -0.022], [0.108, 0.011], [0.140, -0.011]]))
depth = (g[3:].mean() - g[:3].mean())*1000
check(abs(depth - 0.65) < 0.02, f'⑤ 홈 깊이 실측 {depth:.3f} mm (판 윗면 z={g[3]*1000:.2f}, 홈 바닥 z={g[0]*1000:.2f} mm)')
print('\n결과:', 'ALL PASS' if ok else 'FAIL 있음')
