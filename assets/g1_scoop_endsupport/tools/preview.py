import os, sys, glob, numpy as np
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import trimesh, yourdfpy, pyrender
from PIL import Image, ImageDraw, ImageFont
PKG = sys.argv[1]; U = os.path.join(PKG, 'g1_29dof_rev_1_0_scoop_endsupport.urdf')
r = yourdfpy.URDF.load(U, build_scene_graph=True, load_meshes=False)
cfg = {n: 0.0 for n in r.actuated_joint_names}
for s in ['left', 'right']: cfg[f'{s}_hip_pitch_joint'], cfg[f'{s}_knee_joint'], cfg[f'{s}_ankle_pitch_joint'] = -0.1, 0.3, -0.2
J = ['shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'wrist_roll', 'wrist_pitch', 'wrist_yaw']
for s, q in [('right', [-6, -28, -1, 42, 6, 28, 41]), ('left', [-10, 15, 4, 62, -2, 12, -23])]:
    cfg.update({f'{s}_{j}_joint': np.radians(v) for j, v in zip(J, q)})
r.update_cfg(cfg)
COL = {'gripper_white': (0.95, 0.95, 0.93), 'adapter_dark': (0.30, 0.31, 0.33)}
def mat(rgb, rough=0.6): return pyrender.MetallicRoughnessMaterial(baseColorFactor=[*rgb, 1], metallicFactor=0.0, roughnessFactor=rough, doubleSided=True)
def geom_mesh(g):
    if g.mesh is not None: return trimesh.load(os.path.join(PKG, g.mesh.filename), force='mesh')
    if g.box is not None: return trimesh.creation.box(g.box.size)
    if g.cylinder is not None: return trimesh.creation.cylinder(radius=g.cylinder.radius, height=g.cylinder.length, sections=48)
def scene_links(links=None, rel=None, collision=False):
    Tref = np.linalg.inv(r.get_transform(frame_to=rel, frame_from='pelvis')) if rel else np.eye(4)
    out = []
    for name, ln in r.link_map.items():
        if (links and name not in links) or name == 'logo_link': continue   # 미리보기에서만 로고 제외 (URDF에는 그대로)
        Tl = Tref @ r.get_transform(frame_to=name, frame_from='pelvis')
        for k, el in enumerate(ln.collisions if collision else ln.visuals):
            m = geom_mesh(el.geometry)
            if m is None: continue
            m.apply_transform(Tl @ (el.origin if el.origin is not None else np.eye(4)))
            if collision: rgb = [(0.93, 0.55, 0.35), (0.35, 0.62, 0.90), (0.55, 0.78, 0.45)][k % 3]
            else: rgb = COL.get(el.material.name if el.material is not None else '', (0.70, 0.72, 0.75))
            out.append((m, rgb))
    return out
def look_at(eye, target, up=(0, 0, 1)):
    z = np.asarray(eye, float) - target; z /= np.linalg.norm(z); x = np.cross(up, z); x /= np.linalg.norm(x)
    Tm = np.eye(4); Tm[:3, 0], Tm[:3, 1], Tm[:3, 2], Tm[:3, 3] = x, np.cross(z, x), z, eye; return Tm
def render(items, eye, target, W, H, yfov=30, thr=0.02, key=None, amb=0.32, up=(0, 0, 1), ss=2):
    sc = pyrender.Scene(ambient_light=[amb]*3)
    for m, rgb in items: sc.add(pyrender.Mesh.from_trimesh(m, material=mat(rgb), smooth=False))
    pose = look_at(np.array(eye), np.array(target), up)
    sc.add(pyrender.PerspectiveCamera(yfov=np.radians(yfov), aspectRatio=W/H, znear=0.01, zfar=20), pose=pose)
    sc.add(pyrender.DirectionalLight(intensity=1.8), pose=pose)
    sc.add(pyrender.DirectionalLight(intensity=1.6), pose=key if key is not None else look_at(np.array(target) + [0.3, 0.2, 1.0], np.array(target)))
    rr = pyrender.OffscreenRenderer(W*ss, H*ss); col, dep = rr.render(sc); rr.delete()
    col = col.astype(np.float32); bg = dep <= 0
    t = np.linspace(0, 1, H*ss)[:, None]; G = ((0.84*(1 - t) + 0.93*t)*255)[..., None]*np.ones((1, W*ss, 3)); col[bg] = G[bg]
    d = np.where(bg, 99.0, dep); e = np.zeros_like(bg)
    for ax in (0, 1):
        g = np.abs(np.diff(d, axis=ax)) > thr
        if ax == 0: e[1:] |= g; e[:-1] |= g
        else: e[:, 1:] |= g; e[:, :-1] |= g
    col[e] = col[e]*0.4 + 45*0.6
    return Image.fromarray(col.clip(0, 255).astype(np.uint8)).resize((W, H), Image.LANCZOS)
zs = -min(m.bounds[0, 2] for m, _ in scene_links())
lift = lambda items: [(m.copy().apply_translation([0, 0, zs]), c) for m, c in items]
A = render(lift(scene_links()), [2.5, -0.5, 1.15], [0.06, -0.02, 0.82], 760, 1000, yfov=31)
wr = lambda s: [f'{s}_elbow_link', f'{s}_wrist_roll_link', f'{s}_wrist_pitch_link', f'{s}_wrist_yaw_link']
es = scene_links(wr('left') + ['left_end_support_link'], rel='left_end_support_link')
B = render(es, [0.30, -0.16, 0.20], [0.105, 0.0, -0.03], 760, 480, yfov=36, thr=0.0003, key=look_at(np.array([0.10, -0.6, 0.05]), np.array([0.10, 0, -0.03])))
sp = scene_links(wr('right') + ['right_scoop_link'], rel='right_scoop_link')
C = render(sp, [0.34, 0.26, 0.20], [0.13, 0.0, -0.02], 760, 480, yfov=38, thr=0.001)
cl = scene_links(['right_scoop_link'], rel='right_scoop_link', collision=True)
D = render(cl, [0.34, 0.26, 0.22], [0.13, 0.0, -0.02], 760, 480, yfov=38, thr=0.0006)
cands = sorted(glob.glob('/usr/share/fonts/opentype/noto/NotoSansCJK-*.ttc'))
def font(w, size):
    p = [c for c in cands if w in c][0]
    for i in range(6):
        f = ImageFont.truetype(p, size, index=i)
        if 'KR' in f.getname()[0]: return f
FM, FR = font('Medium', 26), font('Regular', 21)
S = Image.new('RGB', (16 + 760 + 16 + 760 + 16, 16 + 1000 + 16 + 60), (238, 239, 241)); dr = ImageDraw.Draw(S)
S.paste(A, (16, 16)); x2 = 16 + 760 + 16
for k, (im, cap) in enumerate([(B, 'ㄴ 받침 left_end_support_link (로봇 왼손) — 윗면 홈 3줄 깊이 0.65 mm'), (C, 'V 주걱 right_scoop_link (로봇 오른손) — 시각 메시'), (D, '주걱 충돌체 — 볼록 조각 78개 (색은 조각 구분용)')]):
    y = 16 + k*(320 + 20)
    im = im.resize((760, 320*760//760)) if False else im.crop((0, 80, 760, 400))
    S.paste(im, (x2, y)); w = dr.textlength(cap, font=FR); dr.rounded_rectangle([x2 + 8, y + 8, x2 + 22 + w, y + 40], 6, fill=(250, 250, 251)); dr.text((x2 + 15, y + 11), cap, font=FR, fill=(40, 40, 42))
dr.text((24, 26), 'URDF만 불러서 그린 모습', font=FM, fill=(40, 40, 42))
dr.text((16, 16 + 1000 + 22), 'g1_29dof_rev_1_0_scoop_endsupport.urdf 를 yourdfpy로 로드 → 전시 자세 → pyrender 렌더 (공식 메시 + 그리퍼 메시)', font=FR, fill=(90, 90, 94))
os.makedirs(os.path.join(PKG, 'preview'), exist_ok=True); S.save(os.path.join(PKG, 'preview', 'preview.png'), optimize=True); print('saved', S.size)
