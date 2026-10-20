#!/usr/bin/env python
"""Save the best extraction candidates from one or more search files as editor projects, with videos and an index.

    python scripts/save_candidates.py --top 6 build/search_093_c.jsonl:0.93 build/search_rob_a.jsonl:0.93
Each candidate becomes a bundle under motions/ (open it in the editor and run the object-physics preview, or in the Task
Controller page) plus build/candidates/<name>.mp4 and a row in build/candidates/index.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extraction_lab as L  # noqa: E402
import render_extraction as R  # noqa: E402
import save_extraction_project as S  # noqa: E402
from motioncreator.motion import save_bundle  # noqa: E402
from motioncreator.robot import Robot  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MIN_TOWARD = 0.0


def load(spec: str):
    """`path:scale:top` -> the best `top` rows of that search file, judged on the report at `scale`."""
    parts = spec.split(":")
    path, scale = parts[0], float(parts[1]) if len(parts) > 1 else 0.93
    top = int(parts[2]) if len(parts) > 2 else 99
    rows = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        report = (row.get("reports") or {}).get(str(scale)) or row.get("report")
        if (report and not row.get("error") and report["rise_m"] >= 0.05 and report["toward_robot_m"] > MIN_TOWARD
                and L.keyframe_count(row["params"]) <= L.PREFERRED_KEYFRAMES):
            rows.append(({**row, "report": report}, scale, Path(path).stem))
    rows.sort(key=lambda item: -(item[0]["report"]["rise_m"] - 2 * sum(item[0]["report"]["others_moved_m"].values())
                                 + (0.2 if min(item[0].get("scores") or [-1]) > 0.3 else 0)))
    return rows[:top]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    parser.add_argument("--top", type=int, default=6)
    parser.add_argument("--min-rise", type=float, default=0.05)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--tag", default="후보", help="name prefix, so a new batch never overwrites an earlier one")
    parser.add_argument("--min-toward", type=float, default=0.10, help="the target must be carried at least this far toward the robot")
    args = parser.parse_args()
    global MIN_TOWARD
    MIN_TOWARD = args.min_toward
    pool = [item for spec in args.files for item in load(spec)]
    chosen, seen = [], set()
    for row, scale, source in pool:                           # skip near-duplicates of the same maneuver
        key = tuple(round(row["params"][k], 2) for k in ("dx0", "dz0", "depth_m", "th1_deg"))
        if key in seen:
            continue
        seen.add(key)
        chosen.append((row, scale, source))
        if len(chosen) == args.top:
            break
    out = ROOT / "build/candidates"
    out.mkdir(parents=True, exist_ok=True)
    robot = Robot("g1-tools", ("scoop", "scoop"))
    lines = [f"# 라면 꺼내기 {args.tag}", "", "| # | 폴더(motions/) | 영상 | 스택 크기 | 타깃 상승 | 몸쪽 이동 | 끝 접촉력 L/R(N) | 이웃 이동 합(m) | 출처 |", "|---|---|---|---|---|---|---|---|---|"]
    for rank, (row, scale, source) in enumerate(chosen, 1):
        rep = row["report"]
        name = f"라면 꺼내기 {args.tag} #{rank} (상승 {rep['rise_m'] * 100:.0f}cm, 스택 {scale * 100:.0f}%)"
        project = S.build_project(robot, row["params"], scale, name, -0.004, 0.25)
        saved = save_bundle(robot, project, 30)
        folder = ROOT / "motions" / saved["files"][0].split("/")[0]
        S.write_stages(robot, project, folder, scale=scale)
        (folder / "extraction_params.json").write_text(json.dumps({"params": row["params"], "stack_scale": scale, "report": rep,
                                                                   "scores": row.get("scores"), "source": source}, indent=1, ensure_ascii=False))
        video = out / (f"candidate_{rank:02d}.mp4" if args.tag == "후보" else f"{args.tag}_{rank:02d}.mp4")
        if not args.no_video:
            R.render_video(row["params"], scale, str(video))
        others = sum(rep["others_moved_m"].values())
        lines.append(f"| {rank} | `{folder.name}` | `{video.name}` | {scale:.2f} | {rep['rise_m'] * 100:.0f} cm | {rep['toward_robot_m'] * 100:.0f} cm | "
                     f"{rep['end_target_n']['left']:.0f}/{rep['end_target_n']['right']:.0f} | {others:.3f} | {source} |")
        print(lines[-1], flush=True)
    index = out / ("index.md" if args.tag == "후보" else f"index_{args.tag}.md")
    index.write_text("\n".join(lines) + "\n")
    print("index:", index)


if __name__ == "__main__":
    main()
