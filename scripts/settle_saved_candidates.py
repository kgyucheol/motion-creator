#!/usr/bin/env python
"""Re-save extraction candidates (in place, same project id) with the scene settled under gravity."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extraction_lab as L  # noqa: E402
from motioncreator.motion import save_bundle  # noqa: E402
from motioncreator.robot import Robot  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    robot = Robot("g1-tools", ("scoop", "scoop"))
    for folder in sorted(ROOT.glob("motions/라면_꺼내기_*")):
        project = json.loads((folder / "project.json").read_text())
        if project.get("extraction_scene_settled"):
            continue
        extras = {name: (folder / name).read_bytes() for name in ("task_stages.json", "extraction_params.json") if (folder / name).is_file()}
        settled = L.settled_scene(robot, project, __import__("numpy").asarray(project["keyframes"][0]["qpos"]))
        settled["extraction_scene_settled"] = True
        saved = save_bundle(robot, settled, 30)
        target = ROOT / "motions" / saved["files"][0].split("/")[0]
        for name, content in extras.items():
            (target / name).write_bytes(content)
        print(("same folder" if target == folder else f"NEW FOLDER {target.name}"), folder.name[:40], flush=True)


if __name__ == "__main__":
    main()
