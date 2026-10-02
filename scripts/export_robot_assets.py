"""Export the tool-equipped G1 MJCF (+ meshes + TCP handles) from a MotionCreator checkout as plain data.

Run with MotionCreator's Python environment:
    ~/motioncreator/.conda-policy/bin/python scripts/export_robot_assets.py ~/motioncreator

This is the only place that touches MotionCreator code; the project consumes the exported files
(assets/g1_tools/, git-ignored, ~17 MB) and records a sha256 manifest so drift is detectable.
"""
import hashlib
import json
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

source_root = Path(sys.argv[1]).expanduser().resolve()
sys.path.insert(0, str(source_root))
from motioncreator.tool_model import tool_model_xml          # noqa: E402

out = Path(__file__).resolve().parents[1] / "assets" / "g1_tools"
root, handles = tool_model_xml()
mesh_source = Path(root.find("compiler").get("meshdir"))
shutil.rmtree(out, ignore_errors=True)
shutil.copytree(mesh_source, out / "meshes")
root.find("compiler").set("meshdir", "meshes")
ET.indent(root)
(out / "g1_tools.xml").write_text(ET.tostring(root, encoding="unicode"), encoding="utf-8")
(out / "tcp_handles.json").write_text(json.dumps(
    {name: {"link": link, "position_m": list(position), "label": label}
     for name, (link, position, label) in handles.items()}, indent=1, ensure_ascii=False), encoding="utf-8")
manifest = {str(path.relative_to(out)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(out.rglob("*")) if path.is_file()}
(out / "manifest.json").write_text(json.dumps({"source": str(source_root), "files": manifest}, indent=1))
print(f"exported {len(manifest)} files to {out}")
