"""The two packages stay separable: the runtime must not pull in the authoring tool's heavy parts."""
import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# authoring modules the runtime may import; they must stay importable without the server or ML stacks
CORE = {"robot", "tool_model", "scene_geometry", "grasp", "ramen_sequence", "interaction", "motion", "reference",
        "kimodo_format", "grip_geometry", "hand_collision", "sim_scene"}
# the only authoring modules that may import the runtime: they assemble editor simulation + task controller
COMPOSITION = {"controller_session", "controller_api", "pose_adaptive_grasp", "tcp_trace"}
HEAVY = ["fastapi", "uvicorn", "starlette", "pydantic", "torch", "kimodo", "onnxruntime", "protomotions"]


def imports_of(path: Path):
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module


def test_runtime_only_imports_core_authoring_modules():
    offenders = []
    for path in (ROOT / "ramen_controller").rglob("*.py"):
        for name in imports_of(path):
            parts = name.split(".")
            if parts[0] == "motioncreator" and (len(parts) < 2 or parts[1] not in CORE):
                offenders.append(f"{path.relative_to(ROOT)} imports {name}")
    assert not offenders, offenders


def test_only_the_composition_layer_imports_the_runtime():
    offenders = [str(path.relative_to(ROOT)) for path in (ROOT / "motioncreator").glob("*.py")
                 if path.stem not in COMPOSITION and any(name.split(".")[0] == "ramen_controller" for name in imports_of(path))]
    assert not offenders, offenders


def _import_with_heavy_blocked(module_names):
    code = ("import sys\n"
            f"for name in {HEAVY!r}:\n    sys.modules[name] = None\n"
            f"sys.path.insert(0, {str(ROOT)!r})\n"
            f"import importlib\nfor module in {list(module_names)!r}:\n    importlib.import_module(module)\n"
            "print('ok')")
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


def test_runtime_core_imports_without_the_heavy_authoring_stack():
    result = _import_with_heavy_blocked([
        "ramen_controller", "ramen_controller.contracts", "ramen_controller.fsm", "ramen_controller.runner",
        "ramen_controller.vision", "ramen_controller.pose_average", "ramen_controller.motion_bundle",
        "ramen_controller.motion_library", "ramen_controller.operator", "ramen_controller.auto_positioner",
        "ramen_controller.debug_gui"])
    assert result.returncode == 0, result.stderr[-600:]


def test_authoring_core_modules_import_without_server_or_ml_stacks():
    import pytest
    pytest.importorskip("mujoco")
    result = _import_with_heavy_blocked([f"motioncreator.{name}" for name in sorted(CORE)])
    assert result.returncode == 0, result.stderr[-600:]
