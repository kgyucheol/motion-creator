"""Validated local scene-asset import and one-time Blender-to-GLB conversion."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

import numpy as np
import trimesh

from .robot import ROOT

ASSET_ROOT = ROOT / "assets/imported"
CONVERTER = ROOT / "scripts/blender/export_motioncreator_asset.py"
MAX_ASSET_BYTES = 200 * 1024 * 1024
SUPPORTED_SUFFIXES = {".blend", ".glb"}
PIPELINE_VERSION = 7
IDENTITY_XYZW = [0., 0., 0., 1.]
Y_UP_TO_Z_UP_XYZW = [2 ** -.5, 0., 0., 2 ** -.5]
Y_UP_TO_Z_UP = np.array([
    [1., 0., 0., 0.],
    [0., 0., -1., 0.],
    [0., 1., 0., 0.],
    [0., 0., 0., 1.],
])


def _bounds(path: Path, transform: np.ndarray | None = None) -> tuple[list[float], list[float]]:
    loaded = trimesh.load(path, force="scene")
    if transform is not None:
        loaded.apply_transform(transform)
    bounds = np.asarray(loaded.bounds, dtype=float)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
        raise ValueError("3D model bounds could not be determined")
    dimensions = bounds[1] - bounds[0]
    if np.any(dimensions <= 1e-6) or np.any(dimensions > 20):
        raise ValueError("3D model dimensions must be between 1 µm and 20 m")
    return bounds[0].tolist(), bounds[1].tolist()


def _mesh_bounds(mesh: trimesh.Trimesh, transform: np.ndarray | None = None) -> tuple[list[float], list[float]]:
    aligned = mesh.copy()
    if transform is not None:
        aligned.apply_transform(transform)
    bounds = np.asarray(aligned.bounds, dtype=float)
    return bounds[0].tolist(), bounds[1].tolist()


def _part_id(index: int, name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.casefold()).strip("-")[:32] or "mesh"
    return f"part-{index + 1:02d}-{slug}"


def _inscribed_cylinder(dimensions: list[float]) -> dict:
    """Build a Z-axis MuJoCo cylinder rotated onto the longest asset axis."""
    axis = int(np.argmax(dimensions))
    cross = [dimensions[index] for index in range(3) if index != axis]
    diameter = float(min(cross))
    rotations = {
        0: [0., 2 ** -.5, 0., 2 ** -.5],       # local Z -> X
        1: [-(2 ** -.5), 0., 0., 2 ** -.5],    # local Z -> Y
        2: IDENTITY_XYZW,
    }
    return {
        "collision_shape": "cylinder",
        "collision_size": [diameter, diameter, float(dimensions[axis])],
        "collision_quaternion_xyzw": rotations[axis],
    }


def _convex_hull(mesh: trimesh.Trimesh, transform: np.ndarray | None = None) -> dict:
    aligned = mesh.copy()
    if transform is not None:
        aligned.apply_transform(transform)
    hull = aligned.convex_hull
    if len(hull.vertices) > 256:
        points = np.asarray(hull.vertices)
        selected = []
        for axis in range(3):
            for index in (int(np.argmin(points[:, axis])), int(np.argmax(points[:, axis]))):
                if index not in selected:
                    selected.append(index)
        minimum_distance = np.full(len(points), np.inf)
        while len(selected) < 256:
            latest = points[selected[-1]]
            minimum_distance = np.minimum(minimum_distance, np.sum((points - latest) ** 2, axis=1))
            minimum_distance[selected] = -1
            selected.append(int(np.argmax(minimum_distance)))
        hull = trimesh.convex.convex_hull(points[selected])
    bounds = np.asarray(aligned.bounds, dtype=float)
    dimensions = bounds[1] - bounds[0]
    center = (bounds[0] + bounds[1]) / 2
    vertices = (np.asarray(hull.vertices, dtype=float) - center) / dimensions
    return {
        "collision_shape": "convex_hull",
        "collision_hull_vertices": vertices.tolist(),
        "collision_hull_faces": np.asarray(hull.faces, dtype=int).tolist(),
    }


def _import_suggestion(filename: str, dimensions: list[float], properties: dict,
                       mesh: trimesh.Trimesh | None = None,
                       transform: np.ndarray | None = None) -> dict:
    suggestion = _suggestion(filename, dimensions, properties)
    if suggestion["shape"] == "open_box":
        return suggestion
    if suggestion["shape"] == "cylinder":
        collision = _inscribed_cylinder(dimensions)
        if int(np.argmax(dimensions)) == 2:
            return {**suggestion, **collision}
        # Keep the authored visual extents, but use a separately oriented
        # inscribed cylinder for a bundle already lying along X or Y.
        return {"shape": "box", "fixed": False, "size": dimensions,
                "mass_kg": suggestion["mass_kg"], "friction": suggestion["friction"],
                "color": suggestion["color"], **collision}
    if mesh is not None:
        try:
            return {**suggestion, **_convex_hull(mesh, transform)}
        except (ValueError, TypeError, RuntimeError):
            pass
    return suggestion


def _scene_mesh(path: Path) -> trimesh.Trimesh:
    scene = trimesh.load(path, force="scene")
    meshes = []
    for node_name in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node_name]
        mesh = scene.geometry[geometry_name].copy()
        mesh.apply_transform(transform)
        meshes.append(mesh)
    if not meshes:
        raise ValueError("3D model contains no mesh geometry")
    return trimesh.util.concatenate(meshes)


def _asset_parts(folder: Path, source_format: str, properties: dict) -> list[dict]:
    """Describe authored GLB nodes without re-exporting or splitting their meshes.

    Keeping every part in the original GLB preserves embedded textures and materials.
    A carton authored as one node therefore remains one semantic scene object even
    when its mesh contains disconnected floor and wall solids.
    """
    scene = trimesh.load(folder / "model.glb", force="scene")
    axis = Y_UP_TO_Z_UP if source_format == "glb" else None
    candidates: list[tuple[str, trimesh.Trimesh]] = []
    for node_name in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node_name]
        mesh = scene.geometry[geometry_name].copy()
        mesh.apply_transform(transform)
        candidates.append((str(node_name), mesh))
    if len(candidates) <= 1:
        return []
    parts = []
    for index, (name, mesh) in enumerate(candidates):
        identifier = _part_id(index, name)
        lower, upper = _mesh_bounds(mesh, axis)
        dimensions = [upper[component] - lower[component] for component in range(3)]
        suggestion = _import_suggestion(name, dimensions, properties, mesh, axis)
        parts.append({
            "part_id": identifier, "node_name": name, "name": name,
            "url": f"/api/scene-assets/{folder.name}.glb",
            "bounds_min": lower, "bounds_max": upper, "dimensions": dimensions,
            "suggestion": suggestion,
        })
    return parts


def _write_metadata(folder: Path, identifier: str, name: str, source_format: str, properties: dict) -> dict:
    glb = folder / "model.glb"
    direct_glb = source_format == "glb"
    axis_transform = Y_UP_TO_Z_UP_XYZW if direct_glb else IDENTITY_XYZW
    lower, upper = _bounds(glb, Y_UP_TO_Z_UP if direct_glb else None)
    dimensions = [upper[index] - lower[index] for index in range(3)]
    combined_mesh = _scene_mesh(glb)
    metadata = {
        "pipeline_version": PIPELINE_VERSION,
        "asset_id": identifier,
        "name": Path(name).stem,
        "source_name": name,
        "source_format": source_format,
        "source_coordinate_system": "right-handed, +Y up (glTF)" if direct_glb else "right-handed, +Z up (Blender)",
        "project_coordinate_system": "right-handed, +X forward, +Y left, +Z up",
        "axis_transform_xyzw": axis_transform,
        "url": f"/api/scene-assets/{identifier}.glb",
        "bounds_min": lower,
        "bounds_max": upper,
        "dimensions": dimensions,
        "suggestion": _import_suggestion(name, dimensions, properties, combined_mesh,
                                           Y_UP_TO_Z_UP if direct_glb else None),
    }
    # v4 wrote one lossy GLB per part. Parts now reference nodes in the original
    # texture-preserving GLB, so stale generated files must not be used.
    shutil.rmtree(folder / "parts", ignore_errors=True)
    parts = _asset_parts(folder, source_format, properties)
    if parts:
        metadata["parts"] = parts
    (folder / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


def _suggestion(filename: str, dimensions: list[float], properties: dict) -> dict:
    lowered = filename.casefold()
    role = str(properties.get("asset_role", ""))
    if "cardboard_box" in lowered or role == "motion_validation_container":
        thickness = float(properties.get("wall_thickness_m", .02))
        return {"shape": "open_box", "fixed": True, "wall_thickness_m": thickness,
                "size": dimensions, "mass_kg": 2., "friction": .7, "color": "#9b6b3c"}
    if "ramen" in lowered or role == "motion_validation_prop":
        # The prepared scan's long axis is Z, matching the cylinder proxy axis.
        diameter = max(dimensions[0], dimensions[1])
        return {"shape": "cylinder", "fixed": False, "size": [diameter, diameter, dimensions[2]],
                "mass_kg": 1., "friction": .7, "color": "#e8d8bd"}
    return {"shape": "box", "fixed": False, "size": dimensions,
            "mass_kg": 1., "friction": .7, "color": "#c7c9cc"}


def import_scene_asset(content: bytes, filename: str) -> dict:
    name = Path(filename).name
    suffix = Path(name).suffix.lower()
    if name != filename or suffix not in SUPPORTED_SUFFIXES:
        raise ValueError("Only a single .blend or .glb file is supported")
    if not content or len(content) > MAX_ASSET_BYTES:
        raise ValueError("3D model must be between 1 byte and 200 MB")
    # Blender 5 may zstd-compress the entire .blend container.
    if suffix == ".blend" and not (content.startswith(b"BLENDER") or content.startswith(b"\x28\xb5\x2f\xfd")):
        raise ValueError("The uploaded file is not a valid Blender file")
    if suffix == ".glb" and content[:4] != b"glTF":
        raise ValueError("The uploaded file is not a valid binary glTF file")
    identifier = hashlib.sha256(content).hexdigest()[:24]
    folder = ASSET_ROOT / identifier
    folder.mkdir(parents=True, exist_ok=True)
    source = folder / f"source{suffix}"
    glb = folder / "model.glb"
    report_path = folder / "conversion.json"
    metadata_path = folder / "metadata.json"
    if metadata_path.is_file() and glb.is_file():
        return asset_metadata(identifier)
    source.write_bytes(content)
    properties = {}
    if suffix == ".blend":
        blender = shutil.which("blender")
        if not blender:
            raise ValueError("Blender executable is required to import .blend files")
        glb.unlink(missing_ok=True)
        report_path.unlink(missing_ok=True)
        command = [blender, "--background", "--factory-startup", str(source), "--python", str(CONVERTER), "--",
                   "--output", str(glb), "--report", str(report_path)]
        try:
            process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            deadline = time.monotonic() + 180
            while process.poll() is None and not (glb.is_file() and report_path.is_file()):
                if time.monotonic() >= deadline:
                    process.terminate()
                    raise ValueError("Blender conversion exceeded 180 seconds")
                time.sleep(.05)
            if process.poll() is None:
                # Some Linux Blender builds finish export but hang while PulseAudio
                # tears down. The report is written last, so it is safe to stop here.
                process.terminate()
            stdout, stderr = process.communicate(timeout=5)
            if not glb.is_file() or not report_path.is_file():
                message = (stderr or stdout or "Blender conversion failed").strip().splitlines()[-1]
                raise ValueError(f"Blender conversion failed: {message}")
        except subprocess.TimeoutExpired as exc:
            process.kill()
            process.communicate()
            raise ValueError("Blender conversion did not stop cleanly") from exc
        properties = json.loads(report_path.read_text(encoding="utf-8")).get("properties", {})
    else:
        glb.write_bytes(content)
    return _write_metadata(folder, identifier, name, suffix[1:], properties)


def _validate_identifier(identifier: str) -> None:
    if len(identifier) != 24 or any(character not in "0123456789abcdef" for character in identifier):
        raise ValueError("Invalid scene asset identifier")


def asset_metadata(identifier: str) -> dict:
    """Return current axis-aware metadata, upgrading assets imported before v2."""
    _validate_identifier(identifier)
    folder = ASSET_ROOT / identifier
    path = folder / "metadata.json"
    glb = folder / "model.glb"
    if not path.is_file() or not glb.is_file():
        raise FileNotFoundError(identifier)
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("pipeline_version") == PIPELINE_VERSION:
        return metadata
    source_name = str(metadata.get("source_name", "model.glb"))
    source_format = str(metadata.get("source_format", Path(source_name).suffix.lstrip(".") or "glb"))
    properties = {}
    report = folder / "conversion.json"
    if source_format == "blend" and report.is_file():
        properties = json.loads(report.read_text(encoding="utf-8")).get("properties", {})
    return _write_metadata(folder, identifier, source_name, source_format, properties)


def asset_path(identifier: str) -> Path:
    _validate_identifier(identifier)
    path = ASSET_ROOT / identifier / "model.glb"
    if not path.is_file():
        raise FileNotFoundError(identifier)
    return path
