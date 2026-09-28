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
PIPELINE_VERSION = 4
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


def _asset_parts(folder: Path, filename: str, source_format: str, properties: dict) -> list[dict]:
    """Export each GLB node, and known joined carton panels, as editable objects."""
    scene = trimesh.load(folder / "model.glb", force="scene")
    axis = Y_UP_TO_Z_UP if source_format == "glb" else None
    candidates: list[tuple[str, trimesh.Trimesh, bool]] = []
    for node_name in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node_name]
        mesh = scene.geometry[geometry_name].copy()
        mesh.apply_transform(transform)
        lowered = f"{filename} {node_name}".casefold()
        pieces = [mesh]
        split_carton = "cardboard_box" in lowered
        if split_carton:
            merged = mesh.copy()
            merged.merge_vertices()
            components = trimesh.graph.connected_components(
                merged.face_adjacency, nodes=np.arange(len(merged.faces)), min_len=1, engine="scipy")
            if 2 <= len(components) <= 16 and all(len(component) >= 4 for component in components):
                pieces = [merged.submesh([component], repair=False)[0] for component in components]
        for piece_index, piece in enumerate(pieces):
            label = str(node_name) if len(pieces) == 1 else f"{node_name} {piece_index + 1}"
            candidates.append((label, piece, split_carton))
    if len(candidates) <= 1:
        return []
    parts = []
    parts_folder = folder / "parts"
    parts_folder.mkdir(parents=True, exist_ok=True)
    for index, (name, mesh, carton_panel) in enumerate(candidates):
        identifier = _part_id(index, name)
        path = parts_folder / f"{identifier}.glb"
        path.write_bytes(trimesh.exchange.gltf.export_glb(trimesh.Scene(mesh)))
        lower, upper = _mesh_bounds(mesh, axis)
        dimensions = [upper[component] - lower[component] for component in range(3)]
        suggestion = _suggestion(name, dimensions, properties)
        if suggestion['shape'] == 'cylinder' and not np.isclose(dimensions[0], dimensions[1], rtol=.15):
            # A packed bundle may already be lying along X or Y. Keep its exact
            # authored extents; a Z-axis cylinder proxy would stretch the mesh.
            suggestion = {"shape": "box", "fixed": False, "size": dimensions,
                          "mass_kg": suggestion['mass_kg'], "friction": suggestion['friction'],
                          "color": suggestion['color']}
        if carton_panel:
            suggestion = {"shape": "box", "fixed": True, "size": dimensions,
                          "mass_kg": .4, "friction": .7, "color": "#9b6b3c"}
        parts.append({
            "part_id": identifier, "name": name,
            "url": f"/api/scene-assets/{folder.name}/parts/{identifier}.glb",
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
        "suggestion": _suggestion(name, dimensions, properties),
    }
    parts = _asset_parts(folder, name, source_format, properties)
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


def asset_part_path(identifier: str, part_id: str) -> Path:
    _validate_identifier(identifier)
    if not re.fullmatch(r"part-[0-9]{2}-[a-z0-9-]{1,32}", part_id):
        raise ValueError("Invalid scene asset part identifier")
    path = ASSET_ROOT / identifier / "parts" / f"{part_id}.glb"
    if not path.is_file():
        raise FileNotFoundError(part_id)
    return path
