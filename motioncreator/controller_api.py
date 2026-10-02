"""HTTP API for the task-controller page (/controller): FSM + vision mock driving the editor's decoupled-WBC sim."""
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .controller_session import DEFAULT_STAGES, controller_sessions, load_stages
from .decoupled_wbc import ROOT, verify_assets

router = APIRouter()


class SessionInput(BaseModel):
    project: dict
    bundle: str                       # saved project path as listed by /api/saved, e.g. "<folder>/project.json"
    stages: dict | None = None
    options: dict | None = None


class CommandInput(BaseModel):
    action: Literal["start", "stop", "reset", "operator-reset", "step-state", "options"]
    options: dict | None = None


def checked(call):
    try:
        return call()
    except (ValueError, KeyError, TypeError, FileNotFoundError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def bundle_folder(bundle: str) -> Path:
    relative = Path(bundle)
    if relative.is_absolute() or ".." in relative.parts or len(relative.parts) < 2:
        raise ValueError("bundle must be '<saved folder>/project.json'")
    folder = ROOT / "motions" / relative.parts[0]
    missing = [name for name in ("metadata.json", "motion.csv") if not (folder / name).is_file()]
    if missing:
        raise ValueError(f"{relative.parts[0]} is not a saved bundle (missing {', '.join(missing)})")
    return folder


@router.get("/controller", include_in_schema=False)
def page():
    return FileResponse(ROOT / "frontend/dist/index.html")


@router.get("/api/controller/runtime")
def runtime():
    return {"assets": verify_assets(), "default_stages": load_stages(None), "default_stages_file": str(DEFAULT_STAGES.relative_to(ROOT))}


@router.get("/api/controller/stages")
def stages(bundle: str):
    folder = checked(lambda: bundle_folder(bundle))
    own = folder / "task_stages.json"
    return {"stages": checked(lambda: load_stages(folder)), "source": "bundle" if own.is_file() else "default"}


@router.post("/api/controller/sessions")
def create_session(payload: SessionInput):
    def create():
        folder = bundle_folder(payload.bundle)
        identifier, session = controller_sessions.create(payload.project, folder, payload.stages or load_stages(folder),
                                                         payload.options)
        return {"id": identifier, **session.snapshot()}
    return checked(create)


@router.delete("/api/controller/sessions/{identifier}")
def delete_session(identifier: str):
    controller_sessions.delete(identifier)
    return {"deleted": True}


@router.get("/api/controller/sessions/{identifier}")
def session_status(identifier: str):
    return checked(lambda: controller_sessions.get(identifier).snapshot())


@router.post("/api/controller/sessions/{identifier}/command")
def session_command(identifier: str, payload: CommandInput):
    session = checked(lambda: controller_sessions.get(identifier))
    if payload.action == "start": session.start()
    elif payload.action == "stop": session.stop()
    elif payload.action == "reset": checked(session.reset)
    elif payload.action == "operator-reset": session.operator_reset()
    elif payload.action == "step-state": session.step_state()
    else: session.set_options(payload.options or {})
    return session.snapshot()
