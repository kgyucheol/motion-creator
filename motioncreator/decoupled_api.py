"""HTTP and WebSocket API for the decoupled-WBC workbench."""
import asyncio

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .decoupled_wbc import ROOT, save_recording_bundle, sessions, verify_assets


router = APIRouter()


class SessionInput(BaseModel):
    project: dict


def checked(call):
    try:
        return call()
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/decoupled-wbc", include_in_schema=False)
def page():
    return FileResponse(ROOT / "frontend/dist/index.html")


@router.get("/api/decoupled-wbc/runtime")
def runtime():
    return verify_assets()


@router.post("/api/decoupled-wbc/sessions")
def create_session(payload: SessionInput):
    identifier, session = checked(lambda: sessions.create(payload.project))
    return {"id": identifier, **session.snapshot()}


@router.delete("/api/decoupled-wbc/sessions/{identifier}")
def delete_session(identifier: str):
    sessions.delete(identifier)
    return {"deleted": True}


@router.post("/api/decoupled-wbc/sessions/{identifier}/save")
def save_session(identifier: str):
    session = checked(lambda: sessions.get(identifier))
    return checked(lambda: save_recording_bundle(session.recording_copy(), session.project.get("name", "motion")))


@router.websocket("/api/decoupled-wbc/sessions/{identifier}/stream")
async def stream(websocket: WebSocket, identifier: str):
    try:
        session = sessions.get(identifier)
    except ValueError:
        await websocket.close(code=4404)
        return
    await websocket.accept()
    last_revision = -1
    try:
        while True:
            try:
                message = await asyncio.wait_for(websocket.receive_json(), timeout=.04)
                action = message.get("action")
                if action == "play": session.play()
                elif action == "stop": session.stop()
                elif action == "reset": session.reset()
                elif action == "key": session.apply_key(str(message.get("key", "")))
            except asyncio.TimeoutError:
                pass
            snapshot = session.snapshot()
            if snapshot["revision"] != last_revision:
                await websocket.send_json(snapshot)
                last_revision = snapshot["revision"]
    except WebSocketDisconnect:
        sessions.delete(identifier)
