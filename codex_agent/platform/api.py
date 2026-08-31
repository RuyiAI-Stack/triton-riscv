"""FastAPI surface for the Triton-RISCV conversational workbench."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from codex_agent.harness import HarnessAgent, HarnessSettings
from codex_agent.operator_lifecycle import (
    decide_repair_proposal,
    get_repair_proposal,
)
from codex_agent.platform.executor import HarnessRunExecutor
from codex_agent.platform.service import PlatformService
from codex_agent.platform.store import PlatformStore


class SessionRequest(BaseModel):
    title: str = Field(default="新对话", max_length=80)


class MessageRequest(BaseModel):
    content: str = Field(min_length=1, max_length=12000)


class RepairDecisionRequest(BaseModel):
    approve: bool
    reviewer: str = Field(min_length=1, max_length=120)
    note: str = Field(default="", max_length=2000)


def create_app(
    repo_root: Path | None = None,
    harness_agent: HarnessAgent | None = None,
) -> FastAPI:
    root = (repo_root or Path.cwd()).resolve()
    frontend_dist = Path(__file__).parents[1] / "frontend" / "dist"
    react_frontend_available = (frontend_dist / "index.html").is_file()
    store = PlatformStore(root / "agent-results" / "platform.sqlite3")
    settings = harness_agent.settings if harness_agent else HarnessSettings.from_env(root)
    agent = harness_agent or HarnessAgent(settings)
    service = PlatformService(root, store, agent)
    executor = HarnessRunExecutor(root, store, agent)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        yield
        executor.close()

    app = FastAPI(
        title="Triton-RISCV Agent",
        description="Conversational operator development and validation workbench.",
        version="0.2.0",
        lifespan=lifespan,
    )
    app.state.repo_root = root
    app.state.store = store
    app.state.service = service
    app.state.executor = executor
    if react_frontend_available:
        app.mount(
            "/ui-assets",
            StaticFiles(directory=frontend_dist),
            name="ui-assets",
        )

    @app.get("/")
    def index():
        if react_frontend_available:
            return FileResponse(frontend_dist / "index.html")
        return JSONResponse(
            status_code=503,
            content={
                "status": "frontend-not-built",
                "message": "Run npm install and npm run build in codex_agent/frontend.",
            },
        )

    @app.get("/api/health")
    def health() -> dict:
        return {
            "status": "ok",
            "repo_root": root.as_posix(),
            "harness": {
                "provider": settings.provider,
                "model": settings.model,
                "api_configured": bool(settings.api_key),
            },
        }

    @app.get("/api/bootstrap")
    def bootstrap() -> dict:
        return service.bootstrap()

    @app.get("/api/sessions")
    def sessions() -> list[dict]:
        return store.list_sessions()

    @app.post("/api/sessions")
    def create_session(request: SessionRequest) -> dict:
        return service.create_session(request.title)

    @app.get("/api/sessions/{session_id}")
    def session(session_id: str) -> dict:
        try:
            return store.session_bundle(session_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.post("/api/sessions/{session_id}/messages")
    def message(session_id: str, request: MessageRequest) -> dict:
        try:
            return service.handle_message(session_id, request.content)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.get("/api/runs/{run_id}")
    def run(run_id: str) -> dict:
        try:
            return store.get_run(run_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.post("/api/runs/{run_id}/confirm")
    def confirm(run_id: str) -> dict:
        try:
            return executor.confirm(run_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.get("/api/runs/{run_id}/events")
    def events(run_id: str, after: int = Query(default=-1, ge=-1)) -> list[dict]:
        try:
            store.get_run(run_id)
            return store.list_events(run_id, after)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.get("/api/runs/{run_id}/events/stream")
    async def event_stream(run_id: str, after: int = Query(default=-1, ge=-1)) -> StreamingResponse:
        try:
            store.get_run(run_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

        async def generate():
            cursor = after
            idle = 0
            while True:
                rows = store.list_events(run_id, cursor)
                if rows:
                    idle = 0
                    for row in rows:
                        cursor = row["sequence"]
                        yield f"id: {cursor}\ndata: {json.dumps(row, ensure_ascii=False)}\n\n"
                else:
                    idle += 1
                current = store.get_run(run_id)
                if current["status"] in {"completed", "failed", "cancelled"} and not rows:
                    break
                if idle % 15 == 0:
                    yield ": keep-alive\n\n"
                await asyncio.sleep(1)

        return StreamingResponse(generate(), media_type="text/event-stream")

    @app.get("/api/repair-proposals/{proposal_id}")
    def repair_proposal(proposal_id: str) -> dict:
        try:
            return get_repair_proposal(root, proposal_id)
        except (FileNotFoundError, ValueError) as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.post("/api/repair-proposals/{proposal_id}/decision")
    def repair_decision(proposal_id: str, request: RepairDecisionRequest) -> dict:
        try:
            return decide_repair_proposal(
                root,
                proposal_id,
                approve=request.approve,
                reviewer=request.reviewer,
                note=request.note,
            )
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    return app


app = create_app()
