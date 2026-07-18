"""Minimal two-mode local UI with an MJPEG tracking view."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse

from preact.runtime import PREACTRuntime

BOUNDARY = "preact-frame"


def create_app(runtime: PREACTRuntime) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        runtime.start_worker()
        yield
        runtime.close()

    app = FastAPI(title="PREACT", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return _index_html()

    @app.post("/api/play")
    async def play() -> dict[str, object]:
        runtime.play()
        return runtime.status().as_dict()

    @app.post("/api/mode/tracking")
    async def tracking() -> dict[str, object]:
        runtime.start_tracking()
        return runtime.status().as_dict()

    @app.post("/api/mode/landing")
    async def landing() -> dict[str, object]:
        runtime.start_landing()
        return runtime.status().as_dict()

    @app.post("/api/stop")
    async def stop() -> dict[str, object]:
        runtime.stop()
        return runtime.status().as_dict()

    @app.get("/api/status")
    async def status() -> dict[str, object]:
        return runtime.status().as_dict()

    @app.get("/video")
    async def video() -> StreamingResponse:
        return StreamingResponse(
            _mjpeg_stream(runtime),
            media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    return app


async def _mjpeg_stream(runtime: PREACTRuntime) -> AsyncIterator[bytes]:
    previous: bytes | None = None
    while True:
        frame = runtime.latest_jpeg()
        if frame is not None and frame != previous:
            previous = frame
            yield (
                f"--{BOUNDARY}\r\n"
                "Content-Type: image/jpeg\r\n"
                f"Content-Length: {len(frame)}\r\n\r\n"
            ).encode("ascii") + frame + b"\r\n"
        await asyncio.sleep(1.0 / 30.0)


def _index_html() -> str:
    path = Path(__file__).with_name("static") / "index.html"
    return path.read_text(encoding="utf-8")
