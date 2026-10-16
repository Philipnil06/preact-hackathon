"""Minimal two-mode local UI with an MJPEG tracking view."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from preact.runtime import PREACTRuntime

BOUNDARY = "preact-frame"
# Tracking still runs at the camera rate.  The browser only needs a responsive
# operator preview, and decoding 30 MJPEG images/s makes its controls sluggish.
PREVIEW_FPS = 15.0


class CalibrationAdjustment(BaseModel):
    pan_delta_deg: float = Field(default=0.0, ge=-10.0, le=10.0)
    tilt_delta_deg: float = Field(default=0.0, ge=-10.0, le=10.0)


class CalibrationAngles(BaseModel):
    pan_deg: float = Field(ge=1.0, le=180.0)
    tilt_deg: float = Field(ge=1.0, le=180.0)


def create_app(runtime: PREACTRuntime) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        runtime.start_worker()
        yield
        runtime.close()

    app = FastAPI(title="PREACT", lifespan=lifespan)

    def public_status() -> dict[str, object]:
        return {
            **runtime.status().as_dict(),
            "tracking_target": runtime.tracking_target,
            "tracking_target_ready": runtime.tracking_target_ready,
        }

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return _index_html()

    @app.post("/api/play")
    async def play() -> dict[str, object]:
        runtime.play()
        return public_status()

    @app.post("/api/mode/tracking")
    async def tracking() -> dict[str, object]:
        runtime.start_tracking()
        return public_status()

    @app.post("/api/mode/landing")
    async def landing() -> dict[str, object]:
        runtime.start_landing()
        return public_status()

    @app.post("/api/stop")
    async def stop() -> dict[str, object]:
        runtime.stop()
        return public_status()

    @app.get("/api/status")
    async def status() -> dict[str, object]:
        return public_status()

    @app.post("/api/tracking-target/{target}")
    async def tracking_target(target: str) -> dict[str, object]:
        try:
            ready = runtime.select_tracking_target(target)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        value = public_status()
        value["tracking_target_ready"] = ready
        return value

    @app.get("/api/landing-results")
    async def landing_results() -> list[dict[str, object]]:
        return runtime.landing_results()

    @app.get("/api/calibration")
    async def calibration_status() -> dict[str, object]:
        return runtime.pan_tilt_calibration_status()

    @app.post("/api/calibration/start/{point}")
    async def calibration_start(point: str) -> dict[str, object]:
        return runtime.start_pan_tilt_calibration(point)

    @app.post("/api/calibration/adjust")
    async def calibration_adjust(
        adjustment: CalibrationAdjustment,
    ) -> dict[str, object]:
        return runtime.adjust_pan_tilt_calibration(
            pan_delta_deg=adjustment.pan_delta_deg,
            tilt_delta_deg=adjustment.tilt_delta_deg,
        )

    @app.post("/api/calibration/save")
    async def calibration_save() -> dict[str, object]:
        return runtime.save_pan_tilt_calibration_point()

    @app.post("/api/calibration/angles")
    async def calibration_angles(angles: CalibrationAngles) -> dict[str, object]:
        return runtime.set_pan_tilt_calibration_angles(
            pan_deg=angles.pan_deg,
            tilt_deg=angles.tilt_deg,
        )

    @app.post("/api/calibration/finish")
    async def calibration_finish() -> dict[str, object]:
        return runtime.finish_pan_tilt_calibration()

    @app.post("/api/calibration/reset")
    async def calibration_reset() -> dict[str, object]:
        return runtime.reset_pan_tilt_calibration()

    @app.post("/api/calibration/offset")
    async def calibration_offset(
        adjustment: CalibrationAdjustment,
    ) -> dict[str, object]:
        try:
            return runtime.shift_pan_tilt_calibration(
                pan_delta_deg=adjustment.pan_delta_deg,
                tilt_delta_deg=adjustment.tilt_delta_deg,
            )
        except RuntimeError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

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
        await asyncio.sleep(1.0 / PREVIEW_FPS)


def _index_html() -> str:
    path = Path(__file__).with_name("static") / "index.html"
    return path.read_text(encoding="utf-8")
