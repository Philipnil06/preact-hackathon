from __future__ import annotations

import asyncio
import time

import httpx

from preact.__main__ import _build_synthetic_runtime
from preact.web import create_app


def test_minimal_web_ui_can_play_and_stop_synthetic_demo() -> None:
    asyncio.run(_exercise_minimal_web_ui())


async def _exercise_minimal_web_ui() -> None:
    runtime = _build_synthetic_runtime()
    app = create_app(runtime)
    transport = httpx.ASGITransport(app=app)

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://preact.test",
        ) as client:
            page = await client.get("/")
            assert page.status_code == 200
            assert "PREACT" in page.text
            assert "Start tracking" in page.text
            assert "Calculate landing" in page.text
            assert "Calibrate mount" in page.text
            assert "Five-point mount calibration" in page.text
            assert "Move servos" in page.text
            assert "Stop" in page.text
            assert 'src="/video"' in page.text
            assert 'feed.addEventListener("error"' in page.text
            assert "feed.naturalWidth === 0" in page.text
            assert "setInterval(checkFeed, 1000)" in page.text
            assert "Predicting through brief target loss" in page.text

            deadline = time.monotonic() + 2.0
            while runtime.latest_jpeg() is None and time.monotonic() < deadline:
                await asyncio.sleep(0.02)
            idle_preview = runtime.latest_jpeg()
            assert idle_preview is not None
            assert idle_preview.startswith(b"\xff\xd8")

            play = await client.post("/api/play")
            assert play.status_code == 200
            assert play.json()["playing"] is True
            assert play.json()["mode"] == "landing"
            assert runtime.engine.genesis.landing_led_on
            assert not runtime.engine.genesis.tracking_led_on

            deadline = time.monotonic() + 2.0
            status = (await client.get("/api/status")).json()
            while status["observations"] == 0 and time.monotonic() < deadline:
                await asyncio.sleep(0.02)
                status = (await client.get("/api/status")).json()

            assert status["observations"] > 0
            assert runtime.latest_jpeg() is not None

            stop = await client.post("/api/stop")
            assert stop.status_code == 200
            stopped = stop.json()
            assert stopped["playing"] is False
            assert stopped["tracking"] is False
            assert stopped["observations"] == 0
            assert stopped["servo_angle_deg"] == 90.0
            assert runtime.engine.genesis.servo_state.angle_deg == 90.0
            assert not runtime.engine.genesis.tracking_led_on
            assert not runtime.engine.genesis.landing_led_on

            selected = await client.post("/api/tracking-target/balloon")
            assert selected.json()["tracking_target"] == "balloon"
            assert selected.json()["tracking_target_ready"] is True
            tracking = await client.post("/api/mode/tracking")
            assert tracking.status_code == 200
            assert tracking.json()["mode"] == "tracking"
            assert runtime.engine.genesis.tracking_led_on
            assert not runtime.engine.genesis.landing_led_on

            deadline = time.monotonic() + 2.0
            tracking_status = (await client.get("/api/status")).json()
            while (
                tracking_status["target_y_m"] is None
                and time.monotonic() < deadline
            ):
                await asyncio.sleep(0.02)
                tracking_status = (await client.get("/api/status")).json()
            assert tracking_status["target_y_m"] is not None

            await client.post("/api/stop")
