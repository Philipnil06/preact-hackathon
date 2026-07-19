"""JSONL telemetry for tuning and evidence after each demo run."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict
from pathlib import Path

from preact.engine import EngineSnapshot


class SessionLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8", buffering=1)
        self._lock = threading.RLock()
        self._session_id = 0

    def start_session(self) -> None:
        with self._lock:
            self._session_id += 1
            self._write(
                {
                    "event": "session_start",
                    "session_id": self._session_id,
                    "wall_time_s": time.time(),
                }
            )

    def record(self, snapshot: EngineSnapshot) -> None:
        prediction = snapshot.prediction
        fit = asdict(prediction.fit) if prediction.fit is not None else None
        detection = asdict(snapshot.detection) if snapshot.detection is not None else None
        servo = asdict(snapshot.servo_target) if snapshot.servo_target is not None else None
        pan_tilt = (
            asdict(snapshot.pan_tilt_target)
            if snapshot.pan_tilt_target is not None
            else None
        )
        with self._lock:
            self._write(
                {
                    "event": "frame",
                    "session_id": self._session_id,
                    "mode": snapshot.mode.value,
                    "landing_track_id": snapshot.landing_track_id,
                    "landing_locked": snapshot.landing_locked,
                    "timestamp_s": snapshot.timestamp_s,
                    "detection": detection,
                    "detector_state": snapshot.detector_state,
                    "observation_count": snapshot.observation_count,
                    "tof": {
                        "distance_mm": snapshot.tof_distance_mm,
                        "baseline_mm": snapshot.tof_baseline_mm,
                        "trigger": (
                            asdict(snapshot.tof_trigger)
                            if snapshot.tof_trigger is not None
                            else None
                        ),
                    },
                    "prediction": {
                        "valid": prediction.valid,
                        "reason": prediction.reason,
                        "landing_x_m": prediction.landing_x_m,
                        "time_to_impact_s": prediction.time_to_impact_s,
                        "confidence": prediction.confidence,
                        "fit": fit,
                    },
                    "servo_target": servo,
                    "pan_tilt_target": pan_tilt,
                }
            )

    def stop_session(self) -> None:
        with self._lock:
            self._write(
                {
                    "event": "session_stop",
                    "session_id": self._session_id,
                    "wall_time_s": time.time(),
                }
            )

    def close(self) -> None:
        with self._lock:
            if not self._handle.closed:
                self._handle.close()

    def _write(self, value: dict[str, object]) -> None:
        self._handle.write(json.dumps(value, separators=(",", ":")) + "\n")
