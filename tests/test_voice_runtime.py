from __future__ import annotations

import json
from pathlib import Path

from preact.__main__ import _build_synthetic_runtime
from preact.models import Prediction
from preact.voice import VoiceCommand, VoiceDecision


def _decision(command: VoiceCommand, transcript: str) -> VoiceDecision:
    return VoiceDecision(
        command=command,
        transcript=transcript,
        source="local",
        reason="test",
    )


def test_voice_commands_switch_modes_leds_and_idle_safely() -> None:
    runtime = _build_synthetic_runtime()
    genesis = runtime.engine.genesis
    try:
        runtime.handle_voice_decision(
            _decision(
                VoiceCommand.SWITCH_BALLOON,
                "Hey Preact, switch to balloon",
            )
        )
        runtime.handle_voice_decision(
            _decision(
                VoiceCommand.TRACKING,
                "Hey Preact, start tracking",
            )
        )
        assert runtime.status().mode == "tracking"
        assert genesis.tracking_led_on
        assert not genesis.landing_led_on

        runtime.handle_voice_decision(
            _decision(
                VoiceCommand.LANDING,
                "Hey Preact, calculate where this will land",
            )
        )
        assert runtime.status().mode == "landing"
        assert not genesis.tracking_led_on
        assert genesis.landing_led_on

        runtime.handle_voice_decision(
            _decision(VoiceCommand.STOP, "Hey Preact, stop")
        )
        status = runtime.status()
        assert status.mode == "idle"
        assert status.servo_angle_deg == 90.0
        assert status.last_voice_command == "stop"
        assert not genesis.tracking_led_on
        assert not genesis.landing_led_on

        # Stop is deliberately idempotent and always reasserts safe hardware state.
        genesis.servo_state = type(genesis.servo_state)(
            angle_deg=130.0,
            command_id=99,
        )
        runtime.handle_voice_decision(
            _decision(VoiceCommand.STOP, "Hey Preact, stop")
        )
        assert genesis.servo_state.angle_deg == 90.0
        assert genesis.mode.value == "idle"
    finally:
        runtime.close()


def test_tracking_requires_a_selected_ready_target() -> None:
    runtime = _build_synthetic_runtime()
    try:
        runtime.handle_voice_decision(
            _decision(VoiceCommand.TRACKING, "Hey Preact, start tracking")
        )
        assert runtime.status().mode == "idle"
        assert "choose face or balloon" in runtime.status().reason

        runtime.handle_voice_decision(
            _decision(VoiceCommand.SWITCH_FACE, "Hey Preact, switch to my face")
        )
        assert runtime.tracking_target == "face"
        runtime.handle_voice_decision(
            _decision(VoiceCommand.TRACKING, "Hey Preact, start tracking")
        )
        assert runtime.status().mode == "idle"
        assert "reference is not ready" in runtime.status().reason

        runtime.handle_voice_decision(
            _decision(
                VoiceCommand.SWITCH_BALLOON,
                "Hey Preact, switch to balloon",
            )
        )
        runtime.handle_voice_decision(
            _decision(VoiceCommand.TRACKING, "Hey Preact, start tracking")
        )
        assert runtime.status().mode == "tracking"
    finally:
        runtime.close()


def test_landing_marker_expires_after_configured_hold_time() -> None:
    runtime = _build_synthetic_runtime()
    prediction = Prediction(
        valid=True,
        reason="predicted",
        generated_at_s=1.0,
        landing_x_m=1.25,
        time_to_impact_s=0.4,
        confidence=0.8,
        fit=None,
    )
    try:
        runtime._last_landing_prediction = prediction
        runtime._landing_marker_track_id = 7
        runtime._landing_marker_expires_at_s = 11.0

        runtime._expire_landing_marker_if_needed(10.99)
        assert runtime._last_landing_prediction is prediction

        runtime._expire_landing_marker_if_needed(11.0)
        assert runtime._last_landing_prediction is None
        assert runtime._landing_marker_expires_at_s is None
        assert runtime._landing_marker_track_id == 7
    finally:
        runtime.close()


def test_predicted_and_actual_landing_are_saved_for_comparison(
    tmp_path: Path,
) -> None:
    runtime = _build_synthetic_runtime()
    result_path = tmp_path / "landing-results.jsonl"
    runtime.landing_results_path = result_path
    try:
        runtime._last_landing_prediction = Prediction(
            valid=True,
            reason="predicted",
            generated_at_s=1.0,
            landing_x_m=1.25,
            time_to_impact_s=0.4,
            confidence=0.82,
            fit=None,
        )
        runtime._landing_marker_track_id = 4
        runtime._actual_landing_candidate = (1.33, 0.01, 200.0, 300.0)

        runtime._finalize_landing_result()

        saved = json.loads(result_path.read_text(encoding="utf-8"))
        assert saved["predicted_landing_x_m"] == 1.25
        assert saved["actual_landing_x_m"] == 1.33
        assert abs(saved["absolute_error_cm"] - 8.0) < 1e-9
        assert saved["measured"] is True
        assert runtime.landing_results() == [saved]
    finally:
        runtime.close()
