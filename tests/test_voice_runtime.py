from __future__ import annotations

from preact.__main__ import _build_synthetic_runtime
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
