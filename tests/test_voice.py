from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from preact.voice import (
    ClaudeVoiceInterpreter,
    VoiceCommand,
    contains_wake_phrase,
    interpret_exact_command,
    interpret_recoverable_command,
)
from preact.voice_listener import (
    MicrophoneVoiceService,
    UtteranceSegmenter,
    VoiceListenerConfig,
    resample_for_whisper,
)


class FakeMessages:
    def __init__(self, command: str) -> None:
        self.command = command
        self.calls = 0

    def create(self, **kwargs: Any) -> object:
        del kwargs
        self.calls += 1
        block = SimpleNamespace(
            type="tool_use",
            name="set_preact_mode",
            input={"command": self.command, "reason": "understood paraphrase"},
        )
        return SimpleNamespace(content=[block], stop_reason="tool_use")


class FakeClient:
    def __init__(self, command: str) -> None:
        self.messages = FakeMessages(command)


def test_three_canonical_wake_commands_are_local_and_immediate() -> None:
    assert (
        interpret_exact_command("Hey Preact, start tracking!")
        is VoiceCommand.TRACKING
    )
    assert (
        interpret_exact_command("Hey, Pre-Act, calculate where this will land.")
        is VoiceCommand.LANDING
    )
    assert interpret_exact_command("Hey Preact, stop.") is VoiceCommand.STOP
    assert (
        interpret_exact_command("Hey Preact, switch to my face")
        is VoiceCommand.SWITCH_FACE
    )
    assert (
        interpret_exact_command("Hey Preact, switch to balloon")
        is VoiceCommand.SWITCH_BALLOON
    )
    assert (
        interpret_exact_command("Hey Preact, switch to ballon")
        is VoiceCommand.SWITCH_BALLOON
    )
    assert (
        interpret_exact_command("Hey Preact, switch to my phase")
        is VoiceCommand.SWITCH_FACE
    )
    assert interpret_exact_command("Hey Preact, face") is VoiceCommand.SWITCH_FACE
    assert (
        interpret_exact_command("Hey Preact, balloon")
        is VoiceCommand.SWITCH_BALLOON
    )
    assert contains_wake_phrase("HEY PRE ACT start tracking")


def test_no_wake_phrase_is_ignored_without_a_claude_call() -> None:
    client = FakeClient("tracking")
    interpreter = ClaudeVoiceInterpreter(client=client, model="test")

    decision = interpreter.interpret("please start tracking")

    assert decision.command is VoiceCommand.IGNORE
    assert client.messages.calls == 0


def test_canonical_commands_work_without_an_api_client() -> None:
    interpreter = ClaudeVoiceInterpreter(client=None)

    decision = interpreter.interpret("Hey Preact, stop")

    assert decision.command is VoiceCommand.STOP
    assert decision.source == "local"


def test_claude_handles_a_wake_phrase_paraphrase_with_strict_command() -> None:
    client = FakeClient("landing")
    interpreter = ClaudeVoiceInterpreter(client=client, model="test")

    decision = interpreter.interpret(
        "Hey Preact, work out the place where the ball is going to touch down"
    )

    assert decision.command is VoiceCommand.LANDING
    assert decision.source == "claude"
    assert client.messages.calls == 1


def test_energy_segmenter_emits_one_complete_utterance() -> None:
    config = VoiceListenerConfig(
        sample_rate_hz=100,
        energy_threshold=0.1,
        pre_roll_s=0.1,
        end_silence_s=0.2,
        minimum_utterance_s=0.1,
        maximum_utterance_s=2.0,
        noise_calibration_s=0.0,
    )
    segmenter = UtteranceSegmenter(config)
    silence = np.zeros(10, dtype=np.float32)
    speech = np.full(10, 0.5, dtype=np.float32)

    assert segmenter.push(silence) is None
    assert segmenter.push(speech) is None
    assert segmenter.push(speech) is None
    assert segmenter.push(silence) is None
    utterance = segmenter.push(silence)

    assert utterance is not None
    assert len(utterance) >= 40


def test_energy_segmenter_adapts_to_a_measured_noise_floor() -> None:
    config = VoiceListenerConfig(
        sample_rate_hz=100,
        energy_threshold=0.01,
        noise_floor_multiplier=2.5,
        noise_calibration_s=0.2,
        pre_roll_s=0.1,
        end_silence_s=0.2,
        minimum_utterance_s=0.1,
        maximum_utterance_s=2.0,
    )
    segmenter = UtteranceSegmenter(config)
    noise = np.full(10, 0.02, dtype=np.float32)
    speech = np.full(10, 0.09, dtype=np.float32)

    assert segmenter.push(noise) is None
    assert segmenter.push(noise) is None
    assert segmenter.push(speech) is None
    assert segmenter.push(noise) is None
    utterance = segmenter.push(noise)

    assert utterance is not None


def test_audio_is_resampled_to_whisper_rate() -> None:
    input_samples = np.linspace(-1.0, 1.0, 48_000, dtype=np.float32)

    output = resample_for_whisper(
        input_samples,
        input_sample_rate_hz=48_000,
    )

    assert output.dtype == np.float32
    assert len(output) == 16_000


def test_voice_service_stitches_wake_phrase_and_stop_segments() -> None:
    decisions = []
    service = MicrophoneVoiceService(
        interpreter=ClaudeVoiceInterpreter(client=None),
        on_decision=decisions.append,
    )

    first = service.process_transcript("Hey Preact")
    second = service.process_transcript("Stop")

    assert first.command is VoiceCommand.IGNORE
    assert second.command is VoiceCommand.STOP
    assert second.transcript == "Hey Preact Stop"
    assert decisions == [first, second]


def test_voice_service_stitches_wake_phrase_and_target_switch() -> None:
    decisions = []
    service = MicrophoneVoiceService(
        interpreter=ClaudeVoiceInterpreter(client=None),
        on_decision=decisions.append,
    )

    first = service.process_transcript("Hey Preact")
    second = service.process_transcript("Switch to my face")

    assert first.command is VoiceCommand.IGNORE
    assert second.command is VoiceCommand.SWITCH_FACE
    assert second.transcript == "Hey Preact Switch to my face"


def test_specific_command_is_recovered_when_whisper_clips_wake_phrase() -> None:
    decisions = []
    service = MicrophoneVoiceService(
        interpreter=ClaudeVoiceInterpreter(client=None),
        on_decision=decisions.append,
    )

    decision = service.process_transcript("Calculates where this will land.")

    assert decision.command is VoiceCommand.LANDING
    assert decision.source == "local-asr-recovery"
    assert interpret_recoverable_command("please calculate something") is None
    assert (
        interpret_recoverable_command("switch to my face")
        is VoiceCommand.SWITCH_FACE
    )


def test_inflected_landing_command_with_wake_phrase_is_canonical() -> None:
    assert (
        interpret_exact_command("Hey Preact, calculates where this will land")
        is VoiceCommand.LANDING
    )


def test_split_wake_phrase_window_accounts_for_slow_cpu_transcription() -> None:
    service = MicrophoneVoiceService(
        interpreter=ClaudeVoiceInterpreter(client=None),
        on_decision=lambda _: None,
    )

    assert service.config.wake_continuation_s >= 15.0
