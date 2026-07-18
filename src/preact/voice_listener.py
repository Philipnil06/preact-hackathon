"""Local microphone capture, energy segmentation, and faster-whisper transcription."""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from scipy.signal import resample_poly

from preact.voice import (
    ClaudeVoiceInterpreter,
    VoiceCommand,
    VoiceDecision,
    contains_wake_phrase,
    interpret_recoverable_command,
    normalise_transcript,
)

WHISPER_SAMPLE_RATE_HZ = 16_000


class AudioTranscriber(Protocol):
    def transcribe(self, samples: np.ndarray) -> str: ...


@dataclass(frozen=True, slots=True)
class VoiceListenerConfig:
    sample_rate_hz: int = 16_000
    block_duration_s: float = 0.10
    energy_threshold: float = 0.012
    pre_roll_s: float = 0.30
    end_silence_s: float = 0.70
    minimum_utterance_s: float = 0.30
    maximum_utterance_s: float = 5.0
    whisper_model: str = "base.en"
    microphone_device: int | str | None = None
    microphone_channel: int = 0
    wake_continuation_s: float = 15.0
    noise_floor_multiplier: float = 2.5
    noise_calibration_s: float = 0.5

    def __post_init__(self) -> None:
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        if self.microphone_channel < 0:
            raise ValueError("microphone_channel must be non-negative")
        if self.energy_threshold <= 0:
            raise ValueError("energy_threshold must be positive")
        if self.wake_continuation_s <= 0:
            raise ValueError("wake_continuation_s must be positive")
        if self.noise_floor_multiplier <= 1.0:
            raise ValueError("noise_floor_multiplier must be greater than one")
        if self.noise_calibration_s < 0:
            raise ValueError("noise_calibration_s must be non-negative")


class UtteranceSegmenter:
    def __init__(self, config: VoiceListenerConfig | None = None) -> None:
        self.config = config or VoiceListenerConfig()
        self._pre_roll: deque[np.ndarray] = deque()
        self._pre_roll_samples = 0
        self._active: list[np.ndarray] | None = None
        self._active_samples = 0
        self._silence_samples = 0
        self._noise_floor_rms: float | None = None
        self._noise_calibration_samples = 0

    def push(self, samples: np.ndarray) -> np.ndarray | None:
        block = np.asarray(samples, dtype=np.float32).reshape(-1)
        if len(block) == 0:
            return None
        rms = float(np.sqrt(np.mean(np.square(block))))

        if self._active is None and self._calibrating_noise():
            self._append_pre_roll(block)
            self._update_noise_floor(rms)
            self._noise_calibration_samples += len(block)
            return None

        threshold = self.config.energy_threshold
        if self._noise_floor_rms is not None:
            threshold = max(
                threshold,
                self._noise_floor_rms * self.config.noise_floor_multiplier,
            )
        voiced = rms >= threshold

        if self._active is None:
            self._append_pre_roll(block)
            if voiced:
                self._active = list(self._pre_roll)
                self._active_samples = sum(len(value) for value in self._active)
                self._silence_samples = 0
            else:
                self._update_noise_floor(rms)
            return None

        self._active.append(block.copy())
        self._active_samples += len(block)
        self._silence_samples = 0 if voiced else self._silence_samples + len(block)
        maximum = round(
            self.config.maximum_utterance_s * self.config.sample_rate_hz
        )
        ended = self._silence_samples >= round(
            self.config.end_silence_s * self.config.sample_rate_hz
        )
        if self._active_samples < maximum and not ended:
            return None

        utterance = np.concatenate(self._active)
        minimum = round(
            self.config.minimum_utterance_s * self.config.sample_rate_hz
        )
        self.reset()
        return utterance if len(utterance) >= minimum else None

    def reset(self) -> None:
        self._pre_roll.clear()
        self._pre_roll_samples = 0
        self._active = None
        self._active_samples = 0
        self._silence_samples = 0

    def _append_pre_roll(self, block: np.ndarray) -> None:
        self._pre_roll.append(block.copy())
        self._pre_roll_samples += len(block)
        maximum = round(self.config.pre_roll_s * self.config.sample_rate_hz)
        while self._pre_roll and self._pre_roll_samples > maximum:
            removed = self._pre_roll.popleft()
            self._pre_roll_samples -= len(removed)

    def _calibrating_noise(self) -> bool:
        required = round(
            self.config.noise_calibration_s * self.config.sample_rate_hz
        )
        return self._noise_calibration_samples < required

    def _update_noise_floor(self, rms: float) -> None:
        if self._noise_floor_rms is None:
            self._noise_floor_rms = rms
        else:
            self._noise_floor_rms = 0.95 * self._noise_floor_rms + 0.05 * rms


class FasterWhisperTranscriber:
    def __init__(
        self,
        model_name: str,
        *,
        input_sample_rate_hz: int = WHISPER_SAMPLE_RATE_HZ,
    ) -> None:
        from faster_whisper import WhisperModel

        self.input_sample_rate_hz = input_sample_rate_hz
        self.model = WhisperModel(
            model_name,
            device="cpu",
            compute_type="int8",
        )

    def transcribe(self, samples: np.ndarray) -> str:
        samples = resample_for_whisper(
            samples,
            input_sample_rate_hz=self.input_sample_rate_hz,
        )
        segments, _ = self.model.transcribe(
            samples,
            language="en",
            beam_size=2,
            condition_on_previous_text=False,
            initial_prompt=(
                "Voice commands begin with Hey Preact. Expected commands: "
                "Hey Preact, start tracking. Hey Preact, calculate where this "
                "will land. Hey Preact, stop."
            ),
            hotwords="Hey Preact start tracking calculate landing stop",
            vad_filter=False,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()


def resample_for_whisper(
    samples: np.ndarray,
    *,
    input_sample_rate_hz: int,
) -> np.ndarray:
    values = np.asarray(samples, dtype=np.float32).reshape(-1)
    if input_sample_rate_hz == WHISPER_SAMPLE_RATE_HZ:
        return values
    return np.asarray(
        resample_poly(
            values,
            WHISPER_SAMPLE_RATE_HZ,
            input_sample_rate_hz,
        ),
        dtype=np.float32,
    )


class MicrophoneVoiceService:
    def __init__(
        self,
        *,
        interpreter: ClaudeVoiceInterpreter,
        on_decision: Callable[[VoiceDecision], None],
        on_state: Callable[[str], None] | None = None,
        config: VoiceListenerConfig | None = None,
        transcriber: AudioTranscriber | None = None,
    ) -> None:
        self.interpreter = interpreter
        self.on_decision = on_decision
        self.on_state = on_state or (lambda _: None)
        self.config = config or VoiceListenerConfig()
        self._transcriber = transcriber
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._audio: queue.Queue[np.ndarray] = queue.Queue(maxsize=64)
        self._pending_wake_transcript: str | None = None
        self._pending_wake_at_s: float | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="preact-voice",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)

    def process_transcript(self, transcript: str) -> VoiceDecision:
        now = time.monotonic()
        if (
            self._pending_wake_transcript is not None
            and self._pending_wake_at_s is not None
            and now - self._pending_wake_at_s <= self.config.wake_continuation_s
            and not contains_wake_phrase(transcript)
        ):
            transcript = f"{self._pending_wake_transcript} {transcript}"
        self._pending_wake_transcript = None
        self._pending_wake_at_s = None

        if normalise_transcript(transcript) in {"hey preact", "hey pre act"}:
            self._pending_wake_transcript = transcript
            self._pending_wake_at_s = now
            decision = VoiceDecision(
                command=VoiceCommand.IGNORE,
                transcript=transcript,
                source="local",
                reason="wake phrase awaiting command continuation",
            )
            self.on_decision(decision)
            return decision

        decision = self.interpreter.interpret(transcript)
        if decision.command is VoiceCommand.IGNORE and not contains_wake_phrase(
            transcript
        ):
            recovered = interpret_recoverable_command(transcript)
            if recovered is not None:
                decision = VoiceDecision(
                    command=recovered,
                    transcript=transcript,
                    source="local-asr-recovery",
                    reason="canonical command with wake phrase clipped by ASR",
                )
        self.on_decision(decision)
        return decision

    def _run(self) -> None:
        try:
            self.on_state("loading speech model")
            transcriber = self._transcriber or FasterWhisperTranscriber(
                self.config.whisper_model,
                input_sample_rate_hz=self.config.sample_rate_hz,
            )
            if self._stop.is_set():
                return
            import sounddevice as sd

            segmenter = UtteranceSegmenter(self.config)
            block_size = round(
                self.config.sample_rate_hz * self.config.block_duration_s
            )

            def capture(
                input_data: np.ndarray,
                frames: int,
                time_info: Any,
                status: Any,
            ) -> None:
                del frames, time_info, status
                try:
                    self._audio.put_nowait(
                        input_data[:, self.config.microphone_channel].copy()
                    )
                except queue.Full:
                    pass

            self.on_state("listening")
            with sd.InputStream(
                samplerate=self.config.sample_rate_hz,
                blocksize=block_size,
                channels=self.config.microphone_channel + 1,
                dtype="float32",
                device=self.config.microphone_device,
                callback=capture,
            ):
                while not self._stop.is_set():
                    try:
                        block = self._audio.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    utterance = segmenter.push(block)
                    if utterance is None:
                        continue
                    try:
                        self.on_state("transcribing")
                        transcript = transcriber.transcribe(utterance)
                        if transcript:
                            self.process_transcript(transcript)
                    except Exception as error:
                        self.on_state(f"command error: {error}")
                    self.on_state("listening")
        except Exception as error:
            self.on_state(f"voice error: {error}")
