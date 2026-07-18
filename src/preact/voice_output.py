"""ElevenLabs text-to-speech for spoken demo feedback.

This is output only: short spoken announcements such as "landing
predicted" or "locked". It is deliberately kept out of the real-time
geometry and servo loop; call it on state changes, never per frame.
Speech synthesis is a network call and must never gate a servo command.

The key comes from ELEVENLABS_API_KEY. If it is absent the caller should
degrade to silence rather than fail, exactly as Claude commissioning
degrades to a saved detector profile.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx

DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # ElevenLabs stock voice "Rachel"
DEFAULT_MODEL = "eleven_flash_v2_5"        # lowest-latency model, good for demo
DEFAULT_SAMPLE_RATE_HZ = 22050
API_ROOT = "https://api.elevenlabs.io/v1/text-to-speech"
_SUPPORTED_SAMPLE_RATES = (16000, 22050, 44100)


@dataclass(frozen=True, slots=True)
class Speech:
    """Raw PCM signed-16-bit little-endian mono audio and its sample rate."""

    pcm: bytes
    sample_rate_hz: int


class VoiceOutputError(RuntimeError):
    pass


class ElevenLabsVoice:
    """Synthesise short spoken feedback with one ElevenLabs call."""

    def __init__(
        self,
        *,
        client: Any,
        api_key: str,
        voice_id: str = DEFAULT_VOICE_ID,
        model: str = DEFAULT_MODEL,
        sample_rate_hz: int = DEFAULT_SAMPLE_RATE_HZ,
    ) -> None:
        if sample_rate_hz not in _SUPPORTED_SAMPLE_RATES:
            raise VoiceOutputError(
                f"sample_rate_hz must be one of {_SUPPORTED_SAMPLE_RATES}"
            )
        self.client = client
        self.api_key = api_key
        self.voice_id = voice_id
        self.model = model
        self.sample_rate_hz = sample_rate_hz

    @classmethod
    def from_environment(cls) -> ElevenLabsVoice:
        api_key = os.getenv("ELEVENLABS_API_KEY", "").strip()
        if not api_key:
            raise VoiceOutputError(
                "ELEVENLABS_API_KEY is not set; run without spoken feedback"
            )
        voice_id = os.getenv("ELEVENLABS_VOICE_ID", DEFAULT_VOICE_ID).strip()
        model = os.getenv("ELEVENLABS_MODEL", DEFAULT_MODEL).strip()
        try:
            timeout_s = float(os.getenv("ELEVENLABS_TIMEOUT_S", "8.0"))
        except ValueError as error:
            raise VoiceOutputError("ELEVENLABS_TIMEOUT_S must be numeric") from error
        if timeout_s <= 0:
            raise VoiceOutputError("ELEVENLABS_TIMEOUT_S must be positive")
        try:
            sample_rate_hz = int(
                os.getenv("ELEVENLABS_SAMPLE_RATE_HZ", str(DEFAULT_SAMPLE_RATE_HZ))
            )
        except ValueError as error:
            raise VoiceOutputError("ELEVENLABS_SAMPLE_RATE_HZ must be an integer") from error
        return cls(
            client=httpx.Client(timeout=timeout_s),
            api_key=api_key,
            voice_id=voice_id,
            model=model,
            sample_rate_hz=sample_rate_hz,
        )

    @classmethod
    def available(cls) -> bool:
        return bool(os.getenv("ELEVENLABS_API_KEY", "").strip())

    def synthesize(self, text: str) -> Speech:
        text = text.strip()
        if not text:
            raise VoiceOutputError("text must not be empty")
        response = self.client.post(
            f"{API_ROOT}/{self.voice_id}",
            params={"output_format": f"pcm_{self.sample_rate_hz}"},
            headers={"xi-api-key": self.api_key, "accept": "audio/pcm"},
            json={
                "text": text,
                "model_id": self.model,
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
            },
        )
        status = getattr(response, "status_code", None)
        if status != 200:
            detail = getattr(response, "text", "")
            raise VoiceOutputError(
                f"ElevenLabs request failed with status {status}: {detail[:200]}"
            )
        return Speech(pcm=response.content, sample_rate_hz=self.sample_rate_hz)

    def speak(self, text: str) -> None:
        """Synthesise and play. Blocks until playback finishes.

        Never call this from the real-time prediction loop. Import of the
        audio backend is lazy so tests and headless runs need no sound
        device.
        """
        speech = self.synthesize(text)
        import numpy as np
        import sounddevice as sd

        samples = np.frombuffer(speech.pcm, dtype=np.int16)
        sd.play(samples, speech.sample_rate_hz)
        sd.wait()

    def close(self) -> None:
        closer = getattr(self.client, "close", None)
        if callable(closer):
            closer()


def main() -> None:
    """`preact-say "landing predicted"` — spot-check the ElevenLabs key."""
    import argparse

    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(description="Speak text with ElevenLabs")
    parser.add_argument("text", help="text to speak")
    arguments = parser.parse_args()

    if not ElevenLabsVoice.available():
        raise SystemExit("ELEVENLABS_API_KEY is not set; nothing to speak")
    voice = ElevenLabsVoice.from_environment()
    try:
        voice.speak(arguments.text)
    finally:
        voice.close()


if __name__ == "__main__":
    main()
