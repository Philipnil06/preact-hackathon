from __future__ import annotations

from typing import Any

import pytest

from preact.voice_output import (
    DEFAULT_MODEL,
    DEFAULT_VOICE_ID,
    ElevenLabsVoice,
    VoiceOutputError,
)


class FakeResponse:
    def __init__(self, status_code: int, content: bytes = b"", text: str = "") -> None:
        self.status_code = status_code
        self.content = content
        self.text = text


class FakeClient:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.request: dict[str, Any] | None = None
        self.closed = False

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        self.request = {"url": url, **kwargs}
        return self.response

    def close(self) -> None:
        self.closed = True


def _voice(response: FakeResponse) -> tuple[ElevenLabsVoice, FakeClient]:
    client = FakeClient(response)
    voice = ElevenLabsVoice(client=client, api_key="test-key")
    return voice, client


def test_synthesize_returns_pcm_and_sends_expected_request() -> None:
    pcm = b"\x01\x00\x02\x00"
    voice, client = _voice(FakeResponse(200, content=pcm))

    speech = voice.synthesize("landing predicted")

    assert speech.pcm == pcm
    assert speech.sample_rate_hz == 22050
    assert client.request is not None
    assert client.request["url"].endswith(DEFAULT_VOICE_ID)
    assert client.request["headers"]["xi-api-key"] == "test-key"
    assert client.request["params"]["output_format"] == "pcm_22050"
    assert client.request["json"]["text"] == "landing predicted"
    assert client.request["json"]["model_id"] == DEFAULT_MODEL


def test_non_200_raises_with_detail() -> None:
    voice, _ = _voice(FakeResponse(401, text="invalid api key"))
    with pytest.raises(VoiceOutputError, match="status 401"):
        voice.synthesize("hello")


def test_empty_text_rejected_before_any_request() -> None:
    voice, client = _voice(FakeResponse(200, content=b""))
    with pytest.raises(VoiceOutputError, match="must not be empty"):
        voice.synthesize("   ")
    assert client.request is None


def test_unsupported_sample_rate_rejected() -> None:
    with pytest.raises(VoiceOutputError, match="sample_rate_hz"):
        ElevenLabsVoice(client=FakeClient(FakeResponse(200)), api_key="k", sample_rate_hz=48000)


def test_available_reflects_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    assert ElevenLabsVoice.available() is False
    monkeypatch.setenv("ELEVENLABS_API_KEY", "abc")
    assert ElevenLabsVoice.available() is True


def test_from_environment_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    with pytest.raises(VoiceOutputError, match="ELEVENLABS_API_KEY"):
        ElevenLabsVoice.from_environment()


def test_close_closes_client() -> None:
    voice, client = _voice(FakeResponse(200, content=b""))
    voice.close()
    assert client.closed is True
