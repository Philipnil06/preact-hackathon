from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np

from preact.claude_targeting import (
    ClaudeCommissionedDetector,
    ClaudeTargetSelectionError,
    ClaudeTargetSelector,
)


class FakeMessages:
    def __init__(self, tool_input: dict[str, object]) -> None:
        self.tool_input = tool_input
        self.request: dict[str, Any] | None = None

    def create(self, **kwargs: Any) -> object:
        self.request = kwargs
        block = SimpleNamespace(
            type="tool_use",
            name="select_tracking_target",
            input=self.tool_input,
        )
        return SimpleNamespace(content=[block], stop_reason="tool_use")


class FakeClient:
    def __init__(self, tool_input: dict[str, object]) -> None:
        self.messages = FakeMessages(tool_input)


def test_claude_selects_roi_with_strict_tool_and_vision() -> None:
    client = FakeClient(
        {
            "found": True,
            "x": 90,
            "y": 60,
            "width": 80,
            "height": 80,
            "description": "magenta foam ball",
            "confidence": 0.95,
            "reason": "single safe coloured object",
        }
    )
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    selector = ClaudeTargetSelector(client=client, model="test-model")

    selection = selector.select(frame)

    assert selection.found
    assert selection.roi == (90, 60, 80, 80)
    assert client.messages.request is not None
    assert client.messages.request["thinking"] == {"type": "disabled"}
    assert client.messages.request["tools"][0]["strict"] is True
    content = client.messages.request["messages"][0]["content"]
    assert content[0]["type"] == "image"
    assert content[0]["source"]["type"] == "base64"


def test_one_claude_call_commissions_a_normal_local_detector() -> None:
    client = FakeClient(
        {
            "found": True,
            "x": 115,
            "y": 75,
            "width": 90,
            "height": 90,
            "description": "green foam ball",
            "confidence": 0.9,
            "reason": "safe target",
        }
    )
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.circle(frame, (160, 120), 32, (0, 255, 0), -1)
    detector = ClaudeCommissionedDetector(
        ClaudeTargetSelector(client=client, model="test-model")
    )

    first = detector.detect(frame, timestamp_s=0.0)
    second = detector.detect(frame, timestamp_s=0.1)

    assert first is not None
    assert second is not None
    assert detector.commissioned
    assert client.messages.request is not None


def test_environment_factory_requires_an_api_key(monkeypatch: Any) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    try:
        ClaudeTargetSelector.from_environment()
    except ClaudeTargetSelectionError as error:
        assert "ANTHROPIC_API_KEY" in str(error)
    else:
        raise AssertionError("missing API key should fail before a network call")
