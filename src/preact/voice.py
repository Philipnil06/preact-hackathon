"""Wake-phrase command interpretation with deterministic fast paths and Claude."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from anthropic import Anthropic, APIError

from preact.claude_targeting import DEFAULT_CLAUDE_MODEL

VOICE_TOOL_NAME = "set_preact_mode"


class VoiceCommand(StrEnum):
    TRACKING = "tracking"
    LANDING = "landing"
    STOP = "stop"
    IGNORE = "ignore"


@dataclass(frozen=True, slots=True)
class VoiceDecision:
    command: VoiceCommand
    transcript: str
    source: str
    reason: str


class VoiceInterpretationError(RuntimeError):
    pass


class ClaudeVoiceInterpreter:
    def __init__(
        self,
        *,
        client: Any | None,
        model: str = DEFAULT_CLAUDE_MODEL,
    ) -> None:
        self.client = client
        self.model = model

    @classmethod
    def from_environment(cls) -> ClaudeVoiceInterpreter:
        api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        model = os.getenv("ANTHROPIC_MODEL", DEFAULT_CLAUDE_MODEL).strip()
        if not api_key:
            return cls(client=None, model=model)
        try:
            timeout_s = float(os.getenv("ANTHROPIC_TIMEOUT_S", "8.0"))
        except ValueError as error:
            raise VoiceInterpretationError(
                "ANTHROPIC_TIMEOUT_S must be numeric"
            ) from error
        if timeout_s <= 0:
            raise VoiceInterpretationError(
                "ANTHROPIC_TIMEOUT_S must be positive"
            )
        return cls(
            client=Anthropic(
                api_key=api_key,
                timeout=timeout_s,
                max_retries=1,
            ),
            model=model,
        )

    def interpret(self, transcript: str) -> VoiceDecision:
        exact = interpret_exact_command(transcript)
        if exact is not None:
            return VoiceDecision(
                command=exact,
                transcript=transcript,
                source="local",
                reason="canonical wake command",
            )
        if not contains_wake_phrase(transcript):
            return VoiceDecision(
                command=VoiceCommand.IGNORE,
                transcript=transcript,
                source="local",
                reason="wake phrase not present",
            )
        if self.client is None:
            raise VoiceInterpretationError(
                "ANTHROPIC_API_KEY is required only for paraphrased commands; "
                "the three canonical Hey Preact commands work locally"
            )

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=256,
                thinking={"type": "disabled"},
                system=(
                    "Interpret voice commands for a benign foam-ball demo. "
                    "Only the wake phrase 'Hey Preact' grants command intent. "
                    "Choose tracking, landing, stop, or ignore. Tracking means "
                    "follow the visible ball continuously. Landing means "
                    "calculate where a thrown ball will land. Stop means enter "
                    "idle, neutralize the servo, and turn both mode LEDs off. "
                    "Never infer any other action."
                ),
                tools=[_voice_tool()],
                tool_choice={
                    "type": "tool",
                    "name": VOICE_TOOL_NAME,
                    "disable_parallel_tool_use": True,
                },
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Classify this microphone transcript exactly: "
                            f"{transcript!r}"
                        ),
                    }
                ],
            )
        except APIError as error:
            raise VoiceInterpretationError(str(error)) from error

        value = _extract_voice_tool(response)
        try:
            command = VoiceCommand(str(value["command"]))
        except (KeyError, ValueError) as error:
            raise VoiceInterpretationError(
                "Claude returned an invalid voice command"
            ) from error
        return VoiceDecision(
            command=command,
            transcript=transcript,
            source="claude",
            reason=str(value.get("reason", "")).strip(),
        )


def normalise_transcript(transcript: str) -> str:
    value = transcript.casefold()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def contains_wake_phrase(transcript: str) -> bool:
    value = normalise_transcript(transcript)
    return bool(re.search(r"\bhey\s+pre\s*act\b|\bhey\s+preact\b", value))


def interpret_exact_command(transcript: str) -> VoiceCommand | None:
    value = normalise_transcript(transcript)
    wake = re.search(r"\bhey\s+pre\s*act\b|\bhey\s+preact\b", value)
    if wake is None:
        return None
    command = value[wake.end() :].strip()
    if command in {
        "start tracking",
        "track the ball",
        "start tracking the ball",
    }:
        return VoiceCommand.TRACKING
    if command in {
        "calculate where this will land",
        "calculates where this will land",
        "calculate where it will land",
        "calculate landing",
        "where will this land",
    }:
        return VoiceCommand.LANDING
    if command in {
        "stop",
        "stop tracking",
        "stop calculating",
        "go idle",
    }:
        return VoiceCommand.STOP
    return None


def interpret_recoverable_command(transcript: str) -> VoiceCommand | None:
    """Recover a canonical command when ASR clipped only the wake phrase.

    This is intentionally limited to exact, benign demo phrases. Arbitrary
    paraphrases without an audible wake phrase remain ignored.
    """

    value = normalise_transcript(transcript)
    if value in {"start tracking", "start tracking the ball", "track the ball"}:
        return VoiceCommand.TRACKING
    if value in {
        "calculate where this will land",
        "calculates where this will land",
        "calculate where it will land",
        "calculate landing",
        "where will this land",
    }:
        return VoiceCommand.LANDING
    if value in {"stop", "stop tracking", "stop calculating", "go idle"}:
        return VoiceCommand.STOP
    return None


def _voice_tool() -> dict[str, Any]:
    return {
        "name": VOICE_TOOL_NAME,
        "description": "Select one PREACT operating-mode command.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["tracking", "landing", "stop", "ignore"],
                },
                "reason": {"type": "string"},
            },
            "required": ["command", "reason"],
            "additionalProperties": False,
        },
    }


def _extract_voice_tool(response: Any) -> dict[str, Any]:
    for block in response.content:
        if (
            getattr(block, "type", None) == "tool_use"
            and getattr(block, "name", None) == VOICE_TOOL_NAME
            and isinstance(getattr(block, "input", None), dict)
        ):
            return block.input
    raise VoiceInterpretationError(
        f"Claude did not call {VOICE_TOOL_NAME}; "
        f"stop reason: {getattr(response, 'stop_reason', 'unknown')}"
    )
