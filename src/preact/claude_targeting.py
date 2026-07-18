"""Claude-assisted one-time target selection for safe demo objects."""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from anthropic import Anthropic, APIError

from preact.ball_calibration import calibrate_ball_colour
from preact.detection import BallDetector, BallDetectorConfig
from preact.models import Detection

DEFAULT_CLAUDE_MODEL = "claude-sonnet-5"
TARGET_TOOL_NAME = "select_tracking_target"


@dataclass(frozen=True, slots=True)
class TargetSelection:
    found: bool
    roi: tuple[int, int, int, int] | None
    description: str
    confidence: float
    reason: str


class ClaudeTargetSelectionError(RuntimeError):
    pass


class ClaudeTargetSelector:
    """Use one Claude vision call to select a benign, brightly coloured object."""

    def __init__(
        self,
        *,
        client: Any,
        model: str = DEFAULT_CLAUDE_MODEL,
        maximum_image_edge_px: int = 1280,
    ) -> None:
        self.client = client
        self.model = model
        self.maximum_image_edge_px = maximum_image_edge_px

    @classmethod
    def from_environment(cls) -> ClaudeTargetSelector:
        api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise ClaudeTargetSelectionError(
                "ANTHROPIC_API_KEY is not set; use a saved detector profile instead"
            )
        model = os.getenv("ANTHROPIC_MODEL", DEFAULT_CLAUDE_MODEL).strip()
        try:
            timeout_s = float(os.getenv("ANTHROPIC_TIMEOUT_S", "8.0"))
        except ValueError as error:
            raise ClaudeTargetSelectionError(
                "ANTHROPIC_TIMEOUT_S must be numeric"
            ) from error
        if timeout_s <= 0:
            raise ClaudeTargetSelectionError(
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

    def select(self, frame_bgr: np.ndarray) -> TargetSelection:
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("frame_bgr must have shape (height, width, 3)")
        image, scale = _resize_for_claude(
            frame_bgr,
            maximum_edge=self.maximum_image_edge_px,
        )
        ok, encoded = cv2.imencode(
            ".jpg",
            image,
            [int(cv2.IMWRITE_JPEG_QUALITY), 88],
        )
        if not ok:
            raise ClaudeTargetSelectionError("could not encode commissioning image")

        height, width = image.shape[:2]
        response = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            thinking={"type": "disabled"},
            system=(
                "You commission a benign hackathon demonstration that tracks a "
                "hand-thrown foam ball. Select only one safe, inanimate, brightly "
                "coloured ball or similar lightweight demo object. Never select a "
                "person, animal, body part, vehicle, weapon, or hazardous object. "
                "Return the tight pixel bounding box using the required tool."
            ),
            tools=[_target_tool()],
            tool_choice={
                "type": "tool",
                "name": TARGET_TOOL_NAME,
                "disable_parallel_tool_use": True,
            },
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": base64.b64encode(encoded).decode("ascii"),
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                f"The image is {width} by {height} pixels. Select "
                                "the single best safe tracking target. If there is "
                                "no suitable object, set found to false and all "
                                "box coordinates to zero."
                            ),
                        },
                    ],
                }
            ],
        )
        tool_input = _extract_tool_input(response)
        return _selection_from_tool(
            tool_input,
            image_width=width,
            image_height=height,
            inverse_scale=1.0 / scale,
            original_shape=frame_bgr.shape,
        )


class ClaudeCommissionedDetector(BallDetector):
    """A normal local detector after a one-time Claude ROI selection."""

    def __init__(
        self,
        selector: ClaudeTargetSelector,
        *,
        fallback: BallDetectorConfig | None = None,
    ) -> None:
        super().__init__(fallback)
        self.selector = selector
        self.commissioned = False
        self.selection: TargetSelection | None = None
        self.commissioning_error: str | None = None

    def detect(self, frame_bgr: np.ndarray, *, timestamp_s: float) -> Detection | None:
        if not self.commissioned:
            self._commission(frame_bgr)
        return super().detect(frame_bgr, timestamp_s=timestamp_s)

    def _commission(self, frame_bgr: np.ndarray) -> None:
        self.commissioned = True
        try:
            selection = self.selector.select(frame_bgr)
            self.selection = selection
            if selection.found and selection.roi is not None:
                self.config = calibrate_ball_colour(
                    frame_bgr,
                    selection.roi,
                ).config
        except (
            APIError,
            ClaudeTargetSelectionError,
            ValueError,
            OSError,
        ) as error:
            self.commissioning_error = str(error)


def _target_tool() -> dict[str, Any]:
    return {
        "name": TARGET_TOOL_NAME,
        "description": (
            "Select the tight bounding box of one safe, brightly coloured foam "
            "ball or similar lightweight inanimate demo object."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "found": {"type": "boolean"},
                "x": {"type": "integer", "minimum": 0},
                "y": {"type": "integer", "minimum": 0},
                "width": {"type": "integer", "minimum": 0},
                "height": {"type": "integer", "minimum": 0},
                "description": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "reason": {"type": "string"},
            },
            "required": [
                "found",
                "x",
                "y",
                "width",
                "height",
                "description",
                "confidence",
                "reason",
            ],
            "additionalProperties": False,
        },
    }


def _extract_tool_input(response: Any) -> dict[str, Any]:
    for block in response.content:
        if getattr(block, "type", None) == "tool_use":
            if getattr(block, "name", None) != TARGET_TOOL_NAME:
                continue
            value = getattr(block, "input", None)
            if isinstance(value, dict):
                return value
    stop_reason = getattr(response, "stop_reason", "unknown")
    raise ClaudeTargetSelectionError(
        f"Claude did not return {TARGET_TOOL_NAME}; stop reason: {stop_reason}"
    )


def _selection_from_tool(
    value: dict[str, Any],
    *,
    image_width: int,
    image_height: int,
    inverse_scale: float,
    original_shape: tuple[int, ...],
) -> TargetSelection:
    found = bool(value.get("found", False))
    description = str(value.get("description", "")).strip()
    reason = str(value.get("reason", "")).strip()
    confidence = float(np.clip(float(value.get("confidence", 0.0)), 0.0, 1.0))
    if not found:
        return TargetSelection(
            found=False,
            roi=None,
            description=description,
            confidence=confidence,
            reason=reason or "no suitable safe target",
        )

    try:
        x = int(value["x"])
        y = int(value["y"])
        width = int(value["width"])
        height = int(value["height"])
    except (KeyError, TypeError, ValueError) as error:
        raise ClaudeTargetSelectionError("Claude returned an invalid target box") from error
    if width <= 0 or height <= 0:
        raise ClaudeTargetSelectionError("Claude returned an empty target box")
    if x < 0 or y < 0 or x + width > image_width or y + height > image_height:
        raise ClaudeTargetSelectionError("Claude target box lies outside the image")

    original_height, original_width = original_shape[:2]
    scaled_x = int(np.clip(round(x * inverse_scale), 0, original_width - 1))
    scaled_y = int(np.clip(round(y * inverse_scale), 0, original_height - 1))
    scaled_width = int(
        np.clip(round(width * inverse_scale), 1, original_width - scaled_x)
    )
    scaled_height = int(
        np.clip(round(height * inverse_scale), 1, original_height - scaled_y)
    )
    return TargetSelection(
        found=True,
        roi=(scaled_x, scaled_y, scaled_width, scaled_height),
        description=description,
        confidence=confidence,
        reason=reason,
    )


def _resize_for_claude(
    frame_bgr: np.ndarray,
    *,
    maximum_edge: int,
) -> tuple[np.ndarray, float]:
    height, width = frame_bgr.shape[:2]
    scale = min(1.0, maximum_edge / max(width, height))
    if scale == 1.0:
        return frame_bgr, 1.0
    resized = cv2.resize(
        frame_bgr,
        (round(width * scale), round(height * scale)),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale
