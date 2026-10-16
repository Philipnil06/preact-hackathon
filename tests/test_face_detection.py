from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from preact.face_detection import FaceDetector, FaceDetectorConfig


def _local_profile(root: Path) -> Path:
    profile = root / "config/my-face.json"
    if not profile.exists():
        pytest.skip("local biometric commissioning profile is not installed")
    return profile


def test_commissioned_face_profile_accepts_all_five_reference_views() -> None:
    root = Path(__file__).resolve().parents[1]
    config = FaceDetectorConfig.load(_local_profile(root))
    detector = FaceDetector(replace(config, live_bootstrap_frames=0))

    for index in range(1, 6):
        frame = cv2.imread(
            str(root / f"config/face-reference/reference-{index:02d}.jpeg")
        )
        assert frame is not None
        detection = detector.detect(frame, timestamp_s=float(index))
        assert detection is not None
        assert detection.bbox_width_px is not None
        assert detection.bbox_height_px is not None


def test_commissioned_face_profile_rejects_an_empty_frame() -> None:
    root = Path(__file__).resolve().parents[1]
    config = FaceDetectorConfig.load(_local_profile(root))
    detector = FaceDetector(replace(config, live_bootstrap_frames=0))

    assert (
        detector.detect(
            np.zeros((480, 640, 3), dtype=np.uint8),
            timestamp_s=1.0,
        )
        is None
    )
