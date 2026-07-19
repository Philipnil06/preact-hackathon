"""Local face identity tracking using OpenCV YuNet and SFace."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from preact.models import Detection


@dataclass(frozen=True, slots=True)
class FaceDetectorConfig:
    detection_model_path: str
    recognition_model_path: str
    reference_image_paths: tuple[str, ...]
    detection_score_threshold: float = 0.60
    identity_cosine_threshold: float = 0.36
    live_bootstrap_minimum_similarity: float = 0.42
    live_bootstrap_frames: int = 8
    live_bootstrap_directory: str = "work/face-live-bootstrap"
    maximum_reference_dimension_px: int = 640

    @classmethod
    def load(cls, path: str | Path) -> FaceDetectorConfig:
        profile_path = Path(path)
        value = json.loads(profile_path.read_text(encoding="utf-8"))

        def resolved(raw: str) -> str:
            candidate = Path(raw)
            if candidate.is_absolute():
                return str(candidate)
            return str((profile_path.parent / candidate).resolve())

        return cls(
            detection_model_path=resolved(value["detection_model_path"]),
            recognition_model_path=resolved(value["recognition_model_path"]),
            reference_image_paths=tuple(
                resolved(item) for item in value["reference_image_paths"]
            ),
            detection_score_threshold=float(
                value.get("detection_score_threshold", 0.60)
            ),
            identity_cosine_threshold=float(
                value.get("identity_cosine_threshold", 0.36)
            ),
            live_bootstrap_minimum_similarity=float(
                value.get("live_bootstrap_minimum_similarity", 0.42)
            ),
            live_bootstrap_frames=int(value.get("live_bootstrap_frames", 8)),
            live_bootstrap_directory=resolved(
                value.get("live_bootstrap_directory", "../work/face-live-bootstrap")
            ),
            maximum_reference_dimension_px=int(
                value.get("maximum_reference_dimension_px", 640)
            ),
        )


class FaceDetector:
    """Detect faces and accept only the commissioned local identity."""

    def __init__(self, config: FaceDetectorConfig) -> None:
        self.config = config
        self._detector = cv2.FaceDetectorYN.create(
            config.detection_model_path,
            "",
            (320, 320),
            config.detection_score_threshold,
            0.3,
            5000,
        )
        self._recognizer = cv2.FaceRecognizerSF.create(
            config.recognition_model_path,
            "",
        )
        self._reference_features = self._load_reference_features()
        if not self._reference_features:
            raise ValueError("face profile did not yield any detectable reference faces")
        self._live_features: list[np.ndarray] = []
        self._bootstrap_count = 0
        self._last_detection: Detection | None = None
        self._state = "searching"

    @property
    def state(self) -> str:
        return self._state

    def reset(self) -> None:
        self._last_detection = None
        self._live_features.clear()
        self._bootstrap_count = 0
        self._state = "searching"

    def detect(self, frame_bgr: np.ndarray, *, timestamp_s: float) -> Detection | None:
        scaled, scale = self._scaled_frame(frame_bgr)
        faces = self._faces(scaled)
        best: tuple[float, np.ndarray, np.ndarray] | None = None
        references = (*self._reference_features, *self._live_features)
        for face in faces:
            feature = self._feature(scaled, face)
            similarity = max(self._cosine(feature, value) for value in references)
            if best is None or similarity > best[0]:
                best = (similarity, face, feature)
        if best is None or best[0] < self.config.identity_cosine_threshold:
            self._state = "lost" if self._last_detection is not None else "searching"
            self._last_detection = None
            return None

        similarity, face, feature = best
        x, y, width, height = (float(value) / scale for value in face[:4])
        detector_score = float(face[-1])
        identity_quality = float(
            np.clip(
                (similarity - self.config.identity_cosine_threshold)
                / max(1.0 - self.config.identity_cosine_threshold, 1e-6),
                0.0,
                1.0,
            )
        )
        quality = float(np.clip(0.55 * detector_score + 0.45 * identity_quality, 0, 1))
        detection = Detection(
            timestamp_s=timestamp_s,
            x_px=x + 0.5 * width,
            y_px=y + 0.5 * height,
            radius_px=0.5 * float(np.hypot(width, height)),
            area_px2=width * height,
            quality=quality,
            bbox_x_px=x,
            bbox_y_px=y,
            bbox_width_px=width,
            bbox_height_px=height,
        )
        self._last_detection = detection
        self._state = "locked"
        self._bootstrap(frame_bgr, detection, feature, similarity)
        return detection

    def _load_reference_features(self) -> list[np.ndarray]:
        output: list[np.ndarray] = []
        for raw_path in self.config.reference_image_paths:
            image = cv2.imread(raw_path)
            if image is None:
                continue
            scaled, _ = self._scaled_frame(image)
            faces = self._faces(scaled)
            if not faces:
                continue
            face = max(faces, key=lambda value: float(value[2] * value[3]))
            output.append(self._feature(scaled, face))
        return output

    def _scaled_frame(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        height, width = image.shape[:2]
        maximum = max(height, width)
        limit = self.config.maximum_reference_dimension_px
        scale = min(1.0, limit / max(maximum, 1))
        if scale >= 1.0:
            return image, 1.0
        return (
            cv2.resize(
                image,
                (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            ),
            scale,
        )

    def _faces(self, image: np.ndarray) -> list[np.ndarray]:
        height, width = image.shape[:2]
        self._detector.setInputSize((width, height))
        _, faces = self._detector.detect(image)
        return [] if faces is None else [value for value in faces]

    def _feature(self, image: np.ndarray, face: np.ndarray) -> np.ndarray:
        aligned = self._recognizer.alignCrop(image, face)
        return self._recognizer.feature(aligned)

    def _cosine(self, first: np.ndarray, second: np.ndarray) -> float:
        return float(
            self._recognizer.match(
                first,
                second,
                cv2.FaceRecognizerSF_FR_COSINE,
            )
        )

    def _bootstrap(
        self,
        frame_bgr: np.ndarray,
        detection: Detection,
        feature: np.ndarray,
        similarity: float,
    ) -> None:
        if (
            self._bootstrap_count >= self.config.live_bootstrap_frames
            or similarity < self.config.live_bootstrap_minimum_similarity
        ):
            return
        self._live_features.append(feature)
        self._bootstrap_count += 1
        directory = Path(self.config.live_bootstrap_directory)
        directory.mkdir(parents=True, exist_ok=True)
        x = max(0, round(detection.bbox_x_px or 0))
        y = max(0, round(detection.bbox_y_px or 0))
        right = min(frame_bgr.shape[1], round(x + (detection.bbox_width_px or 0)))
        bottom = min(frame_bgr.shape[0], round(y + (detection.bbox_height_px or 0)))
        crop = frame_bgr[y:bottom, x:right]
        if crop.size:
            cv2.imwrite(str(directory / f"live-{self._bootstrap_count:02d}.jpg"), crop)
