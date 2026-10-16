"""Local face identity tracking using OpenCV YuNet and SFace."""

from __future__ import annotations

import json
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from preact.models import Detection


@dataclass(frozen=True, slots=True)
class _FaceAnalysis:
    frame_bgr: np.ndarray
    detection: Detection | None
    feature: np.ndarray | None
    similarity: float | None


@dataclass(frozen=True, slots=True)
class FaceDetectorConfig:
    detection_model_path: str
    recognition_model_path: str
    reference_image_paths: tuple[str, ...]
    head_reference_image_paths: tuple[str, ...] = ()
    detection_score_threshold: float = 0.60
    identity_cosine_threshold: float = 0.36
    live_bootstrap_minimum_similarity: float = 0.42
    live_bootstrap_frames: int = 8
    live_bootstrap_directory: str = "work/face-live-bootstrap"
    live_profile_frames: int = 10
    live_profile_directory: str = "work/person-live-profile"
    live_profile_capture_seconds: float = 15.0
    live_profile_sample_interval_seconds: float = 0.8
    body_tracking_timeout_seconds: float = 6.0
    head_motion_bridge_seconds: float = 0.25
    body_identity_histogram_threshold: float = 0.50
    face_analysis_interval_seconds: float = 0.10
    body_tracker_refresh_seconds: float = 1.0
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
            head_reference_image_paths=tuple(
                resolved(item)
                for item in value.get("head_reference_image_paths", [])
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
            live_profile_frames=int(value.get("live_profile_frames", 10)),
            live_profile_directory=resolved(
                value.get("live_profile_directory", "../work/person-live-profile")
            ),
            live_profile_capture_seconds=float(
                value.get("live_profile_capture_seconds", 15.0)
            ),
            live_profile_sample_interval_seconds=float(
                value.get("live_profile_sample_interval_seconds", 0.8)
            ),
            body_tracking_timeout_seconds=float(
                value.get("body_tracking_timeout_seconds", 6.0)
            ),
            head_motion_bridge_seconds=float(
                value.get("head_motion_bridge_seconds", 0.25)
            ),
            body_identity_histogram_threshold=float(
                value.get("body_identity_histogram_threshold", 0.50)
            ),
            face_analysis_interval_seconds=float(
                value.get("face_analysis_interval_seconds", 0.10)
            ),
            body_tracker_refresh_seconds=float(
                value.get("body_tracker_refresh_seconds", 1.0)
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
        # These are deliberately separate from face embeddings.  A rear or
        # profile view does not reliably contain a detectable face, but it is
        # useful to verify the short-lived visual head lock after a face match.
        self._reference_head_histograms = self._load_reference_head_histograms()
        self._live_features: list[np.ndarray] = []
        self._bootstrap_count = 0
        self._last_detection: Detection | None = None
        self._state = "searching"
        self._profile_capture_started_at_s: float | None = None
        self._profile_capture_last_at_s: float | None = None
        self._profile_capture_count = 0
        self._profile_capture_session: Path | None = None
        self._profile_capture_histograms: list[np.ndarray] = []
        self._profile_capture_state = "waiting_for_turn"
        self._body_tracker: object | None = None
        self._last_face_seen_at_s: float | None = None
        self._last_body_tracker_refresh_at_s: float | None = None
        self._last_head_measurement: Detection | None = None
        self._head_velocity_px_s = (0.0, 0.0)
        self._analysis_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="preact-face-analysis",
        )
        # Disk writes are isolated from the camera loop.  Collecting ten
        # reference views must not cause a visible pause in the live preview.
        self._profile_writer = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="preact-profile-writer",
        )
        self._analysis_future: Future[_FaceAnalysis] | None = None
        self._last_analysis_submitted_at_s: float | None = None
        self._initial_analysis_completed = False

    @property
    def state(self) -> str:
        return self._state

    def profile_capture_status(self) -> dict[str, object]:
        """Status for the automatic local appearance-reference collection."""
        return {
            "state": self._profile_capture_state,
            "captured": self._profile_capture_count,
            "required": self.config.live_profile_frames,
        }

    def reset(self) -> None:
        self._last_detection = None
        self._live_features.clear()
        self._bootstrap_count = 0
        self._state = "searching"
        self._profile_capture_started_at_s = None
        self._profile_capture_last_at_s = None
        self._profile_capture_count = 0
        self._profile_capture_session = None
        self._profile_capture_histograms.clear()
        self._profile_capture_state = "waiting_for_turn"
        self._body_tracker = None
        self._last_face_seen_at_s = None
        self._last_body_tracker_refresh_at_s = None
        self._last_head_measurement = None
        self._head_velocity_px_s = (0.0, 0.0)
        future = self._analysis_future
        self._analysis_future = None
        if future is not None:
            future.cancel()
        self._last_analysis_submitted_at_s = None
        self._initial_analysis_completed = False

    def close(self) -> None:
        self._analysis_executor.shutdown(wait=False, cancel_futures=True)
        self._profile_writer.shutdown(wait=False, cancel_futures=True)

    def detect(self, frame_bgr: np.ndarray, *, timestamp_s: float) -> Detection | None:
        analysis = self._finished_analysis()
        if analysis is not None and analysis.detection is not None:
            assert analysis.feature is not None
            assert analysis.similarity is not None
            detection = replace(analysis.detection, timestamp_s=timestamp_s)
            self._last_detection = detection
            self._state = "locked"
            self._last_face_seen_at_s = timestamp_s
            self._arm_head_tracker(analysis.frame_bgr, detection, timestamp_s)
            self._bootstrap(
                analysis.frame_bgr,
                detection,
                analysis.feature,
                analysis.similarity,
            )
        elif analysis is not None:
            body_detection = self._detect_locked_body(frame_bgr, timestamp_s)
            if body_detection is not None:
                self._last_detection = self._predict_head_position(body_detection)
                self._state = "locked"
                self._begin_back_head_capture(timestamp_s)
                self._capture_profile(
                    frame_bgr,
                    self._last_detection,
                    timestamp_s,
                    head_crop=True,
                )
                return self._last_detection
            self._state = "lost" if self._last_detection is not None else "searching"
            self._last_detection = None
        elif self._last_detection is not None:
            # The lightweight tracker updates every camera frame while YuNet
            # and SFace are busy in their background worker.
            head_detection = self._detect_locked_body(frame_bgr, timestamp_s)
            if head_detection is not None:
                self._last_detection = self._predict_head_position(head_detection)
                self._state = "locked"
                self._capture_profile(
                    frame_bgr,
                    self._last_detection,
                    timestamp_s,
                    head_crop=True,
                )
        self._submit_analysis_if_due(frame_bgr, timestamp_s)
        # Preserve the old immediate-lock behaviour for the first frame only.
        # Every subsequent analysis is asynchronous, so live video never waits.
        if not self._initial_analysis_completed and self._analysis_future is not None:
            self._analysis_future.result()
            return self.detect(frame_bgr, timestamp_s=timestamp_s)
        if self._last_detection is None:
            return None
        return replace(self._last_detection, timestamp_s=timestamp_s)

    def _finished_analysis(self) -> _FaceAnalysis | None:
        future = self._analysis_future
        if future is None or not future.done():
            return None
        self._analysis_future = None
        self._initial_analysis_completed = True
        try:
            return future.result()
        except cv2.error:
            return _FaceAnalysis(np.empty((0, 0, 3), dtype=np.uint8), None, None, None)

    def _submit_analysis_if_due(self, frame_bgr: np.ndarray, timestamp_s: float) -> None:
        if self._analysis_future is not None:
            return
        if (
            self._last_analysis_submitted_at_s is not None
            and timestamp_s - self._last_analysis_submitted_at_s
            < self.config.face_analysis_interval_seconds
        ):
            return
        references = tuple((*self._reference_features, *self._live_features))
        self._analysis_future = self._analysis_executor.submit(
            self._analyse_face,
            frame_bgr.copy(),
            timestamp_s,
            references,
        )
        self._last_analysis_submitted_at_s = timestamp_s

    def _analyse_face(
        self,
        frame_bgr: np.ndarray,
        timestamp_s: float,
        references: tuple[np.ndarray, ...],
    ) -> _FaceAnalysis:
        scaled, scale = self._scaled_frame(frame_bgr)
        faces = self._faces(scaled)
        best: tuple[float, np.ndarray, np.ndarray] | None = None
        for face in faces:
            feature = self._feature(scaled, face)
            similarity = max(self._cosine(feature, value) for value in references)
            if best is None or similarity > best[0]:
                best = (similarity, face, feature)
        if best is None or best[0] < self.config.identity_cosine_threshold:
            return _FaceAnalysis(frame_bgr, None, None, None)
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
        detection = Detection(
            timestamp_s=timestamp_s,
            x_px=x + 0.5 * width,
            y_px=y + 0.5 * height,
            radius_px=0.5 * float(np.hypot(width, height)),
            area_px2=width * height,
            quality=float(np.clip(0.55 * detector_score + 0.45 * identity_quality, 0, 1)),
            bbox_x_px=x,
            bbox_y_px=y,
            bbox_width_px=width,
            bbox_height_px=height,
        )
        return _FaceAnalysis(frame_bgr, detection, feature, similarity)

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

    def _load_reference_head_histograms(self) -> list[np.ndarray]:
        """Load manually commissioned head views for the turn-away tracker.

        The supplied back-of-head image has no face for YuNet to find, so a
        face detector alone cannot prepare this part of the local profile.
        Every image is cropped to its head region before deriving a compact
        HSV appearance histogram; backgrounds and clothing are excluded.
        """
        output: list[np.ndarray] = []
        for raw_path in self.config.head_reference_image_paths:
            image = cv2.imread(raw_path)
            if image is None:
                continue
            crop = self._reference_head_crop(image, raw_path)
            if crop.size:
                output.append(self._appearance_histogram(crop))
        return output

    @staticmethod
    def _reference_head_crop(image: np.ndarray, raw_path: str) -> np.ndarray:
        """Crop the commissioned reference photos around the head only."""
        height, width = image.shape[:2]
        name = Path(raw_path).stem.lower()
        if name == "side-left":
            # Subject is on the right half of this supplied profile photo.
            return image[0 : round(height * 0.66), round(width * 0.34) : width]
        if name == "side-right":
            # Subject is on the left half of this supplied profile photo.
            return image[0 : round(height * 0.66), 0 : round(width * 0.66)]
        # The rear-view reference is already a close, centred head portrait.
        return image[
            round(height * 0.02) : round(height * 0.94),
            round(width * 0.04) : round(width * 0.96),
        ]

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

    def _capture_profile(
        self,
        frame_bgr: np.ndarray,
        detection: Detection,
        timestamp_s: float,
        head_crop: bool = False,
    ) -> None:
        """Save back-of-head views only after a continuous turn-away lock.

        These are deliberately captured only after the local face match.  This
        prevents an unrelated person in the frame from being added to the
        user's local appearance profile.
        """
        if self.config.live_profile_frames <= 0:
            return
        if self._profile_capture_state != "capturing_back_head":
            return
        assert self._profile_capture_started_at_s is not None
        if (
            timestamp_s - self._profile_capture_started_at_s
            > self.config.live_profile_capture_seconds
        ):
            self._profile_capture_state = "complete"
            return
        if self._profile_capture_count >= self.config.live_profile_frames:
            return
        if (
            self._profile_capture_last_at_s is not None
            and timestamp_s - self._profile_capture_last_at_s
            < self.config.live_profile_sample_interval_seconds
        ):
            return
        crop = (
            self._bbox_crop(frame_bgr, detection)
            if head_crop
            else self._head_crop(frame_bgr, detection)
        )
        if crop.size == 0:
            return
        histogram = self._appearance_histogram(crop)
        self._profile_capture_histograms.append(histogram)
        self._profile_capture_count += 1
        self._profile_capture_last_at_s = timestamp_s
        assert self._profile_capture_session is not None
        filename = f"appearance-{self._profile_capture_count:02d}.jpg"
        self._profile_writer.submit(
            cv2.imwrite,
            str(self._profile_capture_session / filename),
            crop.copy(),
        )

    def _begin_back_head_capture(self, timestamp_s: float) -> None:
        """Commission back-of-head views only from an uninterrupted face lock."""
        if self._profile_capture_state != "waiting_for_turn":
            return
        self._profile_capture_started_at_s = timestamp_s
        self._profile_capture_last_at_s = None
        self._profile_capture_session = (
            Path(self.config.live_profile_directory)
            / time.strftime("%Y%m%d-%H%M%S", time.localtime())
        )
        self._profile_capture_session.mkdir(parents=True, exist_ok=True)
        self._profile_capture_state = "capturing_back_head"

    @staticmethod
    def _head_crop(frame_bgr: np.ndarray, detection: Detection) -> np.ndarray:
        """Keep the target on the head rather than the torso or shoulders."""
        face_width = detection.bbox_width_px or 0.0
        face_height = detection.bbox_height_px or 0.0
        if face_width <= 0 or face_height <= 0:
            return frame_bgr[0:0, 0:0]
        center_x = (detection.bbox_x_px or 0.0) + face_width * 0.5
        left = max(0, round(center_x - face_width * 0.7))
        right = min(frame_bgr.shape[1], round(center_x + face_width * 0.7))
        top = max(0, round((detection.bbox_y_px or 0.0) - face_height * 0.25))
        bottom = min(frame_bgr.shape[0], round((detection.bbox_y_px or 0.0) + face_height * 1.3))
        return frame_bgr[top:bottom, left:right]

    @staticmethod
    def _appearance_histogram(crop_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        histogram = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
        return cv2.normalize(histogram, histogram).reshape(-1)

    def _arm_head_tracker(
        self,
        frame_bgr: np.ndarray,
        detection: Detection,
        timestamp_s: float,
    ) -> None:
        """Start a short-lived visual lock around the identified head."""
        if not hasattr(cv2, "TrackerCSRT_create"):
            return
        if (
            self._body_tracker is not None
            and self._last_body_tracker_refresh_at_s is not None
            and timestamp_s - self._last_body_tracker_refresh_at_s
            < self.config.body_tracker_refresh_seconds
        ):
            return
        crop_box = self._head_box(frame_bgr, detection)
        if crop_box is None:
            return
        try:
            tracker = cv2.TrackerCSRT_create()
            tracker.init(frame_bgr, crop_box)
        except cv2.error:
            return
        self._body_tracker = tracker
        self._last_body_tracker_refresh_at_s = timestamp_s

    def _detect_locked_body(
        self,
        frame_bgr: np.ndarray,
        timestamp_s: float,
    ) -> Detection | None:
        """Continue a recent face lock through a turn-away, never search blindly."""
        if (
            self._body_tracker is None
            or self._last_face_seen_at_s is None
            or timestamp_s - self._last_face_seen_at_s
            > self.config.body_tracking_timeout_seconds
        ):
            return None
        try:
            found, box = self._body_tracker.update(frame_bgr)  # type: ignore[union-attr]
        except cv2.error:
            return self._motion_bridge_detection(timestamp_s)
        if not found:
            return self._motion_bridge_detection(timestamp_s)
        x, y, width, height = (float(value) for value in box)
        detection = Detection(
            timestamp_s=timestamp_s,
            x_px=x + width * 0.5,
            y_px=y + height * 0.5,
            radius_px=0.5 * float(np.hypot(width, height)),
            area_px2=width * height,
            quality=0.55,
            bbox_x_px=x,
            bbox_y_px=y,
            bbox_width_px=width,
            bbox_height_px=height,
        )
        crop = self._bbox_crop(frame_bgr, detection)
        if crop.size == 0:
            return None
        # A recently verified face is the stronger identity proof.  Do not
        # compare its tracked crop with rear/profile reference images between
        # face-analysis passes: that makes a frontal lock flicker and feels
        # slow even though the lightweight CSRT tracker has a valid lock.
        recent_face_lock_s = max(
            0.9,
            self.config.face_analysis_interval_seconds * 2.5,
        )
        if timestamp_s - self._last_face_seen_at_s <= recent_face_lock_s:
            return detection
        # Once the face has genuinely been absent for a short period, compare
        # the continued head lock against the commissioned rear/profile views.
        references = (
            *self._reference_head_histograms,
            *self._profile_capture_histograms,
        )
        if not references:
            return detection
        similarity = max(
            cv2.compareHist(
                self._appearance_histogram(crop), reference, cv2.HISTCMP_CORREL
            )
            for reference in references
        )
        if (
            similarity < self.config.body_identity_histogram_threshold
            and self._profile_capture_state != "capturing_back_head"
        ):
            return None
        return detection

    def _motion_bridge_detection(self, timestamp_s: float) -> Detection | None:
        """Carry a verified head trajectory through a very brief tracker gap.

        This is intentionally capped at a quarter second: it absorbs a fast
        turn or single blurred frame, without letting the target drift toward
        an unrelated object when visual tracking has genuinely ended.
        """
        previous = self._last_head_measurement
        if previous is None:
            return None
        elapsed = timestamp_s - previous.timestamp_s
        if elapsed <= 0.0 or elapsed > self.config.head_motion_bridge_seconds:
            return None
        velocity_x, velocity_y = self._head_velocity_px_s
        shift_x = velocity_x * elapsed
        shift_y = velocity_y * elapsed
        return replace(
            previous,
            timestamp_s=timestamp_s,
            x_px=previous.x_px + shift_x,
            y_px=previous.y_px + shift_y,
            bbox_x_px=(previous.bbox_x_px or 0.0) + shift_x,
            bbox_y_px=(previous.bbox_y_px or 0.0) + shift_y,
            quality=min(previous.quality, 0.45),
        )

    def _predict_head_position(self, detection: Detection) -> Detection:
        """Alpha-beta prediction makes the head marker lead fast motion slightly."""
        previous = self._last_head_measurement
        if previous is None:
            self._last_head_measurement = detection
            return detection
        elapsed = detection.timestamp_s - previous.timestamp_s
        if elapsed <= 0.0 or elapsed > 0.5:
            self._last_head_measurement = detection
            return detection
        measured_x = (detection.x_px - previous.x_px) / elapsed
        measured_y = (detection.y_px - previous.y_px) / elapsed
        old_x, old_y = self._head_velocity_px_s
        velocity_x = 0.55 * old_x + 0.45 * measured_x
        velocity_y = 0.55 * old_y + 0.45 * measured_y
        self._head_velocity_px_s = (velocity_x, velocity_y)
        self._last_head_measurement = detection
        lead_s = 0.055
        shift_x = velocity_x * lead_s
        shift_y = velocity_y * lead_s
        return replace(
            detection,
            x_px=detection.x_px + shift_x,
            y_px=detection.y_px + shift_y,
            bbox_x_px=(detection.bbox_x_px or 0.0) + shift_x,
            bbox_y_px=(detection.bbox_y_px or 0.0) + shift_y,
        )

    @classmethod
    def _head_box(
        cls,
        frame_bgr: np.ndarray,
        detection: Detection,
    ) -> tuple[int, int, int, int] | None:
        crop = cls._head_crop(frame_bgr, detection)
        if crop.size == 0:
            return None
        face_width = detection.bbox_width_px or 0.0
        face_height = detection.bbox_height_px or 0.0
        center_x = (detection.bbox_x_px or 0.0) + face_width * 0.5
        left = max(0, round(center_x - face_width * 0.7))
        top = max(0, round((detection.bbox_y_px or 0.0) - face_height * 0.25))
        return (left, top, crop.shape[1], crop.shape[0])

    @staticmethod
    def _bbox_crop(frame_bgr: np.ndarray, detection: Detection) -> np.ndarray:
        x = max(0, round(detection.bbox_x_px or 0))
        y = max(0, round(detection.bbox_y_px or 0))
        right = min(frame_bgr.shape[1], round(x + (detection.bbox_width_px or 0)))
        bottom = min(frame_bgr.shape[0], round(y + (detection.bbox_height_px or 0)))
        return frame_bgr[y:bottom, x:right]
