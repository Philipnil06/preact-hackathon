from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np

from preact.detection import BallDetector, BallDetectorConfig, HSVRange, detection_box


def test_detects_orange_ball_and_builds_bounding_box() -> None:
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.circle(frame, (320, 170), 22, (0, 140, 255), -1, cv2.LINE_AA)

    detection = BallDetector().detect(frame, timestamp_s=1.25)

    assert detection is not None
    assert abs(detection.x_px - 320) < 2
    assert abs(detection.y_px - 170) < 2
    assert detection.radius_px > 20
    x0, y0, x1, y1 = detection_box(detection, frame.shape)
    assert x0 < 320 < x1
    assert y0 < 170 < y1


def test_rejects_non_orange_background() -> None:
    frame = np.full((240, 320, 3), (255, 0, 0), dtype=np.uint8)
    assert BallDetector().detect(frame, timestamp_s=0.0) is None


def test_red_can_profile_rejects_skin_and_large_red_body_distractors() -> None:
    detector = BallDetector(_red_can_config(reacquire_confirmation_frames=1))
    frame = np.full((480, 640, 3), (32, 34, 38), dtype=np.uint8)

    # Red-biased skin remains inside a broad red hue range, but lacks the can's
    # saturation and channel separation.
    skin_bgr = (82, 112, 190)
    cv2.ellipse(frame, (485, 120), (64, 86), 0, 0, 360, skin_bgr, -1)
    cv2.rectangle(frame, (430, 205), (560, 465), skin_bgr, -1)

    # A large saturated-red surface should not steal the target either.
    cv2.rectangle(frame, (0, 90), (150, 390), (15, 15, 225), -1)

    can_centre = (310, 265)
    can_box = cv2.boxPoints(((float(can_centre[0]), float(can_centre[1])), (48, 126), 24))
    cv2.fillConvexPoly(frame, np.int32(can_box), (18, 20, 235), cv2.LINE_AA)

    detection = detector.detect(frame, timestamp_s=1.0)

    assert detection is not None
    assert abs(detection.x_px - can_centre[0]) < 3
    assert abs(detection.y_px - can_centre[1]) < 3
    x0, y0, x1, y1 = detection_box(detection, frame.shape)
    assert x0 < can_centre[0] < x1
    assert y0 < can_centre[1] < y1
    assert x1 - x0 < 180
    assert y1 - y0 < 180


def test_red_can_profile_returns_none_for_skin_without_can() -> None:
    detector = BallDetector(_red_can_config(reacquire_confirmation_frames=1))
    frame = np.full((480, 640, 3), (32, 34, 38), dtype=np.uint8)
    skin_bgr = (82, 112, 190)
    cv2.ellipse(frame, (330, 130), (75, 100), 0, 0, 360, skin_bgr, -1)
    arm = np.array([[130, 240], [205, 210], [360, 420], [285, 450]], np.int32)
    cv2.fillConvexPoly(frame, arm, skin_bgr)

    assert detector.detect(frame, timestamp_s=1.0) is None


def test_red_can_profile_rejects_dark_red_lookalike() -> None:
    detector = BallDetector(_red_can_config(reacquire_confirmation_frames=1))
    frame = np.zeros((300, 500, 3), dtype=np.uint8)
    dark_box = cv2.boxPoints(((250.0, 150.0), (48, 132), 12))
    cv2.fillConvexPoly(frame, np.int32(dark_box), (18, 22, 92), cv2.LINE_AA)

    assert detector.detect(frame, timestamp_s=1.0) is None


def test_temporal_lock_does_not_jump_to_larger_distant_candidate() -> None:
    detector = BallDetector(
        BallDetectorConfig(
            hsv=HSVRange(lower=(0, 120, 50), upper=(10, 255, 255)),
            minimum_area_px2=100.0,
            minimum_circularity=0.2,
            association_max_distance_px=80.0,
            lock_timeout_s=0.45,
        )
    )
    first = np.zeros((300, 640, 3), dtype=np.uint8)
    cv2.circle(first, (100, 150), 18, (10, 10, 235), -1)
    initial = detector.detect(first, timestamp_s=0.0)
    assert initial is not None

    second = np.zeros_like(first)
    cv2.circle(second, (124, 150), 18, (10, 10, 235), -1)
    cv2.circle(second, (510, 150), 52, (10, 10, 235), -1)
    associated = detector.detect(second, timestamp_s=0.1)
    assert associated is not None
    assert abs(associated.x_px - 124) < 2

    distant_only = np.zeros_like(first)
    cv2.circle(distant_only, (510, 150), 52, (10, 10, 235), -1)
    assert detector.detect(distant_only, timestamp_s=0.2) is None
    reacquired = detector.detect(distant_only, timestamp_s=0.7)
    assert reacquired is not None
    assert abs(reacquired.x_px - 510) < 2


def test_reacquisition_requires_configured_consecutive_frames() -> None:
    detector = BallDetector(_red_can_config(reacquire_confirmation_frames=3))
    frame = np.zeros((300, 500, 3), dtype=np.uint8)
    can_box = cv2.boxPoints(((240.0, 150.0), (42, 112), -17))
    cv2.fillConvexPoly(frame, np.int32(can_box), (15, 18, 235), cv2.LINE_AA)

    assert detector.detect(frame, timestamp_s=0.00) is None
    assert detector.detect(frame, timestamp_s=0.04) is None
    assert detector.detect(frame, timestamp_s=0.08) is not None


def test_identity_commissioning_requires_upright_can_and_ignores_lips() -> None:
    detector = BallDetector(_identity_can_config())
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    can_box = cv2.boxPoints(((150.0, 180.0), (46, 124), -8))
    cv2.fillConvexPoly(frame, np.int32(can_box), (15, 18, 235), cv2.LINE_AA)
    cv2.ellipse(frame, (455, 190), (62, 16), 0, 0, 360, (12, 15, 230), -1)

    for index in range(4):
        assert detector.detect(frame, timestamp_s=index / 30.0) is None
        assert detector.state == "acquiring"
    locked = detector.detect(frame, timestamp_s=4 / 30.0)

    assert locked is not None
    assert detector.state == "locked"
    assert abs(locked.x_px - 150) < 3
    assert abs(locked.y_px - 180) < 3


def test_identity_lock_rejects_wrong_size_body_and_lips_after_can_disappears() -> None:
    detector = BallDetector(_identity_can_config())
    can = np.zeros((360, 640, 3), dtype=np.uint8)
    can_box = cv2.boxPoints(((150.0, 180.0), (46, 124), 0))
    cv2.fillConvexPoly(can, np.int32(can_box), (15, 18, 235), cv2.LINE_AA)
    for index in range(5):
        detection = detector.detect(can, timestamp_s=index / 30.0)
    assert detection is not None

    distractors = np.zeros_like(can)
    cv2.rectangle(distractors, (400, 70), (478, 278), (15, 18, 235), -1)
    cv2.ellipse(distractors, (250, 220), (58, 15), 0, 0, 360, (15, 18, 235), -1)

    assert detector.detect(distractors, timestamp_s=1.0) is None
    assert detector.state == "lost"


def test_identity_lock_accepts_same_can_appearance_after_motion() -> None:
    detector = BallDetector(_identity_can_config())
    first = np.zeros((360, 640, 3), dtype=np.uint8)
    first_box = cv2.boxPoints(((150.0, 180.0), (46, 124), 0))
    cv2.fillConvexPoly(first, np.int32(first_box), (15, 18, 235), cv2.LINE_AA)
    for index in range(5):
        detection = detector.detect(first, timestamp_s=index / 30.0)
    assert detection is not None

    moved = np.zeros_like(first)
    moved_box = cv2.boxPoints(((230.0, 165.0), (46, 124), 12))
    cv2.fillConvexPoly(moved, np.int32(moved_box), (15, 18, 235), cv2.LINE_AA)
    moved_detection = detector.detect(moved, timestamp_s=0.2)

    assert moved_detection is not None
    assert detector.state == "locked"
    assert abs(moved_detection.x_px - 230) < 4


def test_green_square_profile_locks_square_and_rejects_human_colours() -> None:
    detector = BallDetector(_green_square_config())
    frame = np.full((360, 640, 3), (32, 34, 38), dtype=np.uint8)
    cv2.ellipse(frame, (465, 130), (72, 94), 0, 0, 360, (82, 112, 190), -1)
    cv2.ellipse(frame, (465, 165), (38, 10), 0, 0, 360, (20, 25, 210), -1)
    cv2.rectangle(frame, (360, 265), (575, 315), (25, 235, 30), -1)
    cv2.rectangle(frame, (120, 125), (195, 200), (35, 245, 40), -1)

    assert detector.detect(frame, timestamp_s=0.00) is None
    assert detector.detect(frame, timestamp_s=0.04) is None
    locked = detector.detect(frame, timestamp_s=0.08)

    assert locked is not None
    assert detector.state == "locked"
    assert abs(locked.x_px - 157.5) < 3
    assert abs(locked.y_px - 162.5) < 3

    without_square = np.full_like(frame, (32, 34, 38))
    cv2.ellipse(
        without_square,
        (465, 130),
        (72, 94),
        0,
        0,
        360,
        (82, 112, 190),
        -1,
    )
    cv2.ellipse(without_square, (465, 165), (38, 10), 0, 0, 360, (20, 25, 210), -1)
    assert detector.detect(without_square, timestamp_s=1.0) is None
    assert detector.state == "lost"


def test_multicolour_ball_requires_multiple_target_panels() -> None:
    config = BallDetectorConfig(
        hsv=HSVRange(lower=(0, 105, 70), upper=(12, 255, 255)),
        additional_hsv_ranges=(
            HSVRange(lower=(16, 55, 80), upper=(40, 255, 255)),
            HSVRange(lower=(40, 75, 45), upper=(92, 255, 255)),
            HSVRange(lower=(94, 75, 45), upper=(135, 255, 255)),
        ),
        minimum_area_px2=180.0,
        minimum_circularity=0.34,
        minimum_hsv_range_count=2,
        minimum_hsv_range_fraction=0.025,
        maximum_aspect_ratio=1.7,
        minimum_rectangularity=0.48,
    )
    detector = BallDetector(config)
    frame = np.full((300, 500, 3), (35, 35, 35), dtype=np.uint8)
    cv2.ellipse(frame, (140, 150), (55, 60), 0, 90, 270, (20, 20, 240), -1)
    cv2.ellipse(frame, (140, 150), (55, 60), 0, -90, 90, (240, 70, 25), -1)

    detected = detector.detect(frame, timestamp_s=0.0)

    assert detected is not None
    assert abs(detected.x_px - 140) < 4
    red_only = np.full_like(frame, (35, 35, 35))
    cv2.circle(red_only, (360, 150), 58, (20, 20, 240), -1)
    detector.reset()
    assert detector.detect(red_only, timestamp_s=0.1) is None


def test_blue_balloon_profile_locks_balloon_and_rejects_skin_and_pastels() -> None:
    detector = BallDetector(_blue_balloon_config())
    frame = np.full((360, 640, 3), (150, 155, 170), dtype=np.uint8)
    cv2.ellipse(frame, (210, 180), (70, 82), 0, 0, 360, (205, 155, 110), -1)
    cv2.ellipse(frame, (470, 150), (65, 88), 0, 0, 360, (105, 145, 195), -1)

    assert detector.detect(frame, timestamp_s=0.00) is None
    assert detector.detect(frame, timestamp_s=0.04) is None
    locked = detector.detect(frame, timestamp_s=0.08)

    assert locked is not None
    assert abs(locked.x_px - 210) < 4
    assert abs(locked.y_px - 180) < 4

    distractors = np.full_like(frame, (150, 155, 170))
    cv2.ellipse(
        distractors,
        (180, 180),
        (72, 84),
        0,
        0,
        360,
        (175, 190, 185),
        -1,
    )
    cv2.ellipse(
        distractors,
        (450, 180),
        (72, 84),
        0,
        0,
        360,
        (180, 170, 205),
        -1,
    )
    assert detector.detect(distractors, timestamp_s=1.0) is None
    assert detector.state == "lost"


def test_reference_image_preloads_identity_and_survives_reset(tmp_path) -> None:
    reference = np.full((240, 320, 3), (150, 155, 170), dtype=np.uint8)
    cv2.ellipse(reference, (160, 120), (58, 68), 0, 0, 360, (205, 155, 110), -1)
    reference_path = tmp_path / "blue-balloon.jpeg"
    assert cv2.imwrite(str(reference_path), reference)
    config = replace(
        _blue_balloon_config(),
        reference_image_path=str(reference_path),
        reference_image_bbox=(100, 50, 120, 140),
        reacquire_confirmation_frames=3,
    )
    detector = BallDetector(config)

    assert detector.state == "searching"
    blank = np.full_like(reference, (150, 155, 170))
    assert detector.detect(blank, timestamp_s=0.0) is None
    assert detector.state == "searching"
    assert detector.detect(reference, timestamp_s=0.1) is not None
    assert detector.state == "locked"

    detector.reset()
    assert detector.state == "searching"
    assert detector.detect(reference, timestamp_s=1.0) is not None
    assert detector.state == "locked"


def test_reference_target_bootstraps_live_and_coasts_through_short_loss(
    tmp_path,
) -> None:
    reference = np.full((300, 500, 3), (150, 155, 170), dtype=np.uint8)
    cv2.ellipse(reference, (100, 150), (45, 52), 0, 0, 360, (205, 155, 110), -1)
    reference_path = tmp_path / "moving-blue-balloon.jpeg"
    assert cv2.imwrite(str(reference_path), reference)
    config = replace(
        _blue_balloon_config(),
        reference_image_path=str(reference_path),
        reference_image_bbox=(50, 92, 100, 116),
        association_max_distance_px=250.0,
        live_bootstrap_frames=2,
        live_bootstrap_update_rate=0.5,
        prediction_coast_s=0.45,
        velocity_update_rate=1.0,
        visual_tracker="none",
    )
    detector = BallDetector(config)

    first = detector.detect(reference, timestamp_s=0.0)
    assert first is not None
    moved = np.full_like(reference, (150, 155, 170))
    cv2.ellipse(moved, (150, 150), (45, 52), 0, 0, 360, (205, 155, 110), -1)
    second = detector.detect(moved, timestamp_s=0.1)
    assert second is not None
    assert detector._live_bootstrap_remaining == 0

    blank = np.full_like(reference, (150, 155, 170))
    predicted = detector.detect(blank, timestamp_s=0.2)
    assert predicted is not None
    assert detector.state == "coasting"
    assert abs(predicted.x_px - 200) < 5

    reacquired_frame = np.full_like(reference, (150, 155, 170))
    cv2.ellipse(
        reacquired_frame,
        (225, 150),
        (45, 52),
        0,
        0,
        360,
        (205, 155, 110),
        -1,
    )
    reacquired = detector.detect(reacquired_frame, timestamp_s=0.25)
    assert reacquired is not None
    assert detector.state == "locked"
    assert abs(reacquired.x_px - 225) < 4

    assert detector.detect(blank, timestamp_s=0.8) is None
    assert detector.state == "lost"


def _red_can_config(*, reacquire_confirmation_frames: int) -> BallDetectorConfig:
    return BallDetectorConfig(
        hsv=HSVRange(lower=(0, 80, 45), upper=(15, 255, 255)),
        additional_hsv_ranges=(
            HSVRange(lower=(165, 80, 45), upper=(179, 255, 255)),
        ),
        minimum_area_px2=450.0,
        maximum_area_px2=18_000.0,
        minimum_circularity=0.12,
        minimum_mean_saturation=78.0,
        minimum_mean_value=115.0,
        dominant_channel="red",
        minimum_channel_excess=18.0,
        minimum_aspect_ratio=1.4,
        maximum_aspect_ratio=4.8,
        minimum_rectangularity=0.25,
        association_max_distance_px=130.0,
        lock_timeout_s=0.65,
        reacquire_confirmation_frames=reacquire_confirmation_frames,
    )


def _identity_can_config() -> BallDetectorConfig:
    return replace(
        _red_can_config(reacquire_confirmation_frames=5),
        identity_lock_enabled=True,
        commissioning_min_bbox_height_px=58.0,
        commissioning_min_vertical_ratio=1.45,
        identity_reacquire_min_vertical_ratio=1.05,
        maximum_appearance_distance=0.34,
        maximum_tracker_appearance_distance=0.54,
        appearance_update_rate=0.04,
        maximum_identity_area_ratio=2.2,
        maximum_identity_aspect_ratio_change=1.8,
        visual_tracker="none",
    )


def _green_square_config() -> BallDetectorConfig:
    return BallDetectorConfig(
        hsv=HSVRange(lower=(35, 70, 100), upper=(92, 255, 255)),
        minimum_area_px2=300.0,
        maximum_area_px2=30_000.0,
        minimum_circularity=0.52,
        minimum_mean_saturation=85.0,
        minimum_mean_value=120.0,
        dominant_channel="green",
        minimum_channel_excess=25.0,
        minimum_aspect_ratio=1.0,
        maximum_aspect_ratio=1.35,
        minimum_rectangularity=0.65,
        association_max_distance_px=150.0,
        lock_timeout_s=0.75,
        reacquire_confirmation_frames=3,
        identity_lock_enabled=True,
        commissioning_min_bbox_height_px=28.0,
        maximum_appearance_distance=0.30,
        maximum_tracker_appearance_distance=0.50,
        maximum_identity_area_ratio=3.0,
        maximum_identity_aspect_ratio_change=1.4,
        visual_tracker="none",
    )


def _blue_balloon_config() -> BallDetectorConfig:
    return BallDetectorConfig(
        hsv=HSVRange(lower=(86, 24, 65), upper=(116, 210, 255)),
        minimum_area_px2=260.0,
        maximum_area_px2=120_000.0,
        minimum_circularity=0.56,
        minimum_mean_saturation=28.0,
        minimum_mean_value=75.0,
        dominant_channel="blue",
        minimum_channel_excess=5.0,
        minimum_aspect_ratio=1.0,
        maximum_aspect_ratio=1.48,
        minimum_rectangularity=0.62,
        association_max_distance_px=190.0,
        lock_timeout_s=0.85,
        reacquire_confirmation_frames=3,
        identity_lock_enabled=True,
        commissioning_min_bbox_height_px=24.0,
        commissioning_min_vertical_ratio=0.68,
        identity_reacquire_min_vertical_ratio=0.60,
        maximum_appearance_distance=0.38,
        maximum_tracker_appearance_distance=0.58,
        maximum_identity_area_ratio=4.0,
        maximum_identity_aspect_ratio_change=1.55,
        visual_tracker="none",
    )
