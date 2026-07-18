"""Offline calibration commands. Nothing opens a camera implicitly."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

from preact.aruco_calibration import (
    ArucoPlaneSpec,
    calibrate_aruco_plane,
    calibrate_plane_from_corners,
    draw_calibration_preview,
    generate_calibration_markers,
)
from preact.ball_calibration import calibrate_ball_colour
from preact.claude_targeting import ClaudeTargetSelector
from preact.detection import BallDetector, BallDetectorConfig, HSVRange


def main() -> None:
    parser = argparse.ArgumentParser(description="PREACT offline calibration")
    commands = parser.add_subparsers(dest="command", required=True)

    markers = commands.add_parser("markers", help="generate four printable ArUco markers")
    markers.add_argument("--output", type=Path, required=True)
    markers.add_argument("--side-pixels", type=int, default=1000)

    plane = commands.add_parser("plane", help="calibrate the throwing plane from an image")
    plane.add_argument("--image", type=Path, required=True)
    plane.add_argument("--output", type=Path, required=True)
    plane.add_argument("--preview", type=Path)
    plane.add_argument("--width-m", type=float, default=3.0)
    plane.add_argument("--height-m", type=float, default=2.0)

    plane_manual = commands.add_parser(
        "plane-manual",
        help="calibrate from four measured image points without ArUco",
    )
    plane_manual.add_argument(
        "--points",
        type=_parse_points,
        required=True,
        metavar='"X,Y;X,Y;X,Y;X,Y"',
        help="bottom-left; bottom-right; top-right; top-left",
    )
    plane_manual.add_argument("--output", type=Path, required=True)
    plane_manual.add_argument("--width-m", type=float, default=3.0)
    plane_manual.add_argument("--height-m", type=float, default=2.0)

    colour = commands.add_parser("colour", help="sample the ball colour from an image")
    colour.add_argument("--image", type=Path, required=True)
    colour.add_argument("--roi", type=_parse_quad, required=True, metavar="X,Y,W,H")
    colour.add_argument("--output", type=Path, required=True)
    colour.add_argument("--preview", type=Path)

    claude = commands.add_parser(
        "colour-claude",
        help="ask Claude to locate a safe demo object, then calibrate locally",
    )
    claude.add_argument("--image", type=Path, required=True)
    claude.add_argument("--output", type=Path, required=True)
    claude.add_argument("--preview", type=Path)
    claude.add_argument("--model", default=None)

    manual = commands.add_parser("colour-manual", help="save explicit OpenCV HSV bounds")
    manual.add_argument("--lower", type=_parse_triple, required=True, metavar="H,S,V")
    manual.add_argument("--upper", type=_parse_triple, required=True, metavar="H,S,V")
    manual.add_argument("--output", type=Path, required=True)

    arguments = parser.parse_args()
    load_dotenv()
    if arguments.command == "markers":
        paths = generate_calibration_markers(
            arguments.output,
            side_pixels=arguments.side_pixels,
        )
        print("\n".join(str(path) for path in paths))
        return
    if arguments.command == "colour-manual":
        config = BallDetectorConfig(
            hsv=HSVRange(lower=arguments.lower, upper=arguments.upper)
        )
        _ensure_parent(arguments.output)
        config.save(arguments.output)
        print(arguments.output)
        return
    if arguments.command == "plane-manual":
        calibration = calibrate_plane_from_corners(
            np.asarray(arguments.points, dtype=np.float64),
            ArucoPlaneSpec(
                width_m=arguments.width_m,
                height_m=arguments.height_m,
            ),
        )
        _ensure_parent(arguments.output)
        calibration.save(arguments.output)
        print(
            f"{arguments.output} "
            f"(RMSE {calibration.reprojection_rmse_m:.4f} m)"
        )
        return

    frame = cv2.imread(str(arguments.image))
    if frame is None:
        raise SystemExit(f"could not read image: {arguments.image}")
    if arguments.command == "plane":
        result = calibrate_aruco_plane(
            frame,
            ArucoPlaneSpec(width_m=arguments.width_m, height_m=arguments.height_m),
        )
        _ensure_parent(arguments.output)
        result.calibration.save(arguments.output)
        if arguments.preview:
            _write_image(arguments.preview, draw_calibration_preview(frame, result))
        print(
            f"{arguments.output} "
            f"(RMSE {result.calibration.reprojection_rmse_m:.4f} m)"
        )
        return

    if arguments.command == "colour-claude":
        if arguments.model:
            os.environ["ANTHROPIC_MODEL"] = arguments.model
        selection = ClaudeTargetSelector.from_environment().select(frame)
        if not selection.found or selection.roi is None:
            raise SystemExit(f"Claude did not find a safe target: {selection.reason}")
        roi = selection.roi
    else:
        roi = arguments.roi

    result = calibrate_ball_colour(frame, roi)
    _ensure_parent(arguments.output)
    result.config.save(arguments.output)
    if arguments.preview:
        preview = frame.copy()
        x, y, width, height = roi
        cv2.rectangle(preview, (x, y), (x + width, y + height), (0, 0, 255), 3)
        detection = BallDetector(result.config).detect(preview, timestamp_s=0.0)
        if detection is not None:
            from preact.detection import draw_tracking_box

            preview = draw_tracking_box(preview, detection)
        _write_image(arguments.preview, preview)
    print(
        f"{arguments.output} "
        f"(hue {result.dominant_hue}, {result.sampled_pixels} sample pixels)"
    )


def _parse_triple(raw: str) -> tuple[int, int, int]:
    values = tuple(int(item.strip()) for item in raw.split(","))
    if len(values) != 3:
        raise argparse.ArgumentTypeError("expected H,S,V")
    return values  # type: ignore[return-value]


def _parse_quad(raw: str) -> tuple[int, int, int, int]:
    values = tuple(int(item.strip()) for item in raw.split(","))
    if len(values) != 4:
        raise argparse.ArgumentTypeError("expected X,Y,W,H")
    return values  # type: ignore[return-value]


def _parse_points(raw: str) -> tuple[tuple[float, float], ...]:
    try:
        points = tuple(
            tuple(float(value.strip()) for value in pair.split(","))
            for pair in raw.split(";")
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError("points must be numeric X,Y pairs") from error
    if len(points) != 4 or any(len(point) != 2 for point in points):
        raise argparse.ArgumentTypeError("expected exactly four X,Y pairs")
    return points  # type: ignore[return-value]


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_image(path: Path, image: object) -> None:
    _ensure_parent(path)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"could not write image: {path}")


if __name__ == "__main__":
    main()
