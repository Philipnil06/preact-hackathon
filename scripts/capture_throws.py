"""Record raw webcam frames plus per-frame timestamps for trajectory calibration."""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise RuntimeError(f"could not open camera {args.camera}")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    capture.set(cv2.CAP_PROP_FPS, args.fps)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    ok, first_frame = capture.read()
    if not ok:
        capture.release()
        raise RuntimeError("camera opened but returned no frame")
    height, width = first_frame.shape[:2]
    video_path = args.output / "throws.avi"
    timestamp_path = args.output / "timestamps.csv"
    metadata_path = args.output / "metadata.json"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        args.fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"could not create {video_path}")

    started_monotonic = time.monotonic()
    started_utc = datetime.now(UTC)
    frame_count = 0
    with timestamp_path.open("w", newline="", encoding="utf-8") as timestamp_file:
        timestamps = csv.writer(timestamp_file)
        timestamps.writerow(("frame_index", "elapsed_s", "captured_at_utc"))
        print(
            json.dumps(
                {
                    "event": "CAPTURE_READY",
                    "video": str(video_path.resolve()),
                    "duration_s": args.duration,
                    "width": width,
                    "height": height,
                }
            ),
            flush=True,
        )
        frame = first_frame
        try:
            while time.monotonic() - started_monotonic < args.duration:
                elapsed_s = time.monotonic() - started_monotonic
                writer.write(frame)
                timestamps.writerow(
                    (
                        frame_count,
                        f"{elapsed_s:.9f}",
                        datetime.now(UTC).isoformat(),
                    )
                )
                frame_count += 1
                ok, frame = capture.read()
                if not ok:
                    break
        finally:
            writer.release()
            capture.release()

    elapsed_s = time.monotonic() - started_monotonic
    metadata = {
        "started_at_utc": started_utc.isoformat(),
        "elapsed_s": elapsed_s,
        "frame_count": frame_count,
        "measured_fps": frame_count / max(elapsed_s, 1e-6),
        "video_fps": args.fps,
        "width": width,
        "height": height,
        "ball_dimensions_mm": [70, 50],
        "floor_marker_distance_m": 1.0,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"event": "CAPTURE_COMPLETE", **metadata}), flush=True)


if __name__ == "__main__":
    main()
