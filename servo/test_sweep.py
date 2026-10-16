"""Exercise one servo through ten endpoint moves, then return to neutral."""

from __future__ import annotations

import argparse
import time

import serial


def send_angle(port: serial.Serial, servo: str, angle: int) -> str:
    port.write(f"{servo}:{angle}\n".encode("ascii"))
    port.flush()
    deadline = time.monotonic() + 1.5
    expected = f"{servo}:{angle}"
    while time.monotonic() < deadline:
        reply = port.readline().decode("utf-8", errors="replace").strip()
        if reply == expected:
            return reply
    raise RuntimeError(f"No acknowledgement for {expected}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="COM3")
    parser.add_argument("--servo", choices=("S1", "S2", "S3"), default="S1")
    parser.add_argument("--interval", type=float, default=3.0)
    parser.add_argument("--moves", type=int, default=10)
    arguments = parser.parse_args()

    if arguments.interval <= 0:
        raise SystemExit("--interval must be positive")
    if arguments.moves <= 0:
        raise SystemExit("--moves must be positive")

    with serial.Serial(arguments.port, 115200, timeout=0.2, write_timeout=1) as port:
        time.sleep(2.0)
        try:
            send_angle(port, arguments.servo, 90)
            for move in range(1, arguments.moves + 1):
                angle = 180 if move % 2 else 0
                acknowledgement = send_angle(port, arguments.servo, angle)
                print(f"Move {move}/{arguments.moves}: {acknowledgement}", flush=True)
                time.sleep(arguments.interval)
        finally:
            acknowledgement = send_angle(port, arguments.servo, 90)
            print(f"Neutral: {acknowledgement}", flush=True)


if __name__ == "__main__":
    main()
