"""Inspect or reset SO101 leader GOAL_SPEED registers without commanding motion.

Stop Studio and all robot owners first. Use the leader's port, with standard
servo IDs 1-6. Support the arm; this script requires torque to already be off.
"""

from __future__ import annotations

import argparse
from typing import Any

_TORQUE_ENABLE = 40
_GOAL_SPEED = 46
_SERVO_IDS = range(1, 7)


def reset_speeds(port: Any, packet: Any, *, reset: bool) -> None:
    """Read all servos before optionally resetting their speed limits to zero."""
    for servo_id in _SERVO_IDS:
        torque, result, error = packet.read1ByteTxRx(port, servo_id, _TORQUE_ENABLE)
        if result != 0 or error != 0:
            raise RuntimeError(f"Cannot read torque for servo {servo_id}: comm={result}, error={error}")
        if torque != 0:
            raise RuntimeError(f"Servo {servo_id} has torque enabled. Stop its controller and support the arm first.")

    for servo_id in _SERVO_IDS:
        speed, result, error = packet.read2ByteTxRx(port, servo_id, _GOAL_SPEED)
        if result != 0 or error != 0:
            raise RuntimeError(f"Cannot read speed for servo {servo_id}: comm={result}, error={error}")
        print(f"Servo {servo_id}: GOAL_SPEED={speed}")

    if not reset:
        print("Read-only check. Add --reset to set GOAL_SPEED=0 (no speed limit).")
        return

    for servo_id in _SERVO_IDS:
        result, error = packet.write2ByteTxRx(port, servo_id, _GOAL_SPEED, 0)
        if result != 0 or error != 0:
            raise RuntimeError(f"Cannot reset speed for servo {servo_id}: comm={result}, error={error}")
        speed, result, error = packet.read2ByteTxRx(port, servo_id, _GOAL_SPEED)
        if result != 0 or error != 0 or speed != 0:
            raise RuntimeError(
                f"Speed reset not confirmed for servo {servo_id}: value={speed}, comm={result}, error={error}"
            )
        print(f"Servo {servo_id}: GOAL_SPEED=0 verified")


def main() -> None:
    """Open the explicitly selected leader port at the standard SO101 baud rate."""
    from scservo_sdk import PacketHandler, PortHandler

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="Leader serial port, preferably /dev/serial/by-id/...")
    parser.add_argument("--reset", action="store_true", help="Reset speed limits; otherwise only read them")
    args = parser.parse_args()
    port = PortHandler(args.port)
    if not port.openPort():
        raise RuntimeError(f"Cannot open {args.port}")
    try:
        if not port.setBaudRate(1_000_000):
            raise RuntimeError("Cannot set baud rate to 1000000")
        reset_speeds(port, PacketHandler(0), reset=args.reset)
    finally:
        port.closePort()


if __name__ == "__main__":
    main()
