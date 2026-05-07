#!/usr/bin/env python3
from __future__ import annotations

"""Check whether Dex3 hand command publishers can match DDS subscribers.

The script sends a harmless "open hand" command for the selected hand and reports
whether the DDS writer successfully matched a subscriber. This is mainly a quick
connectivity and topic-wiring test.
"""

import argparse
import os
import sys


# Make the parent module directory importable when the script is run directly.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from sdk_hand import Dex3HandController, build_hand_msg, hand_open_targets


def parse_args() -> argparse.Namespace:
    # Collect the runtime settings for network, target hand, and publish timing.
    parser = argparse.ArgumentParser(description="Probe whether Dex3 command DDS writers match.")
    parser.add_argument("--iface", default="eth0")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--hand", choices=("right", "left", "both"), default="both")
    parser.add_argument("--timeout-s", type=float, default=3.0)
    parser.add_argument("--hold-s", type=float, default=0.2)
    parser.add_argument("--rate-hz", type=float, default=10.0)
    return parser.parse_args()


def probe_hand(hand: str, args: argparse.Namespace) -> bool:
    # Build a controller and send an "open" command briefly to test DDS matching.
    controller = Dex3HandController(hand=hand, iface=args.iface, domain_id=args.domain_id)
    msg = build_hand_msg(hand_open_targets(hand), kp=0.0, kd=0.0, tau=0.0)
    matched = controller.publish_for(
        msg,
        seconds=args.hold_s,
        rate_hz=args.rate_hz,
        first_write_timeout_s=args.timeout_s,
    )
    print(f"{hand}: command writer matched subscriber: {matched}")
    return matched


def main() -> int:
    # Expand "both" into two separate checks and treat any failure as a non-zero exit.
    args = parse_args()
    hands = ("right", "left") if args.hand == "both" else (args.hand,)
    results = [probe_hand(hand, args) for hand in hands]
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
