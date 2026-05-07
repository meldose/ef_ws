#!/usr/bin/env python3
from __future__ import annotations

"""Inspect DDS discovery traffic and print matching topics, publishers, and subscribers.

This helps beginners see what DDS entities are visible on a chosen interface and
domain. The optional filter keeps the output focused on robot-related topics.
"""

import argparse
import time

from cyclonedds.builtin import BuiltinDataReader, BuiltinTopicDcpsPublication, BuiltinTopicDcpsSubscription, BuiltinTopicDcpsTopic
from cyclonedds.domain import Domain, DomainParticipant

from unitree_sdk2py.core import channel as channel_module


# Override the SDK DDS configuration so discovery uses the interface requested
# on the command line instead of a default interface.
channel_module.ChannelConfigHasInterface = """<?xml version="1.0" encoding="UTF-8" ?>
<CycloneDDS>
  <Domain Id="any">
    <General>
      <Interfaces>
        <NetworkInterface name="$__IF_NAME__$" priority="default" multicast="default"/>
      </Interfaces>
    </General>
  </Domain>
</CycloneDDS>"""


def parse_args() -> argparse.Namespace:
    # Accept the interface, DDS domain, discovery time, and output filter.
    parser = argparse.ArgumentParser(description="List DDS topics discovered on an interface.")
    parser.add_argument("--iface", default="eth0")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--filter", default="dex|hand|lowstate|lowcmd|lf")
    return parser.parse_args()


def text_matches(text: str, pattern: str) -> bool:
    # Small helper so the filter can be provided as a regular expression.
    import re

    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def print_samples(label: str, samples: list[object], pattern: str) -> None:
    # Print only the discovered DDS entities that match the filter expression.
    print(f"\n{label}:")
    found = False
    for sample in samples:
        text = str(sample)
        topic = getattr(sample, "topic_name", "")
        type_name = getattr(sample, "type_name", "")
        combined = f"{topic} {type_name} {text}"
        if pattern and not text_matches(combined, pattern):
            continue
        found = True
        print(f"  topic={topic!r} type={type_name!r}")
    if not found:
        print("  <none>")


def main() -> int:
    # Create a DDS participant, read the built-in discovery topics for a while,
    # then print the collected topics, publications, and subscriptions.
    args = parse_args()
    config = channel_module.ChannelConfigHasInterface.replace("$__IF_NAME__$", args.iface)
    domain = Domain(args.domain_id, config)
    participant = DomainParticipant(args.domain_id)

    topic_reader = BuiltinDataReader(participant, BuiltinTopicDcpsTopic)
    pub_reader = BuiltinDataReader(participant, BuiltinTopicDcpsPublication)
    sub_reader = BuiltinDataReader(participant, BuiltinTopicDcpsSubscription)

    deadline = time.time() + max(0.5, float(args.seconds))
    topics: list[object] = []
    pubs: list[object] = []
    subs: list[object] = []

    # Repeatedly take discovery samples until the requested time expires.
    while time.time() < deadline:
        topics.extend(topic_reader.take(100) or [])
        pubs.extend(pub_reader.take(100) or [])
        subs.extend(sub_reader.take(100) or [])
        time.sleep(0.1)

    print(f"iface={args.iface} domain_id={args.domain_id} seconds={args.seconds}")
    print_samples("Topics", topics, args.filter)
    print_samples("Publications", pubs, args.filter)
    print_samples("Subscriptions", subs, args.filter)
    # Explicit cleanup keeps the temporary DDS domain object from lingering.
    del domain
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
