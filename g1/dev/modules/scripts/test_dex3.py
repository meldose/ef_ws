"""Quick DDS listener test for Dex3 hand state topics.

This script subscribes to several possible left and right hand state topic names,
waits a few seconds, and prints which topics actually produced messages. It is
useful when you are not yet sure which topic naming convention the robot setup
is using.
"""

import time
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandState_

# These are the alternative DDS topic names that may exist on different setups.
TOPICS = {
    "right": (
        "rt/dex3/right/state",
        "rt/lf/dex3/right/state",
        "dex3/right/state",
        "lf/dex3/right/state",
    ),
    "left": (
        "rt/dex3/left/state",
        "rt/lf/dex3/left/state",
        "dex3/left/state",
        "lf/dex3/left/state",
    ),
}

# Track whether each topic produced at least one callback.
seen = {topic: False for topics in TOPICS.values() for topic in topics}

def cb(topic):
    # Build a callback function tied to one specific topic name.
    def inner(msg):
        seen[topic] = True
        print(
            f"{topic}: got state, "
            f"motors={len(msg.motor_state)}, press={len(msg.press_sensor_state)}"
        )
    return inner

# Initialize DDS communication on the selected network interface.
ChannelFactoryInitialize(0, "eth0")

# Subscribe to every candidate topic so we can see which ones are alive.
subs = []
for topics in TOPICS.values():
    for topic in topics:
        sub = ChannelSubscriber(topic, HandState_)
        sub.Init(cb(topic), 10)
        subs.append(sub)

# Give the subscribers a short time window to receive state messages.
end = time.time() + 8
while time.time() < end:
    time.sleep(0.2)

# Print a simple summary at the end.
print("seen:")
for topic, value in seen.items():
    print(f"  {topic}: {value}")
