#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import platform
import signal
import sys
import time
from pathlib import Path
from typing import Any

import aiohttp
import psutil


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from sdk_client import Robot


LOGGER = logging.getLogger("altegro_client")
EDGE_RUNTIME_DIR = Path(SCRIPT_DIR).resolve().parents[1] / "edge_runtime"
DEFAULT_CONFIG_PATH = EDGE_RUNTIME_DIR / "config" / "runtime_config.yaml"
DEFAULT_FINGERPRINT_PATH = EDGE_RUNTIME_DIR / "device_identity" / "hardware_fingerprint.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Altegro edge runtime client built on top of the local G1 module wrappers."
    )
    parser.add_argument("--endpoint", default="http://localhost:8080", help="Marketplace or gateway endpoint.")
    parser.add_argument("--device-id", default="G1_Robot_001", help="Logical device identifier.")
    parser.add_argument("--api-key", default="test_api_key", help="API key for the remote gateway.")
    parser.add_argument("--iface", default="eth0", help="Robot SDK network interface.")
    parser.add_argument("--domain-id", type=int, default=0, help="DDS domain id.")
    parser.add_argument("--no-safety-boot", action="store_true", help="Skip safety boot when connecting to the robot.")
    parser.add_argument("--telemetry-interval", type=float, default=15.0, help="Telemetry push interval in seconds.")
    parser.add_argument("--heartbeat-interval", type=float, default=10.0, help="Heartbeat interval in seconds.")
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout in seconds.")
    parser.add_argument("--skill-id", default="navigation_pro", help="Skill id to test for compatibility.")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip the example placeholder skill download after a compatible result.",
    )
    parser.add_argument(
        "--fingerprint-path",
        default=str(DEFAULT_FINGERPRINT_PATH),
        help="Path to the hardware fingerprint JSON file.",
    )
    parser.add_argument(
        "--config-path",
        default=str(DEFAULT_CONFIG_PATH),
        help="Optional runtime config path, used for defaults when present.",
    )
    parser.add_argument("--once", action="store_true", help="Run one registration / telemetry / heartbeat cycle and exit.")
    parser.add_argument(
        "--skip-compatibility",
        action="store_true",
        help="Skip the example compatibility check.",
    )
    return parser.parse_args()


def load_json_file(path: str | os.PathLike[str]) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        return {}
    with file_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {file_path}")
    return data


def load_runtime_defaults(config_path: str | os.PathLike[str]) -> dict[str, Any]:
    file_path = Path(config_path)
    if not file_path.exists():
        return {}
    try:
        import yaml  # type: ignore
    except ImportError:
        LOGGER.debug("PyYAML not available; skipping runtime config load from %s", file_path)
        return {}
    with file_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data if isinstance(data, dict) else {}


class AltegroGatewayClient:
    def __init__(self, endpoint: str, device_id: str, api_key: str, timeout: float = 30.0) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.device_id = device_id
        self.api_key = api_key
        self.timeout = float(timeout)
        self.session: aiohttp.ClientSession | None = None

    async def initialize(self) -> None:
        self.session = aiohttp.ClientSession(
            headers={
                "Content-Type": "application/json",
                "X-Device-ID": self.device_id,
                "X-API-Key": self.api_key,
            },
            timeout=aiohttp.ClientTimeout(total=self.timeout),
        )

    async def close(self) -> None:
        if self.session is not None:
            await self.session.close()
            self.session = None

    async def register_device(self, payload: dict[str, Any]) -> dict[str, Any]:
        assert self.session is not None
        url = f"{self.endpoint}/api/v1/devices/register"
        async with self.session.post(url, json=payload) as response:
            response.raise_for_status()
            return await response.json()

    async def send_telemetry(self, payload: dict[str, Any]) -> dict[str, Any]:
        assert self.session is not None
        url = f"{self.endpoint}/api/v1/telemetry"
        async with self.session.post(url, json=payload) as response:
            response.raise_for_status()
            return await response.json()

    async def send_heartbeat(self, payload: dict[str, Any]) -> dict[str, Any]:
        assert self.session is not None
        url = f"{self.endpoint}/api/v1/devices/{self.device_id}/heartbeat"
        async with self.session.post(url, json=payload) as response:
            response.raise_for_status()
            return await response.json()

    async def check_skill_compatibility(self, skill_id: str) -> dict[str, Any]:
        return await self.check_skill_compatibility_for(skill_id, {})

    async def check_skill_compatibility_for(self, skill_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        assert self.session is not None
        url = f"{self.endpoint}/api/v1/skills/{skill_id}/compatibility"
        request_payload = {"device_id": self.device_id, **payload}
        async with self.session.post(url, json=request_payload) as response:
            response.raise_for_status()
            return await response.json()

    async def download_skill(self, skill_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        assert self.session is not None
        url = f"{self.endpoint}/api/v1/skills/{skill_id}/download"
        async with self.session.post(url, json=payload) as response:
            response.raise_for_status()
            return await response.json()


class RobotTelemetryCollector:
    def __init__(self, robot: Robot, fingerprint: dict[str, Any], runtime_defaults: dict[str, Any]) -> None:
        self.robot = robot
        self.fingerprint = fingerprint
        self.runtime_defaults = runtime_defaults

    def build_device_registration(self) -> dict[str, Any]:
        base = dict(self.fingerprint)
        base.setdefault("manufacturer", "Unitree")
        base.setdefault("model", "G1")
        base.setdefault("device_id", self.robot.iface)
        base["network_interface"] = self.robot.iface
        base["domain_id"] = self.robot.domain_id
        base["runtime"] = "altegro_client"
        return base

    def collect_telemetry(self) -> dict[str, Any]:
        robot_state = self.robot.get_robot_state()
        net_io = psutil.net_io_counters()
        cpu_percent = psutil.cpu_percent(interval=None)
        memory_percent = psutil.virtual_memory().percent
        disk_percent = psutil.disk_usage(os.path.abspath(os.sep)).percent

        battery_level = self.fingerprint.get("battery_level")
        if battery_level is None:
            battery_level = self.fingerprint.get("battery_capacity")

        return {
            "timestamp": int(time.time()),
            "device_id": self.fingerprint.get("device_id"),
            "runtime": "altegro_client",
            "system": {
                "cpu_usage": cpu_percent,
                "memory_usage": memory_percent,
                "disk_usage": disk_percent,
                "network_bytes_sent": net_io.bytes_sent,
                "network_bytes_recv": net_io.bytes_recv,
            },
            "robot": {
                "fsm": robot_state.get("fsm"),
                "mode": robot_state.get("mode"),
                "gait": robot_state.get("gait"),
                "body_height": robot_state.get("body_height"),
                "position": robot_state.get("position"),
                "velocity": robot_state.get("velocity"),
                "yaw": robot_state.get("yaw"),
                "imu": robot_state.get("imu"),
                "odom_pose": robot_state.get("odom_pose"),
                "slam_pose": robot_state.get("slam_pose"),
                "is_moving": robot_state.get("is_moving"),
                "sensor_stale": robot_state.get("sensor_stale"),
                "sensor_timestamps": robot_state.get("sensor_timestamps"),
                "slam_is_running": robot_state.get("slam_is_running"),
                "queued_path_points": robot_state.get("queued_path_points"),
                "joint_count": robot_state.get("joint_count"),
                "battery": battery_level,
            },
        }

    def collect_heartbeat(self) -> dict[str, Any]:
        robot_state = self.robot.get_robot_state()
        return {
            "status": "active",
            "timestamp": int(time.time()),
            "robot_connected": True,
            "fsm": robot_state.get("fsm"),
            "mode": robot_state.get("mode"),
            "is_moving": robot_state.get("is_moving"),
        }

    def build_skill_inventory(self) -> dict[str, Any]:
        skill_settings = self.runtime_defaults.get("skills", {}) if isinstance(self.runtime_defaults.get("skills"), dict) else {}
        updates = self.runtime_defaults.get("updates", {}) if isinstance(self.runtime_defaults.get("updates"), dict) else {}
        network = self.runtime_defaults.get("network", {}) if isinstance(self.runtime_defaults.get("network"), dict) else {}

        hardware = dict(self.fingerprint)
        hardware.setdefault("manufacturer", "Unitree")
        hardware.setdefault("model", "G1")
        hardware["network_interface"] = self.robot.iface
        hardware["domain_id"] = self.robot.domain_id

        software = {
            "runtime": "altegro_client",
            "runtime_version": "0.1.0",
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "os_version": self.fingerprint.get("os_version"),
            "firmware_version": self.fingerprint.get("firmware_version"),
            "skills_repository_path": skill_settings.get("repository_path"),
            "max_concurrent_skills": skill_settings.get("max_concurrent_skills"),
            "default_skill_timeout": skill_settings.get("default_timeout"),
            "fallback_version": updates.get("fallback_version"),
            "network_timeout": network.get("timeout"),
        }
        return {
            "hardware": hardware,
            "software": software,
        }


class AltegroRuntimeClient:
    def __init__(
        self,
        gateway: AltegroGatewayClient,
        telemetry: RobotTelemetryCollector,
        skill_id: str,
        telemetry_interval: float,
        heartbeat_interval: float,
        skip_compatibility: bool = False,
        skip_download: bool = False,
    ) -> None:
        self.gateway = gateway
        self.telemetry = telemetry
        self.skill_id = skill_id
        self.telemetry_interval = float(telemetry_interval)
        self.heartbeat_interval = float(heartbeat_interval)
        self.skip_compatibility = skip_compatibility
        self.skip_download = skip_download
        self.shutdown_event = asyncio.Event()

    async def initialize(self) -> None:
        await self.gateway.initialize()

    async def register(self) -> None:
        payload = self.telemetry.build_device_registration()
        payload.setdefault("device_id", self.gateway.device_id)
        response = await self.gateway.register_device(payload)
        LOGGER.info("Device registration response: %s", response)

    async def push_telemetry_once(self) -> None:
        payload = self.telemetry.collect_telemetry()
        payload["device_id"] = self.gateway.device_id
        response = await self.gateway.send_telemetry(payload)
        LOGGER.info("Telemetry response: %s", response)

    async def push_heartbeat_once(self) -> None:
        payload = self.telemetry.collect_heartbeat()
        response = await self.gateway.send_heartbeat(payload)
        LOGGER.info("Heartbeat response: %s", response)

    async def check_compatibility_once(self) -> None:
        if self.skip_compatibility or not self.skill_id:
            return
        inventory = self.telemetry.build_skill_inventory()
        response = await self.gateway.check_skill_compatibility_for(self.skill_id, inventory)
        LOGGER.info("Skill compatibility for %s: %s", self.skill_id, response)
        if response.get("compatible") and not self.skip_download:
            download_request = {
                "device_id": self.gateway.device_id,
                "compatible": True,
                "compatibility_checked_at": response.get("checked_at"),
                "hardware": inventory.get("hardware"),
                "software": inventory.get("software"),
            }
            download_response = await self.gateway.download_skill(self.skill_id, download_request)
            LOGGER.info("Downloaded placeholder skill package for %s: %s", self.skill_id, download_response)

    async def telemetry_loop(self) -> None:
        while not self.shutdown_event.is_set():
            try:
                await self.push_telemetry_once()
            except Exception as exc:
                LOGGER.warning("Telemetry push failed: %s", exc)
            try:
                await asyncio.wait_for(self.shutdown_event.wait(), timeout=max(0.5, self.telemetry_interval))
            except asyncio.TimeoutError:
                pass

    async def heartbeat_loop(self) -> None:
        while not self.shutdown_event.is_set():
            try:
                await self.push_heartbeat_once()
            except Exception as exc:
                LOGGER.warning("Heartbeat push failed: %s", exc)
            try:
                await asyncio.wait_for(self.shutdown_event.wait(), timeout=max(0.5, self.heartbeat_interval))
            except asyncio.TimeoutError:
                pass

    async def run(self, once: bool = False) -> None:
        await self.register()
        await self.check_compatibility_once()

        if once:
            await self.push_telemetry_once()
            await self.push_heartbeat_once()
            return

        telemetry_task = asyncio.create_task(self.telemetry_loop(), name="telemetry_loop")
        heartbeat_task = asyncio.create_task(self.heartbeat_loop(), name="heartbeat_loop")

        try:
            await self.shutdown_event.wait()
        finally:
            telemetry_task.cancel()
            heartbeat_task.cancel()
            await asyncio.gather(telemetry_task, heartbeat_task, return_exceptions=True)

    async def close(self) -> None:
        await self.gateway.close()

    def shutdown(self) -> None:
        self.shutdown_event.set()


def install_signal_handlers(runtime: AltegroRuntimeClient) -> None:
    loop = asyncio.get_running_loop()

    def _handle_signal(signame: str) -> None:
        LOGGER.info("Received %s, shutting down...", signame)
        runtime.shutdown()

    for signame in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, signame, None)
        if sig is None:
            continue
        try:
            loop.add_signal_handler(sig, lambda s=signame: _handle_signal(s))
        except NotImplementedError:
            pass


async def async_main() -> int:
    args = parse_args()

    runtime_defaults = load_runtime_defaults(args.config_path)
    telemetry_defaults = runtime_defaults.get("telemetry", {}) if isinstance(runtime_defaults.get("telemetry"), dict) else {}
    safety_defaults = runtime_defaults.get("safety", {}) if isinstance(runtime_defaults.get("safety"), dict) else {}

    telemetry_interval = args.telemetry_interval or telemetry_defaults.get("interval", 15.0)
    heartbeat_interval = args.heartbeat_interval or safety_defaults.get("heartbeat_interval", 10.0)

    fingerprint = load_json_file(args.fingerprint_path)
    fingerprint.setdefault("device_id", args.device_id)

    try:
        robot = Robot(
            iface=args.iface,
            domain_id=args.domain_id,
            safety_boot=not args.no_safety_boot,
            auto_start_sensors=True,
        )
    except Exception as exc:
        LOGGER.error("Failed to connect to robot: %s", exc)
        return 1

    gateway = AltegroGatewayClient(
        endpoint=args.endpoint,
        device_id=args.device_id,
        api_key=args.api_key,
        timeout=args.timeout,
    )
    telemetry = RobotTelemetryCollector(robot=robot, fingerprint=fingerprint, runtime_defaults=runtime_defaults)
    runtime = AltegroRuntimeClient(
        gateway=gateway,
        telemetry=telemetry,
        skill_id=args.skill_id,
        telemetry_interval=telemetry_interval,
        heartbeat_interval=heartbeat_interval,
        skip_compatibility=args.skip_compatibility,
        skip_download=args.skip_download,
    )

    install_signal_handlers(runtime)

    try:
        await runtime.initialize()
        await runtime.run(once=args.once)
    except aiohttp.ClientError as exc:
        LOGGER.error("Gateway communication failed: %s", exc)
        return 1
    except Exception as exc:
        LOGGER.error("Altegro runtime client failed: %s", exc, exc_info=True)
        return 1
    finally:
        await runtime.close()

    return 0


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return asyncio.run(async_main())


if __name__ == "__main__":
    raise SystemExit(main())
