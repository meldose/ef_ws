#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import threading
import time
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse


LOGGER = logging.getLogger("altegro_server")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Placeholder Altegro skill-store server used to test scripts/altegro_client.py."
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host.")
    parser.add_argument("--port", type=int, default=8080, help="Bind port.")
    parser.add_argument(
        "--default-skill-status",
        choices=("compatible", "incompatible"),
        default="compatible",
        help="Fallback compatibility result returned for unknown skills.",
    )
    return parser.parse_args()


PLACEHOLDER_SKILLS: dict[str, dict[str, Any]] = {
    "navigation_pro": {
        "skill_id": "navigation_pro",
        "name": "Navigation Pro",
        "version": "0.1.0",
        "summary": "Placeholder navigation skill package served by the Altegro test store.",
        "entrypoint": "skills/navigation_pro/main.py",
        "requirements": {
            "manufacturer": "Unitree",
            "model": "G1",
            "min_os_version": "3.2.0",
            "min_firmware_version": "1.0.0",
            "required_sensors": ["lidar"],
        },
        "artifacts": [
            {
                "path": "skills/navigation_pro/main.py",
                "sha256": "placeholder-navigation-pro",
            }
        ],
    }
}


@dataclass
class AltegroServerState:
    default_skill_status: str = "compatible"
    skills: dict[str, dict[str, Any]] = field(default_factory=lambda: dict(PLACEHOLDER_SKILLS))
    registrations: list[dict[str, Any]] = field(default_factory=list)
    telemetry: list[dict[str, Any]] = field(default_factory=list)
    heartbeats: dict[str, dict[str, Any]] = field(default_factory=dict)
    compatibility_checks: list[dict[str, Any]] = field(default_factory=list)
    downloads: list[dict[str, Any]] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "skill_ids": sorted(self.skills.keys()),
                "registration_count": len(self.registrations),
                "telemetry_count": len(self.telemetry),
                "heartbeat_devices": sorted(self.heartbeats.keys()),
                "compatibility_checks": list(self.compatibility_checks),
                "downloads": list(self.downloads),
                "last_registration": self.registrations[-1] if self.registrations else None,
                "last_telemetry": self.telemetry[-1] if self.telemetry else None,
                "last_heartbeat": self._latest_heartbeat(),
            }

    def _latest_heartbeat(self) -> dict[str, Any] | None:
        if not self.heartbeats:
            return None
        device_id = max(
            self.heartbeats,
            key=lambda key: self.heartbeats[key].get("received_at", 0),
        )
        return self.heartbeats[device_id]


class AltegroRequestHandler(BaseHTTPRequestHandler):
    server_version = "AltegroTestServer/0.1"

    def _json_response(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            return {}
        raw_body = self.rfile.read(content_length)
        if not raw_body:
            return {}
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON body: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("Expected a JSON object payload")
        return payload

    def _state(self) -> AltegroServerState:
        return self.server.state  # type: ignore[attr-defined]

    def _skill(self, skill_id: str) -> dict[str, Any] | None:
        return self._state().skills.get(skill_id)

    @staticmethod
    def _compare_versions(actual: Any, minimum: Any) -> bool:
        def _normalize(raw: Any) -> tuple[int, ...]:
            text = str(raw or "").strip()
            if not text:
                return ()
            parts: list[int] = []
            for chunk in text.split("."):
                try:
                    parts.append(int(chunk))
                except ValueError:
                    break
            return tuple(parts)

        actual_parts = _normalize(actual)
        minimum_parts = _normalize(minimum)
        if not minimum_parts:
            return True
        if not actual_parts:
            return False
        width = max(len(actual_parts), len(minimum_parts))
        padded_actual = actual_parts + (0,) * (width - len(actual_parts))
        padded_minimum = minimum_parts + (0,) * (width - len(minimum_parts))
        return padded_actual >= padded_minimum

    def _evaluate_skill_compatibility(self, skill: dict[str, Any], payload: dict[str, Any], device_id: str) -> dict[str, Any]:
        requirements = skill.get("requirements", {})
        hardware = payload.get("hardware", {}) if isinstance(payload.get("hardware"), dict) else {}
        software = payload.get("software", {}) if isinstance(payload.get("software"), dict) else {}
        sensors = hardware.get("sensors", [])
        sensor_set = set(sensors) if isinstance(sensors, list) else set()
        failures: list[str] = []

        manufacturer = str(hardware.get("manufacturer", "")).strip()
        model = str(hardware.get("model", "")).strip()
        os_version = software.get("os_version") or hardware.get("os_version")
        firmware_version = software.get("firmware_version") or hardware.get("firmware_version")

        expected_manufacturer = requirements.get("manufacturer")
        if expected_manufacturer and manufacturer != expected_manufacturer:
            failures.append(f"manufacturer={manufacturer or 'unknown'} expected={expected_manufacturer}")

        expected_model = requirements.get("model")
        if expected_model and model != expected_model:
            failures.append(f"model={model or 'unknown'} expected={expected_model}")

        min_os_version = requirements.get("min_os_version")
        if min_os_version and not self._compare_versions(os_version, min_os_version):
            failures.append(f"os_version={os_version or 'unknown'} below minimum={min_os_version}")

        min_firmware_version = requirements.get("min_firmware_version")
        if min_firmware_version and not self._compare_versions(firmware_version, min_firmware_version):
            failures.append(f"firmware_version={firmware_version or 'unknown'} below minimum={min_firmware_version}")

        required_sensors = requirements.get("required_sensors", [])
        if isinstance(required_sensors, list):
            missing = [sensor for sensor in required_sensors if sensor not in sensor_set]
            if missing:
                failures.append(f"missing_sensors={missing}")

        default_status = self._state().default_skill_status
        compatible = not failures and default_status == "compatible"
        status = "compatible" if compatible else "incompatible"
        return {
            "skill_id": skill.get("skill_id"),
            "device_id": device_id,
            "compatible": compatible,
            "status": status,
            "checked_at": int(time.time()),
            "requirements": requirements,
            "evaluated_hardware": hardware,
            "evaluated_software": software,
            "reasons": failures,
            "download_path": f"/api/v1/skills/{skill.get('skill_id')}/download" if compatible else None,
        }

    def log_message(self, format: str, *args: Any) -> None:
        LOGGER.info("%s - %s", self.address_string(), format % args)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/health":
            self._json_response(
                HTTPStatus.OK,
                {"status": "ok", "timestamp": int(time.time())},
            )
            return

        if path == "/api/v1/debug/state":
            self._json_response(HTTPStatus.OK, self._state().snapshot())
            return

        if path == "/api/v1/skills":
            catalog = []
            for skill in self._state().skills.values():
                catalog.append(
                    {
                        "skill_id": skill.get("skill_id"),
                        "name": skill.get("name"),
                        "version": skill.get("version"),
                        "summary": skill.get("summary"),
                        "requirements": skill.get("requirements", {}),
                    }
                )
            self._json_response(HTTPStatus.OK, {"skills": catalog})
            return

        if path.startswith("/api/v1/skills/") and path.endswith("/compatibility"):
            skill_id = path.split("/")[4]
            skill = self._skill(skill_id)
            if skill is None:
                self._json_response(
                    HTTPStatus.NOT_FOUND,
                    {"error": "skill_not_found", "skill_id": skill_id},
                )
                return
            response = {
                "skill_id": skill_id,
                "requirements": skill.get("requirements", {}),
                "download_path": f"/api/v1/skills/{skill_id}/download",
                "status": "inventory_required",
                "message": "POST hardware/software inventory to this endpoint to evaluate compatibility.",
            }
            self._json_response(HTTPStatus.OK, response)
            return

        if path.startswith("/api/v1/skills/") and path.endswith("/download"):
            skill_id = path.split("/")[4]
            skill = self._skill(skill_id)
            if skill is None:
                self._json_response(
                    HTTPStatus.NOT_FOUND,
                    {"error": "skill_not_found", "skill_id": skill_id},
                )
                return
            self._json_response(
                HTTPStatus.OK,
                {
                    "skill_id": skill.get("skill_id"),
                    "name": skill.get("name"),
                    "version": skill.get("version"),
                    "entrypoint": skill.get("entrypoint"),
                    "artifacts": skill.get("artifacts", []),
                    "message": "Placeholder skill package metadata. Download is gated by server-side compatibility checks against the client inventory.",
                },
            )
            return

        self._json_response(
            HTTPStatus.NOT_FOUND,
            {"error": "not_found", "path": path},
        )

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        try:
            payload = self._read_json_body()
        except ValueError as exc:
            self._json_response(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return

        if path.startswith("/api/v1/skills/") and path.endswith("/compatibility"):
            skill_id = path.split("/")[4]
            skill = self._skill(skill_id)
            if skill is None:
                self._json_response(
                    HTTPStatus.NOT_FOUND,
                    {"error": "skill_not_found", "skill_id": skill_id},
                )
                return
            device_id = payload.get("device_id") or self.headers.get("X-Device-ID", "unknown_device")
            response = self._evaluate_skill_compatibility(skill, payload, str(device_id))
            with self._state().lock:
                self._state().compatibility_checks.append(response)
            LOGGER.info(
                "Compatibility check skill=%s device=%s compatible=%s",
                skill_id,
                device_id,
                response["compatible"],
            )
            self._json_response(HTTPStatus.OK, response)
            return

        if path == "/api/v1/devices/register":
            device_id = payload.get("device_id") or self.headers.get("X-Device-ID", "unknown_device")
            entry = {
                "device_id": device_id,
                "headers": self._request_headers(),
                "payload": payload,
                "received_at": int(time.time()),
            }
            with self._state().lock:
                self._state().registrations.append(entry)
            LOGGER.info("Registered device=%s", device_id)
            self._json_response(
                HTTPStatus.OK,
                {
                    "status": "registered",
                    "device_id": device_id,
                    "registered_at": entry["received_at"],
                    "message": "Device registration accepted by placeholder Altegro skill store.",
                },
            )
            return

        if path == "/api/v1/telemetry":
            device_id = payload.get("device_id") or self.headers.get("X-Device-ID", "unknown_device")
            entry = {
                "device_id": device_id,
                "headers": self._request_headers(),
                "payload": payload,
                "received_at": int(time.time()),
            }
            with self._state().lock:
                self._state().telemetry.append(entry)
            LOGGER.info("Telemetry received device=%s", device_id)
            self._json_response(
                HTTPStatus.OK,
                {
                    "status": "accepted",
                    "device_id": device_id,
                    "received_at": entry["received_at"],
                    "telemetry_count": len(self._state().telemetry),
                },
            )
            return

        if path.startswith("/api/v1/devices/") and path.endswith("/heartbeat"):
            device_id = path.split("/")[4]
            entry = {
                "device_id": device_id,
                "headers": self._request_headers(),
                "payload": payload,
                "received_at": int(time.time()),
            }
            with self._state().lock:
                self._state().heartbeats[device_id] = entry
            LOGGER.info("Heartbeat received device=%s status=%s", device_id, payload.get("status"))
            self._json_response(
                HTTPStatus.OK,
                {
                    "status": "alive",
                    "device_id": device_id,
                    "received_at": entry["received_at"],
                },
            )
            return

        if path.startswith("/api/v1/skills/") and path.endswith("/download"):
            skill_id = path.split("/")[4]
            skill = self._skill(skill_id)
            if skill is None:
                self._json_response(
                    HTTPStatus.NOT_FOUND,
                    {"error": "skill_not_found", "skill_id": skill_id},
                )
                return
            device_id = payload.get("device_id") or self.headers.get("X-Device-ID", "unknown_device")
            compatibility = self._evaluate_skill_compatibility(skill, payload, str(device_id))
            if not compatibility["compatible"]:
                self._json_response(
                    HTTPStatus.CONFLICT,
                    {
                        "error": "incompatible_skill_request",
                        "skill_id": skill_id,
                        "device_id": device_id,
                        "message": "Current hardware/software inventory does not satisfy this skill.",
                        "reasons": compatibility.get("reasons", []),
                    },
                )
                return
            download_entry = {
                "device_id": device_id,
                "skill_id": skill_id,
                "payload": payload,
                "received_at": int(time.time()),
            }
            with self._state().lock:
                self._state().downloads.append(download_entry)
            self._json_response(
                HTTPStatus.OK,
                {
                    "status": "ready",
                    "device_id": device_id,
                    "skill_id": skill_id,
                    "package": {
                        "name": skill.get("name"),
                        "version": skill.get("version"),
                        "entrypoint": skill.get("entrypoint"),
                        "artifacts": skill.get("artifacts", []),
                    },
                },
            )
            return

        self._json_response(
            HTTPStatus.NOT_FOUND,
            {"error": "not_found", "path": path},
        )

    def _request_headers(self) -> dict[str, str]:
        return {key: value for key, value in self.headers.items()}


class AltegroThreadingHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], handler_class: type[BaseHTTPRequestHandler], state: AltegroServerState) -> None:
        super().__init__(server_address, handler_class)
        self.state = state


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    state = AltegroServerState(default_skill_status=args.default_skill_status)
    server = AltegroThreadingHTTPServer((args.host, args.port), AltegroRequestHandler, state)

    stop_event = threading.Event()

    def _shutdown(signum: int, _frame: Any) -> None:
        LOGGER.info("Received signal %s, shutting down server.", signum)
        stop_event.set()
        server.shutdown()

    for signame in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, signame, None)
        if sig is not None:
            signal.signal(sig, _shutdown)

    LOGGER.info("Starting Altegro placeholder skill store on http://%s:%d", args.host, args.port)
    LOGGER.info("Health endpoint: GET /health")
    LOGGER.info("Debug state endpoint: GET /api/v1/debug/state")
    LOGGER.info("Skill catalog endpoint: GET /api/v1/skills")

    try:
        server.serve_forever()
    finally:
        server.server_close()
        LOGGER.info("Server stopped.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
