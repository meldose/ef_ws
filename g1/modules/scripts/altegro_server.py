#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import signal
import threading
import time
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse


LOGGER = logging.getLogger("altegro_server")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preliminary Altegro gateway server used to test scripts/altegro_client.py."
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host.")
    parser.add_argument("--port", type=int, default=8080, help="Bind port.")
    parser.add_argument(
        "--default-skill-status",
        choices=("compatible", "incompatible"),
        default="compatible",
        help="Compatibility result returned for unknown skills.",
    )
    return parser.parse_args()


@dataclass
class AltegroServerState:
    default_skill_status: str = "compatible"
    registrations: list[dict[str, Any]] = field(default_factory=list)
    telemetry: list[dict[str, Any]] = field(default_factory=list)
    heartbeats: dict[str, dict[str, Any]] = field(default_factory=dict)
    compatibility_checks: list[dict[str, Any]] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "registration_count": len(self.registrations),
                "telemetry_count": len(self.telemetry),
                "heartbeat_devices": sorted(self.heartbeats.keys()),
                "compatibility_checks": list(self.compatibility_checks),
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

        if path.startswith("/api/v1/skills/") and path.endswith("/compatibility"):
            skill_id = path.split("/")[4]
            query = parse_qs(parsed.query)
            device_id = query.get("device_id", ["unknown_device"])[0]
            compatible = self._state().default_skill_status == "compatible"
            response = {
                "skill_id": skill_id,
                "device_id": device_id,
                "compatible": compatible,
                "status": self._state().default_skill_status,
                "checked_at": int(time.time()),
                "requirements": {
                    "manufacturer": "Unitree",
                    "model": "G1",
                },
            }
            with self._state().lock:
                self._state().compatibility_checks.append(response)
            LOGGER.info("Compatibility check skill=%s device=%s", skill_id, device_id)
            self._json_response(HTTPStatus.OK, response)
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
                    "message": "Device registration accepted by preliminary server.",
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

    LOGGER.info("Starting Altegro preliminary server on http://%s:%d", args.host, args.port)
    LOGGER.info("Health endpoint: GET /health")
    LOGGER.info("Debug state endpoint: GET /api/v1/debug/state")

    try:
        server.serve_forever()
    finally:
        server.server_close()
        LOGGER.info("Server stopped.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
