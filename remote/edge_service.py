"""Small HTTP service exposing a remote edge-style backend."""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from adapters.edge_adapter import EdgeAdapter
from core.task_model import TaskRequest
from descriptors.capability_schema import Locality, ResetMode


class RemoteEdgeServiceState:
    def __init__(self) -> None:
        self.adapter = EdgeAdapter(backend_id="remote-edge-backend")
        descriptor = self.adapter.describe().model_copy(deep=True)
        descriptor.display_name = "Remote Edge Twin Backend"
        descriptor.policy.locality = Locality.FOG
        descriptor.custom_metadata["endpoint_kind"] = "http_remote"
        descriptor.custom_metadata["endpoint_transport"] = "http"
        self.descriptor = descriptor


class EdgeServiceHandler(BaseHTTPRequestHandler):
    state = RemoteEdgeServiceState()

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
            return
        if self.path == "/describe":
            self._send_json(200, self.state.descriptor.model_dump(mode="json"))
            return
        if self.path == "/telemetry":
            telemetry = self.state.adapter.collect_telemetry()
            telemetry["endpoint_kind"] = "http_remote"
            self._send_json(200, telemetry)
            return
        self._send_json(404, {"error": f"Unknown endpoint: {self.path}"})

    def do_POST(self) -> None:  # noqa: N802
        payload = self._read_json_body()
        if self.path == "/prepare":
            task = TaskRequest.model_validate(payload["task"])
            result = self.state.adapter.prepare(task)
            self._send_json(200, result.model_dump(mode="json"))
            return
        if self.path == "/invoke":
            task = TaskRequest.model_validate(payload["task"])
            result = self.state.adapter.invoke(task)
            self._send_json(200, result.model_dump(mode="json"))
            return
        if self.path == "/reset":
            mode_raw = payload.get("mode")
            mode = ResetMode(mode_raw) if mode_raw else None
            success = self.state.adapter.reset(mode=mode)
            self._send_json(200, {"success": success})
            return
        if self.path == "/recalibrate":
            success = self.state.adapter.recalibrate()
            self._send_json(200, {"success": success})
            return
        self._send_json(404, {"error": f"Unknown endpoint: {self.path}"})

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def _read_json_body(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length) if content_length > 0 else b"{}"
        return json.loads(raw_body.decode("utf-8"))

    def _send_json(self, status_code: int, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), EdgeServiceHandler)
    print(f"Remote edge service listening on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
