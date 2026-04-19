"""HTTP-backed adapter exposing an externalized edge-style backend."""

from __future__ import annotations

import json
from urllib.request import Request, urlopen

from adapters.base_adapter import AdapterInvocationResult, AdapterPreparationResult, BaseAdapter
from core.task_model import TaskRequest
from descriptors.capability_schema import ResetMode, SubstrateDescriptor


class RemoteEdgeAdapter(BaseAdapter):
    """Adapter that talks to a remote HTTP service instead of an in-process twin."""

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")
        descriptor_payload = self._request_json("GET", "/describe")
        descriptor = SubstrateDescriptor.model_validate(descriptor_payload)
        super().__init__(descriptor=descriptor)

    def describe(self) -> SubstrateDescriptor:
        return self.descriptor

    def prepare(self, task: TaskRequest) -> AdapterPreparationResult:
        payload = self._request_json("POST", "/prepare", {"task": task.model_dump(mode="json")})
        return AdapterPreparationResult.model_validate(payload)

    def invoke(self, task: TaskRequest) -> AdapterInvocationResult:
        payload = self._request_json("POST", "/invoke", {"task": task.model_dump(mode="json")})
        return AdapterInvocationResult.model_validate(payload)

    def collect_telemetry(self) -> dict[str, float | int | str | bool | None]:
        payload = self._request_json("GET", "/telemetry")
        return payload

    def reset(self, mode: ResetMode | None = None) -> bool:
        payload = {"mode": str(mode) if mode is not None else None}
        response = self._request_json("POST", "/reset", payload)
        return bool(response.get("success", False))

    def recalibrate(self) -> bool:
        response = self._request_json("POST", "/recalibrate", {})
        return bool(response.get("success", False))

    def _request_json(self, method: str, path: str, payload: dict | None = None) -> dict:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            self._base_url + path,
            data=body,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=5.0) as response:
            return json.loads(response.read().decode("utf-8"))
