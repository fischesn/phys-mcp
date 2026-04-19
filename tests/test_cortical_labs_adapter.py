from __future__ import annotations

from adapters.cortical_labs_adapter import CorticalLabsAdapter
from demos.common import make_cortical_task


class _FakeRecording:
    file_location = "./fake_recording.h5"

    def stop(self):
        return self


class _FakeNeurons:
    def __init__(self):
        self.stim_calls = []

    def record(self):
        return _FakeRecording()

    def stim(self, *args):
        self.stim_calls.append(args)


class _FakeContext:
    def __init__(self, neurons):
        self._neurons = neurons

    def __enter__(self):
        return self._neurons

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeCL:
    def __init__(self):
        self.neurons = _FakeNeurons()

    def open(self, **kwargs):
        return _FakeContext(self.neurons)

    def get_system_attributes(self):
        return {"project_id": "fake-project"}


def test_cortical_labs_adapter_prepare_and_invoke():
    adapter = CorticalLabsAdapter(cl_module=_FakeCL())
    task = make_cortical_task()

    prep = adapter.prepare(task)
    assert prep.prepared is True

    result = adapter.invoke(task)
    assert result.backend_id == adapter.backend_id()
    assert result.output_payload["interface"] == "cortical-labs-cl-api"
    assert result.output_payload["recording_path"] == "./fake_recording.h5"

    telemetry = adapter.collect_telemetry()
    assert telemetry["sdk_available"] is True
    assert telemetry["health_status"] == "ready"


def test_cortical_labs_adapter_reports_unavailable_without_sdk():
    adapter = CorticalLabsAdapter(cl_module=None)
    task = make_cortical_task(task_id="pytest-cortical-offline")

    prep = adapter.prepare(task)
    assert prep.prepared is False
    assert "SDK not available" in prep.details

    telemetry = adapter.collect_telemetry()
    assert telemetry["sdk_available"] is False
    assert telemetry["health_status"] == "offline"
