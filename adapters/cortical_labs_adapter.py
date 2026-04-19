"""Adapter targeting the public Cortical Labs CL API / CL SDK Simulator.

This adapter is intentionally optional. It allows phys-MCP to target an existing,
digitally addressable wetware interface when the ``cl`` module from the Cortical
Labs SDK is available. In environments without the SDK, the adapter remains
importable and reports itself as unavailable at preparation time.
"""

from __future__ import annotations

import os
import time
from importlib import import_module
from typing import Any

from adapters.base_adapter import AdapterInvocationResult, AdapterPreparationResult, BaseAdapter
from core.task_model import TaskRequest
from descriptors.capability_schema import (
    CapabilityDescriptor,
    IOContract,
    IOEncoding,
    LifecycleContract,
    Locality,
    ObservabilityLevel,
    PolicyConstraints,
    ResetMode,
    SignalModality,
    SubstrateClass,
    SubstrateDescriptor,
    TelemetryContract,
    TelemetryField,
    TenancyModel,
    TimingContract,
    TimingRegime,
    TrainingMode,
    TwinBinding,
)


class CorticalLabsAdapter(BaseAdapter):
    """Adapter for the Cortical Labs CL API and CL SDK Simulator.

    The adapter maps phys-MCP task requests onto the public Python entry points
    documented for ``cl.open()``, ``neurons.record()``, and ``neurons.stim()``.
    It is designed as a best-effort integration example rather than as a fully
    validated wetware control stack.
    """

    def __init__(
        self,
        backend_id: str = 'cortical-labs-backend',
        *,
        cl_module: Any | None = None,
        take_control: bool = True,
        prefer_simulator: bool = True,
        replay_path: str | None = None,
        accelerated_time: bool = False,
    ) -> None:
        self._cl = cl_module if cl_module is not None else self._try_import_cl()
        self._take_control = take_control
        self._prefer_simulator = prefer_simulator
        self._replay_path = replay_path
        self._accelerated_time = accelerated_time
        self._last_telemetry: dict[str, float | int | str | bool | None] = {
            'health_status': 'offline' if self._cl is None else 'ready',
            'drift_score': 0.12,
            'age_of_information_ms': None,
            'last_latency_ms': None,
            'loop_timeout_events': 0,
            'recording_enabled': False,
            'sdk_available': self._cl is not None,
            'simulator_preferred': prefer_simulator,
        }
        descriptor = self._build_descriptor(backend_id=backend_id)
        super().__init__(descriptor=descriptor)

    def describe(self) -> SubstrateDescriptor:
        return self.descriptor

    def prepare(self, task: TaskRequest) -> AdapterPreparationResult:
        if self._cl is None:
            self._last_telemetry['health_status'] = 'offline'
            return AdapterPreparationResult(
                prepared=False,
                details=(
                    'Cortical Labs SDK not available. Install cl-sdk or run on a CL1 system '
                    'to enable this adapter.'
                ),
            )

        self._configure_environment(task)
        supervision_required = self.descriptor.policy.human_supervision_required
        if supervision_required and not task.human_supervision_available:
            self._last_telemetry['health_status'] = 'degraded'
            return AdapterPreparationResult(
                prepared=False,
                details='Human supervision required for this wetware session but not available.',
            )

        self._last_telemetry['health_status'] = 'ready'
        self._last_telemetry['recording_enabled'] = bool(task.metadata.get('record', True))
        return AdapterPreparationResult(
            prepared=True,
            details='CL API available; adapter configured for live or simulator-backed session.',
        )

    def invoke(self, task: TaskRequest) -> AdapterInvocationResult:
        if self._cl is None:
            raise RuntimeError('Cortical Labs SDK is not installed.')

        self._configure_environment(task)
        channel = int(task.metadata.get('channel', 0))
        current_ua = float(task.metadata.get('current_uA', 6.0))
        observation_window_ms = float(task.metadata.get('observation_window_ms', 50.0))
        burst_count = int(task.metadata.get('burst_count', 1))
        burst_frequency_hz = float(task.metadata.get('burst_frequency_hz', 20.0))
        record = bool(task.metadata.get('record', True))

        start = time.perf_counter()
        recording_path = None
        system_attributes: dict[str, Any] = {}
        notes = []

        with self._cl.open(take_control=self._take_control, wait_until_recordable=True) as neurons:
            recording = neurons.record() if record and hasattr(neurons, 'record') else None
            burst_design = self._build_burst_design(burst_count=burst_count, burst_frequency_hz=burst_frequency_hz)
            try:
                if burst_design is None:
                    neurons.stim(channel, current_ua)
                else:
                    neurons.stim(channel, current_ua, burst_design)
            except TypeError:
                # Fall back to the simpler documented scalar-current form.
                neurons.stim(channel, current_ua)
                notes.append('Fell back to scalar-current stimulation call.')

            time.sleep(max(observation_window_ms, 0.0) / 1000.0)

            get_attrs = getattr(self._cl, 'get_system_attributes', None)
            if callable(get_attrs):
                try:
                    system_attributes = dict(get_attrs())
                except Exception:
                    system_attributes = {}

            if recording is not None:
                stop_result = recording.stop()
                recording_path = getattr(stop_result, 'file_location', None) or getattr(recording, 'file_location', None)

        execution_latency_ms = (time.perf_counter() - start) * 1000.0
        self._last_telemetry.update(
            {
                'health_status': 'ready',
                'age_of_information_ms': 0.0,
                'last_latency_ms': execution_latency_ms,
                'recording_enabled': record,
            }
        )

        output_payload = {
            'interface': 'cortical-labs-cl-api',
            'channel': channel,
            'current_uA': current_ua,
            'burst_count': burst_count,
            'burst_frequency_hz': burst_frequency_hz,
            'observation_window_ms': observation_window_ms,
            'recording_path': recording_path,
            'system_attributes': system_attributes,
            'simulator_preferred': self._prefer_simulator,
        }
        return AdapterInvocationResult(
            backend_id=self.backend_id(),
            task_id=task.task_id,
            output_payload=output_payload,
            confidence=None,
            execution_latency_ms=execution_latency_ms,
            backend_state='ready',
            notes='; '.join(notes) if notes else 'Cortical Labs CL API stimulation/recording cycle.',
        )

    def collect_telemetry(self) -> dict[str, float | int | str | bool | None]:
        return dict(self._last_telemetry)

    def reset(self, mode: ResetMode | None = None) -> bool:
        # The public CL API does not expose one uniform physical reset primitive.
        # For the prototype we model reset as clearing transient session state.
        self._last_telemetry['health_status'] = 'ready' if self._cl is not None else 'offline'
        self._last_telemetry['age_of_information_ms'] = 0.0 if self._cl is not None else None
        return self._cl is not None

    def recalibrate(self) -> bool:
        # Likewise treated as a session-level recovery step for the prototype.
        if self._cl is None:
            self._last_telemetry['health_status'] = 'offline'
            return False
        self._last_telemetry['drift_score'] = 0.08
        self._last_telemetry['health_status'] = 'ready'
        return True

    @staticmethod
    def _try_import_cl() -> Any | None:
        try:
            return import_module('cl')
        except Exception:
            return None

    def _configure_environment(self, task: TaskRequest) -> None:
        replay_path = task.metadata.get('cl_replay_path', self._replay_path)
        if replay_path:
            os.environ['CL_SDK_REPLAY_PATH'] = str(replay_path)
        if self._prefer_simulator:
            os.environ.setdefault('CL_SDK_ACCELERATED_TIME', '1' if self._accelerated_time else '0')

    def _build_burst_design(self, *, burst_count: int, burst_frequency_hz: float) -> Any | None:
        if burst_count <= 1 or self._cl is None:
            return None
        burst_cls = getattr(self._cl, 'BurstDesign', None)
        if burst_cls is None:
            return None
        for args, kwargs in [
            ((), {'count': burst_count, 'frequency_hz': burst_frequency_hz}),
            ((burst_count, burst_frequency_hz), {}),
        ]:
            try:
                return burst_cls(*args, **kwargs)
            except TypeError:
                continue
        return None

    @staticmethod
    def _build_descriptor(backend_id: str) -> SubstrateDescriptor:
        return SubstrateDescriptor(
            backend_id=backend_id,
            display_name='Cortical Labs CL API Backend',
            version='0.1.0',
            description=(
                'Optional adapter targeting the public Cortical Labs CL API / CL SDK Simulator '
                'for stimulation, recording, and closed-loop wetware interaction.'
            ),
            input_contracts=[
                IOContract(
                    name='stimulation_input',
                    modality=SignalModality.SPIKES,
                    encoding=IOEncoding.EVENT_STREAM,
                    description='Spike-oriented stimulation requests mapped onto CL API stimulation calls.',
                ),
                IOContract(
                    name='control_input',
                    modality=SignalModality.CONTROL_SIGNAL,
                    encoding=IOEncoding.JSON,
                    required=False,
                    description='Optional session and recording configuration metadata.',
                ),
            ],
            output_contracts=[
                IOContract(
                    name='recording_and_state',
                    modality=SignalModality.SPIKES,
                    encoding=IOEncoding.JSON,
                    description='Recording metadata and session-visible wetware state.',
                )
            ],
            timing=TimingContract(
                regime=TimingRegime.MILLISECONDS,
                typical_latency_ms=1.0,
                latency_jitter_ms=0.5,
                warmup_required=True,
                streaming_supported=True,
            ),
            lifecycle=LifecycleContract(
                supported_reset_modes=[ResetMode.SOFT_RESET, ResetMode.REST, ResetMode.RECALIBRATE],
                reprogrammable=True,
                recalibration_supported=True,
                stateful=True,
                notes='Physical reset and wetware handling remain application-specific; the adapter models session-level recovery.',
            ),
            telemetry=TelemetryContract(
                metrics=[
                    TelemetryField(name='last_latency_ms', units='ms', description='Most recent CL API round-trip latency.', lower_is_better=True),
                    TelemetryField(name='loop_timeout_events', units='count', description='Timeout or jitter events detected at the adapter layer.', lower_is_better=True),
                    TelemetryField(name='drift_score', units='fraction', description='Lightweight wetware drift/readiness proxy.', lower_is_better=True),
                ],
                supports_health_status=True,
                supports_confidence=False,
                supports_drift_reporting=True,
                supports_age_of_information=True,
            ),
            twin_binding=TwinBinding(
                twin_kind='external_api_or_simulator',
                fidelity_level='api-targeted',
                calibration_confidence=0.75,
                twin_notes='Targets the public CL API and its simulator; not part of the reported quantitative evaluation.',
            ),
            policy=PolicyConstraints(
                locality=Locality.LAB,
                tenancy=TenancyModel.RESERVED,
                safety_notes='Wetware access with explicit stimulation and recording semantics.',
                exclusive_access_required=True,
                human_supervision_required=True,
            ),
            capability=CapabilityDescriptor(
                substrate_class=SubstrateClass.WETWARE,
                supported_task_types=['monitoring', 'control', 'temporal_inference'],
                training_mode=TrainingMode.HYBRID,
                observability=ObservabilityLevel.PARTIAL,
                stochastic=True,
                resettable=True,
                programmable=True,
                health_sensitive=True,
                repeated_invocation_supported=True,
            ),
            custom_metadata={
                'paper_role': 'existing wetware API integration target',
                'sdk_package': 'cl-sdk',
            },
        )
