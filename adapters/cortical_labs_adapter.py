"""Cortical Labs adapter for the phys-MCP prototype."""

from __future__ import annotations

import time

from adapters.base_adapter import AdapterInvocationResult, AdapterPreparationResult, BaseAdapter
from backends.cortical.cl_client import (
    CLClient,
    CorticalLabsInvocationError,
    CorticalLabsUnavailableError,
)
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
    def __init__(self, backend_id: str = "cortical-labs-backend") -> None:
        self._client = CLClient()
        self._session_open = False
        self._last_backend_latency_ms: float | None = None
        self._last_observation_latency_ms: float | None = None
        self._last_recording_artifact: dict | None = None
        self._last_health_status: str = "unknown"
        self._last_readiness_state: str = "unavailable"
        self._last_channel_count: int | None = None
        self._last_fps: float | None = None
        self._last_age_of_information_ms: float | None = None
        self._last_prepare_timestamp: float | None = None

        descriptor = self._build_descriptor(backend_id=backend_id)
        super().__init__(descriptor=descriptor)

    def describe(self) -> SubstrateDescriptor:
        return self.descriptor

    def prepare(self, task: TaskRequest) -> AdapterPreparationResult:
        human_supervision_available = getattr(task, "human_supervision_available", True)
        if not human_supervision_available:
            self._last_readiness_state = "rejected"
            self._last_health_status = "unknown"
            return AdapterPreparationResult(
                prepared=False,
                details="Cortical Labs backend requires human supervision.",
            )

        if not self._client.is_available():
            self._session_open = False
            self._last_readiness_state = "unavailable"
            self._last_health_status = "unknown"
            return AdapterPreparationResult(
                prepared=False,
                details="Cortical Labs SDK is not installed or importable.",
            )

        try:
            info = self._client.open_session()
        except CorticalLabsUnavailableError as exc:
            self._session_open = False
            self._last_readiness_state = "unavailable"
            self._last_health_status = "unknown"
            return AdapterPreparationResult(prepared=False, details=str(exc))

        self._session_open = True
        self._last_readiness_state = "ready"
        self._last_health_status = "healthy"
        self._last_channel_count = info.channel_count
        self._last_fps = info.fps
        self._last_age_of_information_ms = 0.0
        self._last_prepare_timestamp = time.perf_counter()

        details = "Cortical Labs session opened successfully."
        if info.channel_count is not None:
            details += f" channels={info.channel_count}"
        if info.fps is not None:
            details += f", fps={info.fps}"

        return AdapterPreparationResult(prepared=True, details=details)

    def invoke(self, task: TaskRequest) -> AdapterInvocationResult:
        if not self._session_open:
            return AdapterInvocationResult(
                backend_id=self.backend_id(),
                task_id=task.task_id,
                output_payload={},
                confidence=None,
                execution_latency_ms=0.0,
                backend_state="unavailable",
                notes="Cortical Labs session is not open; call prepare() first.",
            )

        channel, amplitude_ua = self._extract_stimulation(task)
        observation_window_ms = self._extract_observation_window(task)
        pre_delay_ms = self._extract_pre_delay(task)

        try:
            result = self._client.stimulate_and_record(
                channel=channel,
                amplitude_ua=amplitude_ua,
                observation_window_ms=observation_window_ms,
                pre_delay_ms=pre_delay_ms,
            )
        except CorticalLabsInvocationError as exc:
            self._last_health_status = "degraded"
            self._last_readiness_state = "ready"
            return AdapterInvocationResult(
                backend_id=self.backend_id(),
                task_id=task.task_id,
                output_payload={},
                confidence=None,
                execution_latency_ms=0.0,
                backend_state="error",
                notes=str(exc),
            )

        self._last_backend_latency_ms = result.backend_latency_ms
        self._last_observation_latency_ms = result.observation_latency_ms
        self._last_recording_artifact = result.recording_artifact
        self._last_health_status = "healthy"
        self._last_readiness_state = "ready"
        self._last_age_of_information_ms = 0.0

        output_payload = {
            "response_fingerprint": result.response_summary.get(
                "response_fingerprint",
                "recording_completed",
            ),
            "observation_window_ms": observation_window_ms,
            "stim_channel": channel,
            "stim_amplitude_ua": amplitude_ua,
            "recording_artifact": result.recording_artifact,
            "raw_backend_metadata": result.raw_backend_metadata,
        }

        notes = "Cortical Labs stimulation/recording cycle completed."
        if result.recording_artifact and result.recording_artifact.get("path"):
            notes += f" recording_path={result.recording_artifact['path']}"

        return AdapterInvocationResult(
            backend_id=self.backend_id(),
            task_id=task.task_id,
            output_payload=output_payload,
            confidence=0.75 if result.success else 0.0,
            execution_latency_ms=result.backend_latency_ms,
            backend_state="ready",
            notes=notes,
        )

    def collect_telemetry(self) -> dict[str, float | int | str | bool | None]:
        health = self._client.get_health_status()
        readiness_state = str(health.get("readiness_state", self._last_readiness_state))
        health_status = str(health.get("health_status", self._last_health_status))
        channel_count = health.get("channel_count", self._last_channel_count)
        fps = health.get("fps", self._last_fps)

        telemetry = {
            "readiness_state": readiness_state,
            "health_status": health_status,
            "backend_latency_ms": self._last_backend_latency_ms,
            "observation_latency_ms": self._last_observation_latency_ms,
            "channel_count": channel_count,
            "fps": fps,
            "drift_score": 0.0 if self._session_open else None,
            "age_of_information_ms": self._last_age_of_information_ms,
            "sdk_available": self._client.is_available(),
        }

        if self._last_recording_artifact and self._last_recording_artifact.get("path"):
            telemetry["recording_path"] = self._last_recording_artifact["path"]

        return telemetry

    def reset(self, mode: ResetMode | None = None) -> bool:
        self._client.close_session()
        self._session_open = False
        self._last_readiness_state = "unavailable"
        self._last_health_status = "unknown"
        self._last_backend_latency_ms = None
        self._last_observation_latency_ms = None
        self._last_recording_artifact = None
        self._last_age_of_information_ms = None
        return True

    def recalibrate(self) -> bool:
        self._client.close_session()
        self._session_open = False
        self._last_readiness_state = "unavailable"
        self._last_health_status = "unknown"
        if not self._client.is_available():
            return False
        try:
            info = self._client.open_session()
        except CorticalLabsUnavailableError:
            return False
        self._session_open = True
        self._last_readiness_state = "ready"
        self._last_health_status = "healthy"
        self._last_channel_count = info.channel_count
        self._last_fps = info.fps
        self._last_age_of_information_ms = 0.0
        self._last_prepare_timestamp = time.perf_counter()
        return True

    @staticmethod
    def _extract_stimulation(task: TaskRequest) -> tuple[int, float]:
        pattern = task.metadata.get("stimulation_pattern", {})
        channels = pattern.get("channels", [1])
        amplitude = pattern.get("amplitude", 0.4)
        try:
            channel = int(channels[0]) if channels else 1
        except (TypeError, ValueError, IndexError):
            channel = 1
        try:
            amplitude_ua = float(amplitude)
        except (TypeError, ValueError):
            amplitude_ua = 0.4
        return channel, amplitude_ua

    @staticmethod
    def _extract_observation_window(task: TaskRequest) -> int:
        raw_value = task.metadata.get("observation_window_ms", 100)
        try:
            return int(raw_value)
        except (TypeError, ValueError):
            return 100

    @staticmethod
    def _extract_pre_delay(task: TaskRequest) -> int:
        raw_value = task.metadata.get("pre_delay_ms", 20)
        try:
            return int(raw_value)
        except (TypeError, ValueError):
            return 20

    @staticmethod
    def _build_descriptor(backend_id: str) -> SubstrateDescriptor:
        return SubstrateDescriptor(
            backend_id=backend_id,
            display_name="Cortical Labs CL API Backend",
            version="0.1.1",
            description=(
                "Optional adapter targeting the public Cortical Labs CL API / "
                "CL SDK Simulator for stimulation, recording, and closed-loop "
                "wetware interaction."
            ),
            input_contracts=[
                IOContract(
                    name="stimulation_input",
                    modality=SignalModality.SPIKES,
                    encoding=IOEncoding.EVENT_STREAM,
                    description="Spike-oriented stimulation requests mapped onto CL API stimulation calls.",
                ),
                IOContract(
                    name="control_input",
                    modality=SignalModality.CONTROL_SIGNAL,
                    encoding=IOEncoding.JSON,
                    required=False,
                    description="Optional session and recording configuration metadata.",
                ),
            ],
            output_contracts=[
                IOContract(
                    name="recording_and_state",
                    modality=SignalModality.SPIKES,
                    encoding=IOEncoding.JSON,
                    description="Recording metadata and session-visible wetware state.",
                )
            ],
            timing=TimingContract(
                regime=TimingRegime.MILLISECONDS,
                typical_latency_ms=100.0,
                latency_jitter_ms=20.0,
                warmup_required=True,
                streaming_supported=True,
            ),
            lifecycle=LifecycleContract(
                supported_reset_modes=[ResetMode.SOFT_RESET, ResetMode.REST, ResetMode.RECALIBRATE],
                reprogrammable=True,
                recalibration_supported=True,
                stateful=True,
                notes="Physical reset and wetware handling remain application-specific; the adapter models session-level recovery.",
            ),
            telemetry=TelemetryContract(
                metrics=[
                    TelemetryField(
                        name="backend_latency_ms",
                        units="ms",
                        description="Most recent CL API round-trip latency including the observation cycle.",
                        lower_is_better=True,
                    ),
                    TelemetryField(
                        name="observation_latency_ms",
                        units="ms",
                        description="Latency measured from stimulation until the observation cycle completes.",
                        lower_is_better=True,
                    ),
                    TelemetryField(
                        name="readiness_state",
                        units="state",
                        description="Current readiness state of the Cortical Labs session.",
                        lower_is_better=None,
                    ),
                    TelemetryField(
                        name="health_status",
                        units="state",
                        description="Current wetware/session health status exposed by the adapter.",
                        lower_is_better=None,
                    ),
                    TelemetryField(
                        name="recording_path",
                        units="path",
                        description="Path to the most recent recording artifact when available.",
                        lower_is_better=None,
                    ),
                    TelemetryField(
                        name="channel_count",
                        units="count",
                        description="Channel count reported by the active CL session.",
                        lower_is_better=None,
                    ),
                    TelemetryField(
                        name="fps",
                        units="frames_per_second",
                        description="Frames per second reported by the CL session.",
                        lower_is_better=None,
                    ),
                    TelemetryField(
                        name="drift_score",
                        units="fraction",
                        description="Lightweight wetware drift/readiness proxy.",
                        lower_is_better=True,
                    ),
                    TelemetryField(
                        name="age_of_information_ms",
                        units="ms",
                        description="Approximate age of the current telemetry snapshot.",
                        lower_is_better=True,
                    ),
                ],
                supports_health_status=True,
                supports_confidence=False,
                supports_drift_reporting=True,
                supports_age_of_information=True,
            ),
            twin_binding=TwinBinding(
                twin_kind="external_api_or_simulator",
                fidelity_level="api_targeted",
                calibration_confidence=0.75,
                twin_notes="Targets the public CL API and its simulator; not part of the reported quantitative evaluation.",
            ),
            policy=PolicyConstraints(
                locality=Locality.LAB,
                tenancy=TenancyModel.RESERVED,
                safety_notes="Wetware access with explicit stimulation and recording semantics.",
                exclusive_access_required=True,
                human_supervision_required=True,
            ),
            capability=CapabilityDescriptor(
                substrate_class=SubstrateClass.WETWARE,
                supported_task_types=["monitoring", "control", "temporal_inference"],
                training_mode=TrainingMode.HYBRID,
                observability=ObservabilityLevel.PARTIAL,
                stochastic=True,
                resettable=True,
                programmable=True,
                health_sensitive=True,
                repeated_invocation_supported=True,
            ),
            custom_metadata={
                "paper_role": "existing wetware API integration target",
                "sdk_package": "cl-sdk",
            },
        )
