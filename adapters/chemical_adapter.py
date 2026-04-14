"""Chemical backend adapter for the phys-MCP prototype."""

from __future__ import annotations

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
from twins.chemical_twin import ChemicalTwin


class ChemicalAdapter(BaseAdapter):
    """Adapter exposing a chemical/dna-inspired backend to the control plane."""

    def __init__(self, backend_id: str = "chemical-backend", twin: ChemicalTwin | None = None) -> None:
        self._twin = twin or ChemicalTwin()
        descriptor = self._build_descriptor(backend_id=backend_id)
        super().__init__(descriptor=descriptor)

    def describe(self) -> SubstrateDescriptor:
        return self.descriptor

    def prepare(self, task: TaskRequest) -> AdapterPreparationResult:
        input_level = self._extract_input_level(task)
        prepared, details = self._twin.prepare(input_level=input_level)
        return AdapterPreparationResult(prepared=prepared, details=details)

    def invoke(self, task: TaskRequest) -> AdapterInvocationResult:
        input_level = self._extract_input_level(task)
        result = self._twin.run(input_level=input_level)
        return AdapterInvocationResult(
            backend_id=self.backend_id(),
            task_id=task.task_id,
            output_payload=result.output_payload,
            confidence=result.confidence,
            execution_latency_ms=result.execution_latency_ms,
            backend_state=result.backend_state,
            notes="Chemical/dna-inspired digital twin invocation.",
        )

    def collect_telemetry(self) -> dict[str, float | int | str | bool | None]:
        return self._twin.telemetry()

    def reset(self, mode: ResetMode | None = None) -> bool:
        return self._twin.reset(mode=mode)

    def recalibrate(self) -> bool:
        return self._twin.recalibrate()

    @staticmethod
    def _extract_input_level(task: TaskRequest) -> float:
        raw_value = task.metadata.get("input_level", 1.0)
        try:
            return float(raw_value)
        except (TypeError, ValueError):
            return 1.0

    @staticmethod
    def _build_descriptor(backend_id: str) -> SubstrateDescriptor:
        return SubstrateDescriptor(
            backend_id=backend_id,
            display_name="Chemical Twin Backend",
            version="0.1.0",
            description="A concentration-driven backend with slow convergence and explicit flush/recharge semantics.",
            input_contracts=[
                IOContract(
                    name="chemical_input",
                    modality=SignalModality.CONCENTRATION,
                    encoding=IOEncoding.VECTOR,
                    shape_hint=[1],
                    units="relative concentration",
                    description="Input concentration level for the simulated reaction network.",
                )
            ],
            output_contracts=[
                IOContract(
                    name="chemical_output",
                    modality=SignalModality.CONCENTRATION,
                    encoding=IOEncoding.JSON,
                    required=True,
                    description="Normalized concentration-derived result payload.",
                )
            ],
            timing=TimingContract(
                regime=TimingRegime.SECONDS,
                typical_latency_ms=9500.0,
                latency_jitter_ms=800.0,
                warmup_required=True,
                streaming_supported=False,
            ),
            lifecycle=LifecycleContract(
                supported_reset_modes=[ResetMode.FLUSH, ResetMode.RECHARGE, ResetMode.HARD_RESET],
                reprogrammable=True,
                recalibration_supported=True,
                stateful=True,
                notes="Repeated use increases contamination until the backend is flushed or reset.",
            ),
            telemetry=TelemetryContract(
                metrics=[
                    TelemetryField(name="contamination_level", units="fraction", description="Residual contamination proxy."),
                    TelemetryField(name="last_convergence_time_ms", units="ms", description="Most recent convergence time.", lower_is_better=True),
                    TelemetryField(name="drift_score", units="fraction", description="Combined calibration/contamination drift proxy.", lower_is_better=True),
                ],
                supports_health_status=True,
                supports_confidence=True,
                supports_drift_reporting=True,
                supports_age_of_information=True,
            ),
            twin_binding=TwinBinding(
                twin_kind="ode_simulation",
                fidelity_level="lightweight",
                calibration_confidence=0.90,
                twin_notes="Simple reaction-dynamics twin used for control-plane validation.",
            ),
            policy=PolicyConstraints(
                locality=Locality.LAB,
                tenancy=TenancyModel.SINGLE_TENANT,
                safety_notes="Represents wet-lab style resource usage with explicit lifecycle operations.",
                exclusive_access_required=False,
                human_supervision_required=False,
            ),
            capability=CapabilityDescriptor(
                substrate_class=SubstrateClass.CHEMICAL,
                supported_task_types=["classification", "sensing", "monitoring", "optimization"],
                training_mode=TrainingMode.EX_SITU,
                observability=ObservabilityLevel.PARTIAL,
                stochastic=True,
                resettable=True,
                programmable=True,
                health_sensitive=False,
                repeated_invocation_supported=True,
            ),
            custom_metadata={"paper_role": "slow concentration-driven substrate regime"},
        )
