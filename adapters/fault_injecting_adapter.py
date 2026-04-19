"""Fault-injecting adapter wrapper used for robustness evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field

from adapters.base_adapter import AdapterInvocationResult, AdapterPreparationResult, BaseAdapter
from core.task_model import TaskRequest
from descriptors.capability_schema import ResetMode, SubstrateDescriptor


@dataclass
class FaultProfile:
    """One configurable fault profile for an adapter wrapper."""

    prepare_failure_message: str | None = None
    invoke_failure_message: str | None = None
    drop_telemetry_fields: set[str] = field(default_factory=set)
    override_telemetry: dict[str, float | int | str | bool | None] = field(default_factory=dict)
    one_shot: bool = False


class FaultInjectingAdapter(BaseAdapter):
    """Adapter wrapper that injects preparation, invocation, or telemetry faults."""

    def __init__(self, wrapped: BaseAdapter, fault_profile: FaultProfile | None = None) -> None:
        self._wrapped = wrapped
        self._fault_profile = fault_profile or FaultProfile()
        super().__init__(descriptor=wrapped.describe())

    def describe(self) -> SubstrateDescriptor:
        return self._wrapped.describe()

    def configure(self, fault_profile: FaultProfile) -> None:
        self._fault_profile = fault_profile

    def clear_faults(self) -> None:
        self._fault_profile = FaultProfile()

    def prepare(self, task: TaskRequest) -> AdapterPreparationResult:
        if self._fault_profile.prepare_failure_message is not None:
            message = self._fault_profile.prepare_failure_message
            self._consume_one_shot_if_needed()
            return AdapterPreparationResult(prepared=False, details=message)
        return self._wrapped.prepare(task)

    def invoke(self, task: TaskRequest) -> AdapterInvocationResult:
        if self._fault_profile.invoke_failure_message is not None:
            message = self._fault_profile.invoke_failure_message
            self._consume_one_shot_if_needed()
            raise RuntimeError(message)
        return self._wrapped.invoke(task)

    def collect_telemetry(self) -> dict[str, float | int | str | bool | None]:
        telemetry = dict(self._wrapped.collect_telemetry())
        modified = False
        for key in self._fault_profile.drop_telemetry_fields:
            if key in telemetry:
                telemetry.pop(key, None)
                modified = True
        if self._fault_profile.override_telemetry:
            telemetry.update(self._fault_profile.override_telemetry)
            modified = True
        if modified:
            self._consume_one_shot_if_needed()
        return telemetry

    def reset(self, mode: ResetMode | None = None) -> bool:
        return self._wrapped.reset(mode=mode)

    def recalibrate(self) -> bool:
        return self._wrapped.recalibrate()

    def _consume_one_shot_if_needed(self) -> None:
        if self._fault_profile.one_shot:
            self.clear_faults()
