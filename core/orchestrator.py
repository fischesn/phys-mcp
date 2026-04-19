"""Main control-plane orchestrator for the phys-MCP prototype."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from adapters.base_adapter import (
    AdapterInvocationResult,
    AdapterPreparationResult,
    BaseAdapter,
)
from core.matcher import BackendMatcher, MatchCandidate, MatchReport
from core.task_model import TaskRequest
from core.twin_registry import TwinRegistry
from descriptors.capability_schema import ResetMode


class OrchestrationDecision(BaseModel):
    """Selection result returned before or during execution."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    selected_backend_id: str | None = None
    selected_score: float | None = None
    used_fallback: bool = False
    ranked_candidates: list[MatchCandidate] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class OrchestrationRunResult(BaseModel):
    """End-to-end result of an orchestrated task execution."""

    model_config = ConfigDict(extra="forbid")

    decision: OrchestrationDecision
    preparation: AdapterPreparationResult | None = None
    invocation: AdapterInvocationResult | None = None
    telemetry_before: dict[str, float | int | str | bool | None] = Field(default_factory=dict)
    telemetry_after: dict[str, float | int | str | bool | None] = Field(default_factory=dict)
    recovery_actions: list[str] = Field(default_factory=list)
    validation_failures: list[str] = Field(default_factory=list)
    success: bool = False
    failure_reason: str | None = None


class PhysMCPOrchestrator:
    """Coordinates discovery, matching, invocation, and recovery."""

    def __init__(
        self,
        registry: TwinRegistry | None = None,
        matcher: BackendMatcher | None = None,
    ) -> None:
        self._registry = registry or TwinRegistry()
        self._matcher = matcher or BackendMatcher()

    @property
    def registry(self) -> TwinRegistry:
        """Expose the underlying registry."""
        return self._registry

    def register_adapter(self, adapter: BaseAdapter, *, overwrite: bool = False) -> None:
        """Register one backend adapter with the control plane."""
        self._registry.register(adapter=adapter, overwrite=overwrite)

    def discover_backends(self) -> list[dict]:
        """Return published descriptors for all known backends."""
        return [descriptor.to_public_dict() for descriptor in self._registry.list_descriptors()]

    def plan_task(self, task: TaskRequest) -> MatchReport:
        """Rank all known backends for the given task request."""
        descriptors = self._registry.list_descriptors()
        if task.direct_backend_id is not None:
            descriptors = [descriptor for descriptor in descriptors if descriptor.backend_id == task.direct_backend_id]
        runtime_state = self._registry.telemetry_snapshot()
        return self._matcher.rank_backends(
            task=task,
            descriptors=descriptors,
            runtime_state=runtime_state,
        )

    def execute_task(self, task: TaskRequest) -> OrchestrationRunResult:
        """Execute one task through the selected backend.

        If allowed by the task request, the orchestrator may fall back to the next
        accepted candidate when preparation, invocation, or postcondition checks fail.
        """
        report = self.plan_task(task)
        decision = OrchestrationDecision(
            task_id=task.task_id,
            ranked_candidates=report.candidates,
            notes=[],
        )

        accepted_candidates = report.accepted_candidates()
        if not accepted_candidates:
            if task.direct_backend_id and not self._registry.has_backend(task.direct_backend_id):
                decision.notes.append(f"Directed backend '{task.direct_backend_id}' is not registered.")
                failure_reason = f"Directed backend '{task.direct_backend_id}' not found."
            else:
                decision.notes.append("No accepted backend candidates were found.")
                failure_reason = "No compatible backend found."
            return OrchestrationRunResult(
                decision=decision,
                success=False,
                failure_reason=failure_reason,
            )

        candidate_queue = accepted_candidates
        last_failure_reason: str | None = None

        for index, candidate in enumerate(candidate_queue):
            adapter = self._registry.get_adapter(candidate.backend_id)
            decision.selected_backend_id = candidate.backend_id
            decision.selected_score = candidate.score
            decision.used_fallback = index > 0

            if decision.used_fallback:
                decision.notes.append(
                    f"Using fallback candidate '{candidate.backend_id}' after earlier failure."
                )
            else:
                decision.notes.append(
                    f"Selected primary candidate '{candidate.backend_id}'."
                )

            telemetry_before = adapter.collect_telemetry()
            preparation = adapter.prepare(task)
            if not preparation.prepared:
                last_failure_reason = (
                    f"Preparation failed for backend '{candidate.backend_id}': {preparation.details}"
                )
                decision.notes.append(last_failure_reason)
                if not task.allow_fallback:
                    return OrchestrationRunResult(
                        decision=decision,
                        preparation=preparation,
                        telemetry_before=telemetry_before,
                        success=False,
                        failure_reason=last_failure_reason,
                    )
                continue

            try:
                invocation = adapter.invoke(task)
            except Exception as exc:  # pragma: no cover - defensive path for prototype robustness
                last_failure_reason = (
                    f"Invocation failed for backend '{candidate.backend_id}': {exc}"
                )
                decision.notes.append(last_failure_reason)
                if not task.allow_fallback:
                    return OrchestrationRunResult(
                        decision=decision,
                        preparation=preparation,
                        telemetry_before=telemetry_before,
                        success=False,
                        failure_reason=last_failure_reason,
                    )
                continue

            telemetry_after = adapter.collect_telemetry()
            validation_failures = self._validate_postconditions(task, invocation, telemetry_after)
            recovery_actions = self._maybe_recover(adapter, telemetry_after)

            if validation_failures:
                last_failure_reason = (
                    f"Postcondition validation failed for backend '{candidate.backend_id}': "
                    + "; ".join(validation_failures)
                )
                decision.notes.append(last_failure_reason)
                if not task.allow_fallback:
                    return OrchestrationRunResult(
                        decision=decision,
                        preparation=preparation,
                        invocation=invocation,
                        telemetry_before=telemetry_before,
                        telemetry_after=telemetry_after,
                        recovery_actions=recovery_actions,
                        validation_failures=validation_failures,
                        success=False,
                        failure_reason=last_failure_reason,
                    )
                continue

            return OrchestrationRunResult(
                decision=decision,
                preparation=preparation,
                invocation=invocation,
                telemetry_before=telemetry_before,
                telemetry_after=telemetry_after,
                recovery_actions=recovery_actions,
                validation_failures=validation_failures,
                success=True,
            )

        return OrchestrationRunResult(
            decision=decision,
            success=False,
            failure_reason=last_failure_reason or "All accepted candidates failed.",
        )

    def reset_backend(self, backend_id: str, mode: ResetMode | None = None) -> bool:
        """Reset one backend through its adapter."""
        adapter = self._registry.get_adapter(backend_id)
        return adapter.reset(mode=mode)

    def recalibrate_backend(self, backend_id: str) -> bool:
        """Recalibrate one backend through its adapter."""
        adapter = self._registry.get_adapter(backend_id)
        return adapter.recalibrate()

    @staticmethod
    def _validate_postconditions(
        task: TaskRequest,
        invocation: AdapterInvocationResult,
        telemetry_after: dict[str, float | int | str | bool | None],
    ) -> list[str]:
        failures: list[str] = []

        if invocation.confidence is not None and invocation.confidence < task.min_confidence:
            failures.append(
                f"confidence {invocation.confidence:.3f} is below required threshold {task.min_confidence:.3f}"
            )

        if task.required_telemetry_fields:
            missing = sorted(
                field for field in task.required_telemetry_fields if field not in telemetry_after
            )
            if missing:
                failures.append("missing required telemetry fields: " + ", ".join(missing))

        if task.max_twin_age_ms is not None:
            age = telemetry_after.get("age_of_information_ms")
            if not isinstance(age, (int, float)):
                failures.append("age_of_information_ms is required but missing")
            elif age > task.max_twin_age_ms:
                failures.append(
                    f"age_of_information_ms {age:.2f} exceeds bound {task.max_twin_age_ms:.2f}"
                )

        if task.continuous_monitoring_required and "health_status" not in telemetry_after:
            failures.append("continuous monitoring required but health_status is missing")

        if telemetry_after.get("health_status") == "offline":
            failures.append("backend transitioned to offline state")

        return failures

    @staticmethod
    def _maybe_recover(
        adapter: BaseAdapter,
        telemetry_after: dict[str, float | int | str | bool | None],
    ) -> list[str]:
        """Trigger lightweight recovery actions based on telemetry hints.

        This logic is intentionally small and heuristic-based. It only demonstrates
        that the control plane can react to backend state.
        """
        actions: list[str] = []

        drift_score = telemetry_after.get("drift_score")
        if isinstance(drift_score, (int, float)) and drift_score > 0.8:
            if adapter.recalibrate():
                actions.append("Triggered recalibration due to high drift_score.")

        health_status = telemetry_after.get("health_status")
        if health_status == "degraded":
            if adapter.reset():
                actions.append("Triggered reset due to degraded health_status.")

        return actions
