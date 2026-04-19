"""Backend matching and ranking logic for the phys-MCP prototype."""

from __future__ import annotations

from typing import Iterable

from pydantic import BaseModel, ConfigDict, Field

from core.task_model import TaskRequest
from descriptors.capability_schema import Locality, SubstrateDescriptor, TelemetryField


class MatchCandidate(BaseModel):
    """One ranked backend candidate for a task request."""

    model_config = ConfigDict(extra="forbid")

    backend_id: str
    display_name: str
    accepted: bool
    score: float = Field(default=0.0, description="Higher is better.")
    reasons: list[str] = Field(default_factory=list)
    rejection_reasons: list[str] = Field(default_factory=list)

    def explanation_lines(self) -> list[str]:
        """Return a human-readable explanation for demo output."""
        lines = [f"[{self.backend_id}] score={self.score:.2f} accepted={self.accepted}"]
        lines.extend(f"  + {reason}" for reason in self.reasons)
        lines.extend(f"  - {reason}" for reason in self.rejection_reasons)
        return lines


class MatchReport(BaseModel):
    """Complete ranking report for a task request."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    candidates: list[MatchCandidate] = Field(default_factory=list)

    def accepted_candidates(self) -> list[MatchCandidate]:
        """Return accepted candidates ordered by score."""
        return [candidate for candidate in self.candidates if candidate.accepted]

    def best_candidate(self) -> MatchCandidate | None:
        """Return the highest-ranked accepted candidate, if any."""
        accepted = self.accepted_candidates()
        return accepted[0] if accepted else None


class BackendMatcher:
    """Explainable rule-based matcher for substrate selection.

    The matcher is intentionally heuristic, but it now considers both static
    descriptors and lightweight runtime state such as drift, health, and the age
    of information exposed by the adapters.
    """

    def rank_backends(
        self,
        task: TaskRequest,
        descriptors: Iterable[SubstrateDescriptor],
        runtime_state: dict[str, dict[str, float | int | str | bool | None]] | None = None,
    ) -> MatchReport:
        """Rank all available backends for the given task."""
        state = runtime_state or {}
        candidates = [
            self.score_descriptor(task, descriptor, runtime_state=state.get(descriptor.backend_id, {}))
            for descriptor in descriptors
        ]
        candidates.sort(key=lambda item: (item.accepted, item.score), reverse=True)
        return MatchReport(task_id=task.task_id, candidates=candidates)

    def score_descriptor(
        self,
        task: TaskRequest,
        descriptor: SubstrateDescriptor,
        runtime_state: dict[str, float | int | str | bool | None] | None = None,
    ) -> MatchCandidate:
        """Score one backend descriptor against one task request."""
        runtime = runtime_state or {}
        reasons: list[str] = []
        rejection_reasons: list[str] = []
        score = 0.0

        task_type = task.normalized_task_type()
        supported_modalities = {str(contract.modality) for contract in descriptor.input_contracts}
        required_modalities = {str(modality) for modality in task.required_input_modalities}
        declared_telemetry = self._declared_telemetry_names(descriptor)

        # Hard checks first.
        if task.direct_backend_id is not None and descriptor.backend_id != task.direct_backend_id:
            rejection_reasons.append(
                f"Task explicitly targets backend '{task.direct_backend_id}', not '{descriptor.backend_id}'."
            )

        if not descriptor.supports_task_type(task_type):
            rejection_reasons.append(
                f"Task type '{task_type}' is not listed in supported_task_types."
            )

        if not required_modalities.intersection(supported_modalities):
            rejection_reasons.append(
                "No overlap between required input modalities and backend input contracts."
            )

        if task.repeated_invocation_expected and not descriptor.capability.repeated_invocation_supported:
            rejection_reasons.append(
                "Task expects repeated invocation, but the backend does not declare repeated_invocation_supported."
            )

        if task.required_telemetry_fields:
            missing = sorted(set(task.required_telemetry_fields) - declared_telemetry)
            if missing:
                rejection_reasons.append(
                    "Required telemetry fields are not declared by the backend: " + ", ".join(missing)
                )

        if task.max_twin_age_ms is not None and not descriptor.telemetry.supports_age_of_information:
            rejection_reasons.append(
                "Task requires an age-of-information bound, but the backend does not expose it."
            )

        if descriptor.policy.human_supervision_required and not task.human_supervision_available:
            rejection_reasons.append(
                "Backend requires human supervision, but the task does not declare it as available."
            )

        health_status = str(runtime.get("health_status")) if runtime.get("health_status") is not None else None
        if health_status == "offline":
            rejection_reasons.append("Runtime state reports backend as offline.")

        drift_score = runtime.get("drift_score")
        if isinstance(drift_score, (int, float)) and drift_score >= 0.95:
            rejection_reasons.append(
                f"Runtime drift_score {drift_score:.2f} exceeds the admissible threshold for new sessions."
            )

        age_of_information = runtime.get("age_of_information_ms")
        if (
            task.max_twin_age_ms is not None
            and isinstance(age_of_information, (int, float))
            and age_of_information > task.max_twin_age_ms
        ):
            rejection_reasons.append(
                f"Runtime age_of_information_ms {age_of_information:.2f} exceeds task bound {task.max_twin_age_ms:.2f}."
            )

        accepted = not rejection_reasons
        if not accepted:
            return MatchCandidate(
                backend_id=descriptor.backend_id,
                display_name=descriptor.display_name,
                accepted=False,
                score=0.0,
                reasons=reasons,
                rejection_reasons=rejection_reasons,
            )

        score += 50.0
        reasons.append("Base acceptance score for passing hard compatibility checks.")

        if descriptor.supports_task_type(task_type):
            score += 20.0
            reasons.append(f"Backend supports task type '{task_type}'.")

        matched_modalities = required_modalities.intersection(supported_modalities)
        if matched_modalities:
            score += 15.0
            reasons.append(
                f"Input modality overlap found: {', '.join(sorted(matched_modalities))}."
            )

        latency = descriptor.timing.typical_latency_ms
        budget = task.latency_budget_ms

        if latency <= budget:
            score += 20.0
            reasons.append(
                f"Typical latency {latency:.2f} ms fits within budget {budget:.2f} ms."
            )
        else:
            overshoot_ratio = latency / budget
            penalty = min(30.0, 8.0 * overshoot_ratio)
            score -= penalty
            reasons.append(
                f"Latency exceeds budget ({latency:.2f} ms vs {budget:.2f} ms), penalty {penalty:.2f}."
            )

        if task.prefers_low_variability():
            if descriptor.capability.stochastic:
                score -= 8.0
                reasons.append("Task prefers low variability, but backend is stochastic.")
            else:
                score += 6.0
                reasons.append("Task prefers low variability and backend is comparatively deterministic.")
        else:
            if descriptor.capability.stochastic:
                score += 2.0
                reasons.append("Task tolerates variability; stochastic backend is acceptable.")

        if task.continuous_monitoring_required:
            if descriptor.telemetry.supports_health_status:
                score += 8.0
                reasons.append("Continuous monitoring requested and health telemetry is available.")
            else:
                score -= 10.0
                reasons.append("Continuous monitoring requested, but health telemetry is limited.")

            if descriptor.telemetry.supports_drift_reporting:
                score += 4.0
                reasons.append("Backend exposes drift reporting for supervision.")

        if task.required_telemetry_fields:
            score += 2.0 * len(task.required_telemetry_fields)
            reasons.append(
                "Backend declares required telemetry fields: " + ", ".join(task.required_telemetry_fields)
            )

        if task.max_twin_age_ms is not None:
            score += 6.0
            reasons.append("Backend exposes age-of-information required by the task.")

        if task.reset_free_preferred:
            if descriptor.lifecycle.stateful and descriptor.lifecycle.supported_reset_modes:
                score -= 8.0
                reasons.append("Task prefers reset-free execution, but backend has explicit reset semantics.")
            else:
                score += 4.0
                reasons.append("Backend is comparatively reset-light for this prototype.")

        if task.preferred_locality is not None:
            requested_locality = str(task.preferred_locality)
            available_locality = str(descriptor.policy.locality)
            if available_locality == requested_locality:
                score += 6.0
                reasons.append(
                    f"Backend locality matches preferred locality '{requested_locality}'."
                )
            else:
                mismatch_penalty = self._locality_penalty(
                    requested=requested_locality,
                    available=available_locality,
                )
                score -= mismatch_penalty
                reasons.append(
                    f"Backend locality '{available_locality}' differs from preferred "
                    f"'{requested_locality}', penalty {mismatch_penalty:.2f}."
                )

        if descriptor.capability.health_sensitive:
            score -= 5.0
            reasons.append("Backend is health-sensitive and may require more careful lifecycle handling.")

        if health_status == "degraded":
            score -= 20.0
            reasons.append("Runtime health status is degraded, reducing selection score.")
        elif health_status == "ready":
            score += 4.0
            reasons.append("Runtime health status is ready.")

        if isinstance(drift_score, (int, float)):
            if drift_score <= 0.25:
                score += 6.0
                reasons.append(f"Runtime drift_score {drift_score:.2f} is low.")
            elif drift_score <= 0.75:
                penalty = 12.0 * float(drift_score)
                score -= penalty
                reasons.append(
                    f"Runtime drift_score {drift_score:.2f} incurs penalty {penalty:.2f}."
                )

        if isinstance(age_of_information, (int, float)) and task.max_twin_age_ms is not None:
            freshness_ratio = float(age_of_information) / float(task.max_twin_age_ms)
            if freshness_ratio <= 0.5:
                score += 4.0
                reasons.append("Runtime twin state is comfortably within the freshness budget.")
            else:
                penalty = 6.0 * freshness_ratio
                score -= penalty
                reasons.append(
                    f"Runtime age-of-information consumes much of the freshness budget, penalty {penalty:.2f}."
                )

        return MatchCandidate(
            backend_id=descriptor.backend_id,
            display_name=descriptor.display_name,
            accepted=True,
            score=max(score, 0.0),
            reasons=reasons,
            rejection_reasons=rejection_reasons,
        )

    @staticmethod
    def _declared_telemetry_names(descriptor: SubstrateDescriptor) -> set[str]:
        names = {field.name for field in descriptor.telemetry.metrics}
        if descriptor.telemetry.supports_health_status:
            names.add("health_status")
        if descriptor.telemetry.supports_confidence:
            names.update({"confidence", "last_confidence", "calibration_confidence"})
        if descriptor.telemetry.supports_drift_reporting:
            names.add("drift_score")
        if descriptor.telemetry.supports_age_of_information:
            names.add("age_of_information_ms")
        return names

    @staticmethod
    def _locality_penalty(requested: str, available: str) -> float:
        """Return a simple locality mismatch penalty."""
        if requested == available:
            return 0.0

        if requested == str(Locality.EDGE) and available in {str(Locality.FOG), str(Locality.LOCAL)}:
            return 3.0
        if requested == str(Locality.FOG) and available in {str(Locality.EDGE), str(Locality.CLOUD), str(Locality.LOCAL)}:
            return 3.0
        if requested == str(Locality.CLOUD) and available in {str(Locality.FOG), str(Locality.LOCAL)}:
            return 2.0
        if requested == str(Locality.LAB) and available == str(Locality.LOCAL):
            return 2.0

        return 6.0
