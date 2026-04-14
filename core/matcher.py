"""Backend matching and ranking logic for the phys-MCP prototype."""

from __future__ import annotations

from typing import Iterable

from pydantic import BaseModel, ConfigDict, Field

from core.task_model import TaskRequest
from descriptors.capability_schema import Locality, SubstrateDescriptor


def _enumish_value(value: object) -> str:
    """Return a stable string representation for either enum-like or plain values."""
    return getattr(value, "value", value)  # type: ignore[return-value]


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

    The matcher is intentionally simple. It is designed to make the control-plane
    selection logic transparent and easy to analyze in the paper.
    """

    def rank_backends(
        self,
        task: TaskRequest,
        descriptors: Iterable[SubstrateDescriptor],
    ) -> MatchReport:
        """Rank all available backends for the given task."""
        candidates = [self.score_descriptor(task, descriptor) for descriptor in descriptors]
        candidates.sort(key=lambda item: (item.accepted, item.score), reverse=True)
        return MatchReport(task_id=task.task_id, candidates=candidates)

    def score_descriptor(self, task: TaskRequest, descriptor: SubstrateDescriptor) -> MatchCandidate:
        """Score one backend descriptor against one task request."""
        reasons: list[str] = []
        rejection_reasons: list[str] = []
        score = 0.0

        task_type = task.normalized_task_type()
        supported_modalities = {str(contract.modality) for contract in descriptor.input_contracts}
        required_modalities = {str(modality) for modality in task.required_input_modalities}

        # Hard checks first.
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
                score -= 15.0
                reasons.append("Task prefers low variability, but backend is declared stochastic.")
            else:
                score += 8.0
                reasons.append("Task prefers low variability and backend is not declared stochastic.")
        elif descriptor.capability.stochastic:
            score += 2.0
            reasons.append("Task tolerates stochasticity and backend may exploit it.")

        if task.min_confidence > 0.0:
            if descriptor.telemetry.supports_confidence:
                score += 5.0
                reasons.append("Backend exposes confidence-related telemetry.")
            else:
                score -= 8.0
                reasons.append("Task requests confidence awareness, but backend does not expose it.")

        if task.continuous_monitoring_required:
            if descriptor.timing.streaming_supported:
                score += 8.0
                reasons.append("Continuous monitoring requested and streaming is supported.")
            else:
                score -= 10.0
                reasons.append("Continuous monitoring requested, but streaming is not supported.")

            if descriptor.telemetry.supports_health_status:
                score += 4.0
                reasons.append("Backend exposes health status for ongoing supervision.")

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

        return MatchCandidate(
            backend_id=descriptor.backend_id,
            display_name=descriptor.display_name,
            accepted=True,
            score=max(score, 0.0),
            reasons=reasons,
            rejection_reasons=rejection_reasons,
        )

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
