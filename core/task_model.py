"""Application-facing task model for the phys-MCP prototype."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from descriptors.capability_schema import Locality, SignalModality


class TaskKind(str, Enum):
    """High-level task categories used by the control plane."""

    CLASSIFICATION = "classification"
    TEMPORAL_INFERENCE = "temporal_inference"
    OPTIMIZATION = "optimization"
    SENSING = "sensing"
    MONITORING = "monitoring"
    CONTROL = "control"


class OutputPreference(str, Enum):
    """Desired form of the result returned to the application."""

    LABEL = "label"
    SCORE = "score"
    VECTOR = "vector"
    STATE_ESTIMATE = "state_estimate"
    TELEMETRY_AWARE_RESULT = "telemetry_aware_result"


class TaskRequest(BaseModel):
    """A normalized task request submitted to the phys-MCP orchestrator."""

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    task_id: str = Field(..., description="Stable or externally visible task identifier.")
    task_kind: TaskKind = Field(..., description="Logical task category.")
    summary: str = Field(..., description="Short human-readable description of the requested operation.")
    required_input_modalities: list[SignalModality] = Field(
        default_factory=list,
        description="Input modalities acceptable to the application.",
    )
    preferred_output: OutputPreference = Field(
        default=OutputPreference.SCORE,
        description="Preferred output representation.",
    )
    latency_budget_ms: float = Field(
        default=1000.0,
        gt=0.0,
        description="Maximum acceptable end-to-end latency in milliseconds.",
    )
    min_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum acceptable confidence threshold, if exposed.",
    )
    stochasticity_tolerance: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="How much stochastic variability the application tolerates.",
    )
    continuous_monitoring_required: bool = Field(
        default=False,
        description="Whether the application requires ongoing telemetry-aware observation.",
    )
    repeated_invocation_expected: bool = Field(
        default=False,
        description="Whether the task is expected to be executed repeatedly in a short interval.",
    )
    reset_free_preferred: bool = Field(
        default=False,
        description="Whether the application prefers avoiding explicit reset/recovery steps.",
    )
    preferred_locality: Locality | None = Field(
        default=None,
        description="Desired deployment locality, if any.",
    )
    allow_fallback: bool = Field(
        default=True,
        description="Whether the orchestrator may reroute to a secondary backend if needed.",
    )
    direct_backend_id: str | None = Field(
        default=None,
        description="If set, restrict execution to this concrete backend identifier.",
    )
    max_twin_age_ms: float | None = Field(
        default=None,
        gt=0.0,
        description="Maximum acceptable age of information / twin staleness if exposed.",
    )
    required_telemetry_fields: list[str] = Field(
        default_factory=list,
        description="Telemetry fields that must be available for the task to be admissible.",
    )
    human_supervision_available: bool = Field(
        default=False,
        description="Whether required human supervision is available for this session.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional application-specific task metadata.",
    )

    @field_validator("task_id", "summary")
    @classmethod
    def validate_non_empty_strings(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("task_id and summary must not be empty.")
        return stripped

    @field_validator("direct_backend_id")
    @classmethod
    def validate_direct_backend_id(cls, value: str | None) -> str | None:
        if value is None:
            return value
        stripped = value.strip()
        if not stripped:
            raise ValueError("direct_backend_id must not be empty when provided.")
        return stripped

    @field_validator("required_input_modalities")
    @classmethod
    def validate_modalities(cls, value: list[SignalModality]) -> list[SignalModality]:
        if not value:
            raise ValueError("required_input_modalities must contain at least one modality.")
        return list(dict.fromkeys(value))

    @field_validator("required_telemetry_fields")
    @classmethod
    def validate_required_telemetry_fields(cls, value: list[str]) -> list[str]:
        normalized = [item.strip() for item in value if item.strip()]
        return list(dict.fromkeys(normalized))

    @model_validator(mode="after")
    def validate_monitoring_vs_latency(self) -> "TaskRequest":
        if self.continuous_monitoring_required and self.latency_budget_ms < 1.0:
            raise ValueError(
                "continuous_monitoring_required is not meaningful with a sub-millisecond latency budget "
                "in this prototype model."
            )
        return self

    def normalized_task_type(self) -> str:
        """Return the normalized task type string used by backend capability matching."""
        return str(self.task_kind)

    def short_label(self) -> str:
        """Return a concise label for logs and demo output."""
        return f"{self.task_id}:{str(self.task_kind)}"

    def prefers_low_variability(self) -> bool:
        """Return True if the task strongly prefers deterministic behavior."""
        return self.stochasticity_tolerance < 0.25
