"""Abstract adapter interface for phys-MCP backend integrations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from core.task_model import TaskRequest
from descriptors.capability_schema import ResetMode, SubstrateDescriptor


class AdapterPreparationResult(BaseModel):
    """Outcome of a backend preparation step."""

    model_config = ConfigDict(extra="forbid")

    prepared: bool = Field(..., description="Whether preparation succeeded.")
    details: str = Field(default="", description="Short preparation summary.")


class AdapterInvocationResult(BaseModel):
    """Normalized result returned by a backend adapter invocation."""

    model_config = ConfigDict(extra="forbid")

    backend_id: str = Field(..., description="Identifier of the backend that produced the result.")
    task_id: str = Field(..., description="Identifier of the task that was executed.")
    output_payload: dict[str, Any] = Field(
        default_factory=dict,
        description="Backend output payload normalized into a Python dictionary.",
    )
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional confidence value reported by the backend.",
    )
    execution_latency_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Observed or estimated execution latency.",
    )
    backend_state: str = Field(
        default="ready",
        description="Backend-reported lifecycle state after invocation.",
    )
    notes: str | None = Field(default=None, description="Optional result notes.")


class BaseAdapter(ABC):
    """Common adapter interface used by the phys-MCP control plane.

    Each concrete adapter translates generic control-plane operations into the
    substrate-specific behavior of one backend or one digital twin.
    """

    def __init__(self, descriptor: SubstrateDescriptor) -> None:
        self._descriptor = descriptor

    @property
    def descriptor(self) -> SubstrateDescriptor:
        """Return the immutable descriptor published by this adapter."""
        return self._descriptor

    def backend_id(self) -> str:
        """Return the backend identifier."""
        return self._descriptor.backend_id

    @abstractmethod
    def describe(self) -> SubstrateDescriptor:
        """Return the published substrate descriptor."""
        raise NotImplementedError

    @abstractmethod
    def prepare(self, task: TaskRequest) -> AdapterPreparationResult:
        """Prepare the backend for task execution.

        Examples include warmup, priming, loading weights, or validating the
        task against backend-specific preconditions.
        """
        raise NotImplementedError

    @abstractmethod
    def invoke(self, task: TaskRequest) -> AdapterInvocationResult:
        """Execute the task through the backend and return a normalized result."""
        raise NotImplementedError

    @abstractmethod
    def collect_telemetry(self) -> dict[str, float | int | str | bool | None]:
        """Return the latest telemetry snapshot from the backend."""
        raise NotImplementedError

    @abstractmethod
    def reset(self, mode: ResetMode | None = None) -> bool:
        """Reset or recover the backend state.

        If *mode* is None, the adapter may choose a sensible default reset mode.
        """
        raise NotImplementedError

    @abstractmethod
    def recalibrate(self) -> bool:
        """Trigger backend recalibration if supported."""
        raise NotImplementedError
