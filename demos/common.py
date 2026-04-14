"""Shared demo helpers for the phys-MCP prototype."""

from __future__ import annotations

import sys
from pathlib import Path


def bootstrap_project_root() -> Path:
    """Add the project root to ``sys.path`` and return it."""
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


PROJECT_ROOT = bootstrap_project_root()

from adapters.chemical_adapter import ChemicalAdapter
from adapters.edge_adapter import EdgeAdapter
from adapters.wetware_adapter import WetwareAdapter
from core.orchestrator import PhysMCPOrchestrator
from core.task_model import OutputPreference, TaskKind, TaskRequest
from descriptors.capability_schema import SignalModality


def build_default_orchestrator() -> PhysMCPOrchestrator:
    """Create an orchestrator with the three default demo backends."""
    orchestrator = PhysMCPOrchestrator()
    orchestrator.register_adapter(ChemicalAdapter())
    orchestrator.register_adapter(WetwareAdapter())
    orchestrator.register_adapter(EdgeAdapter())
    return orchestrator


def make_edge_task(task_id: str = "task-edge") -> TaskRequest:
    """Create a representative low-latency edge classification task."""
    return TaskRequest(
        task_id=task_id,
        task_kind=TaskKind.CLASSIFICATION,
        summary="Low-latency edge classification task.",
        required_input_modalities=[SignalModality.DIGITAL_VECTOR],
        preferred_output=OutputPreference.SCORE,
        latency_budget_ms=20.0,
        min_confidence=0.2,
        metadata={"input_vector": [0.1, 0.3, 0.5, 0.7]},
    )


def make_wetware_task(task_id: str = "task-wetware") -> TaskRequest:
    """Create a representative closed-loop temporal inference task."""
    return TaskRequest(
        task_id=task_id,
        task_kind=TaskKind.TEMPORAL_INFERENCE,
        summary="Closed-loop stimulation and observation task.",
        required_input_modalities=[SignalModality.SPIKES],
        preferred_output=OutputPreference.TELEMETRY_AWARE_RESULT,
        latency_budget_ms=120.0,
        continuous_monitoring_required=True,
        min_confidence=0.2,
        metadata={
            "stimulation_strength": 0.65,
            "observation_window_ms": 140.0,
        },
    )


def make_chemical_task(task_id: str = "task-chemical", input_level: float = 1.4) -> TaskRequest:
    """Create a representative concentration-driven sensing task."""
    return TaskRequest(
        task_id=task_id,
        task_kind=TaskKind.SENSING,
        summary="Concentration-driven sensing and steady-state inference task.",
        required_input_modalities=[SignalModality.CONCENTRATION],
        preferred_output=OutputPreference.STATE_ESTIMATE,
        latency_budget_ms=15000.0,
        min_confidence=0.2,
        metadata={"input_level": input_level},
    )


def print_header(title: str) -> None:
    """Print a demo section header."""
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def print_match_report(report) -> None:
    """Pretty-print a matcher report."""
    print(f"Task: {report.task_id}")
    for candidate in report.candidates:
        for line in candidate.explanation_lines():
            print(line)


def print_run_summary(run_result) -> None:
    """Pretty-print a task execution result."""
    print(f"Success: {run_result.success}")
    print(f"Selected backend: {run_result.decision.selected_backend_id}")
    print(f"Used fallback: {run_result.decision.used_fallback}")
    if run_result.invocation is not None:
        print(f"Latency (ms): {run_result.invocation.execution_latency_ms:.3f}")
        print(f"Confidence: {run_result.invocation.confidence}")
        print(f"Output payload: {run_result.invocation.output_payload}")
    print(f"Telemetry after: {run_result.telemetry_after}")
    if run_result.recovery_actions:
        print("Recovery actions:")
        for action in run_result.recovery_actions:
            print(f"  - {action}")
    if run_result.failure_reason:
        print(f"Failure reason: {run_result.failure_reason}")
    if run_result.decision.notes:
        print("Decision notes:")
        for note in run_result.decision.notes:
            print(f"  - {note}")
