"""Evaluate the Gemini-based phys-MCP agent.

This script performs a small number of agent-driven runs and records:

- the user goal
- the structured plan returned by Gemini
- whether execution succeeded
- which backend was selected
- whether fallback was used
- execution latency
- observation latency
- recording artifact / recording path
- readiness and health telemetry after execution

The purpose is not to benchmark LLM quality broadly, but to document that the
agent-facing control-plane path operates end-to-end in a reproducible way.

Expected location:
    evaluation/evaluate_gemini_agent.py

Run from the project root:
    python -m evaluation.evaluate_gemini_agent
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def bootstrap_project_root() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


PROJECT_ROOT = bootstrap_project_root()
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

from agent.gemini_agent import PhysMCPGeminiAgent  # noqa: E402


def build_user_goals() -> list[str]:
    return [
        (
            "Probe whether the cultured network produces a stable response under a "
            "candidate stimulation pattern. Prefer Cortical Labs. Use a short "
            "observation window and do not enable fallback."
        ),
        (
            "Run a directed Cortical Labs stimulation test with a moderate channel "
            "index and moderate amplitude. Keep the observation window short and "
            "avoid fallback."
        ),
        (
            "Test a slightly stronger candidate stimulation through the Cortical Labs "
            "backend, keeping human supervision enabled and using a short observation "
            "window."
        ),
        (
            "Use phys-MCP to run a Cortical Labs screening task with a single channel, "
            "a compact observation window, and no fallback."
        ),
        (
            "Execute a short wetware screening task on the Cortical Labs backend and "
            "summarize the resulting telemetry and recording path."
        ),
    ]


def summarize_agent_result(user_goal: str, result) -> dict[str, Any]:
    run_result = result.run_result
    payload = run_result.get("output_payload", {}) or {}
    telemetry_before = run_result.get("telemetry_before", {}) or {}
    telemetry_after = run_result.get("telemetry_after", {}) or {}
    recording_artifact = payload.get("recording_artifact", {}) or {}

    return {
        "user_goal": user_goal,
        "plan": result.plan,
        "resources": result.resources,
        "success": run_result.get("success"),
        "selected_backend": run_result.get("selected_backend"),
        "used_fallback": run_result.get("used_fallback"),
        "failure_reason": run_result.get("failure_reason"),
        "decision_notes": run_result.get("decision_notes", []),
        "validation_failures": run_result.get("validation_failures", []),
        "recovery_actions": run_result.get("recovery_actions", []),
        "execution_latency_ms": run_result.get("execution_latency_ms"),
        "confidence": run_result.get("confidence"),
        "response_fingerprint": payload.get("response_fingerprint"),
        "stim_channel": payload.get("stim_channel"),
        "stim_amplitude_ua": payload.get("stim_amplitude_ua"),
        "observation_window_ms": payload.get("observation_window_ms"),
        "recording_name": recording_artifact.get("name"),
        "recording_path": telemetry_after.get("recording_path") or recording_artifact.get("path"),
        "backend_latency_ms": telemetry_after.get("backend_latency_ms"),
        "observation_latency_ms": telemetry_after.get("observation_latency_ms"),
        "readiness_before": telemetry_before.get("readiness_state"),
        "health_before": telemetry_before.get("health_status"),
        "readiness_after": telemetry_after.get("readiness_state"),
        "health_after": telemetry_after.get("health_status"),
        "channel_count": telemetry_after.get("channel_count"),
        "fps": telemetry_after.get("fps"),
        "drift_score": telemetry_after.get("drift_score"),
        "age_of_information_ms": telemetry_after.get("age_of_information_ms"),
        "sdk_available": telemetry_after.get("sdk_available"),
        "summary": result.summary,
    }


def write_results(rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = RESULTS_DIR / f"gemini_agent_results_{timestamp}.json"
    csv_path = RESULTS_DIR / f"gemini_agent_results_{timestamp}.csv"

    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    fieldnames = [
        "user_goal",
        "success",
        "selected_backend",
        "used_fallback",
        "failure_reason",
        "execution_latency_ms",
        "confidence",
        "response_fingerprint",
        "stim_channel",
        "stim_amplitude_ua",
        "observation_window_ms",
        "recording_name",
        "recording_path",
        "backend_latency_ms",
        "observation_latency_ms",
        "readiness_before",
        "health_before",
        "readiness_after",
        "health_after",
        "channel_count",
        "fps",
        "drift_score",
        "age_of_information_ms",
        "sdk_available",
        "summary",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})

    return json_path, csv_path


def print_summary(rows: list[dict[str, Any]], json_path: Path, csv_path: Path) -> None:
    print()
    print("=" * 80)
    print("Gemini agent evaluation summary")
    print("=" * 80)

    success_count = sum(1 for r in rows if r["success"])
    print(f"Runs: {len(rows)}")
    print(f"Successful runs: {success_count}/{len(rows)}")

    for idx, row in enumerate(rows, start=1):
        print("-" * 80)
        print(
            f"run-{idx}: success={row['success']}, "
            f"backend={row['selected_backend']}, "
            f"backend_latency_ms={row['backend_latency_ms']}, "
            f"observation_latency_ms={row['observation_latency_ms']}, "
            f"recording_path={row['recording_path']}"
        )

    print("-" * 80)
    print(f"JSON results: {json_path}")
    print(f"CSV results:  {csv_path}")


def main() -> None:
    agent = PhysMCPGeminiAgent()
    user_goals = build_user_goals()

    rows: list[dict[str, Any]] = []
    for goal in user_goals:
        result = agent.run(goal)
        rows.append(summarize_agent_result(goal, result))

    json_path, csv_path = write_results(rows)
    print_summary(rows, json_path, csv_path)


if __name__ == "__main__":
    main()
