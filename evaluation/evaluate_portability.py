"""Evaluate portability and schema consistency across backends."""

from __future__ import annotations

from common import RESULTS_DIR, save_csv, save_json
from plots import save_bar_chart

from demos.common import build_default_orchestrator, make_chemical_task, make_edge_task, make_wetware_task


def evaluate() -> dict:
    orchestrator = build_default_orchestrator()

    tasks = [
        ("edge", make_edge_task(task_id="port-edge")),
        ("wetware", make_wetware_task(task_id="port-wetware")),
        ("chemical", make_chemical_task(task_id="port-chemical")),
    ]

    run_rows: list[dict] = []
    descriptor_dicts = orchestrator.discover_backends()
    descriptor_key_sets = [set(descriptor.keys()) for descriptor in descriptor_dicts]
    shared_descriptor_keys = set.intersection(*descriptor_key_sets)
    union_descriptor_keys = set.union(*descriptor_key_sets)

    invocation_key_sets = []
    for backend_name, task in tasks:
        run_result = orchestrator.execute_task(task)
        if not run_result.success:
            raise RuntimeError(f"Portability run failed for {backend_name}: {run_result.failure_reason}")

        invocation_keys = set(run_result.invocation.model_dump().keys())
        invocation_key_sets.append(invocation_keys)

        run_rows.append(
            {
                "backend_role": backend_name,
                "selected_backend": run_result.decision.selected_backend_id,
                "success": run_result.success,
                "metadata_key_count": len(task.metadata),
                "required_modalities": ",".join(str(value) for value in task.required_input_modalities),
                "output_keys": ",".join(sorted(run_result.invocation.output_payload.keys())),
            }
        )

    shared_invocation_keys = set.intersection(*invocation_key_sets)
    union_invocation_keys = set.union(*invocation_key_sets)

    payload = {
        "application_api_call": "PhysMCPOrchestrator.execute_task(task)",
        "successful_runs": len(run_rows),
        "total_runs": len(tasks),
        "descriptor_shared_key_ratio": len(shared_descriptor_keys) / len(union_descriptor_keys),
        "invocation_shared_key_ratio": len(shared_invocation_keys) / len(union_invocation_keys),
        "shared_descriptor_keys": sorted(shared_descriptor_keys),
        "shared_invocation_keys": sorted(shared_invocation_keys),
        "runs": run_rows,
    }

    save_json(RESULTS_DIR / "portability_results.json", payload)
    save_csv(RESULTS_DIR / "portability_runs.csv", run_rows)

    labels = [row["backend_role"] for row in run_rows]
    values = [row["metadata_key_count"] for row in run_rows]
    save_bar_chart(
        labels=labels,
        values=values,
        title="Backend-specific metadata footprint per task",
        ylabel="Number of metadata keys",
        output_path=RESULTS_DIR / "portability_metadata_bar_chart.png",
    )

    return payload


def main() -> None:
    payload = evaluate()
    print("Portability evaluation complete.")
    print(f"Successful runs: {payload['successful_runs']} / {payload['total_runs']}")
    print(f"Descriptor shared key ratio: {payload['descriptor_shared_key_ratio']:.3f}")
    print(f"Invocation shared key ratio: {payload['invocation_shared_key_ratio']:.3f}")
    for row in payload["runs"]:
        print(
            f"{row['backend_role']}: selected={row['selected_backend']}, "
            f"metadata_keys={row['metadata_key_count']}, "
            f"output_keys={row['output_keys']}"
        )


if __name__ == "__main__":
    main()
