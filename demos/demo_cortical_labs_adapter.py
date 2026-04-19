"""Demonstrate the optional Cortical Labs adapter target."""

from __future__ import annotations

from common import (
    build_live_target_orchestrator,
    make_cortical_task,
    print_header,
    print_match_report,
    print_run_summary,
)


def main() -> None:
    orchestrator = build_live_target_orchestrator(include_cortical_labs=True)
    task = make_cortical_task()

    print_header("Cortical Labs adapter: discovery and directed task")
    report = orchestrator.plan_task(task)
    print_match_report(report)

    run_result = orchestrator.execute_task(task)
    print_run_summary(run_result)


if __name__ == "__main__":
    main()
