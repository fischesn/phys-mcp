"""Minimal Gemini-based agent for the current phys-MCP prototype.

This version is aligned with the current repository state:
- resource discovery uses PhysMCPOrchestrator.discover_backends()
- runtime execution goes only through phys-MCP
- the agent prefers the Cortical Labs backend but does not call CL APIs directly

Expected location:
    agent/gemini_agent.py

Expected run command from project root:
    python -m agent.gemini_agent
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def bootstrap_project_root() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


PROJECT_ROOT = bootstrap_project_root()

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from google.genai import Client  # noqa: E402
from google.genai.types import GenerateContentConfig  # noqa: E402

from demos.common import build_live_target_orchestrator, make_cortical_task  # noqa: E402


PLANNING_PROMPT = """You are a planner for a phys-MCP control-plane client.
Return ONLY valid JSON with this schema:

{
  "action": "run_cortical_screen",
  "arguments": {
    "preferred_backend_id": "cortical-labs-backend",
    "channel": <int>,
    "amplitude": <float>,
    "observation_window_ms": <int>,
    "pre_delay_ms": <int>,
    "allow_fallback": <bool>,
    "human_supervision_available": <bool>
  },
  "rationale": "<short string>"
}

Rules:
- Use action exactly "run_cortical_screen"
- Prefer backend_id "cortical-labs-backend"
- Keep amplitude between 0.1 and 1.0
- Keep channel as a positive integer
- observation_window_ms between 50 and 500
- pre_delay_ms between 0 and 100
- Unless the user explicitly requests otherwise:
  - allow_fallback = false
  - human_supervision_available = true
- Output JSON only, no markdown, no explanation.
"""

SUMMARY_PROMPT_TEMPLATE = """You are summarizing a phys-MCP execution result.

User goal:
{user_goal}

Structured plan:
{plan_json}

Execution result:
{result_json}

Write a short, technically clear summary for a researcher.
Mention:
- whether the run succeeded
- which backend was used
- backend latency
- observation latency
- recording path if available
- whether fallback was used
Keep it to 4-8 sentences.
"""


@dataclass
class AgentResult:
    plan: dict[str, Any]
    resources: list[dict[str, Any]]
    run_result: dict[str, Any]
    summary: str


class PhysMCPGeminiAgent:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-2.5-pro",
    ) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise RuntimeError("Missing GEMINI_API_KEY in environment.")

        self.client = Client(api_key=self.api_key)
        self.model = model
        self.orchestrator = build_live_target_orchestrator(include_cortical_labs=True)

    def discover_resources(self) -> list[dict[str, Any]]:
        return self.orchestrator.discover_backends()

    def plan(self, user_goal: str) -> dict[str, Any]:
        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                PLANNING_PROMPT,
                "",
                f"User goal: {user_goal}",
            ],
            config=GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json",
            ),
        )
        text = response.text or ""
        return json.loads(text)

    def execute_plan(self, plan: dict[str, Any]) -> dict[str, Any]:
        action = plan.get("action")
        if action != "run_cortical_screen":
            raise RuntimeError(f"Unsupported action: {action!r}")

        args = plan.get("arguments", {})

        task = make_cortical_task(
            task_id="agent-cortical-run",
            direct_backend_id=args.get("preferred_backend_id", "cortical-labs-backend"),
            allow_fallback=bool(args.get("allow_fallback", False)),
        )

        task.human_supervision_available = bool(
            args.get("human_supervision_available", True)
        )
        task.metadata["stimulation_pattern"] = {
            "channels": [int(args.get("channel", 1))],
            "amplitude": float(args.get("amplitude", 0.4)),
        }
        task.metadata["observation_window_ms"] = int(
            args.get("observation_window_ms", 100)
        )
        task.metadata["pre_delay_ms"] = int(args.get("pre_delay_ms", 20))

        run_result = self.orchestrator.execute_task(task)
        invocation = run_result.invocation
        payload = invocation.output_payload if invocation is not None else {}

        return {
            "success": run_result.success,
            "selected_backend": run_result.decision.selected_backend_id,
            "used_fallback": run_result.decision.used_fallback,
            "failure_reason": run_result.failure_reason,
            "decision_notes": list(run_result.decision.notes or []),
            "validation_failures": list(run_result.validation_failures or []),
            "recovery_actions": list(run_result.recovery_actions or []),
            "execution_latency_ms": getattr(invocation, "execution_latency_ms", None),
            "confidence": getattr(invocation, "confidence", None),
            "output_payload": payload,
            "telemetry_before": run_result.telemetry_before or {},
            "telemetry_after": run_result.telemetry_after or {},
        }

    def summarize(
        self,
        user_goal: str,
        plan: dict[str, Any],
        run_result: dict[str, Any],
    ) -> str:
        prompt = SUMMARY_PROMPT_TEMPLATE.format(
            user_goal=user_goal,
            plan_json=json.dumps(plan, indent=2),
            result_json=json.dumps(run_result, indent=2),
        )
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=GenerateContentConfig(
                temperature=0.2,
            ),
        )
        return (response.text or "").strip()

    def run(self, user_goal: str) -> AgentResult:
        resources = self.discover_resources()
        plan = self.plan(user_goal)
        run_result = self.execute_plan(plan)
        summary = self.summarize(user_goal, plan, run_result)
        return AgentResult(
            plan=plan,
            resources=resources,
            run_result=run_result,
            summary=summary,
        )


def main() -> None:
    user_goal = (
        "Probe whether the cultured network produces a stable response under a "
        "candidate stimulation pattern. Prefer Cortical Labs. Use a short "
        "observation window and do not enable fallback."
    )

    agent = PhysMCPGeminiAgent()
    result = agent.run(user_goal)

    print("=" * 80)
    print("Gemini phys-MCP agent result")
    print("=" * 80)
    print("\nPlan:")
    print(json.dumps(result.plan, indent=2))
    print("\nDiscovered resources:")
    print(json.dumps(result.resources, indent=2)[:4000])
    print("\nExecution result:")
    print(json.dumps(result.run_result, indent=2))
    print("\nSummary:")
    print(result.summary)


if __name__ == "__main__":
    main()
