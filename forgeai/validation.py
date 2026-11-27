from __future__ import annotations

from pathlib import Path
from typing import Any

from .tools import TOOL_SPECS
from .utils import read_json


REQUIRED_FILES = [
    "run_manifest.json",
    "requirements/requirements.json",
    "groundability/groundability_report.json",
    "final_agent.json",
    "evals/generated_eval_set.json",
    "scorecards/all_scorecards.json",
    "iterations/all_iteration_logs.json",
    "report.md",
]


def validate_run(run_dir: str | Path, expected_tools: list[str] | None = None) -> dict[str, Any]:
    run_dir = Path(run_dir)
    missing = [name for name in REQUIRED_FILES if not (run_dir / name).is_file()]
    if missing:
        raise ValueError(f"Missing required artifacts: {missing}")
    eval_set = read_json(run_dir / "evals/generated_eval_set.json")
    cases = eval_set.get("cases", [])
    categories = {case.get("category") for case in cases}
    if len(cases) < 20:
        raise ValueError(f"Evaluation set has only {len(cases)} cases")
    if not {"answerable", "refusal", "edge"}.issubset(categories):
        raise ValueError(f"Evaluation categories are incomplete: {sorted(categories)}")
    for case in cases:
        if case.get("expected_behavior") == "answer":
            if not case.get("expected_sources") or not case.get("expected_facts"):
                raise ValueError(f"Answerable case lacks grounding: {case.get('id')}")
    agent = read_json(run_dir / "final_agent.json")
    tool_names = [tool.get("name") for tool in agent.get("tools", [])]
    unknown_tools = [name for name in tool_names if name not in TOOL_SPECS]
    if unknown_tools:
        raise ValueError(f"Unknown tools attached: {unknown_tools}")
    if expected_tools is not None and set(tool_names) != set(expected_tools):
        raise ValueError(f"Expected tools {expected_tools}, got {tool_names}")
    scorecards = read_json(run_dir / "scorecards/all_scorecards.json")
    iteration_logs = read_json(run_dir / "iterations/all_iteration_logs.json")
    patch_count = sum(1 for log in iteration_logs if log.get("type") == "targeted_patch")
    if patch_count > 3:
        raise ValueError(f"Optimizer exceeded 3 iterations: {patch_count}")
    for log in iteration_logs:
        if log.get("type") == "targeted_patch":
            if "score_before" not in log or "score_after" not in log or not log.get("changes"):
                raise ValueError(f"Incomplete iteration log: {log.get('iteration')}")
    return {
        "run_dir": str(run_dir),
        "case_count": len(cases),
        "categories": sorted(categories),
        "tools": tool_names,
        "evaluated_versions": len(scorecards),
        "optimizer_iterations": patch_count,
        "final_pass_rate": scorecards[-1]["pass_rate"],
        "status": "valid",
    }
