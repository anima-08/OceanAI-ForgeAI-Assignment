from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .documents import chunk_documents, load_documents
from .evals import generate_eval_set
from .generator import generate_agent
from .intake import analyze_intake, apply_answers, write_requirements
from .judge import evaluate_agent
from .llm import OllamaClient
from .optimizer import targeted_patch
from .utils import read_json, sha256_file, sha256_text, slugify, utc_now, write_json


PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"


def _prepare_output(output_dir: Path, documents: list[Any]) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Output directory is not empty: {output_dir}")
    for folder in (
        "input",
        "knowledge",
        "agents",
        "requirements",
        "groundability",
        "evals",
        "scorecards",
        "evaluations",
        "iterations",
    ):
        (output_dir / folder).mkdir(parents=True, exist_ok=True)
    names = set()
    for document in documents:
        if document.name in names:
            raise ValueError(f"Duplicate document filename is not supported: {document.name}")
        names.add(document.name)
        shutil.copy2(document.path, output_dir / "knowledge" / document.name)


def _read_brief(brief: str) -> tuple[str, str]:
    path = Path(brief)
    if path.is_file():
        return path.read_text(encoding="utf-8"), str(path.resolve())
    return brief, "inline"


def _write_report(
    output_dir: Path,
    requirements: dict[str, Any],
    groundability: dict[str, Any],
    eval_set: dict[str, Any],
    scorecards: list[dict[str, Any]],
    final_agent: dict[str, Any] | None,
    target_pass_rate: float = 0.85,
) -> None:
    final = scorecards[-1] if scorecards else None
    lines = [
        "# ForgeAI Quality Report",
        "",
        f"Generated at: {utc_now()}",
        "",
        "## Outcome",
        "",
    ]
    if final is None:
        lines.extend(
            [
                "Status: **blocked before generation**",
                "",
                "The intake groundability guard found blocking gaps. No agent or evaluation set was generated.",
                "",
            ]
        )
    else:
        status = "target met" if final["pass_rate"] >= target_pass_rate else "target not met; best available agent retained"
        lines.extend(
            [
                f"Status: **{status}**",
                "",
                f"- Agent: `{final_agent.get('name') if final_agent else 'unknown'}` version `{final_agent.get('version') if final_agent else 'n/a'}`",
                f"- Evaluation cases: {eval_set['case_count']}",
                f"- Final passed: {final['passed']}/{final['total_cases']}",
                f"- Final pass rate: {final['pass_rate']:.1%}",
                f"- Final mean score: {final['mean_score']:.3f}",
                "",
                "## Score progression",
                "",
                "| Iteration | Passed | Pass rate | Mean score |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for scorecard in scorecards:
            lines.append(
                f"| {scorecard['iteration']} | {scorecard['passed']}/{scorecard['total_cases']} | "
                f"{scorecard['pass_rate']:.1%} | {scorecard['mean_score']:.3f} |"
            )
        lines.extend(["", "## Remaining failures", ""])
        if final["failures"]:
            for failure in final["failures"]:
                lines.append(
                    f"- `{failure['case_id']}` ({failure['failure_category']}, score {failure['score']:.3f}): "
                    f"{failure['reason']}"
                )
        else:
            lines.append("- No cases failed in the final evaluation. This is a measured result on the generated set, not a guarantee for unseen data.")
        lines.extend(
            [
                "",
                "## Groundability and scope",
                "",
                f"- Intake support status: {'supported' if groundability.get('supported') else 'not supported'}",
                f"- Must-answer topics: {', '.join(requirements.get('must_answer_topics', [])) or 'none specified'}",
                f"- Must-refuse topics: {', '.join(requirements.get('must_refuse_topics', [])) or 'none specified'}",
                "",
                "## Tools attached",
                "",
            ]
        )
        if final_agent and final_agent.get("tools"):
            for tool in final_agent["tools"]:
                lines.append(f"- `{tool['name']}`: {tool.get('rationale', '')}")
        else:
            lines.append("- No external tools were attached; the agent uses document retrieval only.")
        lines.extend(
            [
                "",
                "## Judge safeguards",
                "",
                "- Deterministic behavior, citation, expected-source, retrieval, and fact-coverage checks can veto the local LLM judge.",
                "- The judge sees one anonymized candidate at a time under a fixed rubric; verbosity and confidence are not scored.",
                "- The local judge may still make semantic mistakes, so per-case evidence is retained in `evaluations/`.",
                "",
                "## Known limitations",
                "",
                "- Evaluation questions are generated from the supplied documents and can inherit source ambiguity or model-generation errors.",
                "- Lexical BM25 retrieval can miss synonyms or rare phrasing that dense embeddings might capture.",
                "- Passing this set does not establish production readiness, security, legal compliance, or factual correctness outside the supplied corpus.",
                "",
            ]
        )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run_factory(
    brief: str,
    docs_dir: str | Path,
    output_dir: str | Path,
    *,
    model: str = "llama3.2:latest",
    base_url: str = "http://127.0.0.1:11434",
    answers: dict[str, str] | None = None,
    interactive: bool = False,
    max_iterations: int = 3,
    target_pass_rate: float = 0.85,
    use_llm_judge: bool = True,
) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    source_documents = load_documents(docs_dir)
    _prepare_output(output_dir, source_documents)
    brief_text, brief_source = _read_brief(brief)
    (output_dir / "input" / "brief.txt").write_text(brief_text, encoding="utf-8")
    if answers:
        write_json(output_dir / "input" / "answers.json", answers)
    documents = load_documents(output_dir / "knowledge")
    client = OllamaClient(model=model, base_url=base_url, timeout=600)
    name = slugify(output_dir.name)
    started = utc_now()
    manifest: dict[str, Any] = {
        "factory": "ForgeAI",
        "factory_version": "1.0.0",
        "started_at": started,
        "model": model,
        "ollama_base_url": base_url,
        "brief_source": brief_source,
        "brief_sha256": sha256_text(brief_text),
        "documents": [
            {"name": document.name, "sha256": document.sha256, "characters": len(document.text)}
            for document in documents
        ],
        "prompts": {
            path.name: sha256_file(path)
            for path in sorted(PROMPT_DIR.glob("*.txt"))
        },
        "parameters": {
            "max_optimizer_iterations": max_iterations,
            "target_pass_rate": target_pass_rate,
            "use_llm_judge": use_llm_judge,
            "interactive": interactive,
        },
        "status": "running",
    }
    write_json(output_dir / "run_manifest.json", manifest)

    requirements, groundability, questions = analyze_intake(
        client,
        brief_text,
        documents,
        PROMPT_DIR / "intake_prompt.txt",
        name,
    )
    requirements, dialogue = apply_answers(requirements, questions, answers, interactive=interactive)
    write_json(output_dir / "requirements" / "requirements.json", requirements)
    write_json(output_dir / "requirements" / "intake_dialogue.json", dialogue)
    write_requirements(output_dir / "requirements" / "requirements_and_groundability.json", requirements, groundability)
    write_json(output_dir / "groundability" / "groundability_report.json", groundability)
    print(f"Intake complete: {len(questions)} clarifying question(s); supported={groundability['supported']}", flush=True)

    if not groundability.get("supported"):
        manifest["status"] = "blocked_groundability"
        manifest["completed_at"] = utc_now()
        write_json(output_dir / "run_manifest.json", manifest)
        _write_report(output_dir, requirements, groundability, {"case_count": 0}, [], None, target_pass_rate)
        return {"status": "blocked_groundability", "output_dir": str(output_dir), "groundability": groundability}

    chunking = {"chunk_size_words": 180, "overlap_words": 30}
    chunks = chunk_documents(documents, **chunking)
    eval_set = generate_eval_set(
        client,
        requirements,
        documents,
        chunks,
        PROMPT_DIR / "eval_generation_prompt.txt",
        output_dir / "evals" / "generated_eval_set.json",
        output_dir / "evals" / "generation_evidence",
    )
    print(f"Generated {eval_set['case_count']} evaluation cases", flush=True)

    agent = generate_agent(
        client,
        requirements,
        documents,
        PROMPT_DIR / "agent_generation_prompt.txt",
        output_dir / "agents" / "agent_v0.json",
    )
    print(f"Generated agent v{agent['version']} with tools: {[tool['name'] for tool in agent['tools']]}", flush=True)
    scorecards = []
    iteration_logs = []
    current_agent = agent
    current_agent_path = output_dir / "agents" / "agent_v0.json"
    iteration = 0
    pending_patch: dict[str, Any] | None = None
    while True:
        scorecard = evaluate_agent(
            current_agent_path,
            eval_set,
            client,
            client,
            PROMPT_DIR / "judge_prompt.txt",
            output_dir / "evaluations" / f"iteration_{iteration}",
            iteration,
            use_llm_judge=use_llm_judge,
        )
        scorecards.append(scorecard)
        previous_scorecard = scorecards[-2] if len(scorecards) >= 2 else None
        write_json(output_dir / "scorecards" / f"scorecard_iteration_{iteration}.json", scorecard)
        print(
            f"Iteration {iteration}: {scorecard['passed']}/{scorecard['total_cases']} "
            f"({scorecard['pass_rate']:.1%})",
            flush=True,
        )
        if iteration == 0:
            iteration_logs.append(
                {
                    "iteration": 0,
                    "type": "baseline",
                    "score_before": None,
                    "score_after": scorecard["pass_rate"],
                    "changes": [],
                    "failures_addressed": [],
                }
            )
            write_json(output_dir / "iterations" / "iteration_0.json", iteration_logs[-1])
        elif pending_patch is not None and previous_scorecard is not None:
            addressed = pending_patch["failures_addressed"]
            iteration_logs.append(
                {
                    "iteration": iteration,
                    "type": "targeted_patch",
                    "agent_version": current_agent["version"],
                    "score_before": previous_scorecard["pass_rate"],
                    "score_after": scorecard["pass_rate"],
                    "mean_score_before": previous_scorecard["mean_score"],
                    "mean_score_after": scorecard["mean_score"],
                    "changes": pending_patch["changes"],
                    "failures_addressed": addressed,
                    "new_failure_ids": [
                        failure["case_id"] for failure in scorecard["failures"]
                        if failure["case_id"] not in addressed
                    ],
                }
            )
            write_json(output_dir / "iterations" / f"iteration_{iteration}.json", iteration_logs[-1])
        if scorecard["pass_rate"] >= target_pass_rate or iteration >= max_iterations:
            break
        patched, changes = targeted_patch(current_agent, scorecard)
        if not changes:
            iteration_logs.append(
                {
                    "iteration": iteration + 1,
                    "type": "no_targeted_change_available",
                    "score_before": scorecard["pass_rate"],
                    "score_after": scorecard["pass_rate"],
                    "changes": [],
                    "failures_addressed": [],
                }
            )
            write_json(output_dir / "iterations" / f"iteration_{iteration + 1}.json", iteration_logs[-1])
            break
        pending_patch = {
            "changes": changes,
            "failures_addressed": sorted({failure["case_id"] for failure in scorecard["failures"]}),
        }
        current_agent = patched
        current_agent_path = output_dir / "agents" / f"agent_v{current_agent['version']}.json"
        write_json(current_agent_path, current_agent)
        iteration += 1

    final_agent_path = output_dir / "final_agent.json"
    shutil.copy2(current_agent_path, final_agent_path)
    write_json(output_dir / "scorecards" / "all_scorecards.json", scorecards)
    write_json(output_dir / "iterations" / "all_iteration_logs.json", iteration_logs)
    write_json(
        output_dir / "quality_report.json",
        {
            "generated_at": utc_now(),
            "target_pass_rate": target_pass_rate,
            "scorecards": scorecards,
            "remaining_failures": scorecards[-1]["failures"],
            "final_agent": str(final_agent_path.relative_to(output_dir)),
            "eval_set": str((output_dir / "evals" / "generated_eval_set.json").relative_to(output_dir)),
        },
    )
    _write_report(output_dir, requirements, groundability, eval_set, scorecards, current_agent, target_pass_rate)
    manifest["status"] = "completed"
    manifest["completed_at"] = utc_now()
    manifest["final_pass_rate"] = scorecards[-1]["pass_rate"]
    manifest["final_agent"] = "final_agent.json"
    manifest["evaluation_cases"] = eval_set["case_count"]
    write_json(output_dir / "run_manifest.json", manifest)
    return {
        "status": "completed",
        "output_dir": str(output_dir),
        "agent": str(final_agent_path),
        "pass_rate": scorecards[-1]["pass_rate"],
        "scorecards": scorecards,
    }
