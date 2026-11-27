from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .llm import OllamaClient
from .retrieval import tokenize
from .runtime import AgentRunner
from .utils import extract_json, utc_now, write_json


def _token_coverage(expected: list[str], answer: str) -> float:
    answer_tokens = set(tokenize(answer))
    if not expected:
        return 1.0
    scores = []
    for fact in expected:
        fact_tokens = set(tokenize(fact))
        if not fact_tokens:
            scores.append(1.0)
            continue
        overlap = len(fact_tokens & answer_tokens) / len(fact_tokens)
        expected_numbers = set(re.findall(r"\d+(?:\.\d+)?", fact))
        answer_numbers = set(re.findall(r"\d+(?:\.\d+)?", answer))
        if expected_numbers and not expected_numbers.issubset(answer_numbers):
            overlap *= 0.6
        scores.append(overlap)
    return sum(scores) / len(scores)


def deterministic_judge(case: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
    behavior = case["expected_behavior"]
    answer = str(actual.get("answer", ""))
    refused = bool(actual.get("refused", False))
    citations = set(actual.get("citations", []))
    expected_sources = set(case.get("expected_sources", []))
    retrieved_sources = set(actual.get("retrieved_sources", []))
    citation_recall = len(citations & expected_sources) / len(expected_sources) if expected_sources else 1.0
    retrieval_hit = bool(expected_sources & retrieved_sources) if expected_sources else True
    fact_coverage = _token_coverage(case.get("expected_facts", []), answer) if behavior == "answer" else 1.0
    if behavior == "answer":
        behavior_ok = not refused
    elif behavior == "refuse":
        behavior_ok = refused
    else:
        behavior_ok = refused and bool(re.search(r"clarif|which|what|which|specify|more detail|could you", answer, flags=re.I))
    invalid_citations = sorted(citations - set(actual.get("retrieved_sources", [])))
    deterministic_score = (
        0.45 * fact_coverage
        + 0.30 * citation_recall
        + 0.15 * float(behavior_ok)
        + 0.10 * float(retrieval_hit)
    )
    hard_gate = behavior_ok and (behavior != "answer" or (citation_recall == 1.0 and fact_coverage >= 0.35))
    return {
        "behavior_ok": behavior_ok,
        "citation_recall": round(citation_recall, 3),
        "retrieval_hit": retrieval_hit,
        "fact_coverage": round(fact_coverage, 3),
        "invalid_citations": invalid_citations,
        "deterministic_score": round(deterministic_score, 3),
        "hard_gate": hard_gate,
    }


def _llm_judgements(
    client: OllamaClient,
    prompt_path: Path,
    cases: list[dict[str, Any]],
    actuals: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    prompt_template = prompt_path.read_text(encoding="utf-8")
    payload = [
        {
            "id": case["id"],
            "expected_behavior": case["expected_behavior"],
            "expected_sources": case.get("expected_sources", []),
            "expected_facts": case.get("expected_facts", []),
            "rubric": case.get("rubric", ""),
            "question": case["question"],
            "candidate_answer": {
                "answer": actual.get("answer", ""),
                "citations": actual.get("citations", []),
                "refused": actual.get("refused", False),
            },
        }
        for case, actual in zip(cases, actuals)
    ]
    judgements: dict[str, dict[str, Any]] = {}
    records = []
    for start in range(0, len(payload), 6):
        batch = payload[start : start + 6]
        prompt = prompt_template.format(payload=json.dumps(batch, ensure_ascii=False, indent=2))
        try:
            value, response = client.chat_json(prompt, temperature=0.0, max_tokens=1800)
            items = value.get("judgements", []) if isinstance(value, dict) else []
            if not isinstance(items, list):
                items = []
            for item in items:
                if not isinstance(item, dict) or item.get("id") not in {case["id"] for case in batch}:
                    continue
                judgements[str(item["id"])] = {
                    "semantic_correct": bool(item.get("semantic_correct", False)),
                    "grounded": bool(item.get("grounded", False)),
                    "appropriate_behavior": bool(item.get("appropriate_behavior", False)),
                    "reason": str(item.get("reason", "")),
                }
            records.append(
                {
                    "case_ids": [case["id"] for case in batch],
                    "latency_ms": response.latency_ms,
                    "returned_judgements": len(items),
                    "raw_response": response.text,
                }
            )
        except Exception as exc:
            records.append({"case_ids": [case["id"] for case in batch], "error": str(exc)})
    return judgements, {"batches": records, "returned": len(judgements), "expected": len(cases)}


def classify_failure(case: dict[str, Any], actual: dict[str, Any], deterministic: dict[str, Any], llm: dict[str, Any] | None) -> str:
    if case["expected_behavior"] == "answer" and actual.get("refused"):
        return "wrong_refusal"
    if not deterministic["retrieval_hit"]:
        return "retrieval_miss"
    if deterministic["invalid_citations"] or (llm and not llm.get("grounded", True)):
        return "hallucination"
    if llm and not llm.get("appropriate_behavior", True):
        return "prompt_defect"
    return "prompt_defect"


def evaluate_agent(
    agent_path: str | Path,
    eval_set: dict[str, Any],
    client: OllamaClient,
    judge_client: OllamaClient,
    judge_prompt_path: str | Path,
    output_dir: str | Path,
    iteration: int,
    use_llm_judge: bool = True,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runner = AgentRunner(agent_path, client=client)
    actuals = []
    cases = eval_set["cases"]
    batch_size = 4
    for start in range(0, len(cases), batch_size):
        batch = cases[start : start + batch_size]
        actuals.extend(runner.answer_batch([case["question"] for case in batch]))
        print(
            f"  evaluated {min(start + batch_size, len(cases))}/{len(cases)}: "
            + ", ".join(case["id"] for case in batch),
            flush=True,
        )
    llm_judgements: dict[str, dict[str, Any]] = {}
    judge_metadata: dict[str, Any] = {"enabled": use_llm_judge, "provider": "deterministic"}
    if use_llm_judge:
        llm_judgements, judge_metadata = _llm_judgements(
            judge_client, Path(judge_prompt_path), eval_set["cases"], actuals
        )
        judge_metadata.update({"enabled": True, "provider": "ollama+deterministic"})
    results = []
    for case, actual in zip(eval_set["cases"], actuals):
        deterministic = deterministic_judge(case, actual)
        llm = llm_judgements.get(case["id"])
        llm_score = None
        if llm:
            llm_score = sum(
                float(llm.get(key, False)) for key in ("semantic_correct", "grounded", "appropriate_behavior")
            ) / 3
        if llm_score is None:
            final_score = deterministic["deterministic_score"]
            passed = deterministic["hard_gate"]
        else:
            final_score = 0.70 * deterministic["deterministic_score"] + 0.30 * llm_score
            passed = (
                deterministic["hard_gate"]
                and llm.get("semantic_correct", False)
                and llm.get("grounded", False)
                and llm.get("appropriate_behavior", False)
                and final_score >= 0.72
            )
        failure_category = None if passed else classify_failure(case, actual, deterministic, llm)
        results.append(
            {
                "case_id": case["id"],
                "category": case["category"],
                "question": case["question"],
                "expected_behavior": case["expected_behavior"],
                "expected_sources": case.get("expected_sources", []),
                "expected_facts": case.get("expected_facts", []),
                "actual": actual,
                "deterministic": deterministic,
                "llm_judge": llm,
                "llm_score": round(llm_score, 3) if llm_score is not None else None,
                "score": round(final_score, 3),
                "passed": passed,
                "failure_category": failure_category,
            }
        )
    passed_count = sum(1 for result in results if result["passed"])
    scorecard = {
        "iteration": iteration,
        "timestamp": utc_now(),
        "total_cases": len(results),
        "passed": passed_count,
        "failed": len(results) - passed_count,
        "pass_rate": round(passed_count / len(results), 4) if results else 0.0,
        "mean_score": round(sum(result["score"] for result in results) / len(results), 4) if results else 0.0,
        "by_category": {
            category: {
                "total": sum(1 for result in results if result["category"] == category),
                "passed": sum(1 for result in results if result["category"] == category and result["passed"]),
            }
            for category in sorted({result["category"] for result in results})
        },
        "failure_categories": {
            category: sum(1 for result in results if result["failure_category"] == category)
            for category in sorted({result["failure_category"] for result in results if result["failure_category"]})
        },
        "failures": [
            {
                "case_id": result["case_id"],
                "category": result["category"],
                "failure_category": result["failure_category"],
                "score": result["score"],
                "reason": (
                    result["llm_judge"].get("reason")
                    if result["llm_judge"]
                    else "Deterministic behavior, citation, fact, or retrieval gate failed."
                ),
            }
            for result in results
            if not result["passed"]
        ],
        "judge": {
            **judge_metadata,
            "bias_defenses": [
                "single anonymized candidate per case removes position comparisons",
                "judge is not shown the generating agent or model identity",
                "fixed rubric ignores verbosity and confidence",
                "deterministic citation, expected-source, behavior, and fact-coverage gates can veto the LLM judge",
                "the same judge model is not trusted without deterministic evidence",
            ],
        },
    }
    write_json(output_dir / "responses.json", results)
    write_json(output_dir / "scorecard.json", scorecard)
    write_json(output_dir / "judge_metadata.json", judge_metadata)
    return scorecard
