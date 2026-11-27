from __future__ import annotations

from copy import deepcopy
from typing import Any


PATCH_TEXT = {
    "retrieval_miss": (
        "Before answering, check every retrieved passage for the requested topic. If the first passages are not "
        "relevant, use the remaining evidence and combine all relevant sources rather than answering from one passage."
    ),
    "hallucination": (
        "Every factual claim must be directly supported by a retrieved passage. Do not add assumptions, examples, "
        "numbers, or consequences that are not stated. Cite the exact filename beside the evidence you used."
    ),
    "wrong_refusal": (
        "Do not refuse an in-scope question merely because the answer requires careful reading. Refuse only when the "
        "request is out of scope or no supplied evidence supports it. If evidence exists, answer it with citations."
    ),
    "prompt_defect": (
        "Answer the user's actual question directly. Preserve qualifiers, exceptions, negation, units, eligibility, "
        "and approval conditions from the source; do not replace a specific question with a generic summary."
    ),
}


def targeted_patch(agent: dict[str, Any], scorecard: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    patched = deepcopy(agent)
    failures = scorecard.get("failures", [])
    categories = sorted({failure["failure_category"] for failure in failures if failure.get("failure_category")})
    changes = []
    for category in categories:
        failure_ids = [failure["case_id"] for failure in failures if failure.get("failure_category") == category]
        if category == "retrieval_miss":
            retrieval = patched["knowledge"]["retrieval"]
            before = deepcopy(retrieval)
            retrieval["top_k"] = min(12, int(retrieval.get("top_k", 4)) + 2)
            retrieval["min_score"] = 0.0
            changes.append(
                {
                    "type": "retrieval_configuration",
                    "category": category,
                    "before": before,
                    "after": deepcopy(retrieval),
                    "why": "Expected sources were absent from retrieved evidence.",
                    "failure_ids": failure_ids,
                }
            )
        if category in PATCH_TEXT:
            addition = PATCH_TEXT[category]
            if addition not in patched["system_prompt"]:
                before = patched["system_prompt"]
                patched["system_prompt"] = before.rstrip() + "\n\n" + addition
                changes.append(
                    {
                        "type": "system_prompt_patch",
                        "category": category,
                        "before_length": len(before),
                        "after_length": len(patched["system_prompt"]),
                        "added_instruction": addition,
                        "why": f"Targeted correction for {len(failure_ids)} {category.replace('_', ' ')} failure(s).",
                        "failure_ids": failure_ids,
                    }
                )
    patched["version"] = int(agent.get("version", 1)) + 1
    patched["optimization_history"] = deepcopy(agent.get("optimization_history", []))
    patched["optimization_history"].append(
        {
            "from_version": agent.get("version", 1),
            "to_version": patched["version"],
            "score_before": scorecard.get("pass_rate"),
            "categories": categories,
            "changes": changes,
        }
    )
    return patched, changes
