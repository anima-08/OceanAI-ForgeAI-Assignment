from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from .documents import Document, document_digest, document_inventory
from .llm import OllamaClient
from .utils import write_json


DEFAULT_CAPABILITIES = ["document_qa"]


def _list_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def _normalise_requirements(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Requirements must be a JSON object")
    capabilities = value.get("required_capabilities") or DEFAULT_CAPABILITIES
    if isinstance(capabilities, str):
        capabilities = [capabilities]
    allowed = {"document_qa", "csv_lookup", "calculator", "current_date", "public_weather"}
    capabilities = [item for item in capabilities if item in allowed]
    if "document_qa" not in capabilities:
        capabilities.insert(0, "document_qa")
    return {
        "name": str(value.get("name") or name),
        "scope": str(value.get("scope") or "Answer questions grounded in the supplied documents."),
        "audience": str(value.get("audience") or "Requester and authorized users"),
        "tone": str(value.get("tone") or "clear, concise, and professional"),
        "must_answer_topics": _list_strings(value.get("must_answer_topics")),
        "must_refuse_topics": _list_strings(value.get("must_refuse_topics")),
        "required_capabilities": capabilities,
        "success_criteria": _list_strings(value.get("success_criteria")),
    }


def infer_capabilities(brief: str, requirements: dict[str, Any], documents: list[Document]) -> list[str]:
    text = " ".join(
        [brief, requirements.get("scope", ""), " ".join(requirements.get("must_answer_topics", []))]
    ).lower()
    capabilities = ["document_qa"]
    has_csv = any(document.kind == "csv" for document in documents)
    if has_csv and re.search(r"\b(catalogue|catalog|csv|sku|stock|inventory|spare|part|price|lookup|search)\b", text):
        capabilities.append("csv_lookup")
    if re.search(r"\b(calculate|calculation|compute|total|sum|cost|quote|estimate|percentage|percent|discount|tax|convert|arithmetic|multiply|divide)\b", text):
        capabilities.append("calculator")
    if re.search(r"\b(current date|today's date|today|what day is it|date now)\b", text):
        capabilities.append("current_date")
    if re.search(r"\b(weather|forecast|temperature|rainfall|rain|wind)\b", text):
        capabilities.append("public_weather")
    return capabilities


def deterministic_groundability(
    requirements: dict[str, Any], documents: list[Document], groundability: dict[str, Any]
) -> dict[str, Any]:
    corpus = " ".join(document.name + " " + document.text for document in documents).lower()
    generic_terms = {"policy", "document", "provide", "including", "question", "assistant", "staff", "company", "information", "rules", "topics"}
    gaps = []
    for gap in groundability.get("gaps", []):
        if not isinstance(gap, dict):
            continue
        terms = [
            term for term in re.findall(r"[a-zA-Z0-9]+", str(gap.get("request", "")).lower())
            if len(term) > 3 and term not in generic_terms
        ]
        if gap.get("severity") == "blocking" and terms and any(term in corpus for term in terms):
            gaps.append(
                {
                    **gap,
                    "severity": "warning",
                    "reason": str(gap.get("reason", "")) + " Deterministic corpus overlap found; retained as a warning.",
                    "detector": "model_gap_with_deterministic_counterevidence",
                }
            )
        else:
            gaps.append(gap)
    for topic in requirements.get("must_answer_topics", []):
        terms = [
            term for term in re.findall(r"[a-zA-Z0-9]+", topic.lower())
            if len(term) > 3 and term not in generic_terms
        ]
        if terms and not any(term in corpus for term in terms):
            gaps.append(
                {
                    "request": topic,
                    "reason": "No supplied document contains a matching topic term.",
                    "severity": "blocking",
                    "detector": "deterministic_topic_overlap",
                }
            )
    has_blocking_gap = any(gap.get("severity") == "blocking" for gap in gaps)
    return {**groundability, "gaps": gaps, "supported": not has_blocking_gap}


def analyze_intake(
    client: OllamaClient,
    brief: str,
    documents: list[Document],
    prompt_path: str | Path,
    name: str,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, str]]]:
    prompt_template = Path(prompt_path).read_text(encoding="utf-8")
    prompt = prompt_template.format(
        brief=brief,
        inventory=document_inventory(documents),
        digest=document_digest(documents),
    )
    result, _ = client.chat_json(prompt, temperature=0.05, max_tokens=3000)
    if not isinstance(result, dict):
        raise ValueError("Intake response must be a JSON object")
    requirements = _normalise_requirements(result.get("requirements", {}), name)
    requirements["required_capabilities"] = infer_capabilities(brief, requirements, documents)
    groundability = result.get("groundability") if isinstance(result.get("groundability"), dict) else {}
    groundability = {
        "supported": bool(groundability.get("supported", True)),
        "gaps": groundability.get("gaps", []) if isinstance(groundability.get("gaps", []), list) else [],
        "evidence": groundability.get("evidence", []) if isinstance(groundability.get("evidence", []), list) else [],
    }
    groundability = deterministic_groundability(requirements, documents, groundability)
    questions = result.get("clarifying_questions") or []
    questions = [
        {
            "id": str(item.get("id", f"q{index}")),
            "question": str(item.get("question", "")),
            "why": str(item.get("why", "")),
            "default": str(item.get("default", "")),
        }
        for index, item in enumerate(questions[:5], 1)
        if isinstance(item, dict) and item.get("question")
    ]
    return requirements, groundability, questions


def apply_answers(
    requirements: dict[str, Any],
    questions: list[dict[str, str]],
    answers: dict[str, str] | None,
    *,
    interactive: bool,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    answers = answers or {}
    dialogue = []
    clarified = dict(requirements)
    clarification_notes = []
    for question in questions:
        answer = answers.get(question["id"])
        if interactive and answer is None:
            print(f"\n{question['id']}: {question['question']}")
            if question["why"]:
                print(f"   Why: {question['why']}")
            answer = input("   Answer (blank uses the safe default): ").strip()
        if not answer:
            answer = question["default"]
        dialogue.append({**question, "answer": answer, "source": "requester" if answers.get(question["id"]) else "default"})
        clarification_notes.append(f"{question['question']} {answer}")
    clarified["clarifications"] = clarification_notes
    return clarified, dialogue


def write_requirements(path: str | Path, requirements: dict[str, Any], groundability: dict[str, Any]) -> None:
    write_json(path, {"requirements": requirements, "groundability": groundability})
