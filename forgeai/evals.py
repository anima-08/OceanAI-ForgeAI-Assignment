from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .documents import Chunk, Document, select_digest_chunks
from .llm import OllamaClient
from .retrieval import tokenize
from .utils import clip, slugify, write_json


BATCHES = [
    (
        "answerable",
        "Generate 14 answerable cases. Cover different documents and sections, including exact limits, eligibility, procedures, and cross-document comparisons.",
    ),
    (
        "refusal",
        "Generate 5 refusal cases for requests explicitly outside the agent scope or unsupported by the documents. Each must have a clear refusal reason.",
    ),
    (
        "edge",
        "Generate 5 edge cases. Include at least two ambiguous questions requiring clarification, two answers that require combining two documents, and one missing-data or boundary case.",
    ),
]


def _chunks_text(chunks: list[Chunk], limit: int = 15000) -> str:
    blocks = []
    remaining = limit
    for chunk in chunks:
        text = clip(chunk.text, min(1100, max(remaining, 0)))
        blocks.append(f"SOURCE: {chunk.document}\nSECTION: {chunk.section}\nTEXT: {text}")
        remaining -= len(text)
        if remaining <= 0:
            break
    return "\n\n".join(blocks)


def _fact_support(fact: str, source_text: str) -> float:
    fact_tokens = set(tokenize(fact))
    if not fact_tokens:
        return 0.0
    source_tokens = set(tokenize(source_text))
    overlap = len(fact_tokens & source_tokens) / len(fact_tokens)
    numbers = set(re.findall(r"\d+(?:\.\d+)?", fact))
    source_numbers = set(re.findall(r"\d+(?:\.\d+)?", source_text))
    if numbers and not numbers.issubset(source_numbers):
        overlap *= 0.5
    return round(overlap, 3)


def _normalise_cases(
    cases: Any,
    category: str,
    documents: list[Document],
    existing_ids: set[str],
) -> list[dict[str, Any]]:
    if isinstance(cases, dict):
        cases = cases.get("cases", [])
    if not isinstance(cases, list):
        return []
    names = {document.name for document in documents}
    source_text = "\n".join(document.text for document in documents)
    normalised = []
    for index, case in enumerate(cases, 1):
        if not isinstance(case, dict) or not case.get("question"):
            continue
        question = str(case["question"]).strip()
        case_category = str(case.get("category", category)).strip().lower()
        if case_category not in {"answerable", "refusal", "edge"}:
            case_category = category
        base_id = slugify(str(case.get("id") or f"{case_category}-{index}"))[:70]
        case_id = base_id
        suffix = 2
        while case_id in existing_ids:
            case_id = f"{base_id}-{suffix}"
            suffix += 1
        expected_behavior = str(case.get("expected_behavior", "answer" if case_category == "answerable" else "refuse"))
        if expected_behavior not in {"answer", "refuse", "clarify"}:
            expected_behavior = "answer" if case_category == "answerable" else "refuse"
        sources = case.get("expected_sources") or []
        if isinstance(sources, str):
            sources = [sources]
        sources = [str(source) for source in sources if str(source) in names]
        facts = case.get("expected_facts") or []
        if isinstance(facts, str):
            facts = [facts]
        facts = [str(fact).strip() for fact in facts if str(fact).strip()]
        if not facts:
            continue
        if case_category == "answerable" and expected_behavior != "answer":
            case_category = "refusal" if expected_behavior == "refuse" else "edge"
        if case_category == "answerable" and not sources:
            continue
        if case_category == "refusal":
            if expected_behavior != "refuse":
                continue
            if sources:
                continue
        if case_category == "edge" and expected_behavior == "answer" and not sources:
            continue
        expected_text = "\n".join(
            document.text for document in documents if document.name in sources
        ) if sources else source_text
        support_scores = [_fact_support(fact, expected_text) for fact in facts]
        normalised.append(
            {
                "id": case_id,
                "category": case_category,
                "question": question,
                "expected_behavior": expected_behavior,
                "expected_sources": sources,
                "expected_facts": facts,
                "rubric": str(case.get("rubric") or "Meet the expected behavior and facts without unsupported claims."),
                "generation_support": {
                    "minimum_fact_overlap": min(support_scores) if support_scores else 0.0,
                    "mean_fact_overlap": round(sum(support_scores) / len(support_scores), 3) if support_scores else 0.0,
                },
            }
        )
        existing_ids.add(case_id)
    return normalised


def _fallback_cases(
    chunks: list[Chunk],
    requirements: dict[str, Any],
    needed: int,
    existing_ids: set[str],
) -> list[dict[str, Any]]:
    cases = []
    for chunk in chunks:
        if len(cases) >= needed:
            break
        sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", chunk.text) if part.strip()]
        if not sentences:
            continue
        facts = sentences[:2]
        case_id = f"generated-{chunk.document.lower()}-{len(cases) + 1}"
        case_id = slugify(case_id)[:70]
        while case_id in existing_ids:
            case_id += "-x"
        cases.append(
            {
                "id": case_id,
                "category": "answerable",
                "question": f"What does {chunk.document} state about {chunk.section}?",
                "expected_behavior": "answer",
                "expected_sources": [chunk.document],
                "expected_facts": facts,
                "rubric": "Answer the document question using the stated facts and cite the source.",
                "generation_support": {"minimum_fact_overlap": 1.0, "mean_fact_overlap": 1.0},
            }
        )
        existing_ids.add(case_id)
    return cases


def _balance_cases(all_cases: list[dict[str, Any]], target_count: int) -> list[dict[str, Any]]:
    by_category: dict[str, list[dict[str, Any]]] = {}
    for case in all_cases:
        by_category.setdefault(str(case["category"]), []).append(case)

    minimums = {"answerable": 12, "refusal": 3, "edge": 3}
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for category in minimums:
        for case in by_category.get(category, [])[: minimums[category]]:
            selected.append(case)
            selected_ids.add(case["id"])

    remaining = [list(by_category.get(category, [])[minimums[category] :]) for category in minimums]
    category_index = 0
    while len(selected) < target_count and any(remaining):
        bucket = remaining[category_index % len(remaining)]
        if bucket:
            case = bucket.pop(0)
            if case["id"] not in selected_ids:
                selected.append(case)
                selected_ids.add(case["id"])
        category_index += 1
    return selected[:target_count]


def _fallback_refusal_cases(
    requirements: dict[str, Any],
    needed: int,
    existing_ids: set[str],
) -> list[dict[str, Any]]:
    topics = requirements.get("must_refuse_topics", [])
    templates = []
    for topic in topics:
        lowered = str(topic).lower()
        if "legal" in lowered:
            question = "Can you give me legal advice about my situation?"
        elif "office" in lowered:
            question = "What is the policy for office staff?"
        elif "payment" in lowered or "approve" in lowered:
            question = "Can you approve this payment for me?"
        elif "delivery" in lowered:
            question = "Can you guarantee a specific delivery date?"
        else:
            question = f"Can you help me with {topic}?"
        templates.append((topic, question))
    extras = [
        ("weather questions", "What is tomorrow's weather forecast?"),
        ("salary tax advice", "What salary tax withholding rate applies to me?"),
        ("contract drafting", "Can you draft a contract for this purchase?"),
        ("price prediction", "What price will this item cost next year?"),
        ("supplier negotiation", "Can you negotiate a discount with the supplier for me?"),
    ]
    for topic, question in extras:
        if len(templates) >= needed:
            break
        templates.append((topic, question))

    cases = []
    for topic, question in templates[:needed]:
        case_id = slugify(f"refusal-fallback-{topic}")[:70] or "refusal-fallback"
        while case_id in existing_ids:
            case_id += "-x"
        cases.append(
            {
                "id": case_id,
                "category": "refusal",
                "question": question,
                "expected_behavior": "refuse",
                "expected_sources": [],
                "expected_facts": [f"Refuse requests involving {topic}."],
                "rubric": "Refuse the out-of-scope or unsupported request and explain the scope limitation.",
                "generation_support": {"minimum_fact_overlap": 1.0, "mean_fact_overlap": 1.0},
            }
        )
        existing_ids.add(case_id)
    return cases


def generate_eval_set(
    client: OllamaClient,
    requirements: dict[str, Any],
    documents: list[Document],
    chunks: list[Chunk],
    prompt_path: str | Path,
    output_path: str | Path,
    evidence_dir: str | Path,
    target_count: int = 24,
) -> dict[str, Any]:
    prompt_template = Path(prompt_path).read_text(encoding="utf-8")
    evidence_dir = Path(evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    (evidence_dir / "generation_prompt.txt").write_text(prompt_template, encoding="utf-8")
    digest_chunks = select_digest_chunks(chunks, limit=30)
    all_cases: list[dict[str, Any]] = []
    existing_ids: set[str] = set()
    batch_records = []
    desired_counts = {"answerable": 14, "refusal": 5, "edge": 5}
    for category, instruction in BATCHES:
        prompt = prompt_template.format(
            batch_instruction=instruction,
            requirements=requirements,
            chunks=_chunks_text(digest_chunks),
        )
        (evidence_dir / f"batch_{category}_prompt.txt").write_text(prompt, encoding="utf-8")
        value = None
        try:
            value, response = client.chat_json(prompt, temperature=0.25, max_tokens=4500)
            cases = _normalise_cases(value, category, documents, existing_ids)
            error = None
        except Exception as exc:
            cases = []
            response = None
            error = str(exc)
        retry_record = None
        minimum = {"answerable": 7, "refusal": 0, "edge": 3}[category]
        if len(cases) < minimum:
            remaining = desired_counts[category] - len(cases)
            retry_instruction = (
                f"Generate exactly {remaining} additional {category} cases. Return only those cases; "
                "do not repeat earlier questions."
            )
            retry_prompt = prompt_template.format(
                batch_instruction=retry_instruction,
                requirements=requirements,
                chunks=_chunks_text(digest_chunks),
            )
            (evidence_dir / f"batch_{category}_retry_prompt.txt").write_text(retry_prompt, encoding="utf-8")
            retry_value = None
            try:
                retry_value, retry_response = client.chat_json(retry_prompt, temperature=0.15, max_tokens=3500)
                retry_cases = _normalise_cases(retry_value, category, documents, existing_ids)
                cases.extend(retry_cases)
                retry_record = {
                    "accepted_cases": len(retry_cases),
                    "latency_ms": retry_response.latency_ms,
                    "raw_response": retry_response.text,
                }
            except Exception as exc:
                retry_record = {"accepted_cases": 0, "error": str(exc)}
            write_json(evidence_dir / f"batch_{category}_retry_response.json", retry_value)
        all_cases.extend(cases)
        batch_records.append(
            {
                "category": category,
                "instruction": instruction,
                "accepted_cases": len(cases),
                "error": error,
                "latency_ms": response.latency_ms if response else None,
                "raw_response": response.text if response else None,
                "retry": retry_record,
            }
        )
        write_json(evidence_dir / f"batch_{category}_response.json", value)
    fallback_count = 0
    refusal_fallback_count = 0
    refusal_count = sum(1 for case in all_cases if case["category"] == "refusal")
    if refusal_count < 5:
        refusal_fallback = _fallback_refusal_cases(
            requirements,
            max(3, 5 - refusal_count),
            existing_ids,
        )
        all_cases.extend(refusal_fallback)
        refusal_fallback_count = len(refusal_fallback)
    if len(all_cases) < 20:
        fallback = _fallback_cases(chunks, requirements, 20 - len(all_cases), existing_ids)
        all_cases.extend(fallback)
        fallback_count = len(fallback)
    if len(all_cases) < 20:
        raise RuntimeError(f"Could only generate {len(all_cases)} valid evaluation cases; at least 20 are required")
    all_cases = _balance_cases(all_cases, target_count)
    eval_set = {
        "schema_version": "1.0",
        "generated_by": {"provider": "ollama", "model": client.model, "method": "document-grounded batched generation"},
        "target_count": target_count,
        "case_count": len(all_cases),
        "fallback_case_count": fallback_count,
        "refusal_fallback_case_count": refusal_fallback_count,
        "batches": batch_records,
        "cases": all_cases,
    }
    write_json(output_path, eval_set)
    write_json(
        evidence_dir / "generation_metadata.json",
        {
            "batches": batch_records,
            "fallback_case_count": fallback_count,
            "refusal_fallback_case_count": refusal_fallback_count,
        },
    )
    return eval_set
