from __future__ import annotations

from pathlib import Path
from typing import Any

from .documents import Document, document_digest, document_inventory
from .llm import OllamaClient
from .tools import TOOL_SPECS
from .utils import write_json


CAPABILITY_TO_TOOLS = {
    "csv_lookup": "csv_lookup",
    "calculator": "calculator",
    "current_date": "current_date",
    "public_weather": "public_weather",
}


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _validate_agent(value: Any, documents: list[Document], requirements: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Agent definition must be a JSON object")
    document_names = {document.name for document in documents}
    knowledge = value.get("knowledge") if isinstance(value.get("knowledge"), dict) else {}
    listed_documents = knowledge.get("documents") or []
    if isinstance(listed_documents, str):
        listed_documents = [listed_documents]
    listed_documents = [name for name in listed_documents if name in document_names]
    if not listed_documents:
        listed_documents = sorted(document_names)
    chunking = knowledge.get("chunking") if isinstance(knowledge.get("chunking"), dict) else {}
    retrieval = knowledge.get("retrieval") if isinstance(knowledge.get("retrieval"), dict) else {}
    chunk_size = max(60, min(_safe_int(chunking.get("chunk_size_words", 180), 180), 500))
    overlap = max(0, min(_safe_int(chunking.get("overlap_words", 30), 30), chunk_size // 2))
    top_k = max(1, min(_safe_int(retrieval.get("top_k", 4), 4), 12))
    min_score = max(0.0, min(_safe_float(retrieval.get("min_score", 0.0), 0.0), 10.0))

    allowed_tools = {
        CAPABILITY_TO_TOOLS[capability]
        for capability in requirements.get("required_capabilities", [])
        if capability in CAPABILITY_TO_TOOLS
    }
    generated_tools = value.get("tools") or []
    tools = []
    for tool in generated_tools:
        if not isinstance(tool, dict) or tool.get("name") not in TOOL_SPECS:
            continue
        if tool["name"] not in allowed_tools:
            continue
        config = tool.get("config") if isinstance(tool.get("config"), dict) else {}
        if tool["name"] == "csv_lookup":
            csv_name = config.get("csv")
            if csv_name not in document_names or not str(csv_name).lower().endswith(".csv"):
                csv_name = next((name for name in sorted(document_names) if name.lower().endswith(".csv")), None)
            if not csv_name:
                continue
            config = {
                "csv": csv_name,
                "limit": max(1, min(_safe_int(config.get("limit", 10), 10), 25)),
            }
        tools.append({"name": tool["name"], "config": config, "rationale": str(tool.get("rationale", ""))})
    for tool_name in sorted(allowed_tools):
        if any(tool["name"] == tool_name for tool in tools):
            continue
        config: dict[str, Any] = {}
        if tool_name == "csv_lookup":
            csv_name = next((name for name in sorted(document_names) if name.lower().endswith(".csv")), None)
            if not csv_name:
                continue
            config = {"csv": csv_name, "limit": 10}
        tools.append(
            {
                "name": tool_name,
                "config": config,
                "rationale": "Required by the deterministic brief capability audit.",
            }
        )

    output_contract = value.get("output_contract") if isinstance(value.get("output_contract"), dict) else {}
    required_fields = ["answer", "citations", "confidence", "refused"]
    system_prompt = str(value.get("system_prompt") or "").strip()
    if not system_prompt:
        system_prompt = (
            "You are a grounded assistant. Answer only from supplied evidence. Cite exact source filenames. "
            "Refuse unsupported or out-of-scope requests. Never invent facts."
        )
    model = _as_dict(value.get("model"))
    return {
        "name": str(value.get("name") or requirements["name"]),
        "version": _safe_int(value.get("version", 1), 1),
        "description": str(value.get("description") or requirements["scope"]),
        "model": {
            "provider": "ollama",
            "name": str(model.get("name") or "llama3.2:latest"),
            "temperature": max(0.0, min(_safe_float(model.get("temperature", 0.1), 0.1), 1.0)),
        },
        "system_prompt": system_prompt,
        "knowledge": {
            "enabled": bool(knowledge.get("enabled", True)),
            "documents_dir": "knowledge",
            "documents": sorted(listed_documents),
            "chunking": {
                "strategy": "paragraph" if chunking.get("strategy") == "paragraph" else "paragraph",
                "chunk_size_words": chunk_size,
                "overlap_words": overlap,
            },
            "retrieval": {
                "method": "bm25",
                "top_k": top_k,
                "min_score": min_score,
            },
        },
        "tools": tools,
        "output_contract": {
            "format": "json",
            "required_fields": required_fields,
            "citation_rule": str(output_contract.get("citation_rule") or "Cite exact source filenames for factual answers."),
            "refusal_rule": str(output_contract.get("refusal_rule") or "Refuse unsupported or out-of-scope requests."),
        },
        "scope": {
            "must_answer_topics": requirements.get("must_answer_topics", []),
            "must_refuse_topics": requirements.get("must_refuse_topics", []),
        },
    }


def generate_agent(
    client: OllamaClient,
    requirements: dict[str, Any],
    documents: list[Document],
    prompt_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    prompt_template = Path(prompt_path).read_text(encoding="utf-8")
    registry = [
        {"name": name, **spec} for name, spec in TOOL_SPECS.items()
    ]
    prompt = prompt_template.format(
        tool_registry=registry,
        requirements=requirements,
        inventory=document_inventory(documents),
        digest=document_digest(documents, per_document=1800, total=14000),
    )
    value, _ = client.chat_json(prompt, temperature=0.15, max_tokens=3500)
    agent = _validate_agent(value, documents, requirements)
    agent["model"]["name"] = client.model
    write_json(output_path, agent)
    return agent
