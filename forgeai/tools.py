from __future__ import annotations

import ast
import csv
import json
import math
import operator
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable


@dataclass
class ToolResult:
    name: str
    ok: bool
    data: Any
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "ok": self.ok, "data": self.data, "error": self.error}


TOOL_SPECS: dict[str, dict[str, Any]] = {
    "csv_lookup": {
        "description": "Search a configured CSV catalogue by text and return matching rows.",
        "parameters": {"query": "string", "limit": "integer, optional"},
    },
    "calculator": {
        "description": "Evaluate a safe arithmetic expression for totals, percentages, and unit conversions.",
        "parameters": {"expression": "string"},
    },
    "current_date": {
        "description": "Return the current UTC date and time.",
        "parameters": {},
    },
    "public_weather": {
        "description": "Get current weather for a named city using the keyless Open-Meteo API.",
        "parameters": {"city": "string"},
    },
}


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def safe_calculate(expression: str) -> float | int:
    expression = expression.strip().replace("^", "**")
    if len(expression) > 200:
        raise ValueError("Expression is too long")
    tree = ast.parse(expression, mode="eval")

    def evaluate(node: ast.AST) -> float | int:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
            left = evaluate(node.left)
            right = evaluate(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 12:
                raise ValueError("Exponent is too large")
            return _BINARY_OPERATORS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
            return _UNARY_OPERATORS[type(node.op)](evaluate(node.operand))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            allowed = {"abs": abs, "round": round, "min": min, "max": max, "sum": sum}
            if node.func.id in allowed and not node.keywords:
                return allowed[node.func.id](*(evaluate(arg) for arg in node.args))
        raise ValueError("Unsupported arithmetic expression")

    result = evaluate(tree)
    if isinstance(result, float) and not math.isfinite(result):
        raise ValueError("Result is not finite")
    return result


def _extract_expression(question: str) -> str:
    cleaned = question.lower().replace(",", "").replace("$", "")
    unit_pattern = re.search(r"(\d+(?:\.\d+)?)\s*(?:units?|items?)\s*(?:at|@)\s*\$?\s*(\d+(?:\.\d+)?)\s*(?:each|per unit)?", cleaned)
    if unit_pattern:
        cleaned = cleaned[: unit_pattern.start()] + f"{unit_pattern.group(1)}*{unit_pattern.group(2)}" + cleaned[unit_pattern.end() :]
    cleaned = re.sub(r"\b(calculate|compute|what is|what's|total|cost of|evaluate)\b", "", cleaned)
    cleaned = re.sub(r"\bplus\b", "+", cleaned)
    cleaned = re.sub(r"\bminus\b", "-", cleaned)
    cleaned = re.sub(r"\b(?:times|multiplied by)\b", "*", cleaned)
    cleaned = re.sub(r"\bdivided by\b", "/", cleaned)
    cleaned = re.sub(r"(\d+(?:\.\d+)?)\s*%?\s*(?:of|x)\s*\$?\s*(\d+(?:\.\d+)?)", r"\1/100*\2", cleaned)
    match = re.search(r"[-+*/%().\d\s^]+", cleaned)
    if not match:
        raise ValueError("No arithmetic expression found")
    expression = re.sub(r"[+\-*/%^]+$", "", match.group(0)).strip()
    if not re.search(r"\d", expression) or not re.search(r"[+*/%^-]", expression):
        raise ValueError("No arithmetic operator found")
    return expression


def _csv_lookup(question: str, config: dict[str, Any], knowledge_dir: Path) -> Any:
    csv_name = config.get("csv")
    if not csv_name:
        raise ValueError("csv_lookup requires a configured csv filename")
    path = (knowledge_dir / csv_name).resolve()
    if knowledge_dir.resolve() not in path.parents or path.suffix.lower() != ".csv":
        raise ValueError("CSV path is outside the knowledge folder or is not a CSV")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    query_terms = [term.lower() for term in re.findall(r"[A-Za-z0-9.-]+", question) if len(term) > 2]
    ranked = []
    for row in rows:
        haystack = " ".join(str(value) for value in row.values()).lower()
        score = sum(1 for term in query_terms if term in haystack)
        if score:
            ranked.append((score, row))
    ranked.sort(key=lambda item: (-item[0], json.dumps(item[1], sort_keys=True)))
    limit = min(int(config.get("limit", 10)), 25)
    return {"file": csv_name, "row_count": len(rows), "matches": [row for _, row in ranked[:limit]]}


def _public_weather(city: str) -> Any:
    city = city.strip()
    if not city:
        raise ValueError("City is required")
    geocode_url = "https://geocoding-api.open-meteo.com/v1/search?" + urllib.parse.urlencode(
        {"name": city, "count": 1, "language": "en", "format": "json"}
    )
    with urllib.request.urlopen(geocode_url, timeout=20) as response:
        geocode = json.loads(response.read().decode("utf-8"))
    results = geocode.get("results", [])
    if not results:
        raise ValueError(f"City not found: {city}")
    location = results[0]
    weather_url = "https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode(
        {
            "latitude": location["latitude"],
            "longitude": location["longitude"],
            "current": "temperature_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m",
            "timezone": "auto",
        }
    )
    with urllib.request.urlopen(weather_url, timeout=20) as response:
        weather = json.loads(response.read().decode("utf-8"))
    return {"location": location["name"], "country": location.get("country"), **weather.get("current", {})}


def choose_tool(question: str, attached_tools: list[dict[str, Any]]) -> tuple[str, dict[str, Any]] | None:
    names = {tool["name"] for tool in attached_tools}
    lowered = question.lower()
    if "public_weather" in names and re.search(r"\b(weather|forecast|temperature|rain)\b", lowered):
        city_match = re.search(r"\b(?:in|for|at)\s+([A-Za-z][A-Za-z .'-]{1,50})", question, flags=re.I)
        city = city_match.group(1).strip(" .?") if city_match else ""
        if city:
            return "public_weather", {"city": city}
    if "current_date" in names and re.search(r"\b(today|current date|what day|date now)\b", lowered):
        return "current_date", {}
    if "calculator" in names and re.search(r"\b(calculate|compute|total|percentage|percent|convert)\b", lowered):
        try:
            return "calculator", {"expression": _extract_expression(question)}
        except ValueError:
            pass
    if "csv_lookup" in names and re.search(r"\b(catalogue|catalog|part|sku|stock|price|item|spare)\b", lowered):
        return "csv_lookup", {"query": question, "limit": 10}
    return None


def execute_tool(
    name: str,
    parameters: dict[str, Any],
    attached_tools: list[dict[str, Any]],
    knowledge_dir: Path,
) -> ToolResult:
    config = next((tool.get("config", {}) for tool in attached_tools if tool["name"] == name), None)
    if config is None:
        return ToolResult(name=name, ok=False, data=None, error="Tool is not attached to this agent")
    try:
        if name == "calculator":
            data = safe_calculate(str(parameters["expression"]))
        elif name == "current_date":
            now = datetime.now(timezone.utc)
            data = {"date": now.date().isoformat(), "datetime": now.isoformat(), "timezone": "UTC"}
        elif name == "csv_lookup":
            data = _csv_lookup(str(parameters.get("query", "")), config, knowledge_dir)
        elif name == "public_weather":
            data = _public_weather(str(parameters.get("city", "")))
        else:
            raise ValueError(f"Unknown tool: {name}")
        return ToolResult(name=name, ok=True, data=data)
    except Exception as exc:
        return ToolResult(name=name, ok=False, data=None, error=str(exc))
