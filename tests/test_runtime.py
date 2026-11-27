from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from forgeai.llm import LLMResponse
from forgeai.runtime import AgentRunner
from forgeai.utils import write_json


class FakeClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses

    def chat(self, prompt: str, **kwargs: object) -> LLMResponse:
        return LLMResponse(text=self.responses.pop(0), latency_ms=1)


class RuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        knowledge = self.root / "knowledge"
        knowledge.mkdir()
        (knowledge / "hr.md").write_text("# Leave\nAnnual leave is 20 days per year.\n", encoding="utf-8")
        write_json(
            self.root / "agent.json",
            {
                "name": "HR",
                "version": 1,
                "model": {"name": "fake", "temperature": 0.0},
                "system_prompt": "Use evidence.",
                "knowledge": {
                    "documents_dir": "knowledge",
                    "chunking": {"chunk_size_words": 50, "overlap_words": 5},
                    "retrieval": {"top_k": 2, "min_score": 0.0},
                },
                "tools": [],
                "output_contract": {},
            },
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_runner_enforces_citations(self) -> None:
        client = FakeClient([json.dumps({"answer": "20 days", "citations": [], "confidence": 0.9, "refused": False})])
        result = AgentRunner(self.root / "agent.json", client=client).answer("How much annual leave?")
        self.assertTrue(result["refused"])
        self.assertEqual(result["citations"], [])

    def test_runner_accepts_valid_citation(self) -> None:
        client = FakeClient([json.dumps({"answer": "20 days", "citations": ["hr.md"], "confidence": 0.9, "refused": False})])
        result = AgentRunner(self.root / "agent.json", client=client).answer("How much annual leave?")
        self.assertFalse(result["refused"])
        self.assertEqual(result["citations"], ["hr.md"])

    def test_runner_resolves_run_level_knowledge_from_agents_folder(self) -> None:
        agents = self.root / "agents"
        agents.mkdir()
        nested_agent = agents / "agent.json"
        nested_agent.write_bytes((self.root / "agent.json").read_bytes())
        client = FakeClient([json.dumps({"answer": "20 days", "citations": ["hr.md"], "confidence": 0.9, "refused": False})])
        result = AgentRunner(nested_agent, client=client).answer("How much annual leave?")
        self.assertFalse(result["refused"])

    def test_batch_runner_returns_one_result_per_question(self) -> None:
        values = [
            {"answer": "20 days", "citations": ["hr.md"], "confidence": 0.9, "refused": False},
            {"answer": "20 days", "citations": ["hr.md"], "confidence": 0.9, "refused": False},
        ]
        client = FakeClient([json.dumps(values)])
        results = AgentRunner(self.root / "agent.json", client=client).answer_batch(["Leave?", "Annual leave?"])
        self.assertEqual(len(results), 2)
        self.assertTrue(all(not result["refused"] for result in results))


if __name__ == "__main__":
    unittest.main()
