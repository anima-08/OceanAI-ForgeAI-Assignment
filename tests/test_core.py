from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from forgeai.documents import Document, chunk_documents, load_documents
from forgeai.evals import _normalise_cases
from forgeai.generator import _validate_agent
from forgeai.intake import deterministic_groundability, infer_capabilities
from forgeai.judge import deterministic_judge
from forgeai.optimizer import targeted_patch
from forgeai.retrieval import BM25Retriever
from forgeai.tools import _extract_expression, safe_calculate
from forgeai.utils import extract_json


class CoreTests(unittest.TestCase):
    def test_extract_json_from_wrapped_text(self) -> None:
        self.assertEqual(extract_json('```json\n{"ok": true}\n```'), {"ok": True})
        self.assertEqual(extract_json('prefix {"ok": true} suffix'), {"ok": True})

    def test_safe_calculator(self) -> None:
        self.assertEqual(safe_calculate("(20 * 3) + 5"), 65)
        with self.assertRaises(ValueError):
            safe_calculate("__import__('os').system('echo unsafe')")

    def test_calculator_extracts_natural_language_expression(self) -> None:
        expression = _extract_expression("Calculate 10 units at $120 each plus $50 freight")
        self.assertEqual(safe_calculate(expression), 1250)

    def test_chunking_and_retrieval(self) -> None:
        document = Document("policy.md", "policy.md", "# Leave\nAnnual leave is 20 days.\n\n# Travel\nTravel reimbursement requires receipts.", "md", "x")
        chunks = chunk_documents([document], chunk_size_words=20, overlap_words=3)
        self.assertEqual(len(chunks), 2)
        results = BM25Retriever(chunks).search("How much annual leave?", top_k=1)
        self.assertEqual(results[0][0].document, "policy.md")
        self.assertIn("20 days", results[0][0].text)

    def test_load_csv_and_pdf_are_supported(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "catalogue.csv"
            path.write_text("sku,price\nA1,10.00\n", encoding="utf-8")
            documents = load_documents(folder)
            self.assertEqual(documents[0].name, "catalogue.csv")
            chunks = chunk_documents(documents)
            self.assertEqual(len(chunks), 1)
            self.assertIn("A1", chunks[0].text)

    def test_agent_tool_allowlist(self) -> None:
        documents = [Document("hr.md", "hr.md", "Leave is 20 days.", "md", "x")]
        requirements = {"name": "HR", "scope": "HR", "required_capabilities": ["document_qa"]}
        value = {
            "name": "HR",
            "system_prompt": "Ground answers.",
            "knowledge": {"documents": ["hr.md"]},
            "tools": [{"name": "public_weather"}, {"name": "calculator"}],
        }
        agent = _validate_agent(value, documents, requirements)
        self.assertEqual(agent["tools"], [])

    def test_agent_adds_required_tool_missing_from_model_output(self) -> None:
        documents = [Document("catalogue.csv", "catalogue.csv", "sku,price\nA,10", "csv", "x")]
        requirements = {"name": "Procurement", "scope": "Quote costs", "required_capabilities": ["document_qa", "calculator"]}
        agent = _validate_agent({"name": "Procurement", "tools": []}, documents, requirements)
        self.assertEqual([tool["name"] for tool in agent["tools"]], ["calculator"])

    def test_eval_normalisation_rejects_unknown_sources(self) -> None:
        documents = [Document("hr.md", "hr.md", "Annual leave is 20 days.", "md", "x")]
        cases = _normalise_cases(
            {"cases": [{"question": "Leave?", "expected_behavior": "answer", "expected_sources": ["missing.md"], "expected_facts": ["20 days"]}]},
            "answerable",
            documents,
            set(),
        )
        self.assertEqual(cases, [])

    def test_deterministic_judge_catches_wrong_refusal(self) -> None:
        case = {"expected_behavior": "answer", "expected_sources": ["hr.md"], "expected_facts": ["20 days"]}
        actual = {"answer": "I cannot help.", "citations": [], "refused": True, "retrieved_sources": ["hr.md"]}
        result = deterministic_judge(case, actual)
        self.assertFalse(result["hard_gate"])

    def test_capability_audit_prevents_unneeded_tools(self) -> None:
        hr = [Document("hr.md", "hr.md", "Leave is 20 days.", "md", "x")]
        procurement = [Document("catalogue.csv", "catalogue.csv", "sku,price\nA,10", "csv", "x")]
        self.assertEqual(infer_capabilities("HR policy assistant", {"scope": "HR"}, hr), ["document_qa"])
        self.assertEqual(
            infer_capabilities("Search catalogue and calculate total cost", {"scope": "procurement"}, procurement),
            ["document_qa", "csv_lookup", "calculator"],
        )

    def test_deterministic_groundability_blocks_missing_topic(self) -> None:
        documents = [Document("hr.md", "hr.md", "Leave is 20 days.", "md", "x")]
        requirements = {"must_answer_topics": ["payroll tax withholding"]}
        result = deterministic_groundability(requirements, documents, {"supported": True, "gaps": []})
        self.assertFalse(result["supported"])
        self.assertEqual(result["gaps"][0]["detector"], "deterministic_topic_overlap")

    def test_deterministic_evidence_downgrades_false_model_gap(self) -> None:
        documents = [Document("travel.md", "travel.md", "Travel reimbursement includes regional per-diem rates.", "md", "x")]
        requirements = {"must_answer_topics": ["Travel Reimbursement"]}
        model_report = {
            "supported": False,
            "gaps": [{"request": "Travel Reimbursement", "reason": "Unclear.", "severity": "blocking"}],
        }
        result = deterministic_groundability(requirements, documents, model_report)
        self.assertTrue(result["supported"])
        self.assertEqual(result["gaps"][0]["severity"], "warning")

    def test_targeted_patch_is_not_full_regeneration(self) -> None:
        agent = {"version": 1, "system_prompt": "Be grounded.", "knowledge": {"retrieval": {"top_k": 4, "min_score": 0.1}}}
        scorecard = {"failures": [{"case_id": "a", "failure_category": "retrieval_miss"}]}
        patched, changes = targeted_patch(agent, scorecard)
        self.assertEqual(patched["version"], 2)
        self.assertEqual(patched["knowledge"]["retrieval"]["top_k"], 6)
        self.assertEqual(patched["system_prompt"], agent["system_prompt"] + "\n\n" + patched["system_prompt"].split("\n\n", 1)[1])
        self.assertEqual(changes[0]["failure_ids"], ["a"])


if __name__ == "__main__":
    unittest.main()
