from __future__ import annotations

import unittest
from pathlib import Path


PROMPTS = Path(__file__).resolve().parent.parent / "prompts"


class PromptTemplateTests(unittest.TestCase):
    def test_factory_prompts_format_without_key_errors(self) -> None:
        samples = {
            "intake_prompt.txt": {"brief": "b", "inventory": "i", "digest": "d"},
            "agent_generation_prompt.txt": {"tool_registry": "t", "requirements": "r", "inventory": "i", "digest": "d"},
            "eval_generation_prompt.txt": {"batch_instruction": "b", "requirements": "r", "chunks": "c"},
            "judge_prompt.txt": {"payload": "p"},
            "demo_single_document_synthesis.txt": {"filename": "f.md", "format": "markdown", "context": "c", "requirements": "r"},
        }
        for filename, values in samples.items():
            with self.subTest(filename=filename):
                rendered = (PROMPTS / filename).read_text(encoding="utf-8").format(**values)
                self.assertIn("JSON", rendered)


if __name__ == "__main__":
    unittest.main()
