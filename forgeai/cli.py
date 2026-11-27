from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .llm import OllamaClient
from .pipeline import PROMPT_DIR, run_factory
from .runtime import AgentRunner
from .utils import read_json, write_json
from .validation import validate_run


def _load_answers(path: str | None) -> dict[str, str] | None:
    if not path:
        return None
    value = read_json(path)
    if isinstance(value, dict):
        return {str(key): str(answer) for key, answer in value.items()}
    if isinstance(value, list):
        return {
            str(item["id"]): str(item["answer"])
            for item in value
            if isinstance(item, dict) and "id" in item and "answer" in item
        }
    raise ValueError("Answers file must be a JSON object or a list of {id, answer} objects")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="forgeai", description="ForgeAI: build, test, and improve grounded agents")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run the full agent factory")
    run.add_argument("--brief", required=True, help="Brief text or path to a text file")
    run.add_argument("--docs", required=True, help="Folder containing source documents")
    run.add_argument("--out", required=True, help="Empty output folder for this run")
    run.add_argument("--model", default="llama3.2:latest")
    run.add_argument("--base-url", default="http://127.0.0.1:11434")
    run.add_argument("--answers", help="JSON file with requester answers for unattended runs")
    run.add_argument("--interactive", action="store_true", help="Ask up to five clarifying questions")
    run.add_argument("--max-iterations", type=int, default=3, choices=range(0, 4))
    run.add_argument("--target-pass-rate", type=float, default=0.85)
    run.add_argument("--no-llm-judge", action="store_true", help="Use deterministic judging only")

    chat = subparsers.add_parser("chat", help="Run any generated agent through the generic chat runner")
    chat.add_argument("--agent", required=True, help="Path to an agent definition JSON")
    chat.add_argument("--question", help="One question; omit for interactive chat")
    chat.add_argument("--model", help="Override the model configured in the agent")
    chat.add_argument("--base-url", default="http://127.0.0.1:11434")

    synth = subparsers.add_parser("synth-demo", help="Generate synthetic demo documents from a committed prompt")
    synth.add_argument("--prompt", required=True, help="Path to a synthesis prompt")
    synth.add_argument("--out", required=True, help="Folder for generated documents")
    synth.add_argument("--model", default="llama3.2:latest")
    synth.add_argument("--base-url", default="http://127.0.0.1:11434")
    synth.add_argument("--max-tokens", type=int, default=7000)

    validate = subparsers.add_parser("validate", help="Check that a run contains all mandatory artifacts")
    validate.add_argument("--run", required=True, help="ForgeAI run directory")
    validate.add_argument("--expected-tools", nargs="*", default=None, help="Exact tool names expected on the final agent")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        result = run_factory(
            args.brief,
            args.docs,
            args.out,
            model=args.model,
            base_url=args.base_url,
            answers=_load_answers(args.answers),
            interactive=args.interactive,
            max_iterations=args.max_iterations,
            target_pass_rate=args.target_pass_rate,
            use_llm_judge=not args.no_llm_judge,
        )
        print(json.dumps({key: value for key, value in result.items() if key != "scorecards"}, indent=2))
        return 0 if result["status"] == "completed" else 2
    if args.command == "chat":
        client = OllamaClient(model=args.model or "llama3.2:latest", base_url=args.base_url)
        runner = AgentRunner(args.agent, client=client)
        if args.question:
            print(json.dumps(runner.answer(args.question), indent=2, ensure_ascii=False))
            return 0
        print(f"Loaded {runner.agent.get('name')} v{runner.agent.get('version')}. Type 'exit' to quit.")
        while True:
            try:
                question = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if question.lower() in {"exit", "quit"}:
                break
            if not question:
                continue
            result = runner.answer(question)
            print(json.dumps({key: value for key, value in result.items() if key != "raw_response"}, indent=2, ensure_ascii=False))
        return 0
    if args.command == "validate":
        print(json.dumps(validate_run(args.run, args.expected_tools), indent=2))
        return 0
    if args.command == "synth-demo":
        prompt = Path(args.prompt).read_text(encoding="utf-8")
        client = OllamaClient(model=args.model, base_url=args.base_url)
        value, response = client.chat_json(prompt, temperature=0.3, max_tokens=args.max_tokens)
        if not isinstance(value, dict):
            raise ValueError("Synthesis response must map filenames to content")
        output = Path(args.out)
        output.mkdir(parents=True, exist_ok=True)
        for filename, content in value.items():
            if Path(filename).name != filename:
                raise ValueError(f"Unsafe filename: {filename}")
            if not isinstance(content, str):
                raise ValueError(f"Content for {filename} must be a string")
            (output / filename).write_text(content, encoding="utf-8")
        write_json(output / "synthesis_metadata.json", {"prompt": str(Path(args.prompt).resolve()), "model": args.model, "latency_ms": response.latency_ms})
        print(f"Wrote {len(value)} documents to {output}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
