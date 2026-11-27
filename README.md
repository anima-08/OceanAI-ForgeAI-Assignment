# OceanAI ForgeAI Assignment

ForgeAI is a local GenAI system that turns a one-paragraph brief and a folder of documents into a grounded agent. It drafts the agent, creates an evaluation set from the source documents, runs the agent, judges the results, and applies targeted corrections.

## What is implemented

The project covers the nine mandatory assignment requirements:

1. **Intake interviewer:** Inspects the brief and documents, asks up to five clarifying questions, and produces a structured requirements document.
2. **Groundability guard:** Flags requested topics that the supplied documents cannot support before an agent is generated.
3. **Agent generator:** Produces a declarative agent definition containing the system prompt, knowledge settings, chunking and retrieval parameters, selected tools, and output contract.
4. **Evaluation generator:** Creates at least 20 document-grounded test cases covering answerable questions, required refusals, and edge cases.
5. **Judge:** Combines the local model’s judgement with deterministic behavior, citation, expected-source, retrieval, and fact-coverage checks.
6. **Critic–optimizer loop:** Classifies failures, applies targeted patches instead of regenerating the whole agent, and records each change with before/after scores.
7. **Honest reporting:** Produces scorecards and a report that identifies the agent’s remaining failures and limitations.
8. **Runnable chat runner:** Loads any generated agent definition and answers questions through a generic CLI.
9. **Demonstration inputs:** Includes two deliberately different synthetic document packs: shore-staff HR policies and ship-spares procurement.

## Tool registry

The factory can choose from four tools, but attaches only those required by the brief:

- CSV lookup for catalogue searches
- Safe arithmetic calculator
- Current UTC date
- Keyless Open-Meteo weather lookup

## Setup

Requires Python 3.10 or newer and [Ollama](https://ollama.com/).

```bash
python -m pip install -r requirements.txt
ollama pull llama3.2:latest
```

## Run the factory

Use a brief file, a document folder, and an empty output folder:

```bash
python -m forgeai run \
  --brief demos/hr_shore_staff/input/brief.txt \
  --docs demos/hr_shore_staff/docs \
  --out demos/hr_shore_staff/run \
  --answers demos/hr_shore_staff/input/answers.json \
  --max-iterations 3 \
  --target-pass-rate 0.80
```

Use `--interactive` to ask the requester’s clarifying questions through the terminal. For an unattended run, provide `--answers` or let the factory use the documented safe defaults.

Main run artifacts are written to:

- `requirements/` — requirements and intake dialogue
- `groundability/` — unsupported-request findings
- `agents/` — generated and optimized agent versions
- `evals/` — generated evaluation set and generation evidence
- `evaluations/` — per-case responses
- `scorecards/` — per-iteration results
- `iterations/` — targeted changes and score comparisons
- `report.md` — honest final quality report
- `final_agent.json` — best retained agent definition

## Chat with a generated agent

```bash
python -m forgeai chat --agent demos/hr_shore_staff/run/final_agent.json
```

Pass `--question "..."` for a single non-interactive answer.

## Tests

```bash
python -m unittest discover -s tests
```

## Validate a completed run

```bash
python -m forgeai validate --run demos/hr_shore_staff/run
python -m forgeai validate --run demos/ship_spares_procurement/run --expected-tools csv_lookup calculator
```

Validation checks the mandatory artifact set, evaluation category coverage, answerable grounding, tool allowlist, and the three-iteration optimizer limit.

## Demonstration status

The two demo briefs, source documents, synthesis prompts, and raw synthesis responses are included. Complete end-to-end run folders are committed as they finish on the local machine; the factory and generic runner are already implemented and tested.
