# ForgeAI Quality Report

Generated at: 2026-11-27T05:17:53.315386+00:00

## Outcome

Status: **target met**

- Agent: `HR Policy Assistant` version `2`
- Evaluation cases: 23
- Final passed: 19/23
- Final pass rate: 82.6%
- Final mean score: 0.902

## Score progression

| Iteration | Passed | Pass rate | Mean score |
| --- | ---: | ---: | ---: |
| 0 | 16/23 | 69.6% | 0.855 |
| 1 | 19/23 | 82.6% | 0.902 |

## Remaining failures

- `hr-policy-assistant-3` (prompt_defect, score 0.662): Deterministic behavior, citation, fact, or retrieval gate failed.
- `refusal-legal-advice` (prompt_defect, score 0.850): Deterministic behavior, citation, fact, or retrieval gate failed.
- `refusal-office-staff-policy` (prompt_defect, score 0.850): Deterministic behavior, citation, fact, or retrieval gate failed.
- `refusal-guarantee-approval` (prompt_defect, score 0.850): Deterministic behavior, citation, fact, or retrieval gate failed.

## Groundability and scope

- Intake support status: supported
- Must-answer topics: Leave eligibility, Travel reimbursement, Medical cover, Conduct rules
- Must-refuse topics: Legal advice, Office-staff policy questions, Anything not covered by the policies

## Tools attached

- No external tools were attached; the agent uses document retrieval only.

## Judge safeguards

- Deterministic behavior, citation, expected-source, retrieval, and fact-coverage checks can veto the local LLM judge.
- The judge sees one anonymized candidate at a time under a fixed rubric; verbosity and confidence are not scored.
- The local judge may still make semantic mistakes, so per-case evidence is retained in `evaluations/`.

## Known limitations

- Evaluation questions are generated from the supplied documents and can inherit source ambiguity or model-generation errors.
- Lexical BM25 retrieval can miss synonyms or rare phrasing that dense embeddings might capture.
- Passing this set does not establish production readiness, security, legal compliance, or factual correctness outside the supplied corpus.
