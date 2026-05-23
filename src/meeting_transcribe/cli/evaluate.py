#!/usr/bin/env python3
"""Test-only: score compare-models markdown output against your reference transcript (OpenRouter judge).

Not used in real transcription — production `transcribe` does not read test references.

Usage:
    evaluate-models cases/img_6423/tail.m4a
    evaluate-models cases/img_6423/tail.m4a --compare-md cases/img_6423/compare.md

Default test reference: ``<repo>/test_references/<case_id>/ground_truth.rtl.md`` (``case_id`` = parent
folder of the audio, e.g. ``img_6423``). Override with ``--ground-truth PATH``. Audio under ``runs/…``
may not match ``case_id`` — pass ``--ground-truth`` explicitly in that case.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

from meeting_transcribe.core.compare_md import parse_compare_model_sections
from meeting_transcribe.core.paths import project_root, resolve_ground_truth_path

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
JUDGE_MODEL = "anthropic/claude-opus-4-5"

MODELS = [
    "anthropic/claude-opus-4-5",
    "anthropic/claude-opus-4-7",
    "anthropic/claude-sonnet-4-5",
    "openai/gpt-4.1",
    "openai/gpt-4.1-mini",
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "google/gemini-2.5-pro",
    "google/gemini-2.5-flash",
    "google/gemini-2.0-flash-001",
]

JUDGE_SYSTEM = """You are an expert evaluator of Hebrew/English code-switched meeting transcription quality.
You compare AI-structured meeting notes against a human-corrected ground truth and score on specific rubrics.
Respond only with valid JSON."""

JUDGE_PROMPT = """Evaluate the AI model output below against the ground truth transcript of a Hebrew/English code-switched meeting.

## Ground Truth (human-corrected organized transcript)
{ground_truth}

## Model Output (structured notes to evaluate)
{model_output}

## Equivalence Rule
Hebrew and English variants of the same word count as CORRECT:
- "evaluation" = "אבולואציה" = "אבולואיישן" ✓
- "default" = "דיפולט" ✓
- "reusable" = "ריליזיבל" ✓
A completely different word is always wrong, regardless of language.

## Rubrics

**Rubric 1 — Named Entity Accuracy (1-10, strict, no equivalence rule)**
People, product names, org names, tech tool names.
Key entities: Arin (ארין), Eli (אילי), Gil (גיל), Idan (עידן), Himanshu, Ben Stiller (בן סטילר),
Common Agentic Core (CAF), LangGraph, LangFuse, Concon (קונקון), Converge Consumer, swarm, A2A.
10 = all correct, 1 = most wrong or garbled.

**Rubric 2 — Semantic Integrity (1-10)**
Did any substitution flip or destroy meaning?
Critical example: "האדום" (the red line on the board) vs "האדם" (the person).
"feedback loop" → "פינג פלקטי" = semantic destruction.
10 = no meaning-flipping errors, 1 = several critical meaning breaks.

**Rubric 3 — Completeness (1-10)**
What fraction of key topics appear in the output?
Key topics: org hierarchy (IIH, dual-axis red/blue), Arin as manager, weekly tech lead with Eli,
CAF / Common Agentic Core explanation, LangGraph usage, multi-agent architecture
(guardrail → orchestrator → agents), evaluator / feedback loop, follow-up prompts feature,
swarm / A2A connectivity, demo offer, admin dashboard / LangFuse tracing, product list.
10 = all topics, 1 = major sections missing.

**Rubric 4 — Lexical Fidelity (1-10, with equivalence rule)**
Word-level accuracy. Are key technical concepts correctly captured?
10 = virtually all words correct, 1 = many wrong.

**Rubric 5 — Hallucination (1-10, higher = cleaner)**
Did the model add content not in the original?
10 = no hallucinations, 1 = significant fabrications.

## Output
Return ONLY a JSON object with this exact structure:
{{
  "named_entity_accuracy": {{"score": N, "notes": "brief explanation with examples"}},
  "semantic_integrity": {{"score": N, "notes": "brief explanation with examples"}},
  "completeness": {{"score": N, "notes": "brief explanation"}},
  "lexical_fidelity": {{"score": N, "notes": "brief explanation"}},
  "hallucination": {{"score": N, "notes": "brief explanation"}},
  "overall": N,
  "summary": "1-2 sentence overall assessment"
}}

Overall is your holistic judgment (not a simple average). Weight named_entity_accuracy and semantic_integrity more heavily."""


def judge_model(model_output: str, ground_truth: str) -> dict:
    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=os.environ["OPENROUTER_API_KEY"])
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        max_tokens=4096,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": JUDGE_PROMPT.format(ground_truth=ground_truth, model_output=model_output)},
        ],
    )
    text = response.choices[0].message.content
    start = text.find("{")
    end = text.rfind("}") + 1
    return json.loads(text[start:end])


def print_table(results: list[dict]) -> None:
    rubrics = [
        ("named_entity_accuracy", "Entities"),
        ("semantic_integrity", "Semantic"),
        ("completeness", "Complete"),
        ("lexical_fidelity", "Lexical"),
        ("hallucination", "Halluc."),
    ]
    col_model = max(len(r["model"]) for r in results) + 2
    col_w = 9
    col_cost = 10

    header = f"{'Model':<{col_model}}" + "".join(f"{s:>{col_w}}" for _, s in rubrics) + f"{'Overall':>{col_w}}" + f"{'Cost ($)':>{col_cost}}"
    sep = "─" * len(header)
    print()
    print(sep)
    print(header)
    print(sep)
    for r in results:
        row = f"{r['model']:<{col_model}}"
        for rubric, _ in rubrics:
            score = r.get(rubric, {}).get("score", "—")
            row += f"{score:>{col_w}}"
        row += f"{r.get('overall', '—'):>{col_w}}"
        cost = r.get("cost_usd", 0.0) or 0.0
        cost_s = f"${cost:.4f}" if cost else "n/a"
        row += f"{cost_s:>{col_cost}}"
        print(row)
    print(sep)
    print()


def save_report(results: list[dict], output_path: Path, ground_truth_label: str) -> None:
    lines = [
        "# Transcription Evaluation Report\n",
        f"Judge: {JUDGE_MODEL}  |  Test reference: {ground_truth_label}\n",
    ]
    rubrics = [
        "named_entity_accuracy",
        "semantic_integrity",
        "completeness",
        "lexical_fidelity",
        "hallucination",
    ]
    for r in results:
        lines.append(f"## {r['model']}\n")
        lines.append(f"**Overall: {r.get('overall', '—')}/10** — {r.get('summary', '')}\n")
        for rubric in rubrics:
            data = r.get(rubric, {})
            label = rubric.replace("_", " ").title()
            lines.append(f"- **{label}**: {data.get('score', '—')}/10 — {data.get('notes', '')}")
        lines.append("\n---\n")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test-only: evaluate structured outputs vs your reference (not production)")
    p.add_argument("audio_file", help="Audio path (same case folder as compare.md unless --compare-md is set)")
    p.add_argument(
        "--ground-truth",
        default=None,
        help="Test reference file (default: <repo>/test_references/<parent-of-audio>/ground_truth.rtl.md)",
    )
    p.add_argument(
        "--compare-md",
        default=None,
        help="Output from compare-models (default: compare.md beside audio)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    audio_path = args.audio_file
    if not os.path.exists(audio_path):
        print(f"ERROR: File not found: {audio_path}", file=sys.stderr)
        sys.exit(1)
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    try:
        gt_path = resolve_ground_truth_path(audio_path, args.ground_truth)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        gt_label = str(gt_path.relative_to(project_root()))
    except ValueError:
        gt_label = str(gt_path)

    ground_truth = gt_path.read_text(encoding="utf-8")

    compare_path = Path(args.compare_md).expanduser().resolve() if args.compare_md else Path(audio_path).resolve().parent / "compare.md"
    if not compare_path.is_file():
        print(f"ERROR: compare markdown not found: {compare_path} (run compare-models or pass --compare-md)", file=sys.stderr)
        sys.exit(1)

    try:
        sections = parse_compare_model_sections(compare_path)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print("=== Evaluation ===")
    print(f"Audio:      {audio_path}")
    print(f"Compare:    {compare_path}")
    print(f"Judge:      {JUDGE_MODEL}")
    print()

    results = []
    for model in MODELS:
        if model not in sections:
            print(f"  SKIP {model} — not in {compare_path.name}")
            continue
        model_output = sections[model]
        print(f"  Judging {model}...", end=" ", flush=True)
        t0 = time.time()
        try:
            scores = judge_model(model_output, ground_truth)
            results.append({"model": model, "cost_usd": 0.0, **scores})
            print(f"overall={scores['overall']}  ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"ERROR: {e}")

    if not results:
        print("No results to show.")
        return

    print_table(results)

    output_path = Path(audio_path).parent / "evaluation.md"
    save_report(results, output_path, gt_label)
    print(f"Detailed report saved to: {output_path}")


if __name__ == "__main__":
    main()
