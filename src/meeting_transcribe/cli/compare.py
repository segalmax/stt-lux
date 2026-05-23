#!/usr/bin/env python3
"""Compare multiple LLM models on the same transcript. Output: compare.md beside audio.

Does not use any ground-truth file (that is only for compare-stt / evaluate-models tests).

Usage: compare-models cases/img_6423/tail.m4a
"""

import argparse
import os
import sys
import time
from pathlib import Path

from meeting_transcribe.core.audio import audio_duration_seconds
from meeting_transcribe.core.config import DEFAULT_ORCHESTRATOR_MODEL, DEFAULT_STT_MODEL
from meeting_transcribe.core.orchestrate_context import (
    format_downstream_llm_context,
    load_ref_corpus,
    run_context_orchestrator,
)
from meeting_transcribe.core.pipeline_report import multi_model_notes_preamble
from meeting_transcribe.core.run_paths import ensure_output_dir, new_run_dir
from meeting_transcribe.core.stt import transcribe
from meeting_transcribe.core.structure import correct_transcript, structure_notes_openrouter

DEFAULT_MODELS = [
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

DEFAULT_CORRECTION_MODEL = "google/gemini-2.5-flash"


def run_model(transcript: str, model: str, meeting_context: str) -> dict:
    print(f"  Running {model}...", end=" ", flush=True)
    t0 = time.time()
    text, in_tok, out_tok, elapsed, cost_usd = structure_notes_openrouter(transcript, model, meeting_context)
    result = {
        "model": model,
        "text": text,
        "in_tokens": in_tok,
        "out_tokens": out_tok,
        "elapsed_s": time.time() - t0,
        "cost_usd": cost_usd,
    }
    print(f"done ({result['elapsed_s']:.1f}s)")
    return result


def print_table(results: list[dict], transcript_cost: float, correction_cost: float = 0.0) -> None:
    col_model = max(len(r["model"]) for r in results) + 2
    col_tokens = 20
    col_time = 8
    col_cost = 10
    header = (
        f"{'Model':<{col_model}} {'Tokens (in+out)':<{col_tokens}} "
        f"{'Time':>{col_time}} {'Cost ($)':>{col_cost}}"
    )
    sep = "─" * len(header)
    print()
    print(sep)
    print(header)
    print(sep)
    total_llm_cost = 0.0
    for r in results:
        tokens = f"{r['in_tokens']:,}+{r['out_tokens']:,}"
        time_s = f"{r['elapsed_s']:.1f}s"
        cost = r.get("cost_usd", 0.0) or 0.0
        cost_s = f"${cost:.4f}" if cost else "n/a"
        total_llm_cost += cost
        print(f"{r['model']:<{col_model}} {tokens:<{col_tokens}} {time_s:>{col_time}} {cost_s:>{col_cost}}")
    print(sep)
    total = transcript_cost + correction_cost + total_llm_cost
    corr_str = f"  |  Correction: ${correction_cost:.4f}" if correction_cost else ""
    print(f"Transcription: ${transcript_cost:.4f}{corr_str}  |  LLM: ${total_llm_cost:.4f}  |  Total: ${total:.4f}")
    print()


def build_markdown(transcript: str, results: list[dict], *, pipeline_preamble: str = "") -> str:
    lines = ["# Model Comparison\n"]
    if pipeline_preamble:
        lines.append(pipeline_preamble)
    lines.append("## Summary\n")
    lines.append("| Model | Tokens (in+out) | Time | Cost |")
    lines.append("|-------|----------------|------|------|")
    total_cost = 0.0
    for r in results:
        tokens = f"{r['in_tokens']:,}+{r['out_tokens']:,}"
        time_s = f"{r['elapsed_s']:.1f}s"
        cost = r.get("cost_usd", 0.0) or 0.0
        cost_s = f"${cost:.4f}" if cost else "n/a"
        total_cost += cost
        lines.append(f"| {r['model']} | {tokens} | {time_s} | {cost_s} |")
    lines.append(f"\n*Total LLM cost: ${total_cost:.4f}*\n")
    lines.append("## Transcript\n")
    lines.append(f"```\n{transcript}\n```\n")
    lines.append("## Results by Model\n")
    for r in results:
        cost = r.get("cost_usd", 0.0) or 0.0
        cost_str = f" — ${cost:.4f}" if cost else ""
        lines.append(f"### {r['model']}\n")
        lines.append(r["text"])
        lines.append(
            f"\n*{r['in_tokens']:,} in + {r['out_tokens']:,} out tokens — "
            f"{r['elapsed_s']:.1f}s{cost_str}*\n"
        )
        lines.append("\n---\n")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare LLM models on meeting transcript")
    parser.add_argument("audio_file", help="Path to .m4a audio file")
    parser.add_argument("--models", help="Comma-separated model list (overrides defaults)")
    parser.add_argument("--output", "-o", help="Exact output path (conflicts with --run / --output-dir)")
    parser.add_argument("--run", action="store_true", help="Write to <case>/runs/<timestamp>/compare.md")
    parser.add_argument("--output-dir", dest="output_dir", metavar="DIR", help="Write compare.md inside DIR")
    parser.add_argument("--correct", action="store_true", help="Run holistic correction pass on raw transcript before structuring")
    parser.add_argument("--correct-model", default=DEFAULT_CORRECTION_MODEL, help=f"Model for correction pass (default: {DEFAULT_CORRECTION_MODEL})")
    parser.add_argument(
        "--context",
        default="",
        help="THIS call: who, why, org — name participants when relevant; required with --ref-doc for orchestrator",
    )
    parser.add_argument(
        "--ref-doc",
        action="append",
        default=[],
        metavar="PATH",
        help="Prior material for jargon/background only — not this meeting's transcript; requires non-empty --context",
    )
    parser.add_argument("--orchestrator-model", default=None, help="OpenRouter model for context orchestrator (default from config)")
    parser.add_argument("--keyterms", default="", help="Extra comma-separated terms for STT + correction (merged with orchestrator when --context is set)")
    parser.add_argument("--num-speakers", type=int, default=2, help="Expected number of speakers for ElevenLabs diarization (default: 2)")
    parser.add_argument("--stt-model", default=None, help="STT override (default: elevenlabs/scribe-v2)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audio_path = args.audio_file
    if not os.path.exists(audio_path):
        print(f"ERROR: File not found: {audio_path}", file=sys.stderr)
        sys.exit(1)
    if sum(bool(x) for x in (args.output, args.run, args.output_dir)) > 1:
        print("ERROR: use only one of -o, --run, or --output-dir", file=sys.stderr)
        sys.exit(1)
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY not set", file=sys.stderr)
        sys.exit(1)
    ref_paths: list[str] = list(args.ref_doc or [])
    if ref_paths and not args.context.strip():
        print("ERROR: --ref-doc requires non-empty --context", file=sys.stderr)
        sys.exit(1)

    models = args.models.split(",") if args.models else DEFAULT_MODELS
    models = [m.strip() for m in models]

    if args.output:
        output_path = str(Path(args.output).expanduser().resolve())
    elif args.output_dir:
        output_path = str(ensure_output_dir(args.output_dir) / "compare.md")
    elif args.run:
        output_path = str(new_run_dir(audio_path) / "compare.md")
    else:
        output_path = str(Path(audio_path).resolve().parent / "compare.md")

    print("=== Model Comparison ===")
    print(f"Audio:   {audio_path}")
    print(f"Output:  {output_path}")
    if args.run or args.output_dir:
        print(f"Run dir: {Path(output_path).parent}")
    print(f"Models:  {len(models)}")
    if args.correct:
        print(f"Correct: {args.correct_model}")
    print()

    orch = None
    meeting_context = ""
    ref_doc_count = 0
    if args.context.strip():
        print("── Stage 1: Context orchestrator ──")
        docs = load_ref_corpus(ref_paths)
        ref_doc_count = len(docs)
        orch = run_context_orchestrator(
            meeting_situation=args.context.strip(),
            ref_docs=docs,
            num_speakers=args.num_speakers,
            model=args.orchestrator_model,
        )
        meeting_context = format_downstream_llm_context(args.context.strip(), orch)
        print(f"  Model: {args.orchestrator_model or DEFAULT_ORCHESTRATOR_MODEL}")
        print(f"  Reference docs: {ref_doc_count} | keyterms: {len(orch.keyterms)} | {orch.elapsed_s:.1f}s")
        print(
            "  Participants: "
            + ", ".join(f"{n} ({r})" for n, r in zip(orch.speaker_names, orch.speaker_roles, strict=True))
        )
        print()
    keyterms_list = list(dict.fromkeys((orch.keyterms if orch else []) + [k.strip() for k in args.keyterms.split(",") if k.strip()]))

    stt_model = args.stt_model or DEFAULT_STT_MODEL
    stg = 2 if orch else 1
    print(f"── Stage {stg}: Speech-to-text ──")
    print(f"  Model: {stt_model}")
    print("  One transcript is reused for every structuring model below.")
    transcript, transcription_cost, _stt_meta = transcribe(
        audio_path,
        stt_model=args.stt_model,
        keyterms=keyterms_list,
        num_speakers=args.num_speakers,
    )
    audio_sec = audio_duration_seconds(audio_path)
    print(f"  Duration: {audio_sec:.1f}s | {len(transcript):,} chars | est. ${transcription_cost:.4f}")
    print()

    correction_cost = 0.0
    stg += 1
    if args.correct:
        print(f"── Stage {stg}: Holistic correction (LLM) ──")
        print(f"  Model: {args.correct_model}")
        print("  Full-transcript pass to fix systematic STT errors before structuring.")
        kt = ", ".join(keyterms_list) if keyterms_list else ""
        transcript, correction_cost, corr_elapsed = correct_transcript(
            transcript=transcript,
            context=meeting_context or args.context,
            keyterms=kt,
            model=args.correct_model,
        )
        print(f"  {len(transcript):,} chars | est. ${correction_cost:.4f} | {corr_elapsed:.1f}s")
        print()
        stg += 1

    print(f"── Stage {stg}: Structure notes (one LLM run per model) ──")
    print(f"  Models: {', '.join(models)}")
    results = [run_model(transcript, m, meeting_context) for m in models]

    print_table(results, transcription_cost, correction_cost)

    pre = multi_model_notes_preamble(
        audio_path=str(Path(audio_path).resolve()),
        stt_model=stt_model,
        transcription_cost=transcription_cost,
        transcript_chars=len(transcript),
        audio_duration_s=audio_sec,
        correction_ran=bool(args.correct),
        correction_model=args.correct_model if args.correct else None,
        correction_cost=correction_cost,
        models_ran=models,
        orchestrator_ran=orch is not None,
        orchestrator_model=(args.orchestrator_model or DEFAULT_ORCHESTRATOR_MODEL) if orch else None,
        ref_doc_count=ref_doc_count,
    )
    md = build_markdown(transcript, results, pipeline_preamble=pre)
    Path(output_path).write_text(md, encoding="utf-8")
    print(f"Full output saved to: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
