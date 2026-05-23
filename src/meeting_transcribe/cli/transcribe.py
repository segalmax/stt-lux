#!/usr/bin/env python3
"""
Meeting transcription tool for Hebrew/English code-switched audio.

Does not load ground truth or anything under test_references/ — that is only for evaluate-models / compare-stt.

Usage:
    transcribe cases/img_6423/tail.m4a
    python -m meeting_transcribe.cli.transcribe cases/img_6423/tail.m4a

Model routing (defaults — no fallbacks; missing API keys exit with error):
    - STT: ElevenLabs Scribe v2 (ELEVENLABS_API_KEY)
    - Speaker labels + reconcile + structuring: OpenRouter (OPENROUTER_API_KEY), default anthropic/claude-opus-4-5
    - `notes.md` includes Raw (STT) and Reconciled transcript sections; use `--no-reconcile` to skip the reconcile pass

Use --run to write notes.md under <case>/runs/<timestamp>/. Use --output-dir DIR for a chosen folder.

Document context: pass --ref-doc FILE (repeatable) or --ref-dir DIR to give the orchestrator reference
documents (prior notes, agenda, glossary). Combined with --context (meeting situation string).
  --ref-dir alone: orchestrator infers participants and keyterms from docs (best-effort)
  --context alone: orchestrator uses meeting situation only, no doc background
  both:            richest result — situation identifies participants, docs provide terminology

Requirements: pip install -e .  and  brew install ffmpeg
"""

import argparse
import os
import sys
from pathlib import Path

from meeting_transcribe.core.audio import audio_duration_seconds
from meeting_transcribe.core.config import (
    DEFAULT_OPENROUTER_MODEL,
    DEFAULT_ORCHESTRATOR_MODEL,
    DEFAULT_RECONCILE_MODEL,
    DEFAULT_STT_MODEL,
)
from meeting_transcribe.core.orchestrate_context import (
    format_downstream_llm_context,
    load_ref_corpus,
    load_ref_dir,
    run_context_orchestrator,
)
from meeting_transcribe.core.pipeline_report import transcribe_notes_preamble
from meeting_transcribe.core.run_paths import ensure_output_dir, new_run_dir
from meeting_transcribe.core.stt import transcribe
from meeting_transcribe.core.structure import organize_speakers_openrouter, reconcile_transcript_openrouter, structure_notes
from meeting_transcribe.core.vocab import extract_vocab_from_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe meeting audio and generate structured notes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Models (via OpenRouter, set OPENROUTER_API_KEY):
  anthropic/claude-opus-4-5       default — best for Hebrew/English
  anthropic/claude-sonnet-4-5     faster, cheaper
  google/gemini-2.5-pro           Google alternative
  openai/gpt-4o                   OpenAI alternative
  mistralai/mistral-large          Mistral

Full list: https://openrouter.ai/models
        """,
    )
    parser.add_argument("audio_file", help="Path to .m4a audio file")
    parser.add_argument("--output", "-o", help="Exact output markdown path (conflicts with --run / --output-dir)")
    parser.add_argument("--run", action="store_true", help="Write to <case>/runs/<timestamp>/notes.md")
    parser.add_argument("--output-dir", dest="output_dir", metavar="DIR", help="Write notes.md inside DIR (mkdir if needed)")
    parser.add_argument("--model", "-m", help=f"OpenRouter model ID (default: {DEFAULT_OPENROUTER_MODEL})", default=None)
    parser.add_argument("--no-structure", action="store_true", help="Output raw transcript only, skip LLM structuring")
    parser.add_argument(
        "--prompt-file",
        help="Context file (agenda, glossary, etc.) — key terms extracted for STT; not a test reference",
    )
    parser.add_argument(
        "--no-speaker-labels",
        action="store_true",
        help="Keep STT [speaker_N] lines in the Transcript section (skip LLM name labeling even when orchestrator ran)",
    )
    parser.add_argument(
        "--no-reconcile",
        action="store_true",
        help="After speaker labels, skip the reconciliation pass (reconciled section = labeled transcript; structured notes use labeled)",
    )
    parser.add_argument(
        "--reconcile-model",
        default=None,
        metavar="MODEL",
        help=f"OpenRouter model for reconciled transcript (default: {DEFAULT_RECONCILE_MODEL})",
    )
    parser.add_argument("--speakers", help='Comma-separated names, or "auto" for orchestrator names; if omitted and --context ran, names are applied by default')
    parser.add_argument(
        "--context",
        help="THIS call: who, why, org — names participants here when notes must identify them; enables orchestrator when non-empty",
        default="",
    )
    parser.add_argument(
        "--ref-doc",
        action="append",
        default=[],
        metavar="PATH",
        help="Reference document (MD/TXT) for terminology/org background — repeatable. Use alone or with --context.",
    )
    parser.add_argument(
        "--ref-dir",
        metavar="DIR",
        default=None,
        help="Directory of reference documents (MD/TXT) — all supported files loaded automatically. Combined with any --ref-doc files.",
    )
    parser.add_argument("--orchestrator-model", default=None, help="OpenRouter model for context orchestrator")
    parser.add_argument("--keyterms", help="Comma-separated terms for STT (merged with orchestrator when --context is set)", default="")
    parser.add_argument("--num-speakers", type=int, default=2, help="Expected number of speakers for diarization (default: 2)")
    parser.add_argument(
        "--stt-model",
        default=None,
        help="STT override (default: elevenlabs/scribe-v2; requires ELEVENLABS_API_KEY)",
    )
    parser.add_argument(
        "--language",
        required=True,
        metavar="CODE",
        help=(
            "BCP-47 language code for ElevenLabs STT (e.g. 'he', 'en'). "
            "Pass 'auto' to omit language and let ElevenLabs auto-detect — "
            "recommended for Hebrew/English code-switched audio. Required."
        ),
    )
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
    ref_paths: list[str] = list(args.ref_doc or [])
    if getattr(args, "ref_dir", None):
        try:
            ref_paths += load_ref_dir(args.ref_dir)
            print(f"  Loaded {len(ref_paths)} reference files from: {args.ref_dir}")
        except (FileNotFoundError, ValueError) as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)

    ap = Path(audio_path).resolve()
    if args.output:
        output_path = str(Path(args.output).expanduser().resolve())
    elif args.output_dir:
        output_path = str(ensure_output_dir(args.output_dir) / "notes.md")
    elif args.run:
        output_path = str(new_run_dir(audio_path) / "notes.md")
    else:
        output_path = str(ap.parent / f"{ap.stem}_notes.md")

    print("=== Meeting Transcription Tool ===")
    print(f"Input:    {audio_path}")
    print(f"Output:   {output_path}")
    if args.run or args.output_dir:
        print(f"Run dir:  {Path(output_path).parent}")
    print()

    vocab_prompt = None
    if args.prompt_file:
        if not os.path.exists(args.prompt_file):
            print(f"ERROR: prompt-file not found: {args.prompt_file}", file=sys.stderr)
            sys.exit(1)
        vocab_prompt = extract_vocab_from_file(args.prompt_file)
        print(f"Vocab hint: {vocab_prompt[:120]}...")
        print()

    manual_kt = [k.strip() for k in args.keyterms.split(",") if k.strip()] if args.keyterms else []
    orch = None
    ref_doc_count = 0
    meeting_llm_context = ""
    orchestrator_in = orchestrator_out = 0
    orchestrator_elapsed = 0.0
    orchestrator_cost = 0.0
    om = args.orchestrator_model or DEFAULT_ORCHESTRATOR_MODEL
    has_context = bool(args.context.strip())
    has_ref_docs = bool(ref_paths)
    if has_context or has_ref_docs:
        if not os.environ.get("OPENROUTER_API_KEY"):
            print("ERROR: OPENROUTER_API_KEY required for --context orchestrator", file=sys.stderr)
            sys.exit(1)
        print("── Stage 1: Context orchestrator ──")
        print(f"  Model: {om}")
        docs = load_ref_corpus(ref_paths)
        ref_doc_count = len(docs)
        orch = run_context_orchestrator(
            meeting_situation=args.context.strip(),
            ref_docs=docs,
            num_speakers=args.num_speakers,
            model=args.orchestrator_model,
        )
        orchestrator_in, orchestrator_out = orch.in_tokens, orch.out_tokens
        orchestrator_elapsed, orchestrator_cost = orch.elapsed_s, orch.cost_usd
        meeting_llm_context = orch.llm_context
        print(f"  Reference docs: {ref_doc_count} | keyterms: {len(orch.keyterms)} | {orch.elapsed_s:.1f}s")
        print(
            "  Participants: "
            + ", ".join(f"{n} ({r})" for n, r in zip(orch.speaker_names, orch.speaker_roles, strict=True))
        )
        print()

    keyterms_list = list(dict.fromkeys((orch.keyterms if orch else []) + manual_kt))
    if not keyterms_list and vocab_prompt:
        keyterms_list = [k.strip() for k in vocab_prompt.split(",") if k.strip()]

    speakers: list[str] = []
    if args.speakers and args.speakers.strip().lower() == "auto":
        if not orch:
            print("ERROR: --speakers auto requires non-empty --context (orchestrator)", file=sys.stderr)
            sys.exit(1)
        speakers = list(orch.speaker_names)
    elif args.speakers:
        speakers = [s.strip() for s in args.speakers.split(",") if s.strip()]
    elif orch and not args.no_speaker_labels:
        speakers = list(orch.speaker_names)

    if speakers:
        print(f"Speakers (for labeled transcript): {', '.join(speakers)}")

    stg = 2 if orch else 1
    stt_model = args.stt_model or DEFAULT_STT_MODEL
    print(f"── Stage {stg}: Speech-to-text ──")
    print(f"  Model: {stt_model}")
    print("  Produces diarized text; keyterms and speaker count apply to ElevenLabs Scribe.")
    language_code = None if args.language.strip().lower() == "auto" else args.language.strip()
    if language_code:
        print(f"  Language: {language_code}")
    else:
        print("  Language: auto-detect (code-switched)")
    transcript, transcription_cost, stt_meta = transcribe(
        audio_path,
        vocab_prompt,
        stt_model=args.stt_model,
        keyterms=keyterms_list,
        num_speakers=args.num_speakers,
        language_code=language_code,
    )
    audio_sec = audio_duration_seconds(audio_path)
    print(f"  Audio duration: {audio_sec:.1f}s")
    print(f"  Transcript: {len(transcript):,} characters | est. ${transcription_cost:.4f}")
    print()

    stg += 1
    transcript_raw = transcript
    labeled = transcript_raw
    situation = args.context.strip()
    if orch and situation:
        ctx_for_structure = format_downstream_llm_context(situation, orch)
    else:
        ctx_for_structure = meeting_llm_context or situation or ""

    org_in = org_out = 0
    org_elapsed = 0.0
    org_cost = 0.0
    org_finish = ""
    org_model: str | None = None
    rec_in = rec_out = 0
    rec_elapsed = 0.0
    rec_cost = 0.0
    rec_finish = ""
    reconcile_model: str | None = None
    reconciled: str = ""
    struct_in = struct_out = 0
    struct_elapsed = 0.0
    struct_cost = 0.0
    struct_model: str | None = None
    structured_notes = ""

    if speakers:
        org_model = args.model or DEFAULT_OPENROUTER_MODEL
        print(f"── Stage {stg}: Speaker labels (LLM) ──")
        print(f"  Model: {org_model} (OpenRouter)")
        print("  Map STT speaker IDs to names; wording unchanged.")
        spk_ctx = format_downstream_llm_context(situation, orch) if orch and situation else (meeting_llm_context or situation)
        labeled, org_in, org_out, org_elapsed, org_cost, org_finish = organize_speakers_openrouter(
            transcript_raw, org_model, speakers, spk_ctx
        )
        print(f"  Tokens: {org_in:,} in + {org_out:,} out | {org_elapsed:.1f}s")
        print()
        stg += 1

    transcript_for_structure = labeled
    if speakers and not args.no_reconcile:
        reconcile_model = args.reconcile_model or DEFAULT_RECONCILE_MODEL
        print(f"── Stage {stg}: Reconciled transcript (LLM) ──")
        print(f"  Model: {reconcile_model} (OpenRouter)")
        print("  Split mis-merged speaker lines; wording unchanged.")
        reconciled, rec_in, rec_out, rec_elapsed, rec_cost, rec_finish = reconcile_transcript_openrouter(
            labeled, reconcile_model, speakers, spk_ctx
        )
        transcript_for_structure = reconciled
        print(f"  Tokens: {rec_in:,} in + {rec_out:,} out | {rec_elapsed:.1f}s")
        print()
        stg += 1
    elif speakers and args.no_reconcile:
        reconciled = labeled

    if not args.no_structure:
        print(f"── Stage {stg}: Structured meeting notes (LLM) ──")
        print(f"  Model: {args.model or DEFAULT_OPENROUTER_MODEL} (OpenRouter)")
        print("  Summary, action items, and decisions in markdown.")
        structured_notes, struct_in, struct_out, struct_elapsed, struct_model, struct_cost = structure_notes(
            transcript_for_structure, args.model, ctx_for_structure
        )
        print(f"  Tokens: {struct_in:,} in + {struct_out:,} out | {struct_elapsed:.1f}s")
        print()

    raw_stt_chars = len(transcript_raw)
    labeled_chars = len(labeled) if speakers else raw_stt_chars
    reconciled_chars = len(reconciled) if speakers else 0
    reconcile_ran = bool(speakers) and not args.no_reconcile
    preamble = transcribe_notes_preamble(
        audio_path=str(ap),
        stt_model=stt_model,
        stt_cost=transcription_cost,
        raw_stt_chars=raw_stt_chars,
        labeled_chars=labeled_chars,
        audio_duration_s=audio_sec,
        stt_last_word_end_s=stt_meta.get("last_word_end_s") if stt_meta else None,
        stt_word_count=int(stt_meta["word_count"]) if stt_meta and stt_meta.get("word_count") is not None else None,
        organize_finish_reason=org_finish if speakers else None,
        keyterms_count=len(keyterms_list),
        num_speakers=args.num_speakers,
        no_structure=args.no_structure,
        speaker_organize_ran=bool(speakers),
        organize_model=org_model,
        organize_in_tokens=org_in,
        organize_out_tokens=org_out,
        organize_elapsed_s=org_elapsed,
        organize_cost=org_cost,
        reconcile_ran=reconcile_ran,
        reconcile_model=reconcile_model if reconcile_ran else None,
        reconciled_chars=reconciled_chars if reconcile_ran else 0,
        reconcile_finish_reason=rec_finish if reconcile_ran else None,
        reconcile_in_tokens=rec_in if reconcile_ran else 0,
        reconcile_out_tokens=rec_out if reconcile_ran else 0,
        reconcile_elapsed_s=rec_elapsed if reconcile_ran else 0.0,
        reconcile_cost=rec_cost if reconcile_ran else 0.0,
        structure_notes_ran=not args.no_structure,
        structure_model=struct_model if not args.no_structure else None,
        structure_in_tokens=struct_in,
        structure_out_tokens=struct_out,
        structure_elapsed_s=struct_elapsed,
        structure_cost=struct_cost,
        orchestrator_ran=orch is not None,
        orchestrator_model=om if orch else None,
        ref_doc_count=ref_doc_count,
        orchestrator_in_tokens=orchestrator_in,
        orchestrator_out_tokens=orchestrator_out,
        orchestrator_elapsed_s=orchestrator_elapsed,
        orchestrator_cost=orchestrator_cost,
    )
    if not speakers:
        reconciled_note = (
            "_Named-speaker labeling was not run — use `--context` or `--speakers` to enable labeled and reconciled transcripts._\n\n"
        )
        reconciled_body = ""
    elif args.no_reconcile:
        reconciled_note = "_Reconciliation skipped (`--no-reconcile`); same as labeled transcript._\n\n"
        reconciled_body = reconciled
    else:
        reconciled_note = ""
        reconciled_body = reconciled
    sections = [
        "# Transcript\n\n## Raw (speech-to-text)\n\n"
        f"{transcript_raw}\n\n## Reconciled transcript\n\n{reconciled_note}{reconciled_body}\n"
    ]
    if structured_notes:
        sections.append(f"# Structured Notes\n\n{structured_notes}\n")
    final_output = preamble + "\n---\n\n".join(sections)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_output)

    print(f"Notes saved to: {output_path}")
    print()
    print("=== Cost Summary ===")
    if orch:
        oco = f"  ${orchestrator_cost:.4f}" if orchestrator_cost else "  (see openrouter.ai/activity)"
        print(f"  Orchestrator     {om}")
        print(f"                  {orchestrator_in:,} in + {orchestrator_out:,} out tokens  {orchestrator_elapsed:.1f}s{oco}")
    print(f"  Transcription   {stt_model}  ${transcription_cost:.4f}")
    if speakers:
        oc = f"  ${org_cost:.4f}" if org_cost else "  (see openrouter.ai/activity)"
        print(f"  Speaker labels  {org_model}")
        print(f"                  {org_in:,} in + {org_out:,} out tokens  {org_elapsed:.1f}s{oc}")
    if reconcile_ran and reconcile_model:
        rc = f"  ${rec_cost:.4f}" if rec_cost else "  (see openrouter.ai/activity)"
        print(f"  Reconcile        {reconcile_model}")
        print(f"                  {rec_in:,} in + {rec_out:,} out tokens  {rec_elapsed:.1f}s{rc}")
    if not args.no_structure and struct_model:
        sc = f"  ${struct_cost:.4f}" if struct_cost else "  (see openrouter.ai/activity)"
        print(f"  Structured notes {struct_model}")
        print(f"                  {struct_in:,} in + {struct_out:,} out tokens  {struct_elapsed:.1f}s{sc}")
    total_est = transcription_cost
    if orch:
        total_est += orchestrator_cost
    if speakers:
        total_est += org_cost
    if reconcile_ran:
        total_est += rec_cost
    if not args.no_structure and struct_model:
        total_est += struct_cost
    print(f"  {'─'*45}")
    print(f"  Pipeline total (est.): ${total_est:.4f}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
