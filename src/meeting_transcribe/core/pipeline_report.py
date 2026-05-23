"""Human-readable pipeline stage metadata for markdown exports."""

from __future__ import annotations

from datetime import datetime, timezone


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def transcribe_notes_preamble(
    *,
    audio_path: str,
    stt_model: str,
    stt_cost: float,
    raw_stt_chars: int,
    labeled_chars: int,
    audio_duration_s: float,
    keyterms_count: int,
    num_speakers: int,
    no_structure: bool,
    stt_last_word_end_s: float | None = None,
    stt_word_count: int | None = None,
    organize_finish_reason: str | None = None,
    speaker_organize_ran: bool = False,
    organize_model: str | None = None,
    organize_in_tokens: int = 0,
    organize_out_tokens: int = 0,
    organize_elapsed_s: float = 0.0,
    organize_cost: float = 0.0,
    reconcile_ran: bool = False,
    reconcile_model: str | None = None,
    reconciled_chars: int = 0,
    reconcile_finish_reason: str | None = None,
    reconcile_in_tokens: int = 0,
    reconcile_out_tokens: int = 0,
    reconcile_elapsed_s: float = 0.0,
    reconcile_cost: float = 0.0,
    structure_notes_ran: bool = False,
    structure_model: str | None = None,
    structure_in_tokens: int = 0,
    structure_out_tokens: int = 0,
    structure_elapsed_s: float = 0.0,
    structure_cost: float = 0.0,
    orchestrator_ran: bool = False,
    orchestrator_model: str | None = None,
    ref_doc_count: int = 0,
    orchestrator_in_tokens: int = 0,
    orchestrator_out_tokens: int = 0,
    orchestrator_elapsed_s: float = 0.0,
    orchestrator_cost: float = 0.0,
) -> str:
    """Markdown block at the top of notes.md (before # Transcript)."""
    lines = [
        "## Pipeline",
        "",
        f"- **Generated (UTC):** {utc_timestamp()}",
        f"- **Input audio:** `{audio_path}`",
        "",
    ]
    n = 1
    if orchestrator_ran and orchestrator_model:
        lines += [
            f"### Stage {n} — Context orchestrator",
            "- **Role:** Distill meeting situation + reference docs into keyterms, per-speaker roles, and LLM context — feeds STT and later speaker-label / structured-notes steps.",
            f"- **Model:** `{orchestrator_model}`",
            f"- **Reference documents:** {ref_doc_count}",
            f"- **Tokens:** {orchestrator_in_tokens:,} in + {orchestrator_out_tokens:,} out",
            f"- **Time:** {orchestrator_elapsed_s:.1f}s",
            (
                f"- **Cost (est.):** ${orchestrator_cost:.4f}"
                if orchestrator_cost
                else "- **Cost:** (see openrouter.ai/activity)"
            ),
            "",
        ]
        n += 1
    stt_extra = []
    if stt_word_count is not None:
        stt_extra.append(f"- **STT word count:** {stt_word_count:,}")
    if stt_last_word_end_s is not None:
        cov = f"{stt_last_word_end_s:.1f}s vs audio {audio_duration_s:.1f}s"
        stt_extra.append(f"- **Last word end (API):** {cov}")
    lines += [
        f"### Stage {n} — Speech-to-text",
        "- **Role:** Diarized transcript from audio (speaker labels where supported). Lines may include `[H:MM:SS-H:MM:SS]` per segment when the API returns word timings.",
        f"- **Model:** `{stt_model}`",
        f"- **Audio duration:** {audio_duration_s:.1f}s",
        f"- **Raw STT text length:** {raw_stt_chars:,} characters",
        *stt_extra,
        f"- **Keyterms for STT:** {keyterms_count} term(s)",
        f"- **Expected speakers (hint):** {num_speakers}",
        f"- **Cost (est.):** ${stt_cost:.4f}",
        "",
    ]
    n += 1
    if speaker_organize_ran and organize_model:
        fin = (
            f"- **Finish reason:** `{organize_finish_reason}`"
            if organize_finish_reason
            else "- **Finish reason:** (unknown)"
        )
        lines += [
            f"### Stage {n} — Speaker labels (LLM)",
            "- **Role:** Replace STT speaker IDs with names (Max, Gil, Other 1, …); wording unchanged; preserve segment time spans when present.",
            f"- **Model:** `{organize_model}`",
            f"- **Labeled transcript length:** {labeled_chars:,} characters",
            fin,
            f"- **Tokens:** {organize_in_tokens:,} in + {organize_out_tokens:,} out",
            f"- **Time:** {organize_elapsed_s:.1f}s",
            f"- **Cost (est.):** ${organize_cost:.4f}" if organize_cost else "- **Cost:** (see openrouter.ai/activity)",
            "",
        ]
        n += 1
    if reconcile_ran and reconcile_model:
        rfin = (
            f"- **Finish reason:** `{reconcile_finish_reason}`"
            if reconcile_finish_reason
            else "- **Finish reason:** (unknown)"
        )
        lines += [
            f"### Stage {n} — Reconciled transcript (LLM)",
            "- **Role:** Split lines where diarization merged two speakers; wording unchanged; same format as labeled transcript.",
            f"- **Model:** `{reconcile_model}`",
            f"- **Reconciled transcript length:** {reconciled_chars:,} characters",
            rfin,
            f"- **Tokens:** {reconcile_in_tokens:,} in + {reconcile_out_tokens:,} out",
            f"- **Time:** {reconcile_elapsed_s:.1f}s",
            f"- **Cost (est.):** ${reconcile_cost:.4f}" if reconcile_cost else "- **Cost:** (see openrouter.ai/activity)",
            "",
        ]
        n += 1
    if no_structure:
        if not speaker_organize_ran:
            lines += [
                f"### Stage {n} — Structured output",
                "- **Skipped** (`--no-structure`): transcript only below.",
            ]
    elif structure_notes_ran and structure_model:
        lines += [
            f"### Stage {n} — Structured meeting notes",
            "- **Role:** Summary, action items, and decisions (markdown).",
            f"- **Model:** `{structure_model}` (OpenRouter)",
            f"- **Tokens:** {structure_in_tokens:,} in + {structure_out_tokens:,} out",
            f"- **Time:** {structure_elapsed_s:.1f}s",
            f"- **Cost (est.):** ${structure_cost:.4f}" if structure_cost else "- **Cost:** (see openrouter.ai/activity)",
        ]
    return "\n".join(lines) + "\n\n---\n\n"


def multi_model_notes_preamble(
    *,
    audio_path: str,
    stt_model: str,
    transcription_cost: float,
    transcript_chars: int,
    audio_duration_s: float,
    correction_ran: bool,
    correction_model: str | None,
    correction_cost: float,
    models_ran: list[str],
    orchestrator_ran: bool = False,
    orchestrator_model: str | None = None,
    ref_doc_count: int = 0,
) -> str:
    """Markdown block at top of compare.md (multi-model structuring run, not vs ground truth)."""
    lines = [
        "## Pipeline",
        "",
        f"- **Generated (UTC):** {utc_timestamp()}",
        f"- **Input audio:** `{audio_path}`",
        "",
    ]
    n = 1
    if orchestrator_ran and orchestrator_model:
        lines += [
            f"### Stage {n} — Context orchestrator",
            "- **Role:** Keyterms, per-speaker roles, and shared LLM context from meeting situation + reference docs — feeds STT and structuring models.",
            f"- **Model:** `{orchestrator_model}`",
            f"- **Reference documents:** {ref_doc_count}",
            "",
        ]
        n += 1
    lines += [
        f"### Stage {n} — Speech-to-text",
        "- **Role:** Single transcript shared by all downstream models.",
        f"- **Model:** `{stt_model}`",
        f"- **Audio duration:** {audio_duration_s:.1f}s",
        f"- **Transcript length:** {transcript_chars:,} characters",
        f"- **Cost (est.):** ${transcription_cost:.4f}",
        "",
    ]
    n += 1
    if correction_ran:
        lines += [
            f"### Stage {n} — Holistic correction (optional)",
            "- **Role:** One LLM pass to fix systematic STT errors using context + keyterms.",
            f"- **Model:** `{correction_model}`",
            f"- **Cost (est.):** ${correction_cost:.4f}",
            "",
        ]
        n += 1
    model_list = ", ".join(f"`{m}`" for m in models_ran)
    lines += [
        f"### Stage {n} — Structure notes (per model)",
        "- **Role:** Same transcript, independent structured-note runs for each model.",
        f"- **Models:** {model_list}",
        "",
    ]
    return "\n".join(lines) + "\n---\n\n"
