import os
import time

from openai import OpenAI

from meeting_transcribe.core.config import (
    DEFAULT_OPENROUTER_MODEL,
    OPENROUTER_BASE_URL,
    RECONCILE_MAX_TOKENS,
    SPEAKER_LABEL_MAX_TOKENS,
    STRUCTURE_NOTES_MAX_TOKENS,
)
from meeting_transcribe.core.prompts import (
    CORRECTION_SYSTEM_PROMPT,
    CORRECTION_USER_PROMPT,
    RECONCILE_SYSTEM_PROMPT,
    RECONCILE_USER_PROMPT,
    SPEAKERS_PROMPT_TEMPLATE,
    SPEAKERS_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
)


def organize_speakers_openrouter(
    transcript: str, model: str, speakers: list[str], context: str
) -> tuple[str, int, int, float, float, str]:
    """Organize transcript by speaker via OpenRouter. Returns (text, in_tok, out_tok, elapsed_s, cost_usd, finish_reason)."""
    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=os.environ["OPENROUTER_API_KEY"])
    speakers_str = " and ".join(speakers)
    example = speakers[0] if speakers else "Speaker 1"
    other = speakers[1] if len(speakers) > 1 else "Speaker 2"
    t0 = time.time()
    response = client.chat.completions.create(
        model=model,
        max_tokens=SPEAKER_LABEL_MAX_TOKENS,
        messages=[
            {"role": "system", "content": SPEAKERS_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": SPEAKERS_PROMPT_TEMPLATE.format(
                    speakers=speakers_str,
                    context=context,
                    transcript=transcript,
                    speaker_example=example,
                    other_speaker_example=other,
                ),
            },
        ],
    )
    elapsed = time.time() - t0
    usage = response.usage
    cost_usd = (usage.model_extra or {}).get("cost", 0.0) if usage else 0.0
    choice = response.choices[0]
    text = choice.message.content or ""
    finish = getattr(choice, "finish_reason", None) or ""
    if finish == "length":
        raise RuntimeError(
            "Speaker labels output hit max_tokens and was truncated. "
            f"Raise SPEAKER_LABEL_MAX_TOKENS in config (currently {SPEAKER_LABEL_MAX_TOKENS}) or split the pipeline."
        )
    return text, usage.prompt_tokens, usage.completion_tokens, elapsed, cost_usd, finish


def reconcile_transcript_openrouter(
    labeled_transcript: str, model: str, speakers: list[str], context: str
) -> tuple[str, int, int, float, float, str]:
    """Split mis-merged speaker lines; wording unchanged. Returns (text, in_tok, out_tok, elapsed_s, cost_usd, finish_reason)."""
    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=os.environ["OPENROUTER_API_KEY"])
    speakers_str = ", ".join(speakers)
    t0 = time.time()
    response = client.chat.completions.create(
        model=model,
        max_tokens=RECONCILE_MAX_TOKENS,
        messages=[
            {"role": "system", "content": RECONCILE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": RECONCILE_USER_PROMPT.format(
                    speakers=speakers_str,
                    context=context.strip() if context else "(none)",
                    transcript=labeled_transcript,
                ),
            },
        ],
    )
    elapsed = time.time() - t0
    usage = response.usage
    cost_usd = (usage.model_extra or {}).get("cost", 0.0) if usage else 0.0
    choice = response.choices[0]
    text = choice.message.content or ""
    finish = getattr(choice, "finish_reason", None) or ""
    if finish == "length":
        raise RuntimeError(
            "Reconciled transcript hit max_tokens and was truncated. "
            f"Raise RECONCILE_MAX_TOKENS in config (currently {RECONCILE_MAX_TOKENS})."
        )
    return text, usage.prompt_tokens, usage.completion_tokens, elapsed, cost_usd, finish


def correct_transcript(
    transcript: str,
    context: str,
    keyterms: str,
    model: str,
) -> tuple[str, float, float]:
    """Holistic correction pass on raw transcript. Returns (corrected, cost_usd, elapsed_s)."""
    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=os.environ["OPENROUTER_API_KEY"])
    t0 = time.time()
    response = client.chat.completions.create(
        model=model,
        max_tokens=8192,
        messages=[
            {"role": "system", "content": CORRECTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": CORRECTION_USER_PROMPT.format(
                    context=context,
                    keyterms=keyterms or "(none provided)",
                    transcript=transcript,
                ),
            },
        ],
    )
    elapsed = time.time() - t0
    cost_usd = (response.usage.model_extra or {}).get("cost", 0.0) if response.usage else 0.0
    text = response.choices[0].message.content
    return text, cost_usd, elapsed


def structure_notes_openrouter(
    transcript: str, model: str, meeting_context: str = ""
) -> tuple[str, int, int, float, float]:
    """Structure notes via OpenRouter. Returns (text, in_tokens, out_tokens, elapsed_s, cost_usd)."""
    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=os.environ["OPENROUTER_API_KEY"])
    t0 = time.time()
    mc = meeting_context.strip() if meeting_context else ""
    if not mc:
        mc = "(none provided)"
    user_content = USER_PROMPT_TEMPLATE.format(transcript=transcript, meeting_context=mc)
    response = client.chat.completions.create(
        model=model,
        max_tokens=STRUCTURE_NOTES_MAX_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    elapsed = time.time() - t0
    usage = response.usage
    cost_usd = (usage.model_extra or {}).get("cost", 0.0) if usage else 0.0
    choice = response.choices[0]
    text = choice.message.content or ""
    finish = getattr(choice, "finish_reason", None) or ""
    if finish == "length":
        raise RuntimeError(
            "Structured notes hit max_tokens and were truncated. "
            f"Raise STRUCTURE_NOTES_MAX_TOKENS in config (currently {STRUCTURE_NOTES_MAX_TOKENS})."
        )
    return text, usage.prompt_tokens, usage.completion_tokens, elapsed, cost_usd


def structure_notes(
    transcript: str,
    model: str | None,
    meeting_context: str = "",
) -> tuple[str, int, int, float, str, float]:
    """Structure notes via OpenRouter only (no Anthropic fallback)."""
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is required for structuring notes (no fallback).")
    effective_model = model or DEFAULT_OPENROUTER_MODEL
    text, in_tok, out_tok, elapsed, cost_usd = structure_notes_openrouter(transcript, effective_model, meeting_context)
    return text, in_tok, out_tok, elapsed, effective_model, cost_usd
