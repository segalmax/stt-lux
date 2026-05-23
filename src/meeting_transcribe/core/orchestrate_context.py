"""LLM step: meeting situation + optional reference files → keyterms, speakers, shared LLM context."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

from meeting_transcribe.core.config import (
    DEFAULT_ORCHESTRATOR_MODEL,
    OPENROUTER_BASE_URL,
    REF_DOC_MAX_BYTES,
    REF_DOC_SUFFIXES,
    REF_DOCS_MAX_TOTAL_BYTES,
)


PIPELINE_STAGES_FOR_ORCHESTRATOR = """
How your JSON is consumed downstream (optimize each field for this):
1) keyterms → speech-to-text (ElevenLabs): hint list for spelling/mishearing only; STT does not know identities or roles.
2) STT returns word-level speaker_id clusters as lines like [speaker_0], [speaker_1], … — those are acoustic diarization hints only; they may not match real people or clean turn boundaries.
3) A later speaker-label LLM maps each [speaker_N] block to real names using speaker_names, speaker_roles, and llm_context — keep those three mutually consistent and role-explicit.
4) Structured meeting notes use the labeled transcript plus the same context bundle — front-load who-is-who and roles in llm_context.
"""


@dataclass(frozen=True)
class OrchestratorResult:
    keyterms: list[str]
    speaker_names: list[str]
    speaker_roles: list[str]
    llm_context: str
    in_tokens: int
    out_tokens: int
    elapsed_s: float
    cost_usd: float


def format_downstream_llm_context(meeting_situation: str, orch: OrchestratorResult) -> str:
    """Bundle meeting situation, per-speaker roles, and jargon context for speaker-label, reconcile, and structured-notes LLMs."""
    sit = meeting_situation.strip()
    lines = [
        "### Meeting situation (authoritative for who is who and roles)",
        sit,
        "",
        "### Participants (orchestrator — name and role per speaker)",
    ]
    for name, role in zip(orch.speaker_names, orch.speaker_roles, strict=True):
        lines.append(f"- {name} — {role}")
    lines.extend(["", "### Extra context (jargon / org)", orch.llm_context])
    return "\n".join(lines)


ORCHESTRATOR_SYSTEM = f"""You are a preprocessing assistant for Hebrew/English meeting transcription pipelines.
The meeting situation text is always authoritative for WHO is in THIS recording. Reference files are long and may describe other sessions — prioritize the meeting situation over ref-doc volume when they conflict.
{PIPELINE_STAGES_FOR_ORCHESTRATOR.strip()}
You output ONLY valid JSON matching the schema given in the user message. No markdown fences, no commentary."""


def _load_ref_doc(path: str) -> tuple[str, str]:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Reference document not found: {p}")
    if p.suffix.lower() not in REF_DOC_SUFFIXES:
        raise ValueError(f"Unsupported reference document type {p.suffix!r}; allowed: {REF_DOC_SUFFIXES}")
    raw = p.read_bytes()
    if len(raw) > REF_DOC_MAX_BYTES:
        raise ValueError(f"Reference document too large: {p} ({len(raw)} bytes; max {REF_DOC_MAX_BYTES})")
    return p.name, raw.decode("utf-8")


def load_ref_corpus(paths: list[str]) -> list[tuple[str, str]]:
    if not paths:
        return []
    total = 0
    out: list[tuple[str, str]] = []
    for path in paths:
        name, text = _load_ref_doc(path)
        total += len(text.encode("utf-8"))
        if total > REF_DOCS_MAX_TOTAL_BYTES:
            raise ValueError(f"Total reference corpus exceeds {REF_DOCS_MAX_TOTAL_BYTES} bytes")
        out.append((name, text))
    return out


def load_ref_dir(dir_path: str) -> list[str]:
    """Return sorted list of supported file paths from a directory (non-recursive)."""
    p = Path(dir_path).expanduser().resolve()
    if not p.is_dir():
        raise FileNotFoundError(f"Reference directory not found: {p}")
    files = sorted(
        str(f) for f in p.iterdir()
        if f.is_file() and f.suffix.lower() in REF_DOC_SUFFIXES
    )
    if not files:
        raise ValueError(f"No supported files ({REF_DOC_SUFFIXES}) found in: {p}")
    return files


def _parse_json_object(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}") + 1
    if start < 0 or end <= start:
        raise ValueError("Orchestrator did not return JSON object")
    return json.loads(text[start:end])


def run_context_orchestrator(
    *,
    meeting_situation: str,
    ref_docs: list[tuple[str, str]],
    num_speakers: int,
    model: str | None = None,
) -> OrchestratorResult:
    """Single OpenRouter call → keyterms, speaker_names, speaker_roles, llm_context.

    meeting_situation may be empty when ref_docs are provided — the orchestrator will
    infer participants and context from the documents.  At least one of meeting_situation
    or ref_docs must be non-empty.
    """
    if not meeting_situation.strip() and not ref_docs:
        raise ValueError(
            "Provide --context (meeting situation) and/or --ref-doc / --ref-dir (reference documents). "
            "At least one is required for the orchestrator."
        )
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is required for the context orchestrator.")
    effective = model or DEFAULT_ORCHESTRATOR_MODEL
    doc_blocks = ""
    for name, body in ref_docs:
        doc_blocks += f"\n\n### File: {name}\n\n{body}"

    has_situation = bool(meeting_situation.strip())
    if has_situation:
        rule1 = "1) THIS recording — who is speaking now — comes ONLY from the meeting situation text above. Reference files may be transcripts or notes from PAST or OTHER meetings; do NOT treat their main speakers as participants in THIS audio."
        rule2 = "2) speaker_names must list exactly the people in THIS recording, in a sensible order (e.g. guest first then host). Derive names from the meeting situation. If the situation names people (e.g. coach and employee), those are the speakers — not whoever dominates a reference file."
    else:
        rule1 = "1) No explicit meeting situation was provided. Infer who is in THIS recording from recurring named participants, titles, and described relationships in the reference documents. Use your best judgment."
        rule2 = "2) speaker_names must list exactly the people in THIS recording — derive from document evidence. Use generic names (Participant 1, Participant 2) only if documents give no names."

    rules = f"""
CRITICAL RULES (read before producing JSON):
{rule1}
{rule2}
3) speaker_roles must be parallel to speaker_names (same order, same length): one short role string per person (e.g. coach, new employee). Use "participant" if roles are unclear.
4) keyterms may include names and terms from reference files for STT accuracy. Including a name in keyterms does NOT mean that person is speaking in this recording.
5) llm_context must: (a) open with who is in THIS meeting and their roles; (b) add only entity/org/product/jargon background distilled from reference files for interpreting the transcript.

"""

    if has_situation:
        situation_block = f"Meeting situation (THIS conversation — authoritative for who is on this recording, why, org):\n{meeting_situation.strip()}"
    else:
        situation_block = (
            "Meeting situation: NOT PROVIDED.\n"
            "Infer who is in THIS recording and their roles exclusively from the reference documents below.\n"
            "Treat document titles, recurring speaker names, and described relationships as your best evidence for speaker_names and speaker_roles.\n"
            "If the documents do not clearly identify participants, use generic names like 'Participant 1', 'Participant 2'."
        )

    user_prompt = f"""{situation_block}
{doc_blocks}

Expected number of speakers in THIS audio: {num_speakers}
{rules}
Return a JSON object with exactly these keys:
- "keyterms": array of short strings (names, products, Hebrew/English jargon) for speech-to-text hints; max 80 items; each string max 50 characters; deduplicated.
- "speaker_names": array of exactly {num_speakers} strings — participants in THIS recording only, taken from the meeting situation. Same length as num_speakers.
- "speaker_roles": array of exactly {num_speakers} short strings — one role per speaker_names entry, same order (from the meeting situation; use "participant" if roles are not distinct).
- "llm_context": one markdown-safe string: first restate who is in THIS meeting (from situation), then add jargon/entity background from reference files only as context for terms — not as a second transcript of an old meeting.

JSON only."""

    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=os.environ["OPENROUTER_API_KEY"])
    t0 = time.time()
    response = client.chat.completions.create(
        model=effective,
        max_tokens=4096,
        messages=[
            {"role": "system", "content": ORCHESTRATOR_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
    )
    elapsed = time.time() - t0
    usage = response.usage
    cost_usd = (usage.model_extra or {}).get("cost", 0.0) if usage else 0.0
    raw = response.choices[0].message.content or ""
    data = _parse_json_object(raw)
    keyterms = [str(x).strip()[:50] for x in data.get("keyterms", []) if str(x).strip()]
    keyterms = list(dict.fromkeys(keyterms))[:80]
    names = [str(x).strip() for x in data.get("speaker_names", []) if str(x).strip()]
    if len(names) != num_speakers:
        raise RuntimeError(
            f"Orchestrator returned {len(names)} speaker name(s); expected exactly {num_speakers}: {names!r}"
        )
    roles = [str(x).strip() for x in data.get("speaker_roles", [])]
    if len(roles) != num_speakers:
        raise RuntimeError(
            f"Orchestrator returned {len(roles)} speaker role(s); expected exactly {num_speakers}: {roles!r}"
        )
    if any(not r for r in roles):
        raise RuntimeError(f"Orchestrator returned empty speaker role string in {roles!r}")
    llm_context = str(data.get("llm_context", "")).strip()
    if not llm_context:
        raise RuntimeError("Orchestrator returned empty llm_context")
    return OrchestratorResult(
        keyterms=keyterms,
        speaker_names=names,
        speaker_roles=roles,
        llm_context=llm_context,
        in_tokens=usage.prompt_tokens if usage else 0,
        out_tokens=usage.completion_tokens if usage else 0,
        elapsed_s=elapsed,
        cost_usd=float(cost_usd or 0.0),
    )
