#!/usr/bin/env python3
"""
Test/benchmark: compare STT backends on one file; optional judge vs your reference (not production).

Production transcription is `transcribe` only — it never reads ``test_references/``.

Usage:
    compare-stt cases/img_6423/tail.m4a
    (for judging: default ``<repo>/test_references/<case_id>/ground_truth.rtl.md``, or ``--ground-truth``; ``--skip-judge`` skips the judge)

Supported backends (keys read from env):
    OPENAI_API_KEY    → gpt-4o-transcribe, gpt-4o-mini-transcribe, whisper-1
    GROQ_API_KEY      → groq/whisper-large-v3, groq/whisper-large-v3-turbo
    DEEPGRAM_API_KEY  → deepgram/nova-3 (with Hebrew + Keyterm Prompting)
    GOOGLE_API_KEY    → gemini-2.5-flash (multimodal audio)
"""

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

from meeting_transcribe.core.audio import audio_duration_seconds
from meeting_transcribe.core.paths import project_root, resolve_ground_truth_path
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
JUDGE_MODEL = "anthropic/claude-opus-4-5"

# Cost per minute in USD
COSTS_PER_MIN = {
    "openai/gpt-4o-transcribe":       0.006,
    "openai/gpt-4o-mini-transcribe":  0.003,
    "openai/whisper-1":               0.006,
    "groq/whisper-large-v3":          0.00185,
    "groq/whisper-large-v3-turbo":    0.00067,
    "deepgram/nova-3":                0.00433,
    "google/gemini-2.5-flash":        0.0,
    "elevenlabs/scribe-v2":           0.004,  # ~$0.40/100min
}

KEYTERMS = (
    "Common Agentic Core, CAF, LangGraph, LangFuse, Concon, Converge Consumer, "
    "Arin, Eli, Gil, Idan, Himanshu, Ben Stiller, Tau, swarm, A2A, "
    "guardrail, orchestrator, evaluator, feedback loop, RAG, react agent, data frame, IIH"
)

# ──────────────────────────────────────────────
# STT backends
# ──────────────────────────────────────────────

def transcribe_openai(audio_path: str, model: str, prompt: str | None = None) -> str:
    client = OpenAI()
    with open(audio_path, "rb") as f:
        kwargs = {"model": model, "file": f}
        if prompt:
            kwargs["prompt"] = prompt
        result = client.audio.transcriptions.create(**kwargs)
    return result.text


def transcribe_groq(audio_path: str, model: str) -> str:
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ["GROQ_API_KEY"],
    )
    with open(audio_path, "rb") as f:
        result = client.audio.transcriptions.create(model=model, file=f)
    return result.text


def transcribe_deepgram(audio_path: str, keyterms: list[str]) -> str:
    import urllib.request
    api_key = os.environ["DEEPGRAM_API_KEY"]
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    params = "model=nova-3&language=he&detect_language=true&smart_format=true"
    for term in keyterms:
        params += f"&keyterm={urllib.parse.quote(term)}"

    import urllib.parse
    req = urllib.request.Request(
        f"https://api.deepgram.com/v1/listen?{params}",
        data=audio_bytes,
        headers={
            "Authorization": f"Token {api_key}",
            "Content-Type": "audio/m4a",
        },
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())

    return data["results"]["channels"][0]["alternatives"][0]["transcript"]


def transcribe_elevenlabs(
    audio_path: str,
    keyterms: list[str],
    num_speakers: int,
    language_code: str = "he",
) -> str:
    """Transcribe via ElevenLabs Scribe v2 — #1 on AA-WER leaderboard.

    Args:
        keyterms:     Required. Terms to anchor transcription (names, products, jargon).
                      In future: extract from meeting docs, agenda, participant list.
        num_speakers: Required. Expected speaker count — improves diarization accuracy.
        language_code: ISO-639-1 code. 'he' for Hebrew-dominant code-switching.
    """
    import urllib.request
    import uuid
    api_key = os.environ["ELEVENLABS_API_KEY"]

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    boundary = uuid.uuid4().hex

    def field(name, value):
        return (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
            f"{value}\r\n"
        ).encode()

    body = (
        field("model_id", "scribe_v2")
        + field("diarize", "true")
        + field("language_code", language_code)
        + field("tag_audio_events", "false")
        + field("temperature", "0")
    )

    # Keyterms: one field per term (API expects repeated form fields)
    if keyterms:
        for term in keyterms[:1000]:
            body += field("keyterms", term[:50])

    if num_speakers is not None:
        body += field("num_speakers", str(num_speakers))

    body += (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="audio.m4a"\r\n'
        f"Content-Type: audio/mp4\r\n\r\n"
    ).encode() + audio_bytes + f"\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        "https://api.elevenlabs.io/v1/speech-to-text",
        data=body,
        headers={
            "xi-api-key": api_key,
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode()[:300]}")

    # Reconstruct with speaker labels when diarize=true
    if "words" in data and data["words"]:
        lines = []
        current_speaker = None
        current_words = []
        for w in data["words"]:
            if w.get("type") != "word":
                continue
            spk = w.get("speaker_id", "speaker_0")
            if spk != current_speaker:
                if current_words:
                    lines.append(f"[{current_speaker}] " + " ".join(current_words))
                current_speaker = spk
                current_words = []
            current_words.append(w.get("text", ""))
        if current_words:
            lines.append(f"[{current_speaker}] " + " ".join(current_words))
        return "\n".join(lines)

    return data.get("text", "")


def transcribe_gemini(audio_path: str) -> str:
    """Transcribe via Gemini 2.5 Flash multimodal (inline audio bytes)."""
    import google.generativeai as genai
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content([
        {
            "inline_data": {
                "mime_type": "audio/mp4",
                "data": base64.b64encode(audio_bytes).decode(),
            }
        },
        ("Transcribe this audio exactly as spoken. The speakers mix Hebrew and English freely — "
         "preserve both languages as-is. Do not translate, summarize, or add punctuation that "
         "changes meaning. Output only the transcript text."),
    ])
    return response.text


# ──────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────

def run_stt(audio_path: str, model_id: str, duration_min: float) -> dict:
    print(f"  [{model_id}]...", end=" ", flush=True)
    t0 = time.time()
    error = None
    text = ""

    try:
        if model_id.startswith("openai/"):
            openai_model = model_id.split("/", 1)[1]
            text = transcribe_openai(audio_path, openai_model, prompt=KEYTERMS)

        elif model_id.startswith("groq/"):
            if not os.environ.get("GROQ_API_KEY"):
                raise RuntimeError("GROQ_API_KEY not set")
            groq_model = model_id.split("/", 1)[1]
            text = transcribe_groq(audio_path, groq_model)

        elif model_id == "deepgram/nova-3":
            if not os.environ.get("DEEPGRAM_API_KEY"):
                raise RuntimeError("DEEPGRAM_API_KEY not set")
            keyterms_list = [k.strip() for k in KEYTERMS.split(",")]
            text = transcribe_deepgram(audio_path, keyterms_list)

        elif model_id == "elevenlabs/scribe-v2":
            if not os.environ.get("ELEVENLABS_API_KEY"):
                raise RuntimeError("ELEVENLABS_API_KEY not set")
            keyterms_list = [k.strip() for k in KEYTERMS.split(",")]
            text = transcribe_elevenlabs(audio_path, keyterms=keyterms_list, num_speakers=2)

        elif model_id == "google/gemini-2.5-flash":
            if not os.environ.get("GOOGLE_API_KEY"):
                raise RuntimeError("GOOGLE_API_KEY not set")
            text = transcribe_gemini(audio_path)

        else:
            raise RuntimeError(f"Unknown model: {model_id}")

    except Exception as e:
        error = str(e)

    elapsed = time.time() - t0
    cost = (COSTS_PER_MIN.get(model_id, 0.0) * duration_min) if not error else 0.0

    status = f"ERROR: {error}" if error else f"done ({elapsed:.1f}s, {len(text):,} chars)"
    print(status)

    result = {
        "model": model_id,
        "text": text if not error else f"ERROR: {error}",
        "chars": len(text),
        "elapsed_s": elapsed,
        "cost_usd": cost,
        "error": error,
    }

    return result


# ──────────────────────────────────────────────
# Judge
# ──────────────────────────────────────────────

JUDGE_SYSTEM = """You are an expert evaluator of Hebrew/English code-switched speech-to-text quality.
You compare a raw STT transcript against a human-corrected ground truth and score on rubrics.
Respond only with valid JSON."""

JUDGE_PROMPT = """Evaluate this raw STT transcript against the ground truth.

## Ground Truth (human-corrected)
{ground_truth}

## STT Output to evaluate
{stt_output}

## Equivalence Rule
Hebrew and English variants of the same word count as CORRECT:
- "default" = "דיפולט" ✓  |  "evaluation" = "אבולואציה" ✓  |  "reusable" = "ריליזיבל" ✓
A phonetically unrelated word is always wrong.

## Rubrics

**Named Entity Accuracy (1-10, strict)**
People, products, tools, org names. Key entities:
Arin (ארין), Eli (אילי), Gil (גיל), Idan (עידן), Himanshu, Ben Stiller (בן סטילר),
Common Agentic Core (CAF), LangGraph, LangFuse, Concon, Converge Consumer, swarm, A2A, IIH.
10 = all correct, 1 = most garbled.

**Semantic Integrity (1-10)**
Did any mishearing flip meaning?
Key example: "האדום" (red line on board) → "האדם" (a person) = critical error.
"feedback loop" → nonsense = semantic destruction.
10 = no meaning-flips, 1 = several critical breaks.

**Lexical Fidelity (1-10, with equivalence rule)**
Word-level accuracy across the whole transcript.
10 = virtually all words correct, 1 = heavily garbled.

**Completeness (1-10)**
Did all spoken content make it into the transcript? Penalise missing segments.
10 = full coverage, 1 = large sections missing.

## Output
Return ONLY this JSON:
{{
  "named_entity_accuracy": {{"score": N, "notes": "examples of hits/misses"}},
  "semantic_integrity": {{"score": N, "notes": "examples"}},
  "lexical_fidelity": {{"score": N, "notes": "examples"}},
  "completeness": {{"score": N, "notes": "brief"}},
  "overall": N,
  "summary": "1-2 sentences"
}}
Overall = holistic judgment, weight named_entity + semantic heavily."""


def judge_transcript(stt_text: str, ground_truth: str) -> dict:
    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=os.environ["OPENROUTER_API_KEY"])
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": JUDGE_PROMPT.format(
                ground_truth=ground_truth,
                stt_output=stt_text,
            )},
        ],
    )
    text = response.choices[0].message.content
    return json.loads(text[text.find("{"):text.rfind("}") + 1])


# ──────────────────────────────────────────────
# Output
# ──────────────────────────────────────────────

def print_table(results: list[dict], duration_min: float) -> None:
    rubrics = [
        ("named_entity_accuracy", "Entities"),
        ("semantic_integrity",    "Semantic"),
        ("lexical_fidelity",      "Lexical"),
        ("completeness",          "Complete"),
    ]
    col_model = max(len(r["model"]) for r in results) + 2
    col_w = 9
    col_cost = 10
    col_time = 8

    header = (f"{'Model':<{col_model}}"
              + "".join(f"{s:>{col_w}}" for _, s in rubrics)
              + f"{'Overall':>{col_w}}"
              + f"{'Time':>{col_time}}"
              + f"{'Cost ($)':>{col_cost}}")
    sep = "─" * len(header)
    print()
    print(sep)
    print(header)
    print(sep)
    for r in results:
        if r.get("error"):
            print(f"{r['model']:<{col_model}} {'ERROR: ' + r['error']}")
            continue
        row = f"{r['model']:<{col_model}}"
        for rubric, _ in rubrics:
            score = r.get(rubric, {}).get("score", "—")
            row += f"{score:>{col_w}}"
        row += f"{r.get('overall', '—'):>{col_w}}"
        row += f"{r['elapsed_s']:.1f}s".rjust(col_time)
        cost_s = f"${r['cost_usd']:.4f}" if r['cost_usd'] else "n/a"
        row += f"{cost_s:>{col_cost}}"
        print(row)
    print(sep)
    print()


def save_report(results: list[dict], output_path: Path, ground_truth_label: str) -> None:
    lines = [
        "# STT Model Comparison — Raw Transcript Quality\n",
        f"Judge: {JUDGE_MODEL}  |  Test reference: {ground_truth_label}\n",
    ]
    rubrics = ["named_entity_accuracy", "semantic_integrity", "lexical_fidelity", "completeness"]
    for r in results:
        if r.get("error"):
            lines.append(f"## {r['model']} — ERROR\n{r['error']}\n\n---\n")
            continue
        lines.append(f"## {r['model']}\n")
        lines.append(f"**Overall: {r.get('overall','—')}/10** — {r.get('summary','')}\n")
        for rubric in rubrics:
            d = r.get(rubric, {})
            lines.append(f"- **{rubric.replace('_',' ').title()}**: {d.get('score','—')}/10 — {d.get('notes','')}")
        lines.append(f"\n### Raw Transcript\n```\n{r.get('text','')[:1000]}{'...' if len(r.get('text',''))>1000 else ''}\n```\n")
        lines.append("\n---\n")
    output_path.write_text("\n".join(lines), encoding="utf-8")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

DEFAULT_MODELS = [
    "openai/gpt-4o-transcribe",
    "openai/gpt-4o-mini-transcribe",
    "openai/whisper-1",
    "elevenlabs/scribe-v2",
    "groq/whisper-large-v3",
    "groq/whisper-large-v3-turbo",
    "deepgram/nova-3",
    "google/gemini-2.5-flash",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare STT models on raw transcript quality")
    parser.add_argument("audio_file", help="Audio file path")
    parser.add_argument("--models", help="Comma-separated model list")
    parser.add_argument("--skip-judge", action="store_true", help="Skip LLM judging, just transcribe")
    parser.add_argument(
        "--ground-truth",
        default=None,
        help="Test reference file (default: <repo>/test_references/<parent-of-audio>/ground_truth.rtl.md)",
    )
    args = parser.parse_args()

    audio_path = args.audio_file
    if not os.path.exists(audio_path):
        print(f"ERROR: File not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    models = [m.strip() for m in args.models.split(",")] if args.models else DEFAULT_MODELS

    duration_s = audio_duration_seconds(audio_path)
    duration_min = duration_s / 60

    print(f"=== STT Model Comparison ===")
    print(f"Audio:    {audio_path}  ({duration_min:.1f} min)")
    print(f"Models:   {len(models)}")
    print()

    # Transcribe with each model
    print("Transcribing...")
    results = []
    for model_id in models:
        result = run_stt(audio_path, model_id, duration_min)
        results.append(result)

    if args.skip_judge:
        for r in results:
            print(f"\n{'='*60}\n{r['model']}\n{'='*60}")
            print(r["text"][:500] + "..." if len(r["text"]) > 500 else r["text"])
        return

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
    print(f"\nJudging transcripts ({JUDGE_MODEL})...")
    for r in results:
        if r.get("error"):
            continue
        print(f"  [{r['model']}]...", end=" ", flush=True)
        t0 = time.time()
        try:
            scores = judge_transcript(r["text"], ground_truth)
            r.update(scores)
            print(f"overall={scores['overall']}  ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"ERROR: {e}")

    print_table(results, duration_min)

    output_path = Path(audio_path).parent / "stt_compare.md"
    save_report(results, output_path, gt_label)
    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    main()
