import json
import os
import urllib.error
import urllib.request
import uuid

from openai import OpenAI

from meeting_transcribe.core.audio import audio_duration_seconds, chunk_audio


def _fmt_hms_range(start_s: float | None, end_s: float | None) -> str:
    """Format [H:MM:SS-H:MM:SS] for ElevenLabs word start/end (seconds)."""

    def one(t: float | None) -> str:
        if t is None:
            return "?"
        s = max(0.0, float(t))
        h, rem = divmod(int(s), 3600)
        m, sec = divmod(rem, 60)
        if h:
            return f"{h:d}:{m:02d}:{sec:02d}"
        return f"{m:d}:{sec:02d}"

    return f"[{one(start_s)}-{one(end_s)}]"
from meeting_transcribe.core.config import (
    DEFAULT_STT_MODEL,
    MAX_DURATION_S,
    MAX_FILE_MB,
    TRANSCRIPTION_COST_PER_MIN,
    TRANSCRIPTION_COSTS_PER_MIN,
)


def transcribe_elevenlabs(
    audio_path: str,
    keyterms: list[str],
    num_speakers: int,
    language_code: str | None = "he",
) -> tuple[str, dict[str, float | int | None]]:
    """Transcribe via ElevenLabs Scribe v2 — #1 STT model (AA-WER 2.3%).

    Returns (text, meta) where meta has last_word_end_s, word_count (for STT coverage checks).
    Each line: [speaker_id] [H:MM:SS-H:MM:SS] words... when API provides start/end per word.
    """
    api_key = os.environ["ELEVENLABS_API_KEY"]

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    boundary = uuid.uuid4().hex

    def field(name: str, value: str) -> bytes:
        return (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
            f"{value}\r\n"
        ).encode()

    lang_field = field("language_code", language_code) if language_code else b""
    body = (
        field("model_id", "scribe_v2")
        + field("diarize", "true")
        + lang_field
        + field("tag_audio_events", "false")
        + field("temperature", "0")
    )
    for term in keyterms[:1000]:
        body += field("keyterms", term[:50])
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
        with urllib.request.urlopen(req, timeout=900) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"ElevenLabs HTTP {e.code}: {e.read().decode()[:300]}")

    if "words" in data and data["words"]:
        lines = []
        current_speaker = None
        current_words: list[str] = []
        seg_start: float | None = None
        seg_end: float | None = None
        last_word_end: float | None = None
        word_count = 0
        for w in data["words"]:
            if w.get("type") != "word":
                continue
            word_count += 1
            spk = w.get("speaker_id", "speaker_0")
            w_start = w.get("start")
            w_end = w.get("end")
            if w_end is not None:
                last_word_end = float(w_end)
            if spk != current_speaker:
                if current_words and current_speaker is not None:
                    span = (
                        f" {_fmt_hms_range(seg_start, seg_end)}"
                        if seg_start is not None and seg_end is not None
                        else ""
                    )
                    lines.append(f"[{current_speaker}]{span} " + " ".join(current_words))
                current_speaker = spk
                current_words = []
                seg_start = float(w_start) if w_start is not None else None
                seg_end = float(w_end) if w_end is not None else None
            current_words.append(w.get("text", ""))
            if w_start is not None:
                seg_start = float(w_start) if seg_start is None else seg_start
            if w_end is not None:
                seg_end = float(w_end)
        if current_words and current_speaker is not None:
            span = (
                f" {_fmt_hms_range(seg_start, seg_end)}"
                if seg_start is not None and seg_end is not None
                else ""
            )
            lines.append(f"[{current_speaker}]{span} " + " ".join(current_words))
        text = "\n".join(lines)
        meta: dict[str, float | int | None] = {
            "last_word_end_s": last_word_end,
            "word_count": word_count,
        }
        return text, meta

    return data.get("text", ""), {"last_word_end_s": None, "word_count": 0}


def transcribe_file(client: OpenAI, path: str, prompt: str | None = None) -> str:
    """Transcribe a single audio file via OpenAI. No language lock — critical for Hebrew/English."""
    kwargs = dict(model="gpt-4o-transcribe", file=None)
    if prompt:
        kwargs["prompt"] = prompt
    with open(path, "rb") as f:
        kwargs["file"] = f
        result = client.audio.transcriptions.create(**kwargs)
    return result.text


def transcribe(
    audio_path: str,
    vocab_prompt: str | None = None,
    stt_model: str | None = None,
    keyterms: list[str] | None = None,
    num_speakers: int = 2,
    language_code: str | None = "he",
) -> tuple[str, float, dict[str, float | int | None]]:
    """Transcribe audio. Returns (transcript, cost_usd, meta).

    meta (ElevenLabs): last_word_end_s, word_count — empty dict otherwise.
    language_code: BCP-47 code passed to ElevenLabs (e.g. "he", "en"). Pass None to omit
    (auto-detect) — recommended for Hebrew/English code-switched audio.
    """
    if stt_model is None:
        stt_model = DEFAULT_STT_MODEL

    if stt_model == "elevenlabs/scribe-v2" and not os.environ.get("ELEVENLABS_API_KEY"):
        raise RuntimeError("ELEVENLABS_API_KEY is required for elevenlabs/scribe-v2 (default STT). Set it or pass --stt-model for an OpenAI model.")
    if stt_model != "elevenlabs/scribe-v2" and not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for OpenAI STT models.")

    file_mb = os.path.getsize(audio_path) / (1024 * 1024)
    duration_s = audio_duration_seconds(audio_path)
    duration_min = duration_s / 60
    print(f"Audio file: {file_mb:.1f} MB, {duration_s:.1f}s ({duration_min:.2f} min)")

    meta: dict[str, float | int | None] = {}
    if stt_model == "elevenlabs/scribe-v2":
        print(f"Transcribing ({stt_model})...")
        text, meta = transcribe_elevenlabs(
            audio_path,
            keyterms=keyterms or [],
            num_speakers=num_speakers,
            language_code=language_code,
        )
    else:
        openai_client = OpenAI()  # requires OPENAI_API_KEY (checked above)
        if file_mb <= MAX_FILE_MB and duration_s <= MAX_DURATION_S:
            print(f"Transcribing ({stt_model})...")
            text = transcribe_file(openai_client, audio_path, vocab_prompt)
        else:
            reason = f"{MAX_FILE_MB}MB" if file_mb > MAX_FILE_MB else f"{MAX_DURATION_S}s"
            print(f"File exceeds {reason} limit — chunking ({stt_model})...")
            chunk_paths = chunk_audio(audio_path)
            parts = []
            for i, chunk_path in enumerate(chunk_paths, 1):
                print(f"  Transcribing chunk {i}/{len(chunk_paths)}...")
                parts.append(transcribe_file(openai_client, chunk_path, vocab_prompt))
                os.unlink(chunk_path)
            try:
                os.rmdir(os.path.dirname(chunk_paths[0]))
            except OSError:
                pass
            text = "\n\n".join(parts)

    cost = duration_min * TRANSCRIPTION_COSTS_PER_MIN.get(stt_model, TRANSCRIPTION_COST_PER_MIN)

    return text, cost, meta
