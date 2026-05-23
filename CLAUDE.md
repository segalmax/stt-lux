# stt-lux — Claude Project Memory

Hebrew/English ("Hebrish") code-switched meeting transcription pipeline.
Audio → ElevenLabs STT → Speaker labels → Reconcile → Structured notes.

## Critical rules

- **Never commit `.env`** — contains live API keys (`OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `ELEVENLABS_API_KEY`).
- **Never commit `cases/`** — contains PII (audio, transcripts). Already gitignored.
- **Never commit `test_references/`** — contains PII. Already gitignored.
- **Always use the venv**: `/Users/segalmax/misc-claude/.venv/bin/python`. System Python is too old for union type hints.
- **`--language` is required** on every `transcribe` call — no default, by design. Pass `he`, `en`, or `auto` (auto-detect, best for code-switched).

## Commands

```bash
# Install (one-time)
pip install -e .

# Run transcription
/Users/segalmax/misc-claude/.venv/bin/python -m meeting_transcribe.cli.transcribe \
  cases/<case_id>/audio.m4a \
  --context "..." --num-speakers 2 --language auto --run

# Load .env before running (API keys)
set -a && source .env && set +a

# Extract audio from video (create a new case)
ffmpeg -i "source.mp4" -vn -acodec aac -b:a 128k cases/<case_id>/audio.m4a -y

# Push changes
git add -p && git commit && git push
```

## Project layout

```
src/meeting_transcribe/
  cli/
    transcribe.py        # Main 4-stage pipeline CLI — entry point
    compare.py           # LLM model comparison tool
    compare_stt.py       # STT model comparison tool
    evaluate.py          # Evaluation against ground truth
  core/
    config.py            # All model defaults and cost constants — edit here first
    stt.py               # ElevenLabs + OpenAI STT; transcribe_elevenlabs() + transcribe()
    orchestrate_context.py  # Stage 1: LLM → keyterms, speaker_names, speaker_roles, llm_context
    structure.py         # Stages 3-5: speaker labels, reconcile, structured notes (OpenRouter)
    audio.py             # ffmpeg helpers: duration, chunking
    paths.py             # case_id = parent directory name of audio file
    run_paths.py         # new_run_dir() → cases/<id>/runs/<YYYYMMDD-HHMMSS>/
    pipeline_report.py   # Markdown preamble with cost/token summary
cases/                   # <case_id>/audio.m4a + runs/<timestamp>/notes.md  [gitignored]
test_references/         # Ground truth for evaluation  [gitignored — PII]
docs/
  PIPELINE_INSIGHTS.md   # Benchmark results, STT comparison tables, cost breakdown
  ARCHITECTURE.md
```

## Pipeline stages

1. **Orchestrator** (`--context` / `--ref-doc` / `--ref-dir`) — OpenRouter LLM → keyterms for STT, speaker names/roles, llm_context for downstream stages.
2. **STT** — ElevenLabs Scribe v2 (default, best for Hebrew). Returns word-level diarized lines: `[speaker_0] [0:00-0:05] words...`
3. **Speaker labels** — LLM maps `[speaker_N]` → real names. Skipped if no `--context`.
4. **Reconcile** — LLM splits mis-merged speaker turns. Skip with `--no-reconcile`.
5. **Structured notes** — LLM → summary, action items, decisions. Skip with `--no-structure`.

## Key design decisions

- **Language**: `--language auto` (omit from ElevenLabs) beats `--language he` for code-switched audio per benchmark. `--language` is required, no default.
- **Cache**: SHA256 hash (first 16 chars) of audio file = stable cache key in `.transcribe_cache.json`.
- **Chunking**: Files >25MB or >20min are auto-chunked; ElevenLabs handles 3h+ files.
- **OpenRouter**: All LLM calls go through `https://openrouter.ai/api/v1` with `OPENROUTER_API_KEY`.
- **Case ID**: derived from the audio file's parent directory name (lowercase, underscores).

## API keys (in `.env`)

| Variable | Service |
|---|---|
| `ELEVENLABS_API_KEY` | ElevenLabs STT (Scribe v2) |
| `OPENROUTER_API_KEY` | All LLM stages (orchestrator, labels, reconcile, notes) |
| `OPENAI_API_KEY` | OpenAI STT fallback (gpt-4o-transcribe) |

## Adding a new case

```bash
mkdir -p cases/<case_id>
ffmpeg -i "/path/to/video.mp4" -vn -acodec aac -b:a 128k cases/<case_id>/audio.m4a -y
```
