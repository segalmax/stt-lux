# STT model defaults and costs (per minute of audio)
DEFAULT_STT_MODEL = "elevenlabs/scribe-v2"
TRANSCRIPTION_COSTS_PER_MIN = {
    "elevenlabs/scribe-v2": 0.004,
    "gpt-4o-transcribe": 0.006,
    "gpt-4o-mini-transcribe": 0.003,
    "whisper-1": 0.006,
}
TRANSCRIPTION_COST_PER_MIN = TRANSCRIPTION_COSTS_PER_MIN["gpt-4o-transcribe"]

CHUNK_MINUTES = 10
MAX_FILE_MB = 25
MAX_DURATION_S = 1200  # gpt-4o-transcribe limit is 1400s; chunk at 20min to be safe

DEFAULT_OPENROUTER_MODEL = "anthropic/claude-opus-4-5"
DEFAULT_ORCHESTRATOR_MODEL = "anthropic/claude-opus-4-5"
DEFAULT_RECONCILE_MODEL = "anthropic/claude-opus-4-5"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# OpenRouter chat max_tokens — speaker labeling must fit full diarized text (Hebrew expands output).
SPEAKER_LABEL_MAX_TOKENS = 65536
RECONCILE_MAX_TOKENS = 131072
STRUCTURE_NOTES_MAX_TOKENS = 16384

REF_DOC_MAX_BYTES = 400_000
REF_DOCS_MAX_TOTAL_BYTES = 2_000_000
REF_DOC_SUFFIXES = (".md", ".txt", ".markdown")
