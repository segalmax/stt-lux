# Hebrish Meeting Transcription Pipeline — Insights & Results

> Benchmark: Hebrew/English code-switched meeting — assets under `cases/img_6423/` (`tail.m4a`, `video.mov`)  
> Judge: `anthropic/claude-opus-4-5` via OpenRouter  
> Test reference (benchmark only; not part of production runs): `cases/img_6423/ground_truth.rtl.md`

---

## Pipeline Architecture

```
Audio (.m4a)
     │
     ▼
┌─────────────────────────────────────────┐
│  Stage 1 — STT                          │
│  ElevenLabs Scribe v2  (default)        │
│  keyterms + num_speakers + language     │
└─────────────────────────────────────────┘
     │ raw transcript (diarized)
     ▼
┌─────────────────────────────────────────┐
│  Stage 2 — Holistic Correction  (opt.)  │
│  LLM reads full transcript + context   │
│  + keyterms → fixes STT mishearings    │
│  Default: google/gemini-2.5-flash      │
└─────────────────────────────────────────┘
     │ corrected transcript
     ▼
┌─────────────────────────────────────────┐
│  Stage 3 — Structuring                  │
│  LLM → organized markdown meeting notes│
│  Default: anthropic/claude-opus-4-5    │
└─────────────────────────────────────────┘
     │
     ▼
  notes.md
```

**Key design decisions:**
- Stage 1 language param intentionally set to `he` (Hebrew-dominant) rather than locked, to handle code-switching
- Stage 2 is optional but strongly recommended — lifts Semantic scores from ~4 → 9 across all Stage 3 models
- Stages 2 and 3 could be collapsed into a single prompt for production use (no significant quality loss, one fewer API call)
- All results cached by SHA256 file hash + model ID in `.transcribe_cache.json`

---

## Stage 1 — STT Model Comparison

Scores out of 10, judged against human-corrected ground truth.

| Model | Entities | Semantic | Lexical | Complete | **Overall** | Cost/10min |
|-------|:--------:|:--------:|:-------:|:--------:|:-----------:|:----------:|
| **elevenlabs/scribe-v2** ⭐ | **6** | **6** | **6** | **8** | **6** | $0.04 |
| openai/whisper-1 | 4 | 5 | 5 | 7 | 5 | $0.06 |
| openai/gpt-4o-transcribe | 5 | 4 | 5 | 4 | 4 | $0.06 |
| openai/gpt-4o-mini-transcribe | 3 | 4 | 3 | 3 | 3 | $0.03 |

**Winner: ElevenLabs Scribe v2** — best on every dimension, and cheaper than gpt-4o-transcribe.  
Consistent with Artificial Analysis leaderboard (2.3% AA-WER, ranked #1).

### Why gpt-4o-transcribe underperforms

Despite being OpenAI's newest model, it scored lower on **Semantic** (4/10) and **Completeness** (4/10).  
Root cause: it tends to hallucinate and collapse content in noisy/code-switched audio.  
Whisper-1 (the older model) actually scored better on Completeness (7/10) — it transcribes more verbatim.

### ElevenLabs Scribe v2 — what keyterms fixed

Before keyterms: Entities 5/10  
After keyterms: Entities 6/10 (+20%)  
Correctly anchored: `LangGraph`, `LangFuse`, `Concon`, `Common Agentic Core`, `Ben Stiller`

### Remaining errors (phonetic — keyterms can't fix)

| Error | Correct | Type |
|-------|---------|------|
| `מתחתן נורא גרוע` (gets married very poorly) | `מתפקד נורא גרוע` (functions very poorly) | Semantic flip |
| `Ben Grad` | `RAG` | Phonetic noise |
| `QPR` | `KPI` | Phonetic confusion |
| Garbled opening | — | Overlapping speech |

These require either cleaner audio or the correction pass (Stage 2).

---

## Stage 2 — Correction Pass Impact

The correction pass uses an LLM to read the full transcript holistically before fixing anything.  
Tested with `google/gemini-2.5-flash` as corrector.

| Rubric | Before correction | After correction | Δ |
|--------|:-----------------:|:----------------:|:-:|
| Named Entity | 5 | 7–8 | +2–3 |
| **Semantic Integrity** | **4** | **9** | **+5** |
| Completeness | 4 | 7–8 | +3–4 |
| Lexical Fidelity | 5 | 7–8 | +2–3 |

The semantic jump (+5) is the biggest win — it catches meaning-destroying mishearings that STT produces under noise.

### Correction vs. Collapsing into Stage 3

| Approach | When to use |
|----------|------------|
| **Separate correction pass** | Benchmarking (correct once, run N structuring models with identical input) |
| **Single-pass (correct + structure)** | Production use — give context + keyterms directly to structuring model |

For production, collapse: `"Fix STT errors using these keyterms, then structure as meeting notes"` in one prompt.

---

## Stage 3 — Structuring Model Comparison

Input: gpt-4o-transcribe raw transcript (without correction pass).  
Scores weighted toward Named Entity + Semantic per rubric design.

| Model | Entities | Semantic | Complete | Lexical | Halluc. | **Overall** |
|-------|:--------:|:--------:|:--------:|:-------:|:-------:|:-----------:|
| **anthropic/claude-opus-4-7** | 7 | 9 | **9** | 8 | 7 | **8** |
| **openai/gpt-4.1** | **8** | 9 | 8 | 8 | 7 | **8** |
| openai/gpt-4.1-mini | 7 | 9 | 7 | 8 | 7 | 7 |
| google/gemini-2.5-flash | 7 | 9 | 7 | 8 | **8** | 7 |

**Top tier:** Claude Opus 4.7 and GPT-4.1 tied at 8/10.  
**Note:** All models achieve Semantic 9/10 — the structuring LLM largely compensates for STT's semantic errors. This is Stage 2's job in the separate-pass design.

### Notable model quirks

| Model | Known issue |
|-------|------------|
| claude-opus-4-7 | Renders `Concon` as `Quantcon` (consistent mishearing passed through from STT) |
| openai/gpt-4.1 | Best Named Entity (8/10); misses swarm/A2A and follow-up prompts feature |
| google/gemini-2.5-flash | Misreads `יש פה 25 חברה` (25 companies) as "25 team members" |
| openai/gpt-4.1-mini | Emits random German word (`Ansprechpartner`) — minor hallucination |

---

## ElevenLabs Scribe v2 — Parameter Reference

POST `https://api.elevenlabs.io/v1/speech-to-text`  
Authentication: `xi-api-key` header

| Parameter | Type | Default | Impact | Notes |
|-----------|------|---------|--------|-------|
| `model_id` | string | — | — | Always `"scribe_v2"` |
| `keyterms` | string[] | — | **High** (+20% cost) | Up to 1000 terms, 50 chars each. **Primary lever for document-context injection** |
| `num_speakers` | int | auto | **High** | Tell model how many speakers to expect — improves diarization |
| `language_code` | ISO-639-1 | auto | **Medium** | `"he"` for Hebrew-dominant; auto-detect handles code-switching but explicit is better |
| `diarize` | bool | false | **High** | Enable speaker labels (`[speaker_0]`, `[speaker_1]`) |
| `tag_audio_events` | bool | true | Medium | Set `false` for meetings — suppresses `[laughter]`, `[footsteps]` noise in transcript |
| `temperature` | float 0–2 | — | Low | Set `0` for deterministic/reproducible results |
| `seed` | int | — | Low | Pair with `temperature=0` for exact reproducibility |
| `no_verbatim` | bool | false | Medium | Removes filler words — use carefully in Hebrew (fillers carry meaning) |
| `entity_detection` | — | off | Low | Detects PII/PHI/PCI (+30% cost) — not useful for meeting notes |
| `use_multi_channel` | bool | false | High* | For multi-track recordings — not applicable to mono phone recordings |

### Document-context injection (future use)

When you have pre-meeting documents (agenda, technical specs, participant list):

```python
def keyterms_from_documents(docs: list[str]) -> list[str]:
    """Extract entity names and tech terms from documents to pass as keyterms."""
    import re
    terms = set()
    for doc in docs:
        # English capitalized terms (names, products, acronyms)
        terms.update(re.findall(r'\b[A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+)*\b', doc))
        # Hebrew names (if pre-labeled)
        # ... domain-specific extraction
    return [t[:50] for t in terms if len(t) > 2][:1000]

# Usage
text = transcribe_elevenlabs(
    audio_path,
    keyterms=keyterms_from_documents(agenda_docs),
    num_speakers=len(participants),
)
```

---

## Cost Summary (10-minute meeting, full pipeline)

| Component | Model | Cost |
|-----------|-------|------|
| Stage 1 STT | ElevenLabs Scribe v2 | ~$0.04 |
| Stage 2 Correction | gemini-2.5-flash | ~$0.01 |
| Stage 3 Structuring | claude-opus-4-5 | ~$0.05–0.15 |
| **Total** | | **~$0.10–0.20** |

For comparison: gpt-4o-transcribe (Stage 1 only) = $0.06 with worse results.

---

## Key Takeaways

1. **ElevenLabs Scribe v2 is the best available STT for Hebrew/English code-switching** — wins on all rubrics, cheaper than gpt-4o-transcribe, #1 on Artificial Analysis leaderboard.
2. **The bottleneck is STT, not the structuring LLM** — all Stage 3 models perform similarly once given a clean transcript. Invest in Stage 1/2 quality.
3. **Keyterms are the primary context injection mechanism** — no native "prompt" field in Scribe v2; pass extracted entity names from documents.
4. **The correction pass (Stage 2) is the biggest quality lever** — +5 on Semantic Integrity for a ~$0.01 cost.
5. **For production: collapse Stages 2+3** into a single LLM call with keyterms in the system prompt.
6. **Acoustic noise errors can't be fixed by keyterms** — phonetically similar Hebrew words (מתפקד/מתחתן) need cleaner audio or manual review.
