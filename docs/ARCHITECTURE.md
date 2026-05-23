# meeting-transcribe — architecture

Python package: **`meeting_transcribe`** (`pip install -e .`). Entry points are console scripts defined in [`pyproject.toml`](../pyproject.toml).

## Package layout

```mermaid
flowchart TB
  subgraph cli [cli]
    transcribe[transcribe.py]
    compare[compare.py]
    compare_stt[compare_stt.py]
    evaluate[evaluate.py]
  end
  subgraph core [core]
    orch[orchestrate_context.py]
    stt[stt.py]
    structure[structure.py]
    prompts[prompts.py]
    pipeline_report[pipeline_report.py]
    audio[audio.py]
    run_paths[run_paths.py]
    vocab[vocab.py]
    paths[paths.py]
    hashing[hashing.py]
    compare_md[compare_md.py]
  end
  transcribe --> orch
  transcribe --> stt
  transcribe --> structure
  transcribe --> pipeline_report
  compare --> orch
  compare --> stt
  compare --> structure
  orch --> prompts
  stt --> audio
  structure --> prompts
```

## Main pipeline (`transcribe` CLI)

Default path when `--context` is set: **orchestrator → STT → speaker labels → reconciled transcript (Opus) → structured notes**. Reference docs (`--ref-doc`) feed the orchestrator only (jargon / entities), not as the transcript of *this* recording. Skip reconciliation with `--no-reconcile` (structured notes then use the labeled transcript).

```mermaid
flowchart LR
  subgraph inputs [Inputs]
    A[audio_m4a]
    C[meeting_situation]
    R[ref_docs_optional]
  end
  subgraph stage1 [Stage_1]
    O[Context_orchestrator_OpenRouter]
  end
  subgraph stage2 [Stage_2]
    S[Speech_to_text_ElevenLabs_Scribe_v2]
  end
  subgraph stage3 [Stage_3]
    L[Speaker_labels_LLM_OpenRouter]
  end
  subgraph stage3b [Stage_3b]
    Rc[Reconcile_LLM_OpenRouter_Opus]
  end
  subgraph stage4 [Stage_4]
    N[Structured_notes_LLM_OpenRouter]
  end
  subgraph out [Output]
    M[notes_md]
  end
  A --> S
  C --> O
  R --> O
  O -->|keyterms_num_speakers_hint| S
  O -->|speaker_names_roles_llm_context| L
  O -->|bundled_context| N
  S -->|diarized_text_with_optional_timestamps| L
  L -->|labeled_transcript| Rc
  Rc -->|reconciled_transcript| N
  N --> M
```

### Data passed between stages (conceptual)

| Stage | Produces | Consumed downstream |
|-------|----------|---------------------|
| Orchestrator | `keyterms`, `speaker_names`, `speaker_roles`, `llm_context` | STT (keyterms + `num_speakers`); speaker + reconcile + structure LLMs (full **meeting situation + participants + jargon** bundle via `format_downstream_llm_context`) |
| STT | `[speaker_id] [H:MM:SS-H:MM:SS]?` lines, word timings in API response | Speaker-label prompt |
| Speaker labels | `Name: [span] …` (same words, resolved names) | Reconcile prompt (unless `--no-reconcile`) |
| Reconcile | Same format; splits mis-merged long lines using roles + dialogue | Structured notes prompt; `notes.md` **Reconciled transcript** section |
| Structured notes | Markdown sections (summary, actions, decisions) | `notes.md` |

## Sequence (external services)

```mermaid
sequenceDiagram
  participant User
  participant CLI as transcribe_CLI
  participant OR as OpenRouter
  participant EL as ElevenLabs_STT
  User->>CLI: audio + context + optional_ref_doc
  CLI->>OR: orchestrator_chat JSON_out
  OR-->>CLI: keyterms_speaker_roles_llm_context
  CLI->>EL: multipart_audio_keyterms_diarize
  EL-->>CLI: words_with_speaker_id_and_timestamps
  CLI->>OR: speaker_label_chat full_transcript
  OR-->>CLI: labeled_markdown_lines
  CLI->>OR: reconcile_chat labeled_transcript
  OR-->>CLI: reconciled_markdown_lines
  CLI->>OR: structure_chat reconciled_transcript
  OR-->>CLI: structured_notes
  CLI-->>User: notes_md
```

## Chunking and limits

- **ElevenLabs (default STT):** whole file in one request; lines may include **segment time spans** when the API returns `start` / `end` per word ([`stt.py`](../src/meeting_transcribe/core/stt.py)).
- **OpenAI STT path (non-default):** long files may be **chunked** via [`audio.chunk_audio`](../src/meeting_transcribe/core/audio.py) when over size/duration caps in [`stt.transcribe`](../src/meeting_transcribe/core/stt.py).
- **Speaker labels / structure LLMs:** `max_tokens` and **`finish_reason`** handling are in [`structure.py`](../src/meeting_transcribe/core/structure.py) and [`config.py`](../src/meeting_transcribe/core/config.py); truncation should **fail fast** rather than silently cut.

## Other CLIs

```mermaid
flowchart LR
  compare_models[compare_models] --> one_stt[one_STT_pass]
  one_stt --> many_llm[many_structure_models]
  compare_stt[compare_stt] --> bench[STT_benchmarks_vs_optional_ground_truth]
  evaluate_models[evaluate_models] --> metrics[judge_metrics]
```

- **`compare-models`:** one transcript, several structuring models → `compare.md`.
- **`compare-stt`:** STT comparison / evaluation helpers (see module docstrings).
- **`evaluate-models`:** model evaluation flow (ground truth under test harness paths only).

## Configuration touchpoints

| File | Role |
|------|------|
| [`config.py`](../src/meeting_transcribe/core/config.py) | Default STT/OpenRouter/orchestrator models, cost rates, `SPEAKER_LABEL_MAX_TOKENS`, `STRUCTURE_NOTES_MAX_TOKENS` |
| [`prompts.py`](../src/meeting_transcribe/core/prompts.py) | System/user templates for structure + speaker labeling + optional correction |
| [`orchestrate_context.py`](../src/meeting_transcribe/core/orchestrate_context.py) | Orchestrator JSON schema, pipeline contract text, `format_downstream_llm_context` |

## Related docs

- Benchmark-oriented notes (older stage naming in places): [`PIPELINE_INSIGHTS.md`](PIPELINE_INSIGHTS.md)
