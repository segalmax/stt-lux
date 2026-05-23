"""Repository paths and case-local reference paths."""

from pathlib import Path


def project_root() -> Path:
    here = Path(__file__).resolve().parent
    for p in [here, *here.parents]:
        if (p / "pyproject.toml").exists():
            return p
    raise RuntimeError(
        "meeting_transcribe: could not find project root (no pyproject.toml above %s)" % here
    )


def resolve_ground_truth_path(audio_path: str, explicit: str | None = None) -> Path:
    """Test-only: human-authored reference for scoring (evaluate-models / compare-stt judge).

    Production `transcribe` never calls this.

    Default: ``<project>/test_references/<case_id>/ground_truth.rtl.md`` where ``case_id`` is the
    audio file's parent directory name (e.g. ``cases/img_6423/tail.m4a`` → ``img_6423``).
    Override with ``--ground-truth PATH`` for any path or filename.
    """
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Test reference file not found: {p}")
        return p
    audio = Path(audio_path).expanduser().resolve()
    case_id = audio.parent.name
    default_gt = project_root() / "test_references" / case_id / "ground_truth.rtl.md"
    if default_gt.is_file():
        return default_gt
    raise FileNotFoundError(
        f"Test reference missing: {default_gt} (for audio under …/{case_id}/) or pass --ground-truth PATH"
    )
