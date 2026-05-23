"""Per-run output folders next to case audio (runs/<timestamp>/ or explicit --output-dir)."""

from datetime import datetime
from pathlib import Path


def new_run_dir(audio_path: str) -> Path:
    """`<case>/runs/YYYYMMDD-HHMMSS/` (created)."""
    parent = Path(audio_path).resolve().parent
    run_dir = parent / "runs" / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def ensure_output_dir(path: str) -> Path:
    d = Path(path).expanduser().resolve()
    d.mkdir(parents=True, exist_ok=True)
    return d
