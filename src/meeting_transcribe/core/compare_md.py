"""Parse compare-models markdown output for downstream tools (e.g. evaluate-models)."""

from __future__ import annotations

import re
from pathlib import Path


def parse_compare_model_sections(compare_md: Path) -> dict[str, str]:
    """Extract model id → structured notes body from compare.md."""
    text = compare_md.read_text(encoding="utf-8")
    if "## Results by Model" not in text:
        raise ValueError(f"{compare_md}: missing '## Results by Model' section")
    _, rest = text.split("## Results by Model", 1)
    parts = re.split(r"\n### ", rest)
    out: dict[str, str] = {}
    for chunk in parts[1:]:
        if not chunk.strip():
            continue
        first_nl = chunk.find("\n")
        if first_nl == -1:
            continue
        header = chunk[:first_nl].strip()
        header = re.sub(r"\s*\*\(cached\)\*\s*$", "", header).strip()
        body = chunk[first_nl + 1 :]
        body = re.sub(r"\n\*[0-9,]+ in \+ [0-9,]+ out tokens[^\n]*\n*\s*$", "", body, flags=re.DOTALL)
        body = body.rstrip()
        if body.endswith("---"):
            body = body[:-3].rstrip()
        out[header] = body.strip()
    if not out:
        raise ValueError(f"{compare_md}: no model sections found under '## Results by Model'")
    return out
