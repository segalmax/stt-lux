import re
from pathlib import Path


def extract_vocab_from_file(path: str, max_chars: int = 800) -> str:
    """Extract key proper nouns and tech terms from a context file for transcription hints."""
    text = Path(path).read_text(encoding="utf-8")
    english_terms = re.findall(r"\b[A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+)*\b", text)
    seen = set()
    terms = []
    for t in english_terms:
        if len(t) > 2 and t.lower() not in ("the", "and", "for", "you", "are", "not", "this", "that", "with"):
            if t not in seen:
                seen.add(t)
                terms.append(t)
    vocab = ", ".join(terms)
    return vocab[:max_chars]
